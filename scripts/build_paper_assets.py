#!/usr/bin/env python3
"""Build publication assets from completed, hash-bound local evidence only."""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs/paper/latex"
SOURCES = {
    "stage2": "output/variance-diagnostic-v1/stage2/summary.json",
    "stage3": "output/variance-diagnostic-v1/stage3/summary.json",
    "stage4": "output/variance-diagnostic-v1/stage4/summary.json",
    "stage4_aggregate": "output/variance-diagnostic-v1/stage4/aggregate-summary.json",
    "nll": "output/nll-scoring-diagnostic-v1/aggregate-summary.json",
    "webshop": "output/webshop-development-pilot-v1/aggregate-summary.json",
    "source_comparison": "output/source-comparison-v1/summary.json",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    previous = json.loads((ROOT / "docs/paper/EVIDENCE_SNAPSHOT_v0.2.json").read_text())
    for path, expected in {**previous["sources"], **previous["figures"]}.items():
        if sha(ROOT / path) != expected:
            raise ValueError("Evidence changed: " + path)
    data = {k: json.loads((ROOT / v).read_text()) for k, v in SOURCES.items()}
    if not all(d["complete"] for d in data.values()):
        raise ValueError("Only complete reports may enter the main paper")
    data["stage3"]["results"] = [
        dict(r, policy=r["arm"])
        for r in data["stage3"]["results"]
        if r["arm"] in ("random", "min_nll")
    ]
    for source in ("api", "generic"):
        effect = data["stage3"]["policy_effects"]["mean_success"]["nll_minus_random"][
            source
        ]
        effect["per_initialization"] = [
            dict(initialization_seed=i, mean_difference=v)
            for i, v in zip((101, 202, 303, 404), effect["per_initialization_mean"])
        ]
    figures = PAPER / "figures"
    generated = PAPER / "generated"
    figures.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "axes.axisbelow": True,
        }
    )
    colors = ["#147D92", "#C85B39"]
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.25), sharey=True)
    rows = []
    for col, stage in enumerate(("stage3", "stage4")):
        primary = [r for r in data[stage]["results"] if r["repeat"] == 0]
        if max(r["mean_success"] * 100 for r in primary) > 62:
            raise ValueError(
                "Plot range must retain all observed points and mean labels"
            )
        for row, source in enumerate(("api", "generic")):
            ax = axes[row, col]
            group = [r for r in primary if r["source"] == source]
            pairs = defaultdict(dict)
            for r in group:
                pairs[(r["initialization_seed"], r["sampling_seed"])][r["policy"]] = r
            means = [
                np.mean([p[policy]["mean_success"] for p in pairs.values()]) * 100
                for policy in ("random", "min_nll")
            ]
            for index, (key, pair) in enumerate(sorted(pairs.items())):
                values = [100 * pair[p]["mean_success"] for p in ("random", "min_nll")]
                offset = ((index % 4) - 1.5) * 0.022
                ax.plot(
                    np.array([0, 1]) + offset,
                    values,
                    color="#8B969E",
                    alpha=0.48,
                    lw=0.65,
                    zorder=1,
                )
                for x, value, color in zip((0, 1), values, colors):
                    ax.scatter(
                        x + offset,
                        value,
                        s=11,
                        color=color,
                        alpha=0.75,
                        linewidths=0,
                        zorder=2,
                    )
            for x, value in enumerate(means):
                ax.plot(
                    [x - 0.15, x + 0.15],
                    [value, value],
                    color="#152A38",
                    lw=2.3,
                    zorder=3,
                )
                ax.annotate(
                    f"{value:.1f}",
                    (x, value),
                    xytext=(-14 if x == 0 else 14, 12),
                    textcoords="offset points",
                    ha="right" if x == 0 else "left",
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5),
                )
            delta = means[1] - means[0]
            ax.set_title(
                ("Task matched" if stage == "stage3" else "Task + action matched")
                + f" / {source.upper()}",
                loc="left",
                fontweight="bold",
            )
            ax.set(
                xticks=[0, 1],
                xticklabels=["Random", "Low NLL"],
                xlim=(-0.5, 1.5),
                ylim=(-1, 65),
            )
            ax.grid(axis="y", color="#E5E9EC", lw=0.6)
            if col == 0:
                ax.set_ylabel("Task success (%)")
            ax.text(
                0.98,
                0.94,
                rf"$\Delta={delta:.2f}$ pp",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=.25", fc="white", ec="#D9E0E4", lw=0.5),
            )
            effect = data[stage]["policy_effects"]["mean_success"]["nll_minus_random"][
                source
            ]
            if not math.isclose(delta, effect["mean_difference"] * 100, abs_tol=1e-12):
                raise ValueError("Mean/effect mismatch")
            rows.append((stage, source, means, effect))
    fig.tight_layout(pad=0.7, h_pad=1.2, w_pad=1.2)
    fig.savefig(figures / "selection-outcomes.pdf")
    plt.close(fig)
    lines = []
    for stage, source, means, effect in rows:
        title = "Task matched" if stage == "stage3" else "Task + action"
        lines.append(
            f"{title} & {source.upper()} & {means[0]:.3f} & {means[1]:.3f} & {effect['mean_difference'] * 100:+.3f} & {effect['wins']}/{effect['ties']}/{effect['losses']} "
            + r"\\"
        )
    (generated / "selection-rows.tex").write_text(
        "\\newcommand{\\SelectionRows}{%\n"
        + "\n".join(line + "%" for line in lines)
        + "\n}\n"
    )
    effects = data["stage4"]["policy_effects"]["mean_success"]["nll_minus_random"]
    seed_lines = []
    for a, g in zip(
        effects["api"]["per_initialization"], effects["generic"]["per_initialization"]
    ):
        if a["initialization_seed"] != g["initialization_seed"]:
            raise ValueError("Seed pairing changed")
        seed_lines.append(
            f"{a['initialization_seed']} & {100 * a['mean_difference']:+.3f} & {100 * g['mean_difference']:+.3f} "
            + r"\\"
        )
    (generated / "initialization-rows.tex").write_text(
        "\\newcommand{\\InitializationRows}{%\n"
        + "\n".join(line + "%" for line in seed_lines)
        + "\n}\n"
    )
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.15), sharey=True)
    for ax, source in zip(axes, ("api", "generic")):
        for stage, color, label in [
            ("stage3", "#A7AFB5", "Task matched"),
            ("stage4", "#147D92", "Task + action"),
        ]:
            values = data[stage]["policy_effects"]["mean_success"]["nll_minus_random"][
                source
            ]["per_initialization"]
            x = np.arange(len(values))
            ax.plot(
                x,
                [100 * r["mean_difference"] for r in values],
                marker="o",
                markersize=4,
                lw=1.1,
                color=color,
                label=label,
            )
        ax.axhline(0, color="#152A38", lw=0.8)
        ax.set(
            title=source.upper(),
            xticks=range(6),
            xticklabels=[101, 202, 303, 404, 505, 606],
            xlabel="Initialization/dropout seed",
            ylim=(-24, 3),
        )
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Low NLL - random (pp)")
    axes[0].legend(loc="lower right", frameon=False)
    fig.tight_layout(pad=0.6)
    fig.savefig(figures / "initialization-effects.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(
        1, 2, figsize=(5.5, 2.65), gridspec_kw={"width_ratios": [1.05, 1]}
    )
    conditions = data["nll"]["metrics"]["conditions"]
    names = ["pad32_default", "pad32_explicit", "pad128_default", "pad128_explicit"]
    labels = [
        "Pad 32 / default",
        "Pad 32 / explicit",
        "Pad 128 / default",
        "Pad 128 / explicit",
        "BF16 vs. FP32",
    ]
    means = [conditions[n]["mean_absolute_difference"] for n in names] + [
        data["nll"]["metrics"]["bf16_vs_fp32_mean_absolute_difference"]
    ]
    maxima = [conditions[n]["max_absolute_difference"] for n in names] + [
        data["nll"]["metrics"]["bf16_vs_fp32_max_absolute_difference"]
    ]
    ax = axes[0]
    ax.barh(range(5), means, color=colors[0], height=0.55, label="Mean")
    ax.scatter(
        maxima, range(5), marker="|", s=60, color=colors[1], label="Maximum", zorder=3
    )
    ax.set(
        yticks=range(5),
        yticklabels=labels,
        xlabel="Absolute NLL change",
        xlim=(0, 0.28),
        title="Scoring sensitivity",
    )
    ax.invert_yaxis()
    handles, legend_labels = ax.get_legend_handles_labels()
    ax.grid(axis="x", alpha=0.2)
    ax = axes[1]
    groups = [
        ("stage3", "api"),
        ("stage3", "generic"),
        ("stage4", "api"),
        ("stage4", "generic"),
    ]
    for i, (stage, source) in enumerate(groups):
        changes = [
            r["replaced_content"]
            for r in data["nll"]["selection_changes"]
            if r["stage"] == stage and r["source"] == source
        ]
        ax.scatter(
            changes, [i + (j - 1.5) * 0.07 for j in range(4)], s=18, color=colors[1]
        )
    ax.set(
        yticks=range(4),
        yticklabels=[
            "Task / API",
            "Task / Generic",
            "Action / API",
            "Action / Generic",
        ],
        xlabel="Replaced / 256 records",
        xlim=(0, 12),
        xticks=[0, 4, 8, 12],
        title="Selection changes",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.29, -0.03),
        ncol=2,
        fontsize=7,
    )
    fig.tight_layout(pad=0.65, w_pad=1.1, rect=(0, 0.06, 1, 1))
    fig.savefig(figures / "scoring-diagnostic.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(
        1, 2, figsize=(5.5, 2.65), gridspec_kw={"width_ratios": [0.8, 1.2]}
    )
    groups = [data["webshop"]["groups"][p] for p in ["prompt1", "prompt1_actonly"]]
    rates = [100 * g["mean_success_completed_only"] for g in groups]
    axes[0].bar(range(2), rates, color=colors, width=0.55)
    for i, value in enumerate(rates):
        axes[0].text(i, value + 3, f"{value:.1f}%", ha="center", fontsize=8)
    axes[0].set(
        xticks=range(2),
        xticklabels=["ReAct", "Act-only"],
        ylabel="Success (%)",
        ylim=(0, 100),
        title="Train-only calibration",
    )
    axes[0].grid(axis="y", alpha=0.2)
    offset = np.zeros(2)
    for key, label, color in [
        ("physical_actions", "Physical", "#147D92"),
        ("thought_decisions", "Thought", "#8FBCC1"),
        ("invalid_format_decisions", "Invalid format", "#E5AF83"),
        ("invalid_control_decisions", "Invalid control", "#C85B39"),
    ]:
        values = np.array([g[key] for g in groups])
        axes[1].barh(
            range(2), values, left=offset, height=0.5, color=color, label=label
        )
        offset += values
    axes[1].set(
        yticks=range(2),
        yticklabels=["ReAct", "Act-only"],
        xlabel="Model decisions",
        xlim=(0, 270),
        title="Decision composition",
    )
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.2)
    axes[1].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.52),
        ncol=2,
        fontsize=7,
    )
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.34, top=0.85, wspace=0.62)
    fig.savefig(figures / "webshop-calibration.pdf")
    plt.close(fig)
    sources = {path: sha(ROOT / path) for path in SOURCES.values()}
    sources.update(
        {
            "docs/paper/EVIDENCE_SNAPSHOT_v0.2.json": sha(
                ROOT / "docs/paper/EVIDENCE_SNAPSHOT_v0.2.json"
            )
        }
    )
    evidence = dict(
        status="completed-evidence-only",
        sources=sources,
        figures={p.name: sha(p) for p in sorted(figures.glob("*.pdf"))},
        generated_tables={p.name: sha(p) for p in sorted(generated.glob("*.tex"))},
        numeric_checks=dict(
            stage3_api_pp=-17.7734375,
            stage3_generic_pp=-7.8125,
            stage4_api_pp=100 * effects["api"]["mean_difference"],
            stage4_generic_pp=100 * effects["generic"]["mean_difference"],
        ),
        pending_study_not_used="runs/variance-diagnostic-v1/stage5-canonical",
        raw_records_exported=False,
    )
    (PAPER / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                status="paper_assets_built",
                figures=4,
                tables=2,
                sources_verified=len(previous["sources"]),
            )
        )
    )


if __name__ == "__main__":
    main()
