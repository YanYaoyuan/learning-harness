#!/usr/bin/env python3
"""Plot the completed instrument diagnostic and train-only WebShop calibration."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(fig, folder, pdf_folder, stem, source):
    folder.mkdir(parents=True, exist_ok=True)
    pdf_folder.mkdir(parents=True, exist_ok=True)
    png = folder / (stem + ".png")
    pdf = pdf_folder / (stem + ".pdf")
    if png.exists() or pdf.exists():
        raise ValueError("Preserve existing figure outputs")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    manifest = dict(
        source_file=str(source),
        source_sha256=sha(source),
        plotting_script_sha256=sha(__file__),
        outputs={str(path): sha(path) for path in (png, pdf)},
        uncertainty="Descriptive plots; no confidence intervals",
    )
    (folder / "figure-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def scoring_figure(report):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.8, bottom=0.25, wspace=0.32)
    fig.suptitle("Scoring layout and selected-content sensitivity", fontsize=15, y=0.98)
    fig.text(
        0.5,
        0.9,
        "32 prespecified stress inputs | 853 unique inputs uniformly rescored | 1,194 forwards",
        ha="center",
        fontsize=10,
    )
    conditions = report["metrics"]["conditions"]
    keys = [
        "canonical_repeat",
        "pad32_default",
        "pad32_explicit",
        "pad128_default",
        "pad128_explicit",
    ]
    labels = [
        "Repeat",
        "Pad 32\nDefault",
        "Pad 32\nExplicit",
        "Pad 128\nDefault",
        "Pad 128\nExplicit",
    ]
    ax = axes[0]
    ax.bar(
        range(5),
        [conditions[k]["mean_absolute_difference"] for k in keys],
        color="#4C72B0",
        width=0.6,
        label="Mean absolute difference",
    )
    ax.scatter(
        range(5),
        [conditions[k]["max_absolute_difference"] for k in keys],
        marker="D",
        color="#C44E52",
        label="Maximum absolute difference",
        zorder=4,
    )
    ax.set(
        xticks=range(5),
        xticklabels=labels,
        ylabel="Absolute NLL difference (nat / target token)",
        ylim=(-0.008, 0.28),
        title="(a) BF16 shifts relative to canonical layout",
    )
    ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax = axes[1]
    categories = [
        ("api", "stage3"),
        ("api", "stage4"),
        ("generic", "stage3"),
        ("generic", "stage4"),
    ]
    for index, (source, stage) in enumerate(categories):
        rows = sorted(
            (
                r
                for r in report["selection_changes"]
                if r["source"] == source and r["stage"] == stage
            ),
            key=lambda r: r["sampling_seed"],
        )
        if len(rows) != 4 or any(r["selected_content"] != 256 for r in rows):
            raise ValueError("Four 256-content recipes required per source/stage")
        values = [r["replaced_content"] for r in rows]
        ax.scatter(
            [index + offset for offset in (-0.15, -0.05, 0.05, 0.15)],
            values,
            color="#4C72B0" if source == "api" else "#DD8452",
            s=40,
            zorder=4,
        )
        mean = sum(values) / 4
        ax.plot(
            [index - 0.24, index + 0.24], [mean, mean], color="black", linewidth=2.8
        )
    ax.set(
        xticks=range(4),
        xticklabels=[
            "API\nTask matched",
            "API\nAction matched",
            "Generic\nTask matched",
            "Generic\nAction matched",
        ],
        ylabel="Replaced distinct examples (out of 256)",
        ylim=(0, 12.8),
        title="(b) Changes after uniform rescoring",
    )
    ax.set_yticks(range(0, 13, 2))
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                color="gray",
                linestyle="",
                label="One sampling recipe",
            ),
            Line2D([], [], color="black", linewidth=2.8, label="Mean"),
        ],
        frameon=False,
        fontsize=9,
        loc="upper left",
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
    fig.text(
        0.07,
        0.07,
        "The stress set was chosen for sensitivity; it is not a population-error sample.\nSelection overlap is a data-composition measurement; no retraining was performed.",
        fontsize=9,
    )
    return fig


def webshop_figure(report):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.8, bottom=0.27, wspace=0.32)
    fig.suptitle("WebShop base-model calibration", fontsize=15, y=0.98)
    fig.text(
        0.5,
        0.9,
        "32 shared train goals per prompt | 14 model decisions maximum | Base-only 4B chat/native port",
        ha="center",
        fontsize=10,
    )
    modes = ["prompt1", "prompt1_actonly"]
    groups = [report["groups"][mode] for mode in modes]
    labels = ["ReAct\nPublic prompt", "Act-only\nPublic prompt"]
    ax = axes[0]
    values = [100 * g["mean_success_completed_only"] for g in groups]
    ax.bar(range(2), values, color=["#4C72B0", "#DD8452"], width=0.58)
    for index, (value, group) in enumerate(zip(values, groups)):
        successes = round(
            group["mean_success_completed_only"] * group["completed_episodes"]
        )
        ax.text(
            index,
            value + 3,
            f"{successes}/32\n({value:.1f}%)",
            ha="center",
            fontsize=11,
        )
    ax.set(
        xticks=range(2),
        xticklabels=labels,
        ylabel="Task success (%)",
        ylim=(0, 100),
        title="(a) Base-model starting point",
    )
    ax = axes[1]
    categories = [
        ("physical_actions", "Physical action", "#55A868"),
        ("thought_decisions", "Thought", "#4C72B0"),
        ("invalid_format_decisions", "Invalid format", "#C44E52"),
        ("invalid_control_decisions", "Invalid control", "#DD8452"),
    ]
    bottoms = [0.0, 0.0]
    for key, label, color in categories:
        shares = [100 * g[key] / g["attempted_requests"] for g in groups]
        ax.bar(range(2), shares, bottom=bottoms, width=0.58, label=label, color=color)
        for index, share in enumerate(shares):
            if share >= 8:
                ax.text(
                    index,
                    bottoms[index] + share / 2,
                    f"{share:.1f}%",
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
        bottoms = [a + b for a, b in zip(bottoms, shares)]
    for index, group in enumerate(groups):
        ax.text(
            index, 103, f"N={group['attempted_requests']}", ha="center", fontsize=10
        )
    ax.set(
        xticks=range(2),
        xticklabels=labels,
        ylabel="Share of model decisions (%)",
        ylim=(0, 112),
        title="(b) Decision-budget composition",
    )
    ax.set_yticks(range(0, 101, 20))
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.21),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
    fig.text(
        0.07,
        0.055,
        "Train-only calibration. All 463 requests are accounted for, with 0 transport errors.\nThoughts and invalid outputs consume the fixed decision budget; prompt superiority is not inferred.",
        fontsize=9,
    )
    return fig


def main():
    plt.rcParams.update(
        {"font.size": 10, "axes.titlesize": 12, "font.family": "DejaVu Sans"}
    )
    nll_source = Path("output/nll-scoring-diagnostic-v1/aggregate-summary.json")
    webshop_source = Path("runs/webshop-development-pilot-v1/aggregate-summary.json")
    nll = read(nll_source)
    webshop = read(webshop_source)
    if not (
        nll["complete"]
        and nll["uniform_complete"]
        and nll["base_model_files_after_verified"]
        and webshop["complete"]
        and webshop["completed_episodes"] == 64
    ):
        raise ValueError("Complete, reviewed follow-up results required")
    for review, source, key in [
        (
            "output/research-operations/20261113/nll-scoring-complete-local-check.json",
            nll_source,
            "report_sha256",
        ),
        (
            "output/research-operations/20261113/webshop-pilot-complete-local-check.json",
            webshop_source,
            "authoritative_report_sha256",
        ),
    ]:
        if read(review)[key] != sha(source):
            raise ValueError("Reviewed source changed")
    webshop_out = Path("output/webshop-development-pilot-v1")
    webshop_out.mkdir(parents=True, exist_ok=True)
    copy = webshop_out / "aggregate-summary.json"
    if copy.exists() and sha(copy) != sha(webshop_source):
        raise ValueError("Preserve an existing differing public pilot copy")
    copy.write_bytes(webshop_source.read_bytes())
    manifests = [
        export(
            scoring_figure(nll),
            nll_source.parent,
            Path("output/pdf/nll-scoring-diagnostic-v1"),
            "scoring-layout-and-selection",
            nll_source,
        ),
        export(
            webshop_figure(webshop),
            webshop_out,
            Path("output/pdf/webshop-development-pilot-v1"),
            "base-calibration",
            webshop_source,
        ),
    ]
    print(
        json.dumps(
            dict(status="figures_created_for_visual_review", manifests=manifests),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
