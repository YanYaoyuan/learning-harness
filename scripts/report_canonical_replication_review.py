#!/usr/bin/env python3
"""Compare completed legacy/canonical runs without rewriting either report."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    old_root = Path("output/variance-diagnostic-v1/stage4")
    root = Path("output/variance-diagnostic-v1/stage5-canonical")
    old = json.loads((old_root / "aggregate-summary.json").read_text())
    new = json.loads((root / "aggregate-summary.json").read_text())
    detail = [json.loads((r / "summary.json").read_text()) for r in (old_root, root)]
    if not old["complete"] or not new["complete"] or new["models"] != 96:
        raise ValueError("Both final reports required")
    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(11.6, 3.8), gridspec_kw={"width_ratios": [1, 1, 1.65]}
    )
    colors = ["#147D92", "#C85B39"]
    for ax, source, limit in zip(axes[:2], ("api", "generic"), (40, 15)):
        for p, color, dx, label in zip(
            ("random", "min_nll"), colors, (-0.18, 0.18), ("Random", "Low NLL")
        ):
            values = [100 * d["groups"][source][p]["mean_success"] for d in (old, new)]
            bars = ax.bar(
                np.arange(2) + dx, values, width=0.33, color=color, label=label
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + limit * 0.025,
                    f"{value:.2f}",
                    ha="center",
                    fontsize=8,
                )
        ax.set(
            xticks=[0, 1],
            xticklabels=["Legacy scoring", "Uniform scoring"],
            ylim=(0, limit),
            title=source.upper(),
            ylabel="Post-SFT success (%)",
        )
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    for s, color in zip(("api", "generic"), colors):
        for d, style, label in zip(detail, ("--", "-"), ("legacy", "uniform")):
            rows = d["policy_effects"]["mean_success"]["nll_minus_random"][s][
                "per_initialization"
            ]
            axes[2].plot(
                range(6),
                [100 * r["mean_difference"] for r in rows],
                style,
                marker="o",
                markersize=3.5,
                color=color,
                label=f"{s.upper()} / {label}",
            )
    axes[2].axhline(0, color="black", lw=0.7)
    axes[2].set(
        xticks=range(6),
        xticklabels=[101, 202, 303, 404, 505, 606],
        ylim=(-12, 3),
        ylabel="Low NLL - random (pp)",
        title="Per-initialization differences",
    )
    axes[2].grid(axis="y", alpha=0.15)
    axes[2].legend(frameon=False, fontsize=7.5, loc="lower right", ncol=2)
    fig.suptitle(
        "Completed scoring replication: six initializations × four sampling recipes",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    fig.text(
        0.5,
        0.015,
        "Fixed development tasks; no confidence interval. Both canonical arms retrained; cross-run changes are not a pure scoring effect.",
        ha="center",
        fontsize=8,
    )
    figure = root / "replication-comparison.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    summary = dict(
        old=old,
        uniform=new,
        new_minus_old_pp={
            s: {
                p: 100
                * (
                    new["groups"][s][p]["mean_success"]
                    - old["groups"][s][p]["mean_success"]
                )
                for p in ("random", "min_nll")
            }
            for s in ("api", "generic")
        },
        sources={
            str(r / "aggregate-summary.json"): hashlib.sha256(
                (r / "aggregate-summary.json").read_bytes()
            ).hexdigest()
            for r in (old_root, root)
        },
        figure_sha256=hashlib.sha256(figure.read_bytes()).hexdigest(),
        raw_records_uploaded=False,
    )
    (root / "replication-comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(dict(figure=str(figure), complete=True, new_model_calls=0)))


if __name__ == "__main__":
    main()
