#!/usr/bin/env python3
"""Render aggregate numerical repeatability evidence, without input identities."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path("output/gradient-repeatability-diagnostic-v1")
    summary = json.loads((root / "precision-comparison-summary.json").read_text())
    modes = ("bf16_checkpointing", "bf16_eval", "fp32_math_checkpointing")
    rows = [summary["precision_conditions"][m] for m in modes]
    labels = ["BF16\ncheckpointed", "BF16\neval", "FP32 math\ndeterministic"]
    colors = ["#C85B39", "#C85B39", "#147D92"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    relative = [100 * row["max_relative_l2_error"] for row in rows]
    axes[0].bar(range(3), relative, color=colors, width=0.55)
    for i, value in enumerate(relative):
        axes[0].text(i, value + 0.03, f"{value:.3f}%", ha="center", fontsize=9)
    axes[0].axhline(
        0.1, linestyle="--", color="#555555", label="Prespecified 0.1% limit"
    )
    axes[0].set(ylabel="Max gradient relative L2 error (%)", ylim=(0, 1.13))
    axes[0].legend(fontsize=7, loc="upper right")
    axes[1].scatter(
        range(3), [min(1.0, row["min_cosine"]) for row in rows], c=colors, s=55
    )
    axes[1].axhline(
        0.9999, linestyle="--", color="#555555", label="Prespecified cosine limit"
    )
    axes[1].set(ylabel="Min gradient cosine", ylim=(0.999885, 1.00002))
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[1].legend(fontsize=7, loc="lower right")
    for ax in axes:
        ax.set_xticks(range(3), labels)
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle(
        "Saved checkpoint / two fixed train inputs / no parameter updates", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(root / "gradient-repeatability.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
