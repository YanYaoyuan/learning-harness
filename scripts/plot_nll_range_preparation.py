#!/usr/bin/env python3
"""Plot candidate NLL separation; this figure contains no learning outcomes."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path("output/nll-range-controls-v1")
    report = json.loads((root / "preparation-summary.json").read_text())
    policies = ("low", "middle", "random", "high")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
    for ax, source in zip(axes, ("api", "generic")):
        rows = [r for r in report["rows"] if r["source"] == source]
        for row in rows:
            values = [row["policies"][p]["mean_selected_nll"] for p in policies]
            ax.plot(
                range(4),
                values,
                marker="o",
                alpha=0.65,
                linewidth=1,
                label=str(row["recipe"]),
            )
        ax.set(
            xticks=range(4), xticklabels=policies, title=source.upper(), ylim=(0, 0.34)
        )
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Mean selected target NLL")
    axes[1].legend(title="Recipe seed", fontsize=7, title_fontsize=7)
    fig.suptitle("Prepared selection controls / no training results", fontsize=10)
    fig.tight_layout()
    fig.savefig(root / "nll-range-preparation.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
