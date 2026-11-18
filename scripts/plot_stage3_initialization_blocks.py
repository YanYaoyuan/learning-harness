#!/usr/bin/env python3
"""Plot every paired sampling-cell effect grouped by initialization, without inference."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    source = args.snapshot / "summary.json"
    value = json.loads(source.read_text())
    if not value["interim"] or value["final_study_complete"]:
        raise ValueError("Expected a clearly labelled descriptive interim snapshot")
    rows = value["paired_rows"]
    expected = {
        (s, *key) for s in ("api", "generic") for key in value["matched_seed_cells"]
    }
    keys = [(r["source"], r["initialization_seed"], r["sampling_seed"]) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("Incomplete or duplicate paired cells")
    for r in rows:
        if (
            abs(
                r["success_difference_pp"]
                - 100 * (r["min_nll_success"] - r["random_success"])
            )
            > 1e-9
        ):
            raise ValueError("Stored difference does not match paired scores")
    out = (
        Path("output/variance-diagnostic-v1/stage3/interim-blocks") / args.snapshot.name
    )
    out.mkdir(parents=True, exist_ok=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    inits = sorted({r["initialization_seed"] for r in rows})
    summary = []
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), sharey=True)
    largest = max(abs(r["success_difference_pp"]) for r in rows)
    bound = largest + max(10, largest * 0.25)
    for ax, source_name, color in zip(axes, ("api", "generic"), ("#2878b5", "#e58b2a")):
        for x, init in enumerate(inits):
            group = sorted(
                [
                    r
                    for r in rows
                    if r["source"] == source_name and r["initialization_seed"] == init
                ],
                key=lambda r: r["sampling_seed"],
            )
            delta = [r["success_difference_pp"] for r in group]
            offset = (
                np.linspace(-0.12, 0.12, len(group))
                if len(group) > 1
                else np.array([0.0])
            )
            mean = sum(delta) / len(delta)
            ax.scatter(x + offset, delta, color=color, s=40, zorder=3)
            ax.plot([x - 0.22, x + 0.22], [mean, mean], color="black", linewidth=3)
            ax.annotate(
                f"mean {mean:+.2f}pp",
                (x, max(delta)),
                (0, 16),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
            summary.append(
                dict(
                    source=source_name,
                    initialization_seed=init,
                    sampling_seeds=[r["sampling_seed"] for r in group],
                    mean_difference_pp=mean,
                    wins=sum(v > 0 for v in delta),
                    ties=sum(v == 0 for v in delta),
                    losses=sum(v < 0 for v in delta),
                )
            )
        ax.axhline(0, color="#555555", linestyle="--", linewidth=1)
        ax.set(
            title=source_name.upper(),
            xticks=range(len(inits)),
            xticklabels=inits,
            xlabel="Initialization seed",
            ylabel="Low NLL − random success (pp)",
            ylim=(-bound, bound),
            xlim=(-0.6, len(inits) - 0.4),
        )
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        f"INTERIM: {len(value['matched_seed_cells'])} paired cells, grouped by initialization",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.015,
        "Dots: sampling seeds. Black bars: within-initialization mean. Fixed development tasks; no confidence intervals.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(out / ("initialization-blocks." + ext), dpi=180)
    plt.close(fig)
    with (out / "matched-cells.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    record = dict(
        interim=True,
        complete=False,
        source_snapshot=str(args.snapshot),
        source_summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        initialization_blocks=summary,
        no_outcome_dependent_scheduling=True,
    )
    (out / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    lines = [
        "# 中途结果：按初始化分解",
        "",
        "| 来源 | 初始化 | 采样种子数 | 低NLL−随机平均差 | 胜/平/负 |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in summary:
        lines.append(
            f"| {s['source']} | {s['initialization_seed']} | {len(s['sampling_seeds'])} | {s['mean_difference_pp']:+.2f}pp | {s['wins']}/{s['ties']}/{s['losses']} |"
        )
    lines += [
        "",
        "保留每个配对采样单元；这是同一开发任务集上的描述性分解，未新增检验或改变预算。",
        "",
        "![初始化分解](initialization-blocks.png)",
        "",
    ]
    (out / "RESULTS.zh-CN.md").write_text("\n".join(lines))
    print(
        json.dumps(dict(directory=str(out), initialization_blocks=summary)), flush=True
    )


if __name__ == "__main__":
    main()
