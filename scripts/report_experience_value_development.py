#!/usr/bin/env python3
"""Immutable balanced-block figures and descriptive development endpoints."""

import argparse
from collections import defaultdict
import json

from learning_harness.bounded_queue import save
from learning_harness.common import file_sha
from learning_harness.experience_value_study import (
    NODES,
    POLICIES,
    SOURCES,
    aggregate_cells,
)
from run_experience_value_development import OUT, ROOT, audit_result, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, choices=NODES)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    frozen = validate()
    blocks = args.blocks or 16
    target = OUT / "interim" / f"blocks-{blocks:02d}" if args.blocks else OUT
    if (target / "aggregate-summary.json").exists():
        raise ValueError("Preserve the previous report snapshot")
    cases = [c for c in frozen["cases"] if c["name"] == "base" or c["block"] <= blocks]
    records = [audit_result(c) for c in cases]
    baseline = next(r for r in records if r["case"]["name"] == "base")
    students = [r for r in records if r["case"]["name"] != "base"]
    initial = defaultdict(set)
    for r in students:
        initial[r["case"]["initialization_seed"]].add(
            r["training"]["initial_parameter_sha256"]
        )
    if any(len(v) != 1 for v in initial.values()):
        raise ValueError("Matched initialization weights differ across conditions")
    endpoint = aggregate_cells(students, list(range(1, blocks + 1)))
    public = []
    for r in records:
        public.append(
            dict(
                case=r["case"],
                episodes=r["episodes"],
                successes=r["successes"],
                mean_success=r["mean_success"],
                families=r["families"],
                invalid_format_decisions=r["invalid_format_decisions"],
                usage=r["usage"],
                training=r["training"],
            )
        )
    result = dict(
        complete=True,
        interim=args.blocks is not None,
        development_only=True,
        freeze_sha256=file_sha(ROOT / "freeze.json"),
        complete_blocks=blocks,
        planned_blocks=16,
        completed_students=len(students),
        completed_episodes=len(records) * 128,
        planned_episodes=28800,
        baseline_success=baseline["mean_success"],
        endpoints=endpoint,
        models=public,
        teacher_calls=0,
        confirmation_evaluations=0,
        official_tests_used=False,
        inferential_tests_performed=False,
        raw_records_uploaded=False,
        scope="Fixed development matrix; only complete balanced blocks. No calibrated confirmatory significance or equivalence claim.",
        case_receipt_sha256={
            c["name"]: file_sha(ROOT / "cases" / c["name"] / "worker-result.json")
            for c in cases
        },
    )
    target.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1, 3, figsize=(12, 3.7), gridspec_kw=dict(width_ratios=[1.2, 1.2, 1])
    )
    labels = [
        "Random",
        "Grad high",
        "High NLL",
        "Low NLL",
        "Mid NLL",
        "Grad low",
        "Grad global",
    ]
    for ax, source in zip(axes[:2], SOURCES):
        data = endpoint["sources"][source]
        values = [100 * data[p]["mean_success"] for p in POLICIES]
        ax.bar(
            range(7),
            values,
            color=[
                "#6B7785",
                "#147D92",
                "#C85B39",
                "#BD8D36",
                "#A39A79",
                "#6B8E5B",
                "#8665A4",
            ],
            width=0.7,
        )
        for i, p in enumerate(POLICIES):
            points = [100 * v for v in data[p]["per_initialization"].values()]
            offsets = np.linspace(-0.16, 0.16, len(points)) if len(points) > 1 else [0]
            ax.scatter(i + np.array(offsets), points, s=12, color="black", alpha=0.6)
        ax.axhline(
            100 * baseline["mean_success"],
            color="#555555",
            linestyle="--",
            label="Untrained base",
        )
        ax.set(
            xticks=range(7),
            xticklabels=labels,
            ylim=(0, 100),
            ylabel="Success (%)",
            title=source.upper(),
        )
        ax.tick_params(axis="x", labelrotation=60, labelsize=8)
        ax.grid(axis="y", alpha=0.15)
    contrasts = endpoint["contrasts"]
    names = ("random", "high", "gradient_low")
    for i, name in enumerate(names):
        points = [
            100 * v
            for v in contrasts["gradient_high_minus_" + name]["per_paired_block"]
        ]
        axes[2].scatter(
            np.full(len(points), i), points, s=18, color="#147D92", alpha=0.6
        )
        axes[2].scatter(
            i,
            100 * contrasts["gradient_high_minus_" + name]["mean_difference"],
            marker="D",
            s=45,
            color="#C85B39",
        )
    axes[2].axhline(0, color="#555555", linewidth=1)
    axes[2].set(
        xticks=range(3),
        xticklabels=["vs Random", "vs High NLL", "vs Grad low"],
        ylabel="Gradient-high difference (pp)",
        title="Paired development blocks",
    )
    axes[2].tick_params(axis="x", labelrotation=35, labelsize=8)
    axes[2].grid(axis="y", alpha=0.15)
    fig.suptitle(
        f"Fresh development / {blocks}/16 complete balanced blocks / descriptive results",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(target / "selection-learning.png", dpi=180)
    plt.close(fig)
    lines = [
        "# 新任务经验学习价值：开发结果",
        "",
        f"固定16块中的{blocks}个完整均衡块，{len(students)}个学生、{len(records) * 128}回合。基础模型成功率{baseline['mean_success']:.3%}。",
        "",
        "这是开发研究；不报告未经二值结果校准的p值、少种子bootstrap区间或等效结论。点估计、逐初始化/采样块及所有负向结果保留。",
        "",
        "| 方法 | API成功率 | Generic成功率 |",
        "|---|---:|---:|",
    ]
    for p in POLICIES:
        lines.append(
            f"| {p} | {endpoint['sources']['api'][p]['mean_success']:.3%} | {endpoint['sources']['generic'][p]['mean_success']:.3%} |"
        )
    lines.extend(
        [
            "",
            "![开发对照](selection-learning.png)",
            "",
            "Grad global未控制覆盖，其他条件保持逐位置任务/动作与不同内容配额。全部条件同呈现预算，实际token与准备成本另计；没有开启确认集或官方测试。",
        ]
    )
    (target / "RESULTS.zh-CN.md").write_text("\n".join(lines) + "\n")
    save(target / "aggregate-summary.json", result)
    print(
        json.dumps(
            dict(
                status="descriptive_report_complete",
                balanced_blocks=blocks,
                students=len(students),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
