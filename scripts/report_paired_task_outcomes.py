#!/usr/bin/env python3
"""Describe shared task outcomes for a post-hoc selected paired model cell."""

import argparse
import csv
import json
from pathlib import Path

from learning_harness.common import file_sha

METRICS = (
    "success",
    "steps",
    "take_attempt",
    "move_attempt",
    "any_format_error",
    "literal_no_effect_fraction",
    "exact_command_repetition_fraction",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--source", choices=("api", "generic"), required=True)
    parser.add_argument("--initialization-seed", type=int, required=True)
    parser.add_argument("--sampling-seed", type=int, required=True)
    args = parser.parse_args()
    summary = json.loads((args.audit / "summary.json").read_text())
    with (args.audit / "episodes.csv").open() as stream:
        rows = [
            r
            for r in csv.DictReader(stream)
            if r["source"] == args.source
            and int(r["initialization_seed"]) == args.initialization_seed
            and int(r["sampling_seed"]) == args.sampling_seed
        ]
    indexed, behavior = {}, {}
    for policy in ("random", "min_nll"):
        selected = [r for r in rows if r["policy"] == policy]
        indexed[policy] = {(r["family"], r["task_id"]): r for r in selected}
        if len(selected) != 64 or len(indexed[policy]) != 64:
            raise ValueError("Expected exactly 64 unique task outcomes per policy")
        behavior[policy] = {m: sum(float(r[m]) for r in selected) / 64 for m in METRICS}
        reference = next(
            r
            for r in summary["model_rows"]
            if r["source"] == args.source
            and r["policy"] == policy
            and r["initialization_seed"] == args.initialization_seed
            and r["sampling_seed"] == args.sampling_seed
        )
        if any(abs(behavior[policy][m] - reference[m]) > 1e-12 for m in METRICS):
            raise ValueError(
                "Episode CSV differs from previously audited model aggregates"
            )
    if set(indexed["random"]) != set(indexed["min_nll"]):
        raise ValueError("Task pairs must match by identity, not row order")
    contingency = {}
    for family in ("cool", "heat", "all"):
        cells = [[0, 0], [0, 0]]
        for key, a in indexed["random"].items():
            if family == "all" or key[0] == family:
                b = indexed["min_nll"][key]
                sa, sb = int(a["success"]), int(b["success"])
                if sa not in (0, 1) or sb not in (0, 1):
                    raise ValueError("Success must be binary")
                cells[sa][sb] += 1
        n = sum(sum(row) for row in cells)
        if n != (64 if family == "all" else 32):
            raise ValueError("Expected two equally weighted task families")
        contingency[family] = dict(
            tasks=n,
            both_fail=cells[0][0],
            min_nll_only=cells[0][1],
            random_only=cells[1][0],
            both_success=cells[1][1],
            min_nll_minus_random_pp=100 * (cells[0][1] - cells[1][0]) / n,
            matrix_random_rows_nll_columns=cells,
        )
    delta = 100 * (behavior["min_nll"]["success"] - behavior["random"]["success"])
    if abs(delta - contingency["all"]["min_nll_minus_random_pp"]) > 1e-12:
        raise ValueError(
            "Paired outcomes do not reconcile with reported mean difference"
        )
    out = (
        args.audit
        / f"paired-case-{args.source}-i{args.initialization_seed}-s{args.sampling_seed}"
    )
    out.mkdir(exist_ok=False)
    record = dict(
        exploratory=True,
        case_selected_after_observing_results=True,
        source=args.source,
        initialization_seed=args.initialization_seed,
        sampling_seed=args.sampling_seed,
        models=2,
        episodes=128,
        new_model_calls=0,
        teacher_calls=0,
        official_tests_used=False,
        raw_records_uploaded=False,
        contingency=contingency,
        behavior=behavior,
        provenance={
            str(p): file_sha(p)
            for p in (
                args.audit / "summary.json",
                args.audit / "episodes.csv",
                Path(__file__),
            )
        },
        interpretation="Description of an outcome-selected counterexample on the same development tasks. Attempt rates, no-effect feedback and command repetition are not causal mediators or independent predictors of success.",
    )
    (out / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, family in zip(axes, ("cool", "heat")):
        values = contingency[family]["matrix_random_rows_nll_columns"]
        ax.imshow(values, cmap="Blues", vmin=0, vmax=32)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(values[i][j]),
                    ha="center",
                    va="center",
                    fontsize=18,
                    color="white" if values[i][j] > 16 else "black",
                )
        ax.set(
            xticks=[0, 1],
            xticklabels=["Fail", "Success"],
            yticks=[0, 1],
            yticklabels=["Fail", "Success"],
            xlabel="Low-NLL policy outcome",
            ylabel="Random policy outcome",
            title=f"{family.upper()}: {contingency[family]['min_nll_minus_random_pp']:+.2f}pp",
        )
    fig.suptitle(
        f"POST-HOC CASE: {args.source.upper()}, init {args.initialization_seed}, sample {args.sampling_seed}",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.015,
        "Counts on the same 64 development tasks. Outcome-selected illustration; no new model calls or causal inference.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(out / ("paired-task-outcomes." + ext), dpi=180)
    plt.close(fig)
    lines = [
        "# 事后选取的成对单元：逐任务成功交叉表",
        "",
        f"{args.source}，初始化{args.initialization_seed}，采样{args.sampling_seed}。该单元在观察结果后选取，仅作反例说明。",
        "",
        "| 任务族 | 均失败 | 仅随机成功 | 仅低NLL成功 | 均成功 | 低NLL−随机 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family, r in contingency.items():
        lines.append(
            f"| {family} | {r['both_fail']} | {r['random_only']} | {r['min_nll_only']} | {r['both_success']} | {r['min_nll_minus_random_pp']:+.2f}pp |"
        )
    lines += [
        "",
        "动作尝试、字面无效果反馈和命令重复均来自同批评测轨迹，不能据此归因或作为独立成功预测器。原始回合保留于本地；本报告只含汇总计数。",
        "",
        "![成对任务交叉表](paired-task-outcomes.png)",
        "",
    ]
    (out / "RESULTS.zh-CN.md").write_text("\n".join(lines))
    print(
        json.dumps(dict(output=str(out), contingency=contingency, behavior=behavior)),
        flush=True,
    )


if __name__ == "__main__":
    main()
