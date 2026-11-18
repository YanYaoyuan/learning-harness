#!/usr/bin/env python3
"""Measured-only Stage 0 figures and trace checks after the frozen estimator finishes."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("runs/variance-diagnostic-v1/stage0")
OLD = Path("runs/transfer-factorial-v1")
OUT = Path("output/variance-diagnostic-v1/stage0")
SEEDS = (1101, 1202, 1303)
FAMILIES = ("cool", "heat")
SOURCES = ("api", "generic")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def committed_cell(directory):
    """Reject duplicate tasks and ignore interrupted trace attempts."""
    episodes = read_rows(directory / "episodes.jsonl")
    cells = {row["task_id"]: row for row in episodes}
    if len(cells) != len(episodes):
        raise ValueError("Duplicate committed task: " + str(directory))
    if any(row["status"] != "completed" or row["arm"] != "none" for row in episodes):
        raise ValueError("Errored or assisted episode in " + str(directory))
    valid = {row["attempt_id"] for row in episodes}
    traces = {task: [] for task in cells}
    for row in read_rows(directory / "steps.jsonl"):
        if row["attempt_id"] in valid:
            traces[row["task_id"]].append(tuple(row[key] for key in
                ("step", "action", "observation", "success", "invalid_action")))
    return cells, traces


def flip_counts(cells):
    """Count task outcomes that differ over complete new replicates."""
    if not cells:
        return 0, 0
    tasks = set(cells[0])
    if any(set(cell) != tasks for cell in cells):
        raise ValueError("Replicates contain different task sets")
    return sum(len({bool(cell[task]["success"]) for cell in cells}) > 1 for task in tasks), len(tasks)


def main():
    result = json.loads((ROOT / "result.json").read_text())
    freeze = json.loads((ROOT / "freeze.json").read_text())
    ledger = json.loads((ROOT / "execution.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    flip_rows = []
    for family in FAMILIES:
        for source in SOURCES:
            flipped = total = 0
            for seed in SEEDS:
                cells = [committed_cell(ROOT / f"rep{r}" / family / f"full-{source}-seed{seed}")[0]
                         for r in result["complete_new_replicates"]]
                count, size = flip_counts(cells)
                flipped += count
                total += size
            flip_rows.append(dict(family=family, source=source, flipped=flipped, total=total,
                                  flip_rate=flipped / total if total and len(result["complete_new_replicates"]) >= 2 else None))
    determinism = []
    for source in SOURCES:
        directory = ROOT / "determinism" / "heat" / f"full-{source}-seed1101"
        if not (directory / "summary.json").exists() or not json.loads((directory / "summary.json").read_text())["complete"]:
            determinism.append(dict(source=source, complete=False))
            continue
        old = OLD / "heat/evaluation/main" / f"full-{source}-seed1101/B/step-0000"
        old_cells, old_traces = committed_cell(old)
        new_cells, new_traces = committed_cell(directory)
        if set(old_cells) != set(new_cells):
            raise ValueError("Determinism check changed the task set")
        determinism.append(dict(source=source, complete=True, tasks=len(old_cells),
            success_matches=sum(bool(old_cells[t]["success"]) == bool(new_cells[t]["success"]) for t in old_cells),
            action_observation_trace_matches=sum(old_traces[t] == new_traces[t] for t in old_cells),
            historical_environment_identity_attested=bool(freeze.get("environment_attestation"))))
    summary = dict(complete=True, exploratory=True, population="Legacy full-experience API versus generic; not API versus ReAct",
        perturbation="Fixed task-order permutations at temperature zero; batching/numerical sensitivity only",
        independent_training_seeds=3, new_replicates=result["complete_new_replicates"],
        primary=result["primary"], decision=result["decision"], task_flip_rates=flip_rows,
        determinism_checks=determinism, gpu_hours=ledger["gpu_seconds"] / 3600,
        ledger_status=ledger["status"], include_historical_primary=freeze["include_historical"],
        teacher_calls=0, official_tests_used=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    primary = result["primary"]
    if primary:
        d = np.asarray(result["paired_difference_by_seed_and_replicate"]) * 100
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        x = np.arange(d.shape[1])
        for seed, row in zip(SEEDS, d):
            axes[0].plot(x, row, "o-", label=f"A seed {seed - 1000}")
        axes[0].axhline(0, color="gray", linestyle="--", linewidth=1)
        axes[0].set_xticks(x, result["primary_replicates"])
        axes[0].set_ylabel("API − generic success (pp)")
        axes[0].set_title("Same checkpoints, changed batch membership", loc="left")
        axes[0].legend(frameon=False, fontsize=8, loc="center right", bbox_to_anchor=(1, .65))
        axes[1].bar([0, 1], [primary["replicate_sd_pp"], primary["seed_sd_pp"]],
                    color=["#dc7945", "#2878a6"])
        axes[1].set_xticks([0, 1], ["Batching repeat", "Training seed"])
        axes[1].set_ylabel("Estimated SD (pp)")
        axes[1].set_ylim(0, max(1.0, 1.3 * max(primary["replicate_sd_pp"], primary["seed_sd_pp"])))
        share = primary["share"]
        title = "Share undefined" if share is None else f"Repeat share: {100 * share:.1f}%"
        axes[1].set_title(title, loc="left")
        interval = primary["share_interval"]
        if interval:
            axes[1].text(.5, .97, f"Approx. 90% share interval: {100 * interval[0]:.1f}–{100 * interval[1]:.1f}%",
                         transform=axes[1].transAxes, ha="center", va="top", fontsize=8)
        for ax in axes:
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
        fig.suptitle("Stage 0 — legacy API/generic, fixed development queries")
        fig.text(.5, .015, "Three training seeds; components and interval are uncertain. The branch decision allocates budget, not a mechanism claim.",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .06, 1, .94))
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"variance-decomposition.{ext}", dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    labels = [f"{r['family'].capitalize()} / {r['source']}" for r in flip_rows]
    ax.bar(np.arange(4), [100 * r["flip_rate"] if r["flip_rate"] is not None else np.nan for r in flip_rows],
           color=["#2878a6", "#dc7945"] * 2)
    if len(result["complete_new_replicates"]) < 2:
        ax.text(.5, .5, "Fewer than two repeats: flip rate not estimated", transform=ax.transAxes, ha="center")
    ax.set_xticks(np.arange(4), labels)
    ax.set_ylabel("Checkpoint-task cells that flip (%)")
    ax.set_title("Outcome sensitivity to batch membership", loc="left")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.5, .015, "Each group contains 3 checkpoints × 32 fixed tasks. New complete replicates only.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"task-flips.{ext}", dpi=180)
    plt.close(fig)
    lines = ["# Stage 0：批布局与数值敏感性", "", f"预算分支：{result['decision']}", "",
             "对象是旧的 API−generic 全量经验模型，不是刚完成的 API−ReAct 对照。",
             "改变任务排列、保持温度0及环境种子不变；不能据此估计所有执行随机性。", "",
             f"完整新重复：{result['complete_new_replicates']}；累计 GPU 时间：{summary['gpu_hours']:.3f} 小时。",
             f"主估计纳入历史重复：{freeze['include_historical']}。", "",
             "| 任务族／来源 | 发生结果翻转的 checkpoint-task 单元 |", "|---|---:|"]
    lines += [f"| {r['family']} / {r['source']} | {r['flipped']}/{r['total']} |" for r in flip_rows]
    lines += ["", "方差分量及近似区间见 summary.json；只有三个训练种子，不能将预算分支当作机制结论。",
              "原配置重跑与历史结果的差异也可能包含历史环境变化，单独列于 determinism_checks。", ""]
    (OUT / "RESULTS.zh-CN.md").write_text("\n".join(lines))
    print(json.dumps(dict(output=str(OUT), decision=result["decision"], complete=True)))


if __name__ == "__main__":
    main()
