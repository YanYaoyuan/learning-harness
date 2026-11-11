from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import statistics

from .common import digest, read_jsonl, write_jsonl
from .support import SUPPORT_PREFIX


def summarize(run_dir, bootstrap=2000):
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "run.json").read_text())
    rows = read_jsonl(run_dir / "episodes.jsonl")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["arm"]].append(row)
    metrics = {}
    for arm, records in sorted(grouped.items()):
        by_type = defaultdict(list)
        for row in records:
            by_type[row["task_type"]].append(int(row["success"]))
        metrics[arm] = dict(n=len(records), success_rate=statistics.mean(int(r["success"]) for r in records),
                           macro_task_type_success=statistics.mean(statistics.mean(v) for v in by_type.values()),
                           success_by_type={k: statistics.mean(v) for k, v in sorted(by_type.items())},
                           error_count=sum(r["status"] == "error" for r in records),
                           mean_steps=statistics.mean(r["steps"] for r in records),
                           mean_input_tokens=statistics.mean(r["input_tokens"] for r in records),
                           mean_output_tokens=statistics.mean(r["output_tokens"] for r in records),
                           support_exposure_rate=statistics.mean(int(r["support_injected"]) for r in records))
    # Paired cluster bootstrap: retain all rollout seeds belonging to a task group.
    # This describes rollout uncertainty at ONE checkpoint, not training-seed uncertainty.
    paired = {}
    index = {(r["task_id"], r["rollout_seed"], r["arm"]): r for r in rows}
    if "none" in grouped:
        for arm in grouped:
            if arm == "none":
                continue
            differences = defaultdict(list)
            for row in grouped[arm]:
                baseline = index.get((row["task_id"], row["rollout_seed"], "none"))
                if baseline:
                    differences[row["task_group"]].append(int(row["success"]) - int(baseline["success"]))
            if differences:
                clusters = [statistics.mean(v) for v in differences.values()]
                rng = random.Random(20261109)
                samples = sorted(statistics.mean(rng.choices(clusters, k=len(clusters))) for _ in range(bootstrap))
                paired[arm] = dict(task_group_average_delta=statistics.mean(clusters),
                                   interval_95=[samples[int(0.025 * bootstrap)], samples[int(0.975 * bootstrap)]],
                                   paired_task_groups=len(clusters))
    report = dict(planned=metadata["planned_episodes"], recorded=len(rows),
                  complete=len(rows) == metadata["planned_episodes"],
                  uncertainty="Paired task-group bootstrap at one model checkpoint; not independent training seeds",
                  metrics=metrics, paired_against_none=paired)
    calls_path = run_dir / "calls.jsonl"
    if calls_path.exists():
        calls = read_jsonl(calls_path)
        report["all_attempts_token_accounting"] = dict(calls=len(calls),
            input_tokens=sum(r["input_tokens"] for r in calls),
            output_tokens=sum(r["output_tokens"] for r in calls))
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def export_sft(run_dir, arm, output, seed=20261109):
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    episodes = read_jsonl(run_dir / "episodes.jsonl")
    eligible = {(r["task_id"], r["rollout_seed"]): r["attempt_id"] for r in episodes
                if r["arm"] == arm and r["partition"] == "train" and r["success"] and r["status"] == "completed"}
    steps = read_jsonl(run_dir / "steps.jsonl")
    output_rows = []
    for row in steps:
        if row["arm"] != arm or (row["task_id"], row["rollout_seed"]) not in eligible or row["invalid_action"]:
            continue
        if row["attempt_id"] != eligible[(row["task_id"], row["rollout_seed"])]:
            continue
        if row["partition"] != "train":
            raise ValueError("Held-out data in SFT export")
        messages = [m for m in row["messages"]
                    if not (m["role"] == "user" and m["content"].startswith(SUPPORT_PREFIX))]
        output_rows.append(dict(task_id=row["task_id"], partition="train", source_arm=arm,
                                rollout_seed=row["rollout_seed"], step=row["step"],
                                messages=messages + [{"role": "assistant", "content": row["response"]}]))
    if not output_rows:
        raise ValueError("No successful training episodes; pilot/dev/probe/test cannot be exported for training")
    output_rows.sort(key=lambda r: digest([seed, r["task_id"], r["rollout_seed"], r["step"]]))
    write_jsonl(output, output_rows)
    metadata = dict(baseline="target-view rejection fine-tuning; not a reproduction of EvoIn/Harness-Zero",
                    source_run_fingerprint=run["fingerprint"], source_arm=arm,
                    successful_episodes=len(eligible), examples=len(output_rows), seed=seed)
    Path(str(output) + ".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata

