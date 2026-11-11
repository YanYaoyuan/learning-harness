#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import random

from learning_harness.analysis import export_sft, summarize
from learning_harness.common import digest, read_jsonl, write_jsonl
from learning_harness.support import SUPPORT_PREFIX

ROOT = Path("runs/dual5090-v1")


def configure(stage, arms):
    records = read_jsonl(ROOT / f"{stage}.jsonl")
    for gpu in range(2):
        manifest = ROOT / f"{stage}-gpu{gpu}.jsonl"
        write_jsonl(manifest, records[gpu::2])
        config = json.loads(Path("configs/pilot.json").read_text())
        config.update(protocol_id=f"dual5090-discovery-v1-{stage}", manifest=str(manifest),
                      arms=arms, rollout_seeds=[11, 23] if stage == "train" else [11],
                      max_steps=40, max_student_output_tokens=5120,
                      temperature=0.7 if stage == "train" else 0.0,
                      teacher_bank=str(ROOT / "support.jsonl") if arms != ["none"] else None)
        path = ROOT / "configs" / f"{stage}-gpu{gpu}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2) + "\n")
    return {"stage": stage, "tasks": len(records), "arms": arms}


def merge(stage):
    target = ROOT / stage
    target.mkdir(exist_ok=True)
    metas = [json.loads((ROOT / f"{stage}-gpu{gpu}/run.json").read_text()) for gpu in range(2)]
    if len({(m["model"], m["revision"]) for m in metas}) != 1:
        raise ValueError("Worker models must match")
    comparable = [{k: v for k, v in m["config"].items() if k != "manifest"} for m in metas]
    if comparable[0] != comparable[1]:
        raise ValueError("Worker protocols must match apart from task shard")
    for filename in ("steps.jsonl", "episodes.jsonl", "calls.jsonl"):
        combined = []
        for gpu in range(2):
            worker_rows = read_jsonl(ROOT / f"{stage}-gpu{gpu}" / filename)
            if filename == "episodes.jsonl":
                if len(worker_rows) != metas[gpu]["planned_episodes"] or any(r["status"] == "error" for r in worker_rows):
                    raise ValueError("Both workers must finish without service errors before merging")
            combined += worker_rows
        if filename == "episodes.jsonl":
            keys = [(r["task_id"], r["arm"], r["rollout_seed"]) for r in combined]
            if len(set(keys)) != len(keys):
                raise ValueError("Duplicate episode in merged workers")
        write_jsonl(target / filename, combined)
    metadata = dict(fingerprint=digest([m["fingerprint"] for m in metas]),
                    planned_episodes=sum(m["planned_episodes"] for m in metas),
                    model=metas[0]["model"], revision=metas[0]["revision"], source_workers=metas)
    (target / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return summarize(target)


def corpus(arm):
    target = ROOT / "corpora" / f"{arm}.jsonl"
    export_sft(ROOT / "train", arm, target)
    hints = {r["task_id"]: r["hint"] for r in read_jsonl(ROOT / "support.jsonl")}
    rows = read_jsonl(target)
    for row in rows:
        base = row["messages"]
        row["teacher_messages"] = base[:2] + [dict(role="user", content=SUPPORT_PREFIX + hints[row["task_id"]])] + base[2:]
    write_jsonl(target, rows)
    return {"arm": arm, "examples": len(rows), "source": str(target), "same_examples_across_training_methods": True}


def report():
    result = dict(protocol="dual5090-discovery-v1", support="fixed handwritten public strategy",
                  teacher_api_used=False, final_benchmarks_used=False,
                  scientific_status="exploratory, one model family, one training seed")
    for stage in ("pilot", "train", "probe-base"):
        path = ROOT / stage / "summary.json"
        if path.exists():
            result[stage] = json.loads(path.read_text())
    jobs = []
    indexed = {}
    for path in sorted((ROOT / "trained-eval").glob("*/summary.json")):
        metric = json.loads(path.read_text())
        training = json.loads((ROOT / "checkpoints" / path.parent.name / "experiment.json").read_text())
        jobs.append(dict(checkpoint=path.parent.name, summary=metric, training=training))
        indexed[path.parent.name] = read_jsonl(path.parent / "episodes.jsonl")
    result["trained_evaluation"] = jobs
    result["grid_complete"] = len(jobs) == 6 and all(j["summary"]["complete"] for j in jobs)
    selected = json.loads((ROOT / "selection.json").read_text())["arm"]
    result["training_data_coverage"] = {}
    training_episodes = read_jsonl(ROOT / "train/episodes.jsonl")
    for arm in ("none", selected):
        won = [r for r in training_episodes if r["arm"] == arm and r["success"]]
        counts = defaultdict(int)
        for row in won:
            counts[row["task_type"]] += 1
        result["training_data_coverage"][arm] = dict(successful_episodes=len(won),
            successful_task_groups=len({r["task_group"] for r in won}), success_by_type=dict(counts))
    manifests = {stage: read_jsonl(ROOT / f"{stage}.jsonl") for stage in ("pilot", "train", "probe")}
    groups = {stage: {r["task_group"] for r in rows} for stage, rows in manifests.items()}
    if any(groups[a] & groups[b] for a, b in (("pilot", "train"), ("train", "probe"), ("pilot", "probe"))):
        raise ValueError("Study partitions overlap")
    if any(r["official_split"] != "train" for rows in manifests.values() for r in rows):
        raise ValueError("Discovery must not use official final evaluation splits")
    result["integrity"] = dict(task_groups_disjoint=True, all_tasks_from_official_training_split=True)
    corpus_hashes = defaultdict(set)
    for job in jobs:
        t = job["training"]
        corpus_hashes[t["source_export_meta"]["source_arm"]].add(t["training_data_sha256"])
    if any(len(values) != 1 for values in corpus_hashes.values()):
        raise ValueError("Within-source training methods must share one exact corpus")
    result["integrity"]["within_source_corpus_hashes_match"] = True
    base_path = ROOT / "probe-base/episodes.jsonl"
    if base_path.exists():
        baseline = read_jsonl(base_path)
        result["learning_comparisons"] = {}
        for method in ("target_sft", "mixed_sft", "context_distill"):
            names = [f"{arm}-{method}-seed101" for arm in ("none", selected)]
            if all(name in indexed for name in names):
                rows = [indexed[name] for name in names]
                base_index = {r["task_id"]: r for r in baseline}
                indices = [{r["task_id"]: r for r in group} for group in rows]
                if any(set(i) != set(base_index) for i in indices):
                    raise ValueError("Checkpoint comparison requires identical complete probe tasks")
                clusters = defaultdict(list)
                for task, r in base_index.items():
                    clusters[r["task_group"]].append(int(indices[1][task]["success"]) - int(indices[0][task]["success"]))
                deltas = [statistics.mean(v) for v in clusters.values()]
                rng = random.Random(20261109)
                draws = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(2000))
                result["learning_comparisons"][method] = dict(
                    source_arm_delta=statistics.mean(deltas), paired_task_groups=len(deltas),
                    task_bootstrap_interval_95=[draws[50], draws[1950]],
                    uncertainty="Conditional on one training seed; excludes training variability",
                    none_wins=sum(r["success"] for r in rows[0]),
                    support_wins=sum(r["success"] for r in rows[1]),
                    base_wins=sum(r["success"] for r in baseline), n=len(baseline))
    result["claims_not_established"] = ["multi-seed stability", "cross-model transfer", "external-environment transfer", "a predictive law", "frontier-teacher effectiveness", "improved future-task learning speed", "continual or online RL effectiveness"]
    (ROOT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def select_support():
    metrics = json.loads((ROOT / "pilot/summary.json").read_text())["metrics"]
    preference = ["early", "delayed", "random_timing"]
    best = max(preference, key=lambda arm: metrics[arm]["success_rate"])
    selection = dict(arm=best, criterion="pilot success only; ties early, delayed, random_timing",
                     pilot_metrics=metrics, probe_used_for_selection=False)
    (ROOT / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["configure", "merge", "corpus", "report", "select"])
    parser.add_argument("--stage", choices=["pilot", "train", "probe"])
    parser.add_argument("--arms", nargs="+", default=["none", "early", "random_timing", "delayed"])
    parser.add_argument("--arm")
    args = parser.parse_args()
    if args.command == "configure":
        result = configure(args.stage, args.arms)
    elif args.command == "merge":
        result = merge(args.stage)
    elif args.command == "corpus":
        result = corpus(args.arm)
    elif args.command == "select":
        result = select_support()
    else:
        result = report()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
