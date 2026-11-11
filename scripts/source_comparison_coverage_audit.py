#!/usr/bin/env python3
"""Descriptive coverage and actual matched-SFT exposure; no learner outcome selection."""
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from learning_harness.common import digest, read_jsonl

ROOT = Path("runs/source-comparison-v1")
OUT = Path("output/source-comparison-v1/supplementary/coverage")
SOURCES = ("api", "react")
SEEDS = (101, 202, 303)
TYPES = ("pick_and_place_simple", "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep")


def main():
    index = {r["task_id"]: r for r in read_jsonl(ROOT / "train.jsonl")}
    selection = json.loads((ROOT / "selection.json").read_text())
    recorded_schedules = json.loads((ROOT / "paired-a-schedules.json").read_text())
    spec = importlib.util.spec_from_file_location("frozen_training_schedule", "scripts/train_lora.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    collection, success_sets, pools, by_source = {}, {}, {}, {}
    for source in SOURCES:
        rows = read_jsonl(ROOT / "collection" / source / "tasks.jsonl")
        pairs = {(r["task_id"], r["seed"]) for r in rows}
        if len(rows) != 144 or len(pairs) != 144:
            raise ValueError("Collection is incomplete or duplicated")
        success_sets[source] = {(r["task_id"], r["seed"]) for r in rows if r["success"]}
        collection[source] = {}
        for family in TYPES:
            family_rows = [r for r in rows if index[r["task_id"]]["task_type"] == family]
            won = [r for r in family_rows if r["success"]]
            collection[source][family] = dict(attempts=len(family_rows), successes=len(won),
                success_rate=len(won) / len(family_rows), successful_task_groups=len({r["task_group"] for r in won}))
        pools[source] = read_jsonl(ROOT / "corpora" / f"matched-{source}.jsonl")
        eligible = read_jsonl(ROOT / "corpora" / f"eligible-{source}.jsonl")
        by_source[source] = dict(
            eligible_action_labels=dict(Counter(index[r["task_id"]]["task_type"] for r in eligible)),
            matched_action_labels=dict(Counter(index[r["task_id"]]["task_type"] for r in pools[source])))
    common = success_sets["api"] & success_sets["react"]
    if len(common) != selection["common_task_seed_pairs"]:
        raise ValueError("Successful-pair intersection differs from the frozen selection")
    exposure = []
    for seed in SEEDS:
        paired_keys = []
        for source in SOURCES:
            rows = pools[source]
            order = trainer.fixed_schedule(len(rows), 256, seed)
            keys = [(rows[i]["task_id"], rows[i]["rollout_seed"], rows[i]["matched_index"]) for i in order]
            paired_keys.append(keys)
            meta = json.loads((ROOT / "a-checkpoints" / f"{source}-seed{seed}" / "experiment.json").read_text())
            if hashlib.sha256(json.dumps(order).encode()).hexdigest() != meta["schedule_sha256"]:
                raise ValueError("Reconstructed order differs from the actual training schedule")
            if digest(keys) != recorded_schedules[str(seed)]["paired_training_keys_sha256"]:
                raise ValueError("Paired training keys differ from the frozen record")
            exposure.append(dict(source=source, seed=seed,
                processed_labels=dict(Counter(index[rows[i]["task_id"]]["task_type"] for i in order)),
                unique_examples=len(set(order)), input_tokens=meta["input_tokens_rank0"],
                target_tokens=meta["target_tokens_rank0"]))
        if paired_keys[0] != paired_keys[1]:
            raise ValueError("Actual training exposure is not paired")
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(descriptive=True, learner_outcomes_read=False, collection=collection,
        pair_coverage=dict(common=len(common), api_only=len(success_sets["api"] - common),
                           react_only=len(success_sets["react"] - common)),
        corpus_composition=by_source, actual_training_exposure=exposure,
        caveat="Mutual-success conditioning excludes source-specific coverage. Equal labels do not imply equal tokens or compute.")
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    x = np.arange(3)
    for source, offset, color, label in (("api", -.18, "#2878a6", "Candidate: teacher-assisted"),
                                       ("react", .18, "#dc7945", "Baseline: ReAct self-collection")):
        y = [100 * collection[source][f]["success_rate"] for f in TYPES]
        axes[0].bar(x + offset, y, .34, color=color, label=label)
        for location, family, height in zip(x + offset, TYPES, y):
            entry = collection[source][family]
            axes[0].text(location, height + 1, f"{entry['successes']}/{entry['attempts']}", ha="center", fontsize=8)
    axes[0].set_ylim(0, 110)
    axes[0].set_ylabel("Collection success (%)")
    axes[0].set_title("Collection performance by task type", loc="left")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    values = np.array([[r["processed_labels"].get(f, 0) for f in TYPES]
                       for r in exposure if r["source"] == "api"])
    axes[1].bar(x, values.mean(axis=0), color="#777777", alpha=.65)
    for i, family in enumerate(TYPES):
        axes[1].scatter(i + np.array([-.10, 0, .10]), values[:, i], color="#222222", s=20)
    axes[1].set_ylabel("Action-label presentations")
    axes[1].set_title("Matched SFT: identical counts for both sources", loc="left")
    for ax in axes:
        ax.set_xticks(x, ["Place", "Clean", "Heat"])
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.text(.5, .015, "Left: fresh collection, 144 attempts per source. Right: 256 presentations per seed; dots show three seeds.\nCommon successes: 58 pairs. Teacher-only: 11 pairs. ReAct-only: 52 pairs. No student evaluation outcomes used.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .11, 1, 1))
    fig.savefig(OUT / "coverage-and-exposure.png", dpi=180)
    fig.savefig(OUT / "coverage-and-exposure.pdf")
    plt.close(fig)
    print(json.dumps(dict(output=str(OUT), pair_coverage=result["pair_coverage"], paired_training_keys_verified=True)))


if __name__ == "__main__":
    main()
