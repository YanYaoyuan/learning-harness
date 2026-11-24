#!/usr/bin/env python3
"""Prepare task splits and settings for the dual-GPU pilot."""
from __future__ import annotations

import json
from pathlib import Path

from learning_harness.common import digest, game_path, read_jsonl, write_jsonl
from learning_harness.environment import AlfWorld

ROOT = Path("runs/dual5090-v1")

PUBLIC_STRATEGY = (
    "Track the goal, the object you are carrying, and its required state. "
    "Find the object by visiting observed receptacles and opening containers when needed; "
    "do not invent object locations. Take the object before trying to change its state. "
    "If cleaning, heating or cooling is requested, use the appropriate observed sink, microwave "
    "or fridge before placing it at the goal receptacle. Check the action feedback. "
    "If an action fails, change the action or inspect the current location instead of repeating it. "
    "A transformation does not finish a placement task: carry the processed object to the destination."
)


def subset(path, per_type):
    rows = read_jsonl(path)
    chosen = []
    for family in sorted({r["task_type"] for r in rows}):
        candidates = sorted((r for r in rows if r["task_type"] == family),
                            key=lambda row: digest([20261109, "dual5090-v1", row["task_id"]]))
        chosen.extend(candidates[:per_type])
    return sorted(chosen, key=lambda row: row["task_id"])


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    source = Path("data/manifests/alfworld")
    choices = {"pilot": subset(source / "pilot.jsonl", 3),
               "train": subset(source / "train_pilot.jsonl", 4),
               "probe": subset(source / "probe.jsonl", 3)}
    for name, rows in choices.items():
        write_jsonl(ROOT / f"{name}.jsonl", rows)
    bank = []
    for task in choices["pilot"] + choices["train"]:
        env = AlfWorld(game_path("data/alfworld", task), seed=0)
        try:
            initial = env.reset()
        finally:
            env.close()
        bank.append(dict(task_id=task["task_id"] , game_sha256=task["game_sha256"],
                         public_reset_sha256=digest(initial), hint=PUBLIC_STRATEGY,
                         source="fixed_handwritten_public_strategy", teacher_api_used=False))
    write_jsonl(ROOT / "support.jsonl", bank)
    config = json.loads(Path("configs/lora.json").read_text())
    config.update(max_steps=64, global_batch_size=4, max_sequence_length=2048,
                  training_seeds=[101], distill_kl_weight=0.5)
    (ROOT / "lora.json").write_text(json.dumps(config, indent=2) + "\n")
    receipt = dict(protocol="dual5090-discovery-v1", final_benchmarks_used=False,
                   teacher_api_used=False, support_source="fixed_handwritten_public_strategy",
                   partitions={k: len(v) for k, v in choices.items()},
                   source_arms=["none", "early", "random_timing", "delayed"],
                   training_methods=["target_sft", "mixed_sft", "context_distill"],
                   training_seed=101, interpretation="exploratory single-seed local screening")
    (ROOT / "protocol.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
