#!/usr/bin/env python3
"""Freeze the larger replication before reading its probe results."""
import json
from pathlib import Path

from learning_harness.common import digest, game_path, read_jsonl, write_jsonl
from learning_harness.environment import AlfWorld
from prepare_dual5090 import PUBLIC_STRATEGY

ROOT = Path("runs/dual5090-v2")


def main():
    if (ROOT / "protocol.json").exists():
        raise ValueError("Existing frozen study: do not overwrite")
    ROOT.mkdir(parents=True, exist_ok=True)
    train = read_jsonl("data/manifests/alfworld/train_pilot.jsonl")
    probe = read_jsonl("data/manifests/alfworld/probe.jsonl")
    old_ids = {r["task_id"] for r in read_jsonl("runs/dual5090-v1/probe.jsonl")}
    fresh = [r for r in probe if r["task_id"] not in old_ids]
    assert len(train) == 96 and len(probe) == 40 and len(fresh) == 28
    assert not {r["task_group"] for r in train} & {r["task_group"] for r in probe}
    for name, rows in (("train", train), ("probe", probe), ("fresh-probe", fresh)):
        write_jsonl(ROOT / f"{name}.jsonl", rows)
    bank = []
    for index, task in enumerate(train):
        env = AlfWorld(game_path("data/alfworld", task), seed=0)
        try:
            initial = env.reset()
        finally:
            env.close()
        bank.append(dict(task_id=task["task_id"], game_sha256=task["game_sha256"],
            public_reset_sha256=digest(initial), hint=PUBLIC_STRATEGY,
            source="fixed_handwritten_public_strategy", teacher_api_used=False))
        if (index + 1) % 8 == 0:
            print(f"Prepared public reset checks {index + 1}/96", flush=True)
    write_jsonl(ROOT / "support.jsonl", bank)
    configs = ROOT / "configs"
    configs.mkdir(exist_ok=True)
    for stage, rows in (("collect", train), ("base", probe)):
        for gpu in (0, 1):
            manifest = ROOT / f"{stage}-gpu{gpu}.jsonl"
            write_jsonl(manifest, rows[gpu::2])
            config = json.loads(Path("configs/probe.json").read_text())
            config.update(protocol_id=f"dual5090-v2-{stage}", manifest=str(manifest), batch_size=8,
                arms=["none", "early"] if stage == "collect" else ["none"],
                teacher_bank=str(ROOT / "support.jsonl") if stage == "collect" else None,
                rollout_seeds=[11, 23] if stage == "collect" else [11],
                max_steps=40, max_new_tokens=128, max_student_output_tokens=5120,
                temperature=0.7 if stage == "collect" else 0.0)
            (configs / f"{stage}-gpu{gpu}.json").write_text(json.dumps(config, indent=2) + "\n")
    config = json.loads(Path("runs/dual5090-v1/lora.json").read_text())
    config.update(training_seeds=[101, 202, 303])
    (ROOT / "lora.json").write_text(json.dumps(config, indent=2) + "\n")
    protocol = dict(protocol="dual5090-v2", train_tasks=96, collection_episodes=384,
        eval_tasks=40, fresh_primary_tasks=28, previous_secondary_tasks=12,
        arms=["none", "early"], training_seeds=[101, 202, 303],
        regimes=["raw", "matched"], method="target_sft", optimizer_steps=64,
        matching="Same task_id and rollout_seed successful in both arms; identical action-example count per pair; evenly spaced steps retain endpoints",
        matching_caveat="Common-success conditioning is diagnostic, not an unbiased population causal effect",
        primary_metric="Fresh28 H0 success; per-seed and paired training-seed/task resampling",
        secondary_metric="All40 H0 and previously inspected12, reported separately",
        batch_protocol="batch-independent-sampling-v1", eval_batch_size=8,
        reuse_v1_rollouts=False, final_benchmarks_used=False, teacher_api_used=False,
        decision="No meta-RL or predictive-law claim; first test reproducibility and selection/count explanation")
    (ROOT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
