#!/usr/bin/env python3
"""Prepare the adaptation pilot and freeze its inputs."""
from collections import defaultdict
import json
from pathlib import Path

from learning_harness.common import digest, file_sha, read_jsonl, write_jsonl

ROOT = Path("runs/learning-speed-v1")
FAMILY = "pick_cool_then_place_in_recep"


def main():
    if (ROOT / "design.json").exists():
        raise ValueError("Design already frozen; do not overwrite")
    root_a = Path("runs/dual5090-v2")
    root_f = Path("runs/dual5090-v2-filtered")
    old_train = read_jsonl("data/manifests/alfworld/train_pilot.jsonl")
    blocked = {r["task_group"] for r in old_train}
    # Verify that the proposed A initialization data contain no B-goal examples.
    types = {r["task_id"]: r["task_type"] for r in old_train}
    for arm in ("none", "early"):
        a = read_jsonl(root_f / f"corpora/filtered-{arm}.jsonl")
        assert all(types[r["task_id"]] != FAMILY for r in a)
    candidates = defaultdict(list)
    for row in read_jsonl("data/manifests/alfworld/train.jsonl"):
        if row["task_type"] == FAMILY and row["task_group"] not in blocked:
            candidates[row["task_group"]].append(row)
    groups = sorted(candidates, key=lambda g: digest([20261110, "learning-speed-v1", g]))
    assert len(groups) >= 84
    selected = [min(candidates[g], key=lambda r: digest(["representative", r["task_id"]])) for g in groups[:84]]
    splits = {"b_support_candidates": (selected[:48], "train"),
              "b_calibration": (selected[48:60], "dev"),
              "b_query": (selected[60:84], "probe")}
    ROOT.mkdir(parents=True, exist_ok=True)
    owned = set(blocked)
    for name, (records, partition) in splits.items():
        current = {r["task_group"] for r in records}
        assert len(current) == len(records) and not owned & current
        owned |= current
        write_jsonl(ROOT / f"{name}.jsonl", [dict(r, partition=partition, purpose="development",
            adaptation_role=name, original_partition=r["partition"]) for r in records])
    retention = [r for r in read_jsonl(root_a / "fresh-probe.jsonl") if r["task_type"] != FAMILY]
    assert len(retention) == 21 and not {r["task_group"] for r in retention} & owned
    write_jsonl(ROOT / "a_retention.jsonl", retention)
    initializations = []
    for seed in (101, 202, 303):
        for arm in ("none", "early"):
            checkpoint = root_f / "checkpoints" / f"filtered-{arm}-seed{seed}"
            meta = json.loads((checkpoint / "experiment.json").read_text())
            assert meta["training_seed"] == seed
            initializations.append(dict(condition=f"filtered-{arm}", a_seed=seed,
                a_adapter=str(checkpoint), a_adapter_sha256=file_sha(checkpoint / "adapter_model.safetensors"),
                b_seed=seed + 1000))
        initializations.append(dict(condition="base", a_seed=None, a_adapter=None,
                                   a_adapter_sha256=None, b_seed=seed + 1000))
    design = dict(status="prepared_not_run", protocol="learning-speed-v1",
        question="Does A-stage assistance change adaptation on identical new B data?",
        b_family=FAMILY, b_support_candidate_tasks=48, b_calibration_tasks=12, b_query_tasks=24,
        old_retention_tasks=21, initializations=initializations, checkpoints=[0, 8, 32, 128],
        b_global_batch=4, b_processed_samples=512, b_data_seed=777,
        b_data_identical_across_conditions=True,
        optimizer_reset=True, trainable_parameters="Fresh B LoRA; freeze A adapter; verify numerical equivalence at step zero",
        b_update_parameterization_status="implementation_and_equivalence_check_required_before_training",
        teacher_status="local reference policy first; API deferred; no teacher requests made",
        teacher_input="Public observations only; optional reusable skill demonstration from prior training trajectories",
        minimum_b_data=dict(successful_task_groups=12, eligible_action_examples=200),
        positive_control=dict(condition="base", calibration_only=True, seeds=3,
            required_additional_calibration_wins=2, required_seeds_meeting_gate=2,
            query_not_used_for_tuning=True),
        main_metrics=["B query success at every checkpoint", "normalized B learning-curve AUC",
            "gain AUC relative to each initialization", "B initial score and action NLL", "A retention"],
        limits=["B goal family absent from A fine-tuning examples, not necessarily absent from pretraining",
            "Task-group disjointness is not room disjointness", "One B family is a pilot, not general meta-learning",
            "AUC differences can reflect prior knowledge as well as plasticity", "3 paired A/B seed trajectories do not separate all randomness sources"],
        final_benchmarks_used=False, paid_api_calls=0, model_calls_made=0)
    (ROOT / "design.json").write_text(json.dumps(design, indent=2) + "\n")
    files = list(ROOT.glob("*.jsonl")) + [ROOT / "design.json"]
    (ROOT / "inputs.sha256.json").write_text(json.dumps({str(p): file_sha(p) for p in files}, indent=2) + "\n")
    print(json.dumps(dict(status=design["status"], output=str(ROOT), b_tasks=84, initializations=9,
                          official_final_test_used=False, model_requests=0), indent=2))


if __name__ == "__main__":
    main()
