#!/usr/bin/env python3
"""Seal fresh train-derived task roles after history and public-demo audits."""

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import time

from learning_harness.common import file_sha, game_path, read_jsonl, write_jsonl
from learning_harness.fresh_tasks import (
    COUNTS,
    FAMILIES,
    SEED,
    allocate_roles,
    candidate_groups,
    exclude_history,
    historical_sources,
    public_demo_signatures,
    reset_signature,
)

ROOT = Path("runs/fresh-experience-value-v1")


def reset_only(task):
    from learning_harness.environment import AlfWorld

    env = AlfWorld(game_path("data/alfworld", task), 11, 40)
    try:
        initial = env.reset()
        return dict(signature=reset_signature(initial), game_sha256=task["game_sha256"])
    finally:
        env.close()


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Preparation must explicitly hide CUDA")
    if ROOT.exists():
        raise ValueError("Preserve existing fresh-task preparation; never reroll it")
    source = Path("data/manifests/alfworld/train.jsonl")
    rows = read_jsonl(source)
    paths = historical_sources(ROOT)
    excluded, ids, provenance, historical_records = exclude_history(paths)
    candidates = candidate_groups(rows, excluded, ids)
    public = public_demo_signatures()
    ROOT.mkdir(parents=True, exist_ok=False)
    private = ROOT / "preparation.private.json"
    payload = dict(
        history_files=provenance,
        excluded_groups=sorted(excluded),
        excluded_ids=sorted(ids),
        training_inventory_sha256=file_sha(source),
        public_demo_signatures=sorted(public),
    )
    private.write_text(json.dumps(payload, indent=2) + "\n")
    os.chmod(private, 0o600)
    needed = sum(COUNTS.values())
    eligible, signatures, collisions, resets = {}, {}, 0, 0
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for family in FAMILIES:
            eligible[family] = []
            available = iter(candidates.get(family, []))
            while len(eligible[family]) < needed:
                n = min(16, needed - len(eligible[family]))
                batch = [row for _, row in zip(range(n), available)]
                if not batch:
                    raise ValueError("Exhausted fresh groups after demonstration audit")
                for row, check in zip(batch, pool.map(reset_only, batch)):
                    resets += 1
                    if check["signature"] in public:
                        collisions += 1
                        continue
                    eligible[family].append(row)
                    signatures[row["task_id"]] = check["signature"]
                print(
                    json.dumps(
                        dict(
                            family=family,
                            eligible=len(eligible[family]),
                            planned=needed,
                        )
                    ),
                    flush=True,
                )
    roles = allocate_roles(eligible)
    for role, tasks in roles.items():
        if any(r["task_group"] in excluded or r["task_id"] in ids for r in tasks):
            raise ValueError("Historical task overlap")
        write_jsonl(ROOT / "tasks" / (role + ".jsonl"), tasks)
        os.chmod(ROOT / "tasks" / (role + ".jsonl"), 0o600)
    (ROOT / "public-reset-signatures.private.json").write_text(
        json.dumps(signatures, indent=2) + "\n"
    )
    os.chmod(ROOT / "public-reset-signatures.private.json", 0o600)
    result = dict(
        status="fresh_roles_prepared_not_evaluated",
        prepared_unix=time.time(),
        seed=SEED,
        history_files=len(paths),
        historical_records=historical_records,
        excluded_groups=len(excluded),
        unused_group_counts={f: len(v) for f, v in candidates.items()},
        selected_families=list(FAMILIES),
        reason="Unused cooling groups exhausted and heating insufficient after conservative historical exclusion; placement/cleaning are a declared fresh-family expansion, not a new cooling/heating replication.",
        roles={
            role: dict(
                tasks=len(tasks),
                per_family=COUNTS[role],
                sha256=file_sha(ROOT / "tasks" / (role + ".jsonl")),
            )
            for role, tasks in roles.items()
        },
        public_reset_only_audits=resets,
        demonstration_collisions_excluded=collisions,
        environment_steps=0,
        new_model_calls=0,
        teacher_calls=0,
        official_seen_unseen_read=False,
        private_snapshot_sha256=file_sha(private),
        raw_records_uploaded=False,
    )
    (ROOT / "preparation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
