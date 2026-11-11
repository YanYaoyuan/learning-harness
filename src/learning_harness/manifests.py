from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

from .common import digest, file_sha, write_jsonl

CORE_TYPES = {
    "pick_and_place_simple", "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep",
}
HELDOUT_TYPES = {"pick_two_obj_and_place", "look_at_obj_in_light"}
ALL_TYPES = CORE_TYPES | HELDOUT_TYPES


def prepare(root, out, per_type=10, seed=20261109):
    root, out = Path(root).resolve(), Path(out)
    rows = []
    for split in ("train", "valid_seen", "valid_unseen"):
        base = root / "json_2.1.1" / split
        if not base.is_dir():
            raise FileNotFoundError(f"Missing official split: {base}")
        for metadata in sorted(base.rglob("traj_data.json")):
            game = metadata.with_name("game.tw-pddl")
            if not game.exists() or any(s in str(metadata) for s in ("movable", "Sliced")):
                continue
            task_type = json.loads(metadata.read_text())["task_type"]
            game_data = json.loads(game.read_text())
            if task_type not in ALL_TYPES or not game_data.get("solvable", False):
                continue
            relative = metadata.parent.relative_to(base)
            rows.append(dict(task_id=f"alfworld:{split}/{relative.as_posix()}",
                             official_split=split, task_type=task_type,
                             task_group=f"{split}/{relative.parts[0]}",
                             game_relpath=game.relative_to(root).as_posix(),
                             game_sha256=file_sha(game), metadata_sha256=file_sha(metadata)))
    if not rows:
        raise ValueError("No solvable ALFWorld games")
    train = [r for r in rows if r["official_split"] == "train"]
    groups = defaultdict(list)
    for row in train:
        if row["task_type"] in CORE_TYPES:
            groups[row["task_type"]].append(row["task_group"])
    assigned = {}
    for task_type in sorted(CORE_TYPES):
        ordered = sorted(set(groups[task_type]), key=lambda group: digest([seed, group]))
        if len(ordered) < 3 * per_type + 1:
            raise ValueError(f"Insufficient groups for {task_type}: {len(ordered)}")
        for index, partition in enumerate(("pilot", "dev", "probe")):
            for group in ordered[index * per_type:(index + 1) * per_type]:
                assigned[group] = partition
    partitions = defaultdict(list)
    seen_reserved = set()
    for row in rows:
        if row["official_split"] == "valid_seen":
            partition = "test_seen"
        elif row["official_split"] == "valid_unseen":
            partition = "test_unseen"
        elif row["task_type"] in HELDOUT_TYPES:
            # These tasks are never used for training, teacher calls or selection.
            continue
        else:
            partition = assigned.get(row["task_group"], "train")
            if partition != "train":
                if row["task_group"] in seen_reserved:
                    continue
                seen_reserved.add(row["task_group"])
        purpose = "final_evaluation" if partition.startswith("test_") else "development"
        partitions[partition].append(dict(row, partition=partition, purpose=purpose))
    # Structural separation of train-derived development groups.
    ownership = {}
    for name in ("pilot", "dev", "probe", "train"):
        for row in partitions[name]:
            previous = ownership.setdefault(row["task_group"], name)
            if previous != name:
                raise ValueError("Task-group leakage between train-derived partitions")
    for name in ("test_seen", "test_unseen"):
        partitions[name + "_id_types"] = [r for r in partitions[name] if r["task_type"] in CORE_TYPES]
        partitions[name + "_heldout_types"] = [r for r in partitions[name] if r["task_type"] in HELDOUT_TYPES]
    # A small training pilot, disjoint from the no-update pilot/dev/probe sets.
    partitions["train_pilot"] = []
    for task_type in sorted(CORE_TYPES):
        candidates = sorted((r for r in partitions["train"] if r["task_type"] == task_type),
                            key=lambda r: digest([seed, "train-pilot", r["task_id"]]))
        partitions["train_pilot"].extend(candidates[:24])
    partitions["pilot_smoke"] = []
    for task_type in sorted(CORE_TYPES):
        partitions["pilot_smoke"].append(next(r for r in partitions["pilot"] if r["task_type"] == task_type))
    out.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, records in sorted(partitions.items()):
        records.sort(key=lambda r: r["task_id"])
        target = out / f"{name}.jsonl"
        write_jsonl(target, records)
        outputs[name] = dict(count=len(records), sha256=file_sha(target),
                             task_types=dict(Counter(r["task_type"] for r in records)))
    summary = dict(protocol="alfworld-learning-v1", seed=seed, per_type=per_type,
                   core_types=sorted(CORE_TYPES), heldout_types=sorted(HELDOUT_TYPES),
                   official_counts=dict(Counter(r["official_split"] for r in rows)), partitions=outputs)
    (out / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary

