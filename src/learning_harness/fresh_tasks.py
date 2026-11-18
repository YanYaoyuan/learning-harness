"""Pre-outcome task-group exclusion and disjoint fresh-development allocation."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from .common import digest, file_sha

FAMILIES = ("pick_and_place_simple", "pick_clean_then_place_in_recep")
COUNTS = {"calibration": 16, "development": 64, "confirmation": 64}
SEED = 20261113
PUBLIC_RESET = (
    r"Looking quickly around you, you see (.*?)\.\s*Your task is to:\s*([^\n]+)"
)


def task_group(row):
    if row.get("task_group"):
        return row["task_group"]
    identity = row.get("task_id", "")
    if identity.startswith("alfworld:train/"):
        parts = identity.split(":", 1)[1].split("/")
        if len(parts) >= 3:
            return "/".join(parts[:2])
    return None


def historical_sources(exclude_root):
    paths = []
    pattern = re.compile(
        r"^(?:train|pilot|.*query|.*support.*|.*diagnostic.*|.*calibration.*|.*candidate.*|.*schedule.*|.*examples.*|full-.*|eligible-.*|matched-.*)\.jsonl$"
    )
    for path in Path("runs").rglob("*.jsonl"):
        if path.is_relative_to(exclude_root) or "wandb" in path.parts:
            continue
        if path.name in ("episodes.jsonl", "tasks.jsonl") or pattern.match(path.name):
            paths.append(path)
    paths += [
        Path("data/manifests/alfworld") / (name + ".jsonl")
        for name in ("pilot", "dev", "probe", "train_pilot", "pilot_smoke")
    ]
    return sorted(set(paths))


def exclude_history(paths):
    groups, identities, hashes = set(), set(), {}
    records = 0
    for path in paths:
        raw = path.read_bytes()
        for line in raw.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            group = task_group(row)
            identity = row.get("task_id", "")
            if group and group.startswith("train/"):
                groups.add(group)
            if identity.startswith("alfworld:train/"):
                identities.add(identity)
                records += 1
        hashes[str(path)] = file_sha(path)
    return groups, identities, hashes, records


def candidate_groups(rows, excluded_groups, excluded_ids, seed=SEED):
    by = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["official_split"] != "train" or row["partition"] != "train":
            raise ValueError(
                "Only the original training inventory may supply fresh tasks"
            )
        if row["task_group"] in excluded_groups or row["task_id"] in excluded_ids:
            continue
        by[row["task_type"]][row["task_group"]].append(row)
    result = {}
    for family, groups in by.items():
        order = sorted(groups, key=lambda g: digest([seed, "fresh-group", g]))
        result[family] = [
            sorted(groups[g], key=lambda r: digest([seed, "fresh-game", r["task_id"]]))[
                0
            ]
            for g in order
        ]
    return result


def reset_signature(initial):
    match = re.search(PUBLIC_RESET, initial, re.S)
    if not match:
        raise ValueError(
            "Unrecognized public reset; do not silently skip contamination checks"
        )
    room, goal = match.groups()
    return digest(
        [
            sorted(re.findall(r"[a-z]+ \d+", room)),
            re.sub(r"\s+", " ", goal.lower().strip()),
        ]
    )


def public_demo_signatures():
    texts = list(
        json.loads(
            Path("third_party/ReAct/prompts/alfworld_3prompts.json").read_text()
        ).values()
    )
    texts.append(
        Path(
            "third_party/Reflexion/alfworld_runs/reflexion_few_shot_examples.txt"
        ).read_text()
    )
    return {
        reset_signature(m.group(0))
        for text in texts
        for m in re.finditer(PUBLIC_RESET, text, re.S)
    }


def allocate_roles(eligible, counts=COUNTS):
    result = {role: [] for role in counts}
    for family in FAMILIES:
        candidates = eligible[family]
        if len(candidates) < sum(counts.values()):
            raise ValueError("Insufficient fresh task groups for " + family)
        cursor = 0
        for role, count in counts.items():
            result[role].extend(
                dict(
                    row,
                    partition="fresh_" + role,
                    purpose="development",
                    fresh_role=role,
                    fresh_seed=SEED,
                )
                for row in candidates[cursor : cursor + count]
            )
            cursor += count
    owners = {}
    for role, rows in result.items():
        for row in rows:
            if row["task_group"] in owners:
                raise ValueError("A task group belongs to multiple fresh roles")
            owners[row["task_group"]] = role
        if Counter(row["task_type"] for row in rows) != Counter(
            {f: counts[role] for f in FAMILIES}
        ):
            raise ValueError("Role must be balanced over its declared families")
    return result
