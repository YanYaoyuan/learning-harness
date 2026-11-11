#!/usr/bin/env python3
"""Describe observed binary reward groups; do not extrapolate to larger groups."""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path

from learning_harness.common import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/dual5090-v2")
    args = parser.parse_args()
    root = Path(args.root)
    episodes = read_jsonl(root / "collect/episodes.jsonl")
    result = dict(sampled_rollouts_per_task=2, rollout_seeds=[11, 23], reward="binary environment won",
        assumption_for_learning_signal="A_i = r_i - mean(r_group), with no other reward or loss",
        caveat="Two observed samples per task do not establish zero variance for larger groups or future policies.", arms={})
    for arm in ("none", "early"):
        grouped = defaultdict(list)
        for row in episodes:
            if row["arm"] == arm:
                grouped[row["task_id"]].append(row)
        counts, by_type = Counter(), defaultdict(Counter)
        for rows in grouped.values():
            assert len(rows) == 2 and {r["rollout_seed"] for r in rows} == {11, 23}
            assert all(r["status"] == "completed" for r in rows)
            rewards = [int(r["success"]) for r in rows]
            kind = "mixed" if len(set(rewards)) == 2 else ("both_success" if rewards[0] else "both_failure")
            counts[kind] += 1
            by_type[rows[0]["task_type"]][kind] += 1
        assert counts["both_success"] * 2 + counts["mixed"] == sum(r["success"] for r in episodes if r["arm"] == arm)
        result["arms"][arm] = dict(task_groups=len(grouped), group_counts=dict(counts),
            empirical_nonflat_fraction=counts["mixed"] / len(grouped),
            by_type={key: dict(value) for key, value in by_type.items()})
    (root / "reward-group-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
