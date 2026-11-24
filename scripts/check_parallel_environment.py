#!/usr/bin/env python3
"""Compare environment resets, feedback, and outcomes by replaying reference actions."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from learning_harness.common import game_path, read_jsonl
from learning_harness.remote_environment import RemoteAlfWorld


def main():
    root = Path("runs/dual5090-v2")
    tasks = {r["task_id"]: r for r in read_jsonl(root / "probe.jsonl")}
    episodes = read_jsonl(root / "evaluation/base/episodes.jsonl")
    steps = read_jsonl(root / "evaluation/base/steps.jsonl")
    references = {r["attempt_id"]: [] for r in episodes}
    for row in steps:
        if row["attempt_id"] in references:
            references[row["attempt_id"]].append(row)
    started = time.monotonic()

    def replay(episode):
        rows = sorted(references[episode["attempt_id"]], key=lambda r: r["step"])
        env = RemoteAlfWorld(game_path("data/alfworld", tasks[episode["task_id"]]), episode["rollout_seed"], 40)
        try:
            assert env.reset() == rows[0]["messages"][1]["content"], "Initial observation changed"
            for row in rows:
                observation, success, done = env.step(row["action"])
                assert observation == row["observation"], (episode["task_id"], row["step"], "feedback mismatch")
                assert success == row["success"], "Won bit changed"
            assert bool(success) == episode["success"]
            return len(rows)
        finally:
            env.close()

    with ThreadPoolExecutor(max_workers=8) as pool:
        counts = list(pool.map(replay, episodes))
    result = dict(passed=True, episodes=len(episodes), steps=sum(counts),
        exact_reset_feedback_won_match=True, model_policy_changed=False,
        backend="process-per-environment-v1", seconds=time.monotonic() - started,
        reference="evaluation/base", interpretation="Environment-only execution equivalence; batching order and inference remain unchanged")
    (root / "parallel-environment-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
