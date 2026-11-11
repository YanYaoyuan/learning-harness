from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import uuid

from .api import ChatClient
from .common import digest, file_sha, game_path, read_jsonl
from .environment import AlfWorld
from .support import ARMS, SYSTEM_PROMPT, canonicalize_action, injection_step, messages_for, parse_action


def load_bank(path):
    if path is None:
        return {}
    rows = read_jsonl(path)
    bank = {r["task_id"]: r for r in rows}
    if len(bank) != len(rows):
        raise ValueError("Teacher bank has duplicate tasks")
    return bank


def validate_protocol(config, tasks, evaluation=False):
    if not tasks or len({r["task_id"] for r in tasks}) != len(tasks):
        raise ValueError("Manifest must contain unique tasks")
    if not config["arms"] or len(set(config["arms"])) != len(config["arms"]):
        raise ValueError("Support arms must be nonempty and unique")
    if set(config["arms"]) - ARMS:
        raise ValueError("Unknown support arm")
    if not config["rollout_seeds"] or len(set(config["rollout_seeds"])) != len(config["rollout_seeds"]):
        raise ValueError("Rollout seeds must be nonempty and unique")
    for key in ("max_steps", "max_new_tokens", "max_student_output_tokens", "history_turns"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    support = config["support"]
    if not (0 <= support["delayed_step"] < config["max_steps"] and
            0 <= support["random_first_step"] <= support["random_last_step"] < config["max_steps"]
            and support["visible_steps"] > 0):
        raise ValueError("Support timing is outside episode bounds")
    heldout = any(r["purpose"] == "final_evaluation" for r in tasks)
    if evaluation or heldout:
        if config["arms"] != ["none"] or config.get("teacher_bank"):
            raise ValueError("Independent evaluation must use H0 with no teacher bank")
    return heldout


def rollout(config_path, data_root, output, evaluation=False, limit=None,
            client=None, environment_factory=AlfWorld):
    config = json.loads(Path(config_path).read_text())
    tasks = read_jsonl(config["manifest"])
    heldout = validate_protocol(config, tasks, evaluation)
    if limit:
        if heldout:
            raise ValueError("Do not silently subset final benchmarks; create a named protocol")
        tasks = tasks[:limit]
    bank = load_bank(config.get("teacher_bank"))
    if any(arm != "none" for arm in config["arms"]):
        if any(r["task_id"] not in bank for r in tasks):
            raise ValueError("Teacher bank must cover the complete selected manifest")
    client = client or ChatClient(config["model"])
    revision = os.environ.get("LH_STUDENT_REVISION", "UNPINNED")
    if (evaluation or heldout) and revision == "UNPINNED":
        raise ValueError("Set LH_STUDENT_REVISION to the actual served checkpoint/revision")
    fingerprint = digest(dict(config=config, manifest_sha=file_sha(config["manifest"]),
                              bank_sha=file_sha(config["teacher_bank"]) if config.get("teacher_bank") else None,
                              model=client.model, revision=revision,
                              system_prompt_hash=digest(SYSTEM_PROMPT),
                              endpoint_hash=digest(getattr(client, "base_url", "test-only"))))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "run.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text())["fingerprint"] != fingerprint:
            raise ValueError("Run directory belongs to a different config/model/manifest")
    else:
        metadata_path.write_text(json.dumps(dict(
            fingerprint=fingerprint, config=config, model=client.model, revision=revision,
            created_at=datetime.now(timezone.utc).isoformat(),
            planned_episodes=len(tasks) * len(config["arms"]) * len(config["rollout_seeds"])), indent=2) + "\n")
    episodes_path = output / "episodes.jsonl"
    existing = read_jsonl(episodes_path) if episodes_path.exists() else []
    completed = {(r["task_id"], r["arm"], r["rollout_seed"]) for r in existing}
    if len(completed) != len(existing):
        raise ValueError("Duplicate episode keys in run")
    with (output / "steps.jsonl").open("a") as step_log, episodes_path.open("a") as episode_log, \
            (output / "calls.jsonl").open("a") as call_log:
        for task in tasks:
            for seed in config["rollout_seeds"]:
                order = list(config["arms"])
                random.Random(int(digest([task["task_id"], seed, "arm-order"]), 16)).shuffle(order)
                for arm in order:
                    key = (task["task_id"], arm, seed)
                    if key in completed:
                        continue
                    env = environment_factory(game_path(data_root, task), seed, config["max_steps"])
                    attempt_id = uuid.uuid4().hex
                    history, input_tokens, output_tokens, invalid_actions = [], 0, 0, 0
                    success, injected, status, error_type = False, False, "completed", None
                    start = injection_step(arm, task["task_id"], seed, config["support"])
                    try:
                        initial = env.reset()
                        if arm != "none":
                            card = bank[task["task_id"]]
                            if card["public_reset_sha256"] != digest(initial):
                                raise ValueError("Teacher card was made from a different public reset")
                        for turn in range(config["max_steps"]):
                            remaining = config["max_student_output_tokens"] - output_tokens
                            if remaining <= 0:
                                status = "token_budget"
                                break
                            active = start is not None and start <= turn < start + config["support"]["visible_steps"]
                            hint = bank[task["task_id"]]["hint"] if active else None
                            injected = injected or active
                            messages = messages_for(initial, history, hint, config["history_turns"])
                            response = client.complete(messages, max_tokens=min(config["max_new_tokens"], remaining),
                                                       temperature=config["temperature"], seed=seed + turn)
                            call_log.write(json.dumps(dict(attempt_id=attempt_id, task_id=task["task_id"],
                                arm=arm, rollout_seed=seed, step=turn, input_tokens=response["input_tokens"],
                                output_tokens=response["output_tokens"])) + "\n")
                            call_log.flush()
                            if response["output_tokens"] > min(config["max_new_tokens"], remaining):
                                raise ValueError("Endpoint exceeded requested output budget")
                            input_tokens += response["input_tokens"]
                            output_tokens += response["output_tokens"]
                            raw_action, invalid = parse_action(response["text"])
                            action = canonicalize_action(raw_action)
                            observation, success, done = env.step(action)
                            invalid_actions += int(invalid)
                            event = dict(attempt_id=attempt_id, task_id=task["task_id"], task_type=task["task_type"],
                                         partition=task["partition"], arm=arm, rollout_seed=seed,
                                         step=turn, messages=messages, response=response["text"],
                                         action=action, raw_action=raw_action, observation=observation, success=success,
                                         invalid_action=invalid, support_active=active,
                                         input_tokens=response["input_tokens"], output_tokens=response["output_tokens"],
                                         response_model=response["response_model"])
                            step_log.write(json.dumps(event, ensure_ascii=False) + "\n")
                            step_log.flush()
                            history.append(event)
                            if success or done:
                                break
                    except Exception as exc:
                        status, error_type = "error", type(exc).__name__
                    finally:
                        env.close()
                    record = dict(attempt_id=attempt_id, task_id=task["task_id"], task_group=task["task_group"],
                                  task_type=task["task_type"], partition=task["partition"],
                                  arm=arm, rollout_seed=seed, success=bool(success), status=status,
                                  error_type=error_type, steps=len(history), invalid_actions=invalid_actions,
                                  support_injected=injected, input_tokens=input_tokens, output_tokens=output_tokens)
                    episode_log.write(json.dumps(record) + "\n")
                    episode_log.flush()
                    completed.add(key)
                    print(f"{task['task_id']} arm={arm} seed={seed} success={success} status={status}", flush=True)
                    if status == "error":
                        raise RuntimeError(f"Episode error ({error_type}); inspect logs before continuing")
    return dict(episodes=len(completed), statuses=dict(Counter(r["status"] for r in read_jsonl(episodes_path))))

