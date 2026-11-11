"""Fixed-slot batched rollouts; sampling RNG streams are independent by request."""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import random
import time
import urllib.request
from urllib.parse import urlsplit
import uuid

from .api import ChatClient
from .common import digest, file_sha, game_path, read_jsonl
from .environment import AlfWorld
from .runner import load_bank, validate_protocol
from .support import SYSTEM_PROMPT, canonicalize_action, injection_step, messages_for, parse_action


class BatchClient(ChatClient):
    def __init__(self, settings):
        super().__init__(settings)
        if urlsplit(self.base_url).hostname not in ("127.0.0.1", "localhost"):
            raise ValueError("The internal batch protocol only supports local endpoints")

    def complete_many(self, requests, max_tokens, temperature, revision):
        payload = dict(model=self.model, requests=requests, max_tokens=max_tokens, temperature=temperature)
        request = urllib.request.Request(self.base_url + "/batch/chat/completions",
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=240) as response:
            result = json.load(response)
        if result["revision"] != revision or result["implementation"] != "batch-independent-sampling-v1":
            raise ValueError("Served checkpoint or sampling implementation changed")
        if len(result["results"]) != len(requests):
            raise ValueError("Missing batch result")
        for row in result["results"]:
            if (not isinstance(row["text"], str) or row["response_model"] != self.model or
                not isinstance(row["input_tokens"], int) or row["input_tokens"] <= 0 or
                not isinstance(row["output_tokens"], int) or not 0 < row["output_tokens"] <= max_tokens):
                raise ValueError("Invalid result or token accounting")
        return result


def rollout(config_path, data_root, output, evaluation=False, client=None, environment_factory=None):
    if environment_factory is None:
        from .remote_environment import RemoteAlfWorld
        environment_factory = RemoteAlfWorld
    parallel_env = bool(getattr(environment_factory, "supports_async_step", False))
    config = json.loads(Path(config_path).read_text())
    tasks = read_jsonl(config["manifest"])
    validate_protocol(config, tasks, evaluation)
    batch_size = config["batch_size"]
    if not isinstance(batch_size, int) or not 1 <= batch_size <= 8:
        raise ValueError("Batch size must be between one and eight")
    if config["max_student_output_tokens"] != config["max_steps"] * config["max_new_tokens"]:
        raise ValueError("Batch protocol requires a uniform per-turn output budget")
    bank = load_bank(config.get("teacher_bank"))
    if any(arm != "none" for arm in config["arms"]) and any(t["task_id"] not in bank for t in tasks):
        raise ValueError("Incomplete support bank")
    client = client or BatchClient(config["model"])
    revision = os.environ.get("LH_STUDENT_REVISION", "UNPINNED")
    if revision == "UNPINNED":
        raise ValueError("Pin the local checkpoint revision")
    fingerprint_fields = dict(config=config, manifest_sha=file_sha(config["manifest"]),
        bank_sha=file_sha(config["teacher_bank"]) if config.get("teacher_bank") else None,
        system_prompt_hash=digest(SYSTEM_PROMPT), model=client.model, revision=revision,
        implementation="batch-independent-sampling-v1", endpoint_hash=digest(client.base_url))
    if parallel_env:
        fingerprint_fields["environment_backend"] = "process-per-environment-v1"
    fingerprint = digest(fingerprint_fields)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    meta = output / "run.json"
    planned = len(tasks) * len(config["arms"]) * len(config["rollout_seeds"])
    if meta.exists():
        if json.loads(meta.read_text())["fingerprint"] != fingerprint:
            raise ValueError("Run fingerprint changed")
    else:
        meta.write_text(json.dumps(dict(fingerprint=fingerprint, config=config, model=client.model,
            revision=revision, planned_episodes=planned, batch_protocol="batch-independent-sampling-v1",
            environment_backend="process-per-environment-v1" if parallel_env else "serial"), indent=2) + "\n")
    episodes_path = output / "episodes.jsonl"
    recorded = read_jsonl(episodes_path) if episodes_path.exists() else []
    done_keys = {(r["task_id"], r["arm"], r["rollout_seed"]) for r in recorded}
    if len(done_keys) != len(recorded) or any(r["status"] == "error" for r in recorded):
        raise ValueError("Duplicate or failed episodes; inspect before resuming")
    jobs = []
    for task in tasks:
        for seed in config["rollout_seeds"]:
            arms = list(config["arms"])
            random.Random(int(digest([task["task_id"], seed, "arm-order"]), 16)).shuffle(arms)
            jobs.extend((task, arm, seed) for arm in arms if (task["task_id"], arm, seed) not in done_keys)
    iterator = iter(jobs)
    slots = [None] * batch_size
    batch_index = 0
    timings = dict(environment_initialization_seconds=0.0, model_requests_seconds=0.0,
                   environment_and_logging_seconds=0.0)
    started = time.monotonic()
    with (output / "steps.jsonl").open("a") as steps, episodes_path.open("a") as episodes, \
            (output / "calls.jsonl").open("a") as calls, (output / "batches.jsonl").open("a") as batches:
        try:
            while True:
                phase_start = time.monotonic()
                for slot in range(batch_size):
                    if slots[slot] is not None:
                        continue
                    job = next(iterator, None)
                    if job is None:
                        continue
                    task, arm, seed = job
                    env = environment_factory(game_path(data_root, task), seed, config["max_steps"])
                    initial = None
                    try:
                        initial = None if parallel_env else env.reset()
                        if not parallel_env and arm != "none" and bank[task["task_id"]]["public_reset_sha256"] != digest(initial):
                            raise ValueError("Support reset mismatch")
                    except Exception:
                        env.close()
                        raise
                    slots[slot] = dict(task=task, arm=arm, seed=seed, env=env, initial=initial,
                        history=[], attempt_id=uuid.uuid4().hex, input_tokens=0, output_tokens=0,
                        invalid_actions=0, injected=False,
                        start=injection_step(arm, task["task_id"], seed, config["support"]))
                active = [state for state in slots if state is not None]
                if not active:
                    break
                if parallel_env:
                    for state in active:
                        if state["initial"] is None:
                            state["initial"] = state["env"].reset()
                            if state["arm"] != "none" and bank[state["task"]["task_id"]]["public_reset_sha256"] != digest(state["initial"]):
                                raise ValueError("Support reset mismatch")
                timings["environment_initialization_seconds"] += time.monotonic() - phase_start
                requests, contexts, exposures = [], [], []
                for state in active:
                    turn = len(state["history"])
                    exposed = state["start"] is not None and state["start"] <= turn < state["start"] + config["support"]["visible_steps"]
                    hint = bank[state["task"]["task_id"]]["hint"] if exposed else None
                    messages = messages_for(state["initial"], state["history"], hint, config["history_turns"])
                    requests.append(dict(messages=messages, seed=state["seed"] + turn))
                    contexts.append(messages)
                    exposures.append(exposed)
                phase_start = time.monotonic()
                result = client.complete_many(requests, config["max_new_tokens"], config["temperature"], revision)
                timings["model_requests_seconds"] += time.monotonic() - phase_start
                phase_start = time.monotonic()
                batches.write(json.dumps(dict(batch_index=batch_index, requests=len(active),
                    padded_prompt_tokens=result["padded_prompt_tokens"],
                    attempts=[s["attempt_id"] for s in active])) + "\n")
                batches.flush()
                parsed = []
                for state, response in zip(active, result["results"]):
                    raw_action, invalid = parse_action(response["text"])
                    action = canonicalize_action(raw_action)
                    parsed.append((raw_action, action, invalid))
                    if parallel_env:
                        state["env"].step_async(action)
                for state, response, messages, exposed, (raw_action, action, invalid) in zip(active, result["results"], contexts, exposures, parsed):
                    task, arm, seed = state["task"], state["arm"], state["seed"]
                    turn = len(state["history"])
                    common = dict(attempt_id=state["attempt_id"], task_id=task["task_id"], arm=arm, rollout_seed=seed)
                    calls.write(json.dumps(dict(common, step=turn, batch_index=batch_index,
                        input_tokens=response["input_tokens"], output_tokens=response["output_tokens"])) + "\n")
                    calls.flush()
                    observation, success, done = state["env"].step_wait() if parallel_env else state["env"].step(action)
                    state["input_tokens"] += response["input_tokens"]
                    state["output_tokens"] += response["output_tokens"]
                    state["invalid_actions"] += int(invalid)
                    state["injected"] |= exposed
                    event = dict(common, task_type=task["task_type"], partition=task["partition"], step=turn,
                        messages=messages, response=response["text"], action=action, raw_action=raw_action,
                        observation=observation, success=success, invalid_action=invalid, support_active=exposed,
                        input_tokens=response["input_tokens"], output_tokens=response["output_tokens"],
                        response_model=response["response_model"], batch_index=batch_index)
                    steps.write(json.dumps(event, ensure_ascii=False) + "\n")
                    steps.flush()
                    state["history"].append(event)
                    if success or done or len(state["history"]) >= config["max_steps"]:
                        record = dict(common, task_group=task["task_group"], task_type=task["task_type"],
                            partition=task["partition"], success=bool(success), status="completed", error_type=None,
                            steps=len(state["history"]), invalid_actions=state["invalid_actions"],
                            support_injected=state["injected"], input_tokens=state["input_tokens"], output_tokens=state["output_tokens"])
                        episodes.write(json.dumps(record) + "\n")
                        episodes.flush()
                        state["env"].close()
                        slots[next(i for i, s in enumerate(slots) if s is state)] = None
                        print(f"completed {task['task_id']} {arm} seed={seed} success={success}", flush=True)
                batch_index += 1
                timings["environment_and_logging_seconds"] += time.monotonic() - phase_start
        finally:
            for state in slots:
                if state is not None:
                    state["env"].close()
    timings["total_seconds"] = time.monotonic() - started
    (output / "timings.json").write_text(json.dumps(timings, indent=2) + "\n")
    return dict(planned=planned, recorded=len(read_jsonl(episodes_path)),
                statuses=dict(Counter(r["status"] for r in read_jsonl(episodes_path))))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-root", default="data/alfworld")
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--serial-env", action="store_true")
    args = parser.parse_args()
    print(json.dumps(rollout(args.config, args.data_root, args.out, args.evaluation,
                             environment_factory=AlfWorld if args.serial_env else None), indent=2))


if __name__ == "__main__":
    main()
