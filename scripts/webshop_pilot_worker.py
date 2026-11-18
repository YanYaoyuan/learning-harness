#!/usr/bin/env python3
"""CPU backend, local base-model HTTP requests, exclusive per-episode journals."""

import argparse
import json
import os
from pathlib import Path
import time
import urllib.request

from learning_harness.bounded_queue import save
from learning_harness.webshop_agent import LineStopClient, run_episode
from learning_harness.webshop_factory import build_full_environment, stream_sha
from learning_harness.webshop_pilot import (
    read,
    recover_aggregate,
    require_parent_completed,
    validate_frozen,
    validate_health,
)
from learning_harness.webshop_requests import RequestJournal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    frozen = validate_frozen(root)
    require_parent_completed()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("The environment/agent process must be CPU-only")
    if (root / "backend-ready.json").exists() or (root / "local").exists():
        raise ValueError("Preserve all earlier attempts; no silent resume")
    os.nice(10)
    config = frozen["config"]
    factory = build_full_environment(config["identity_root"], "train")
    save(
        root / "backend-ready.json",
        dict(
            pid=os.getpid(),
            backend=factory.metadata,
            freeze_sha256=stream_sha(root / "freeze.json"),
        ),
    )
    deadline = time.monotonic() + 900
    while not (root / "release.json").exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Dedicated server was not released")
        time.sleep(0.2)
    release = read(root / "release.json")
    alias = "webshop-base-" + stream_sha(root / "freeze.json")[:12]
    base = f"http://127.0.0.1:{frozen['port']}"
    with urllib.request.urlopen(base + "/health", timeout=5) as reply:
        validate_health(json.load(reply), frozen, alias, release["server_pid"])
    if release["freeze_sha256"] != stream_sha(root / "freeze.json"):
        raise ValueError("Mismatched server release")
    os.environ["LH_WEBSHOP_URL"] = base + "/v1"
    os.environ["LH_WEBSHOP_MODEL"] = alias
    os.environ.pop("LH_WEBSHOP_KEY", None)
    client = LineStopClient(
        dict(
            base_url_env="LH_WEBSHOP_URL",
            model_env="LH_WEBSHOP_MODEL",
            api_key_env="LH_WEBSHOP_KEY",
        )
    )
    prompts = read(config["prompt_source"])
    try:
        for case in frozen["cases"]:
            if time.time() >= release["deadline_unix"]:
                raise TimeoutError("GPU cap reached before next episode")
            directory = root / "local" / case["name"]
            directory.mkdir(parents=True, exist_ok=False)
            save(
                root / "worker-state.json",
                dict(status="running", case=case, completed_unix=time.time()),
            )
            with (
                RequestJournal(directory / "requests.jsonl") as journal,
                (directory / "events.jsonl").open("x") as events,
            ):

                def on_event(event):
                    events.write(json.dumps(event) + "\n")
                    events.flush()

                result = run_episode(
                    client,
                    factory.episode(case["global_goal_index"]),
                    prompts[case["mode"]],
                    max_calls=config["max_executed_decisions"],
                    max_new_tokens=config["max_new_tokens"],
                    max_context_characters=config["max_context_characters"],
                    seed=config["rollout_seed"],
                    on_event=on_event,
                    on_request=journal,
                )
            save(
                directory / "result.json",
                dict(
                    case=case,
                    result=result,
                    freeze_sha256=stream_sha(root / "freeze.json"),
                ),
            )
            save(root / "aggregate-summary.json", recover_aggregate(root, frozen))
        if factory._server.user_sessions:
            raise ValueError("Closed episodes left backend sessions")
        save(
            root / "worker-state.json",
            dict(status="completed", completed_unix=time.time()),
        )
    except BaseException as error:
        save(
            root / "worker-state.json",
            dict(
                status="failed",
                error_type=type(error).__name__,
                stopped_unix=time.time(),
            ),
        )
        raise


if __name__ == "__main__":
    main()
