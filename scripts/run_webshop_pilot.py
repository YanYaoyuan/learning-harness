#!/usr/bin/env python3
"""Freeze or run the dedicated, train-only WebShop base-model calibration."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request

from learning_harness.bounded_queue import save
from learning_harness.webshop_factory import stream_sha
from learning_harness.webshop_pilot import (
    IDENTITY_SHA,
    PARENT,
    PARENT_SHA,
    REVISION,
    episode_plan,
    read,
    recover_aggregate,
    require_parent_completed,
    subprocess_environment,
    validate_config,
    validate_frozen,
    validate_health,
)
from learning_harness.webshop_serving import IMPLEMENTATION

CONFIG = Path("configs/webshop-development-pilot-v1.json")
PROTOCOL = Path("docs/WEBSHOP_DEVELOPMENT_PILOT_PROTOCOL_2026-11-13.zh-CN.md")
CODE = [
    "scripts/run_webshop_pilot.py",
    "scripts/webshop_pilot_worker.py",
    "scripts/serve_webshop_base.py",
    "src/learning_harness/webshop_pilot.py",
    "src/learning_harness/webshop_serving.py",
    "src/learning_harness/webshop_agent.py",
    "src/learning_harness/webshop_requests.py",
    "src/learning_harness/webshop_factory.py",
    "src/learning_harness/webshop_public.py",
    "src/learning_harness/api.py",
    "src/learning_harness/bounded_queue.py",
    "src/learning_harness/common.py",
    "src/learning_harness/idle_gpu_contexts.py",
    "scripts/run_action_controls.py",
    "scripts/run_stage0_repeat_eval.py",
    "scripts/action_control_report.py",
    "third_party/ReAct-WebShop/source.json",
    "third_party/ReAct-WebShop/prompts.json",
    "output/webshop-preparation/20261113/react-baseline/source-audit.json",
    "tests/test_webshop_agent.py",
    "tests/test_webshop_requests.py",
    "tests/test_webshop_serving.py",
    "tests/test_webshop_pilot.py",
]


def freeze(root, gpu):
    if (root / "freeze.json").exists() or (root / "execution.json").exists():
        raise ValueError("Never overwrite an existing calibration freeze or attempt")
    config = validate_config(read(CONFIG))
    identity_root = Path(config["identity_root"])
    if stream_sha(identity_root / "freeze.json") != IDENTITY_SHA:
        raise ValueError("Goal identity profile differs")
    if stream_sha(PARENT / "freeze.json") != PARENT_SHA:
        raise ValueError("Unexpected parent study")
    identity = read(identity_root / "freeze.json")
    parent = read(PARENT / "freeze.json")
    model = Path("data/models/Qwen3-4B-Instruct-2507")
    manifest = read(model / "download-manifest.json")
    if manifest["repo"] != config["student_model"] or manifest["revision"] != REVISION:
        raise ValueError("Unexpected base-model manifest")
    weights = {
        entry["path"]: entry["sha256"]
        for entry in manifest["files"]
        if entry["path"].endswith(".safetensors")
    }
    files = dict(identity["files"])
    additional = [
        *CODE,
        str(CONFIG),
        str(PROTOCOL),
        str(identity_root / "freeze.json"),
        str(identity_root / "sealed.json"),
        str(PARENT / "freeze.json"),
        str(model / "download-manifest.json"),
    ]
    for path in additional:
        files[path] = stream_sha(path)
    frozen = dict(
        name="WebShop local base-model train-only calibration",
        config=config,
        cases=episode_plan(config),
        gpu=gpu,
        port=19780 + gpu,
        parent_freeze_sha256=PARENT_SHA,
        model_directory=str(model),
        weights_sha256=hashlib.sha256(
            json.dumps(weights, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        model_python=str(Path(".venv-model/bin/python").absolute()),
        environment_python=str(Path(".venv-webshop/bin/python").absolute()),
        model_runtime=json.loads(
            subprocess.check_output(
                [
                    str(Path(".venv-model/bin/python").absolute()),
                    "-c",
                    "import json; from learning_harness.webshop_serving import model_runtime; print(json.dumps(model_runtime()))",
                ],
                env=subprocess_environment(),
                text=True,
                timeout=30,
            )
        ),
        allowed_background_contexts=parent["allowed_background_contexts"],
        cap_gpu_seconds=3600,
        shutdown_reserve_seconds=20,
        cpu_prepare_timeout_seconds=900,
        server_ready_timeout_seconds=180,
        server_implementation=IMPLEMENTATION,
        adapters=None,
        automatic_retry=False,
        raw_records_uploaded=False,
        protocol_text=PROTOCOL.read_text(),
        protocol_sha256=stream_sha(PROTOCOL),
        files=files,
        frozen_unix=time.time(),
    )
    # Use exclusive creation, including when another operator freezes concurrently.
    with (root / "freeze.json").open("x") as handle:
        handle.write(json.dumps(frozen, indent=2) + "\n")
    validate_frozen(root)
    print(
        json.dumps(
            dict(
                status="frozen_not_executed",
                episodes=64,
                freeze_sha256=stream_sha(root / "freeze.json"),
            )
        ),
        flush=True,
    )


def stop_children(children):
    """Signal all owned process groups first, then wait within one shared grace."""
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    while (
        any(child.poll() is None for child in children) and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for child in children:
        child.wait(timeout=5)


def run(root):
    frozen = validate_frozen(root)
    parent_evidence = require_parent_completed()
    from action_control_report import load_study
    from run_action_controls import gpu_preflight

    load_study(PARENT)
    if (root / "execution.json").exists():
        raise ValueError(
            "Existing attempt retained; inspect it, never retry automatically"
        )
    for name in ("backend-ready.json", "release.json", "local"):
        if (root / name).exists():
            raise ValueError("Earlier partial attempt retained")
    state = dict(
        status="preparing_cpu_backend",
        pid=os.getpid(),
        freeze_sha256=stream_sha(root / "freeze.json"),
        **parent_evidence,
        started_unix=time.time(),
        gpu_seconds=0,
        teacher_calls=0,
        automatic_retry=False,
        children={},
    )
    save(root / "execution.json", state)
    children, logs = [], []
    gpu_started = None

    def start(role, argv, env):
        log = (root / (role + ".log")).open("x")
        logs.append(log)
        child = subprocess.Popen(
            argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        children.append(child)
        state["children"][role] = dict(pid=child.pid, argv=argv)
        save(root / "execution.json", state)
        return child

    def interrupted(*_):
        raise KeyboardInterrupt

    previous = {
        sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        worker = start(
            "worker",
            [
                frozen["environment_python"],
                "scripts/webshop_pilot_worker.py",
                "--root",
                str(root),
            ],
            subprocess_environment(),
        )
        deadline = time.monotonic() + frozen["cpu_prepare_timeout_seconds"]
        while not (root / "backend-ready.json").exists():
            if worker.poll() is not None:
                raise RuntimeError("CPU backend exited before readiness")
            if time.monotonic() >= deadline:
                raise TimeoutError("CPU backend preparation deadline reached")
            time.sleep(0.2)
        ready = read(root / "backend-ready.json")
        if (
            ready["pid"] != worker.pid
            or ready["freeze_sha256"] != state["freeze_sha256"]
            or ready["backend"]["split"] != "train"
            or ready["backend"]["permitted_goals"] != 10587
        ):
            raise ValueError("Wrong CPU backend identity or split")
        state["gpu_preflight"] = gpu_preflight(
            [frozen["gpu"]], frozen["allowed_background_contexts"]
        )
        alias = "webshop-base-" + state["freeze_sha256"][:12]
        gpu_started = time.time()
        state.update(
            status="starting_dedicated_base_server",
            gpu_started_unix=gpu_started,
            gpu=frozen["gpu"],
        )
        save(root / "execution.json", state)
        server = start(
            "server",
            [
                frozen["model_python"],
                "scripts/serve_webshop_base.py",
                "--model",
                frozen["model_directory"],
                "--revision",
                REVISION,
                "--alias",
                alias,
                "--port",
                str(frozen["port"]),
            ],
            subprocess_environment(frozen["gpu"]),
        )
        stop_at = (
            gpu_started + frozen["cap_gpu_seconds"] - frozen["shutdown_reserve_seconds"]
        )
        ready_until = min(stop_at, gpu_started + frozen["server_ready_timeout_seconds"])
        while True:
            if server.poll() is not None or worker.poll() is not None:
                raise RuntimeError("Calibration process exited during server load")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{frozen['port']}/health", timeout=2
                ) as reply:
                    health = json.load(reply)
                validate_health(health, frozen, alias, server.pid)
                break
            except (urllib.error.URLError, TimeoutError):
                if time.time() >= ready_until:
                    raise TimeoutError(
                        "Dedicated base server readiness deadline reached"
                    )
                time.sleep(0.2)
        save(root / "server-health.json", health)
        save(
            root / "release.json",
            dict(
                freeze_sha256=state["freeze_sha256"],
                server_pid=server.pid,
                deadline_unix=stop_at,
            ),
        )
        state["status"] = "running"
        save(root / "execution.json", state)
        while worker.poll() is None:
            if server.poll() is not None:
                raise RuntimeError("Dedicated base server exited during calibration")
            if time.time() >= stop_at:
                state["status"] = "stopped_at_cap"
                break
            state["gpu_seconds"] = time.time() - gpu_started
            state["last_poll_unix"] = time.time()
            save(root / "execution.json", state)
            time.sleep(0.5)
        else:
            state["status"] = "completed" if worker.returncode == 0 else "failed"
    except BaseException as error:
        state.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
        )
        raise
    finally:
        # Prevent a second terminal signal from bypassing child cleanup/accounting.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            stop_children(children)
        finally:
            state.update(
                stopped_unix=time.time(),
                gpu_seconds=0 if gpu_started is None else time.time() - gpu_started,
            )
            state["cap_exceeded"] = state["gpu_seconds"] > frozen["cap_gpu_seconds"]
            state["children_returncodes"] = {
                str(child.pid): child.poll() for child in children
            }
            save(root / "execution.json", state)
            summary = recover_aggregate(root, frozen)
            summary.update(
                execution_status=state["status"],
                gpu_seconds=state["gpu_seconds"],
                cap_exceeded=state["cap_exceeded"],
            )
            save(root / "aggregate-summary.json", summary)
            for log in logs:
                log.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    if (
        state["status"] != "completed"
        or not summary["complete"]
        or state["cap_exceeded"]
    ):
        raise RuntimeError("Calibration incomplete; preserve all partial records")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("runs/webshop-development-pilot-v1")
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    parser.add_argument("--gpu", type=int, choices=[0, 1])
    args = parser.parse_args()
    args.root.resolve().relative_to(Path("runs").resolve())
    if args.freeze != (args.gpu is not None):
        raise ValueError("GPU is required only at freeze, never a run-time override")
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.freeze:
            freeze(args.root, args.gpu)
        elif args.validate_only:
            frozen = validate_frozen(args.root)
            print(
                json.dumps(
                    dict(status="validated_not_executed", episodes=len(frozen["cases"]))
                )
            )
        else:
            for key in ("NO_PROXY", "no_proxy"):
                os.environ[key] = ",".join(
                    filter(None, [os.environ.get(key, ""), "127.0.0.1,localhost,::1"])
                )
            run(args.root)


if __name__ == "__main__":
    main()
