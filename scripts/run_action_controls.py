#!/usr/bin/env python3
"""Run an explicitly frozen action-control plan using shared bounded accounting."""

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import action_control_worker as worker
from action_control_report import load_study, output_path
from learning_harness.bounded_queue import run, save


def gpu_preflight(gpus):
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        text=True,
        timeout=15,
    )
    known = {
        int(row[0].strip()): row[1].strip()
        for row in csv.reader(inventory.splitlines())
        if len(row) == 2
    }
    if any(g not in known for g in gpus):
        raise ValueError("Frozen GPU index is unavailable")
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
        timeout=15,
    )
    selected = {known[g] for g in gpus}
    if any(
        row[0].strip() in selected
        for row in csv.reader(processes.splitlines())
        if len(row) >= 2
    ):
        raise ValueError(
            "A selected GPU already has a compute process; inspect before starting"
        )
    from run_stage0_repeat_eval import port_is_free

    if not all(port_is_free(g) for g in gpus):
        raise ValueError("A selected model-server port is occupied")


def disk_preflight(root, frozen):
    parent = json.loads((worker.parent.ROOT / "freeze.json").read_text())
    a = (
        worker.parent.ROOT
        / "cases"
        / parent["cases"][0]["name"]
        / "A/adapter_model.safetensors"
    )
    b = (
        worker.kernel.OLD
        / "heat/trajectories/full-api-seed1101/step-0000/B/adapter_model.safetensors"
    )
    pending = 0
    for case in frozen["cases"]:
        result = root / "cases" / case["name"] / "worker-result.json"
        if not result.exists() or not json.loads(result.read_text()).get("complete"):
            pending += 1
    required = pending * (a.stat().st_size + b.stat().st_size) + 2 * 1024**3
    available = shutil.disk_usage(root).free
    if available < required:
        raise ValueError(
            f"Need {required} bytes for pending A/B weights and records reserve; have {available}"
        )
    return dict(
        pending_models=pending, required_bytes=required, available_bytes=available
    )


def start_observer(root, out):
    path = root / "job.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    old = previous.get("observer", {})
    observer = None
    if old.get("pid"):
        try:
            argv = [
                v.decode()
                for v in (Path("/proc") / str(old["pid"]) / "cmdline")
                .read_bytes()
                .split(b"\0")
                if v
            ]
        except FileNotFoundError:
            argv = []
        if argv:
            if (
                not any(v.endswith("track_bounded_study.py") for v in argv)
                or "--root" not in argv
                or Path(argv[argv.index("--root") + 1]).resolve() != root.resolve()
            ):
                raise ValueError("Recorded observer PID belongs to a different process")
            observer = old
    if observer is None:
        env = dict(
            os.environ,
            PYTHONPATH=str(Path("src").absolute()),
            CUDA_VISIBLE_DEVICES="",
            LH_CONTROL_STUDY_ROOT=str(root),
        )
        for key in list(env):
            if key.startswith("LH_TEACHER_"):
                env.pop(key)
        argv = [
            str(Path(".venv-wandb/bin/python").absolute()),
            "scripts/track_bounded_study.py",
            "--root",
            str(root),
            "--out",
            str(out),
        ]
        with (root / f"observer-{time.time_ns()}.log").open("x") as log:
            child = subprocess.Popen(
                argv,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        observer = dict(
            pid=child.pid,
            argv=argv,
            started_unix=time.time(),
            report_root_environment=str(root),
        )
    record = dict(
        pipeline=dict(
            pid=os.getpid(), argv=[sys.executable, *sys.argv], started_unix=time.time()
        ),
        observer=observer,
    )
    if previous:
        with (root / "job-history.jsonl").open("a") as handle:
            handle.write(json.dumps(previous) + "\n")
    save(path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    frozen = load_study(args.root)
    config = json.loads((args.root / "a-lora.json").read_text())
    for case in frozen["cases"]:
        recipe, _ = worker.load_recipe(Path(case["schedule"]))
        if (
            recipe.get("candidate_only", True)
            or any(recipe[k] != case[k] for k in ("source", "policy", "sampling_seed"))
            or recipe["presentations"]
            != config["max_steps"] * config["global_batch_size"]
        ):
            raise ValueError("Formal recipe, case or optimizer budget mismatch")
    if args.validate_only:
        print(
            json.dumps(
                dict(
                    status="frozen_plan_validated",
                    cases=len(frozen["cases"]),
                    training_started=False,
                )
            ),
            flush=True,
        )
        return
    lock = (args.root / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    storage = disk_preflight(args.root, frozen)
    gpu_preflight(frozen["gpus"])
    out = output_path(args.root)
    out.mkdir(parents=True, exist_ok=True)
    save(args.root / "storage-preflight.json", storage)
    start_observer(args.root, out)

    def command(case, gpu, seconds):
        return [
            worker.engine.MODEL_PY,
            "scripts/action_control_worker.py",
            "--root",
            str(args.root),
            "--case",
            case["name"],
            "--gpu",
            str(gpu),
            "--timeout",
            str(seconds),
        ]

    ledger = run(
        args.root,
        frozen,
        command,
        worker.environment,
        preparatory_seconds=frozen.get("preparatory_gpu_seconds", 0),
    )
    if ledger["status"] == "evaluation_completed":
        env = dict(
            os.environ,
            PYTHONPATH=str(Path("src").absolute()),
            CUDA_VISIBLE_DEVICES="",
            MPLCONFIGDIR="/tmp/lh-mpl",
        )
        with (out / "report.log").open("w") as log:
            subprocess.run(
                [
                    worker.engine.MODEL_PY,
                    frozen["report_script"],
                    "--root",
                    str(args.root),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )
        ledger["status"] = "completed"
        save(args.root / "execution.json", ledger)


if __name__ == "__main__":
    main()
