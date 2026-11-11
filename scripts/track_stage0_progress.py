#!/usr/bin/env python3
"""Independent progress-only W&B observer; never reads or compares success outcomes."""
import hashlib
import json
import os
from pathlib import Path
import time

import wandb

ROOT = Path("runs/variance-diagnostic-v1/stage0")
OUT = Path("output/variance-diagnostic-v1/stage0")


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 28 * 3600
    while not (ROOT / "freeze.json").exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("No Stage 0 freeze")
        time.sleep(15)
    identity = hashlib.sha256((ROOT / "freeze.json").read_bytes()).hexdigest()[:12]
    target = read(Path("output/wandb/connection.json"))
    base = target.get("base_url", target.get("host"))
    if base:
        os.environ["WANDB_BASE_URL"] = base
    freeze = read(ROOT / "freeze.json")
    run = wandb.init(entity=target["entity"], project=target["project"], id=identity + "-progress",
        name="Stage 0 / progress only", group="variance-diagnostic-v1", job_type="progress",
        resume="allow", mode="online", dir=str(OUT.absolute()), save_code=False,
        settings=wandb.Settings(console="off", disable_code=True, disable_git=True, save_code=False,
            disable_job_creation=True, x_disable_stats=True, init_timeout=60))
    receipt = OUT / "progress-wandb.json"
    receipt.write_text(json.dumps(dict(status="watching", run_id=run.id, run_url=run.url,
        measured_result_run_id=identity, progress_only=True), indent=2) + "\n")
    try:
        while True:
            ledger, queue = read(ROOT / "execution.json"), read(ROOT / "queue/state.json")
            attempts = ledger.get("attempts", [])
            done = [a for a in attempts if a["outcome"] == "completed"]
            completed = sum("/rep" in a["directory"] for a in done)
            reserved_seconds = ledger.get("gpu_seconds", 0.0)
            if ledger.get("open"):
                reserved_seconds += max(0.0, time.time() - ledger["open"]["started_unix"])
            run.log({"progress/completed_estimator_evaluations": completed,
                     "progress/planned_estimator_evaluations": 36,
                     "progress/completed_all_evaluations": len(done),
                     "progress/reserved_gpu_hours": reserved_seconds / 3600,
                     "progress/cap_gpu_hours": freeze["cap_gpu_hours"],
                     "progress/queue_state": queue.get("status", "unknown")})
            status = queue.get("status")
            if status in ("completed", "failed", "blocked_source_failed", "wait_timeout_no_stage0_started", "inconclusive_at_cap"):
                run.summary.update(progress_only=True, termination_state=status,
                                   measured_result_run_id=identity)
                receipt.write_text(json.dumps(dict(status=status, run_id=run.id, run_url=run.url,
                    measured_result_run_id=identity, progress_only=True), indent=2) + "\n")
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("Progress observer deadline")
            time.sleep(30)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
