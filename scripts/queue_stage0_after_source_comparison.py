#!/usr/bin/env python3
"""Run the accepted Stage 0 only after the frozen source study and reports finish."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

SOURCE = Path("runs/source-comparison-v1")
SOURCE_OUTPUT = Path("output/source-comparison-v1")
ROOT = Path("runs/variance-diagnostic-v1/stage0")
QUEUE = ROOT / "queue"
PINNED = [
    "docs/VARIANCE_DIAGNOSTIC_STAGE0_SPEC.md",
    "scripts/run_stage0_repeat_eval.py",
    "scripts/stage0_variance_share.py",
    "scripts/queue_stage0_after_source_comparison.py",
    "scripts/transfer_factorial_report.py",
    "runs/transfer-factorial-v1/postrun-figure-audit.json",
]
EXCEPTION_REASON = (
    "Current hash matches runs/transfer-factorial-v1/postrun-figure-audit.json, which records "
    "a post-run shared-y-label layout change and states numerical results were unchanged. "
    "The original source version is unavailable for an independent diff. This report is not "
    "imported or executed on the Stage 0 evaluation path. Historical checksums are preserved."
)


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def source_ready():
    execution = read(SOURCE / "execution.json")
    if execution.get("status") in {"failed", "matching_gate_failed"}:
        return False, "source_failed"
    if execution.get("status") != "completed":
        return False, "waiting_for_source_pipeline"
    summary = read(SOURCE_OUTPUT / "summary.json")
    finalization = read(SOURCE_OUTPUT / "finalization.json")
    if not (summary.get("complete") and summary.get("token_and_budget_audit_passed")
            and summary.get("completed_evaluation_task_runs") == 4284
            and finalization.get("complete")):
        return False, "waiting_for_source_report_and_audit"
    return True, "source_and_report_complete"


def resources_ready(gpu):
    # A pipeline can reserve both GPUs between individual model-server lifetimes.
    # Therefore port checks supplement source_ready(); they never replace it.
    for port in (19780, 19781):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=.5):
                return False
        except OSError:
            pass
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10,
    )
    if result.returncode:
        return False
    free, utilization = (float(value.strip()) for value in result.stdout.strip().split(","))
    return free >= 24576 and utilization < 20


def verify_pins(job):
    for name, expected in job["files"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError("Changed while queued: " + name)


def run_step(label, arguments, job):
    verify_pins(job)
    save(QUEUE / "state.json", dict(status=label, gpu=job["gpu"], updated_unix=time.time()))
    env = dict(os.environ, PYTHONPATH=str(Path("src").resolve()), MPLCONFIGDIR="/tmp/lh-mpl")
    for name in ("NO_PROXY", "no_proxy"):
        env[name] = ",".join(value for value in (env.get(name, ""), "127.0.0.1", "localhost", "::1") if value)
    for key in list(env):
        if key.startswith("LH_TEACHER_"):
            env.pop(key)
    with (QUEUE / f"{label}.log").open("x") as handle:
        subprocess.run([str(Path(".venv-model/bin/python").absolute()), *arguments],
                       env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--wait-hours", type=float, default=24)
    args = parser.parse_args()
    QUEUE.mkdir(parents=True, exist_ok=True)
    with (QUEUE / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (QUEUE / "job.json").exists() or (ROOT / "freeze.json").exists():
            raise SystemExit("A queue receipt or freeze already exists; inspect it before resuming manually")
        job = dict(gpu=args.gpu, cap_gpu_hours=4.0, include_historical=False,
                   reason="Historical environment identity cannot be attested; new replicates are primary",
                   upstream_commit="34b72c2", wait_hours=args.wait_hours, created_unix=time.time(),
                   files={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in PINNED})
        save(QUEUE / "job.json", job)
        deadline = time.monotonic() + args.wait_hours * 3600
        consecutive_ready = 0
        try:
            while True:
                verify_pins(job)
                ready, reason = source_ready()
                if reason == "source_failed":
                    save(QUEUE / "state.json", dict(status="blocked_source_failed"))
                    return
                if ready and resources_ready(args.gpu):
                    consecutive_ready += 1
                    if consecutive_ready >= 2:
                        break
                    reason = "confirming_idle_resources"
                else:
                    consecutive_ready = 0
                    if ready:
                        reason = "waiting_for_idle_gpu_and_closed_model_ports"
                save(QUEUE / "state.json", dict(status=reason, gpu=args.gpu, updated_unix=time.time()))
                if time.monotonic() >= deadline:
                    save(QUEUE / "state.json", dict(status="wait_timeout_no_stage0_started"))
                    return
                time.sleep(15)
            runner = "scripts/run_stage0_repeat_eval.py"
            run_step("smoke", [runner, "--smoke", "--gpu", str(args.gpu)], job)
            if read(ROOT / "execution.json").get("status") != "smoke_completed":
                raise ValueError("Smoke did not finish successfully; no freeze or full run")
            run_step("freeze", [runner, "--freeze", "--gpu", str(args.gpu), "--cap-gpu-hours", "4",
                                "--hash-exception", "scripts/transfer_factorial_report.py=" + EXCEPTION_REASON], job)
            run_step("evaluation", [runner], job)
            execution = read(ROOT / "execution.json")
            if execution.get("status") == "stopped_at_cap":
                # The estimator uses complete replicates only. Zero complete replicates
                # cannot produce an estimate; leave an explicit inconclusive receipt.
                summaries = list(ROOT.glob("rep*/**/summary.json"))
                if len(summaries) < 12:
                    save(QUEUE / "state.json", dict(status="inconclusive_at_cap", stage0_started=True))
                    return
            run_step("analysis", ["scripts/stage0_variance_share.py"], job)
            result = read(ROOT / "result.json")
            save(QUEUE / "state.json", dict(status="completed", decision=result.get("decision"),
                                           finished_unix=time.time(), result=str(ROOT / "result.json")))
        except BaseException as error:
            save(QUEUE / "state.json", dict(status="failed", error_type=type(error).__name__,
                                           error=str(error)[:500], updated_unix=time.time()))
            raise


if __name__ == "__main__":
    main()
