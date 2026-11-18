#!/usr/bin/env python3
"""Progress-only observer and independent archive finalizer for the Stage 1 grid."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path("runs/variance-diagnostic-v1/stage1")
OUT = Path("output/variance-diagnostic-v1/stage1")


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main(stage="Stage 1", planned=34, study="stage1", report_script="scripts/stage1_variance_report.py", timeout_hours=9):
    import wandb
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = read(ROOT / "freeze.json")
    if not frozen:
        raise ValueError(stage + " is not frozen")
    identity = hashlib.sha256((ROOT / "freeze.json").read_bytes()).hexdigest()[:12]
    target = read(Path("output/wandb/connection.json"))
    base = target.get("base_url", target.get("host"))
    if base:
        os.environ["WANDB_BASE_URL"] = base
    run = wandb.init(entity=target["entity"], project=target["project"], id=identity + "-progress",
        name=stage + " / progress only", group="variance-diagnostic-v1", job_type="progress",
        resume="allow", mode="online", dir=str(OUT.absolute()), save_code=False,
        settings=wandb.Settings(console="off", disable_code=True, disable_git=True, save_code=False,
            disable_job_creation=True, x_disable_stats=True, init_timeout=60))
    receipt = dict(status="watching", run_id=run.id, run_url=run.url,
                   measured_result_run_id=identity, progress_only=True)
    receipt_path = OUT / "progress-wandb.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    deadline = time.monotonic() + timeout_hours * 3600
    terminal = {"completed", "failed", "stopped_at_cap", "interrupted"}
    status = "watching"
    try:
        while True:
            ledger = read(ROOT / "execution.json")
            done = sum(v.get("status") == "completed" for v in ledger.get("cases", {}).values())
            elapsed = ledger.get("gpu_seconds", 0) + sum(max(0, time.time() - v["started_unix"])
                                                         for v in ledger.get("open", {}).values())
            status = ledger.get("status", "not_started")
            if status == "evaluation_completed":
                # The report writes summary.json before its figures. Wait for the controller's
                # completed state, or recover the frozen report if that controller has exited.
                pid = read(ROOT / "job.json").get("pipeline", {}).get("pid")
                alive = True
                if pid:
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        alive = False
                if not alive:
                    env = dict(os.environ, PYTHONPATH=str(Path("src").absolute()), MPLCONFIGDIR="/tmp/lh-mpl")
                    with (OUT / "report-recovery.log").open("w") as log:
                        subprocess.run([str(Path(".venv-model/bin/python").absolute()),
                            report_script], env=env, stdout=log,
                            stderr=subprocess.STDOUT, check=True, timeout=180)
                    status = "report_recovered"
            run.log({"progress/completed_adapters": done, "progress/planned_adapters": planned,
                     "progress/completed_episodes": done * 64,
                     "progress/elapsed_gpu_hours": elapsed / 3600,
                     "progress/cap_gpu_hours": frozen["cap_gpu_hours"], "progress/state": status})
            if status in terminal or status == "report_recovered":
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(stage + " observer deadline")
            time.sleep(30)
        run.summary.update(dict(progress_only=True, termination_state=status,
                                measured_result_run_id=identity, completed_adapters=done))
        receipt.update(status=status, completed_adapters=done)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        run.finish()
    if status not in ("completed", "report_recovered"):
        (OUT / "finalization.json").write_text(json.dumps(dict(status="stopped", execution_state=status), indent=2) + "\n")
        return
    report = read(OUT / "summary.json")
    if not report.get("complete"):
        raise ValueError("Refusing to archive incomplete report")
    finalization = dict(local_report_complete=True, cloud_status="pending")
    final_path = OUT / "finalization.json"
    final_path.write_text(json.dumps(finalization, indent=2) + "\n")
    env = dict(os.environ, PYTHONPATH=str(Path("src").absolute()))
    try:
        with (OUT / "upload.log").open("w") as log:
            subprocess.run([str(Path(".venv-wandb/bin/python").absolute()),
                "scripts/upload_research_results.py", "--study", study], env=env,
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        finalization["cloud_status"] = "uploaded"
    except Exception as error:
        finalization.update(cloud_status="failed", error_type=type(error).__name__)
    final_path.write_text(json.dumps(finalization, indent=2) + "\n")


if __name__ == "__main__":
    main()
