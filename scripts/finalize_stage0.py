#!/usr/bin/env python3
"""Generate Stage 0 figures and archive results after its independent queue finishes."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path("runs/variance-diagnostic-v1/stage0")
OUT = Path("output/variance-diagnostic-v1/stage0")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 28 * 3600
    while True:
        state_path = ROOT / "queue/state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get("status") == "completed" and (ROOT / "result.json").exists():
            break
        if state.get("status") in ("failed", "blocked_source_failed", "wait_timeout_no_stage0_started", "inconclusive_at_cap"):
            (OUT / "finalization.json").write_text(json.dumps(dict(status="stopped", queue_status=state), indent=2) + "\n")
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Stage 0 did not finish within the finalizer deadline")
        time.sleep(15)
    env = dict(os.environ, PYTHONPATH=str(Path("src").absolute()), MPLCONFIGDIR="/tmp/lh-mpl")
    with (OUT / "report.log").open("w") as handle:
        subprocess.run([str(Path(".venv-model/bin/python").absolute()), "scripts/report_stage0.py"],
                       env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    receipt = dict(local_report_complete=True, cloud_status="pending")
    (OUT / "finalization.json").write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        with (OUT / "upload.log").open("w") as handle:
            subprocess.run([str(Path(".venv-wandb/bin/python").absolute()),
                            "scripts/upload_research_results.py", "--study", "stage0"],
                           env=env, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=600)
        receipt["cloud_status"] = "uploaded"
    except Exception as error:
        receipt.update(cloud_status="failed", upload_error_type=type(error).__name__)
    (OUT / "finalization.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
