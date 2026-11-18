#!/usr/bin/env python3
"""Aggregate-only W&B observer; independent of GPU execution."""

import os
import time

from learning_harness.bounded_queue import save
from learning_harness.common import file_sha
from run_global_selection_baselines import OUT, ROOT, read, validate


def main():
    validate()
    connection = read("output/wandb/connection.json")
    host = connection.get("base_url") or connection.get("host")
    if host:
        os.environ["WANDB_BASE_URL"] = host
    import wandb

    fingerprint = file_sha(ROOT / "freeze.json")
    run = wandb.init(
        entity=connection["entity"],
        project=connection["project"],
        id=fingerprint[:12],
        name="Global selection: equal-freedom baselines",
        group="global-selection-baselines-v1",
        job_type="student-learning",
        resume="allow",
        dir=str(OUT.absolute()),
        save_code=False,
        settings=wandb.Settings(
            console="off",
            disable_code=True,
            disable_git=True,
            save_code=False,
            x_disable_stats=True,
            init_timeout=60,
        ),
    )
    save(
        OUT / "wandb.json",
        dict(
            status="watching",
            run_id=run.id,
            run_url=run.url,
            raw_records_uploaded=False,
        ),
    )
    logged = set()
    deadline = time.monotonic() + 36 * 3600
    try:
        while time.monotonic() < deadline:
            ledger = (
                read(ROOT / "execution.json")
                if (ROOT / "execution.json").exists()
                else {}
            )
            completed = {
                n
                for n, v in ledger.get("cases", {}).items()
                if v["status"] == "completed"
            }
            elapsed = ledger.get("gpu_seconds", 0) + sum(
                max(0, time.time() - r["started_unix"])
                for r in ledger.get("open", {}).values()
            )
            run.log(
                {
                    "progress/completed_models": len(completed),
                    "progress/planned_models": 64,
                    "progress/gpu_hours": elapsed / 3600,
                    "progress/state": ledger.get("status", "starting"),
                }
            )
            for name in sorted(completed - logged):
                result = read(ROOT / "cases" / name / "worker-result.json")
                c = result["case"]
                run.log(
                    {
                        f"completed_model/{c['source']}/{c['policy']}/success": result[
                            "mean_success"
                        ],
                        "completed_model/count": len(logged) + 1,
                    }
                )
                logged.add(name)
            status = ledger.get("status")
            if status == "completed":
                names = (
                    "aggregate-summary.json",
                    "global-comparison.png",
                    "RESULTS.zh-CN.md",
                )
                hashes = {n: file_sha(OUT / n) for n in names}
                artifact = wandb.Artifact(
                    "global-selection-" + fingerprint[:12] + "-final",
                    type="research-aggregates",
                )
                for name in names:
                    artifact.add_file(str(OUT / name), name=name)
                run.log_artifact(artifact).wait()
                run.log(
                    {
                        "figures/global-comparison": wandb.Image(
                            str(OUT / "global-comparison.png")
                        )
                    }
                )
                save(
                    OUT / "archival.json",
                    dict(
                        status="uploaded",
                        run_id=run.id,
                        run_url=run.url,
                        uploaded_sha256=hashes,
                        raw_records_uploaded=False,
                    ),
                )
                break
            if status in ("failed", "interrupted", "stopped_at_cap"):
                run.summary["termination_state"] = status
                break
            time.sleep(30)
        else:
            raise TimeoutError("Global observer deadline")
    finally:
        run.finish()


if __name__ == "__main__":
    main()
