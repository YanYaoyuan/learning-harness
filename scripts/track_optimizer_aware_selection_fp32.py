#!/usr/bin/env python3
"""Read-only cloud progress and aggregate-only artifact observer."""

import os
import time

from learning_harness.bounded_queue import save
from learning_harness.common import file_sha
from run_optimizer_aware_selection_fp32 import CASE, OUT, ROOT, read, validate


def main():
    frozen, _ = validate()
    fingerprint = file_sha(ROOT / "freeze.json")
    target = read("output/wandb/connection.json")
    host = target.get("base_url") or target.get("host")
    if host:
        os.environ["WANDB_BASE_URL"] = host
    import wandb

    settings = wandb.Settings(
        console="off",
        disable_code=True,
        disable_git=True,
        save_code=False,
        disable_job_creation=True,
        x_disable_stats=True,
        init_timeout=60,
    )
    run = wandb.init(
        entity=target["entity"],
        project=target["project"],
        id=fingerprint[:12],
        name=frozen["name"],
        group="experience-value-strong-baseline-v1",
        job_type="gradient-features",
        resume="allow",
        mode="online",
        dir=str(OUT.absolute()),
        save_code=False,
        settings=settings,
    )
    try:
        save(
            OUT / "wandb.json",
            dict(
                status="watching",
                run_id=run.id,
                run_url=run.url,
                raw_records_uploaded=False,
            ),
        )
        deadline = time.monotonic() + 7800
        while True:
            ledger = (
                read(ROOT / "execution.json")
                if (ROOT / "execution.json").exists()
                else {}
            )
            path = ROOT / "cases" / CASE / "progress.json"
            progress = read(path) if path.exists() else {}
            elapsed = ledger.get("gpu_seconds", 0) + sum(
                max(0, time.time() - r["started_unix"])
                for r in ledger.get("open", {}).values()
            )
            values = {
                "progress/phase": progress.get("phase", "loading"),
                "progress/updates": progress.get("updates", 0),
                "progress/completed_checkpoints": progress.get(
                    "completed_checkpoints", 0
                ),
                "progress/candidates_scored": progress.get("candidates_scored", 0),
                "progress/gpu_hours": elapsed / 3600,
                "progress/state": ledger.get("status", "not_started"),
            }
            if "last_warmup_loss" in progress:
                values["warmup/last_complete_update_loss"] = progress[
                    "last_warmup_loss"
                ]
            run.log(values)
            state = ledger.get("status", "not_started")
            if state in ("completed", "failed", "interrupted", "stopped_at_cap"):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Observer deadline; do not restart GPU work")
            time.sleep(30)
        if state == "completed":
            validate()
            aggregate = read(OUT / "aggregate-summary.json")
            if not aggregate["complete"] or aggregate["freeze_sha256"] != fingerprint:
                raise ValueError("Aggregate identity mismatch")
            run.summary.update(aggregate)
            run.log(
                {"figures/feature-pilot": wandb.Image(str(OUT / "feature-pilot.png"))}
            )
            names = ("aggregate-summary.json", "feature-pilot.png", "RESULTS.zh-CN.md")
            artifact = wandb.Artifact(
                "optimizer-aware-" + fingerprint[:12] + "-aggregates",
                type="research-aggregates",
            )
            for name in names:
                artifact.add_file(str(OUT / name), name=name)
            run.log_artifact(artifact).wait()
            save(
                OUT / "archival.json",
                dict(
                    status="uploaded",
                    run_id=run.id,
                    run_url=run.url,
                    raw_records_uploaded=False,
                    uploaded_sha256={name: file_sha(OUT / name) for name in names},
                ),
            )
        run.summary.update(dict(termination_state=state, raw_records_uploaded=False))
        save(
            OUT / "wandb.json",
            dict(
                status=state, run_id=run.id, run_url=run.url, raw_records_uploaded=False
            ),
        )
    finally:
        run.finish()


if __name__ == "__main__":
    main()
