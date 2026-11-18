#!/usr/bin/env python3
"""Observe the fresh base calibration; upload only explicitly named aggregates."""

import os
import time

from learning_harness.bounded_queue import save
from learning_harness.common import file_sha, read_jsonl
from run_json_examples_calibration import OUT, ROOT, read, validate


def main():
    frozen = validate()
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
        group="fresh-experience-value-v1",
        job_type="base-calibration",
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
        deadline = time.monotonic() + 4200
        while True:
            ledger = (
                read(ROOT / "execution.json")
                if (ROOT / "execution.json").exists()
                else {}
            )
            state = ledger.get("status", "not_started")
            episodes = sum(
                len(read_jsonl(p)) for p in ROOT.glob("cases/*/evaluation/tasks.jsonl")
            )
            elapsed = ledger.get("gpu_seconds", 0) + sum(
                max(0, time.time() - item["started_unix"])
                for item in ledger.get("open", {}).values()
            )
            run.log(
                {
                    "progress/completed_episodes": episodes,
                    "progress/planned_episodes": 32,
                    "progress/gpu_hours": elapsed / 3600,
                    "progress/state": state,
                }
            )
            # Progress has no partial method-specific success rates.
            if state in ("completed", "failed", "interrupted", "stopped_at_cap"):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Observer deadline; preserve existing experiment")
            time.sleep(30)
        if state == "completed":
            validate()
            aggregate = read(OUT / "aggregate-summary.json")
            if aggregate["freeze_sha256"] != fingerprint or not aggregate["complete"]:
                raise ValueError("Aggregate identity changed")
            run.summary.update(aggregate)
            run.log(
                {
                    "figures/executor-calibration": wandb.Image(
                        str(OUT / "json-interface-calibration.png")
                    )
                }
            )
            names = (
                "aggregate-summary.json",
                "json-interface-calibration.png",
                "RESULTS.zh-CN.md",
            )
            artifact = wandb.Artifact(
                "json-public-examples-" + fingerprint[:12] + "-aggregates",
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
                    uploaded_sha256={n: file_sha(OUT / n) for n in names},
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
