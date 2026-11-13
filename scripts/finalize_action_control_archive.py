#!/usr/bin/env python3
"""Publish a reviewed, completed study after the frozen observer's SDK error."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import time

from action_control_report import load_study, output_path
from learning_harness.bounded_queue import save
from learning_harness.common import file_sha
from learning_harness.research_aggregates import aggregate_report
from supervise_research import process, read


PUBLIC_FILES = (
    "aggregate-summary.json",
    "action-matched-policies.png",
    "action-matched-policies.pdf",
    "RESULTS.zh-CN.md",
)


def publish(root, review_path, online):
    frozen = load_study(root)
    out = output_path(root)
    freeze_sha = file_sha(root / "freeze.json")
    ledger = read(root / "execution.json")
    report = read(out / "summary.json")
    review = read(review_path)
    if (
        ledger.get("status") != "completed"
        or ledger.get("open")
        or len(ledger.get("cases", {})) != 100
        or any(c.get("status") != "completed" for c in ledger["cases"].values())
        or report.get("complete") is not True
        or report.get("cases") != 100
        or report.get("actual_evaluation_episodes") != 6400
        or report.get("freeze_sha256") != freeze_sha
        or review.get("status")
        != "complete_parent_independently_recollected_and_reviewed"
        or review.get("parent_report_sha256") != file_sha(out / "summary.json")
        or review.get("freeze_sha256") != freeze_sha
        or review.get("raw_episodes_reaudited") != 6400
    ):
        raise ValueError("Complete execution and matching independent review required")
    observer = read(root / "job.json").get("observer", {})
    if observer.get("pid"):
        handle = process(observer["pid"], ["track_action_controls.py", str(root)])
        if handle["alive"] and handle["identity_verified"]:
            raise ValueError("The original observer is still live")
    aggregate = aggregate_report(report, frozen, freeze_sha)
    if aggregate != review["independently_recomputed_aggregate"]:
        raise ValueError("Aggregate differs from the independently reviewed result")
    path = out / "aggregate-summary.json"
    if path.exists() and read(path) != aggregate:
        raise ValueError("Preserve an existing, differing aggregate")
    save(path, aggregate)
    hashes = {name: file_sha(out / name) for name in PUBLIC_FILES}
    provenance = dict(
        checked_unix=time.time(),
        reason="Frozen observer used keyword arguments with SummaryDict.update",
        frozen_observer_sha256=file_sha("scripts/track_action_controls.py"),
        publisher_sha256=file_sha(Path(__file__)),
        freeze_sha256=freeze_sha,
        parent_report_sha256=file_sha(out / "summary.json"),
        independent_review_sha256=file_sha(review_path),
        public_files_sha256=hashes,
        gpu_execution=False,
        scientific_inputs_changed=False,
        raw_records_uploaded=False,
    )
    save(out / "archive-recovery-provenance.json", provenance)
    if not online:
        print(json.dumps(dict(status="prepared", public_files_sha256=hashes)))
        return
    save(
        root / "archive-recovery-job.json",
        dict(pid=os.getpid(), script=__file__, started_unix=time.time()),
    )
    import wandb

    target = read(Path("output/wandb/connection.json"))
    host = target.get("base_url", target.get("host"))
    if host:
        os.environ["WANDB_BASE_URL"] = host
    run = None
    save(out / "finalization.json", dict(local_report_complete=True, cloud_status="pending"))
    try:
        run = wandb.init(
            entity=target["entity"],
            project=target["project"],
            id=freeze_sha[:12],
            name=frozen["name"] + " / aggregate results",
            group="task-action-controls-v1",
            job_type="aggregate-results",
            resume="allow",
            mode="online",
            dir=str(out.absolute()),
            save_code=False,
            settings=wandb.Settings(
                console="off",
                disable_code=True,
                disable_git=True,
                save_code=False,
                disable_job_creation=True,
                x_disable_stats=True,
                init_timeout=60,
            ),
        )
        run.summary.update(aggregate)
        run.log({"figures/action-matched-policies": wandb.Image(str(out / PUBLIC_FILES[1]))})
        artifact = wandb.Artifact(
            "task-action-controls-" + freeze_sha[:12] + "-aggregates",
            type="research-aggregates",
        )
        for name in PUBLIC_FILES:
            artifact.add_file(str(out / name), name=name)
        run.log_artifact(artifact).wait()
        if any(file_sha(out / name) != sha for name, sha in hashes.items()):
            raise ValueError("Public files changed during upload")
        save(
            out / "archival.json",
            dict(
                status="uploaded",
                run_id=run.id,
                run_url=run.url,
                aggregate_only=True,
                raw_records_uploaded=False,
                uploaded_sha256=hashes,
                recovery_publisher_sha256=provenance["publisher_sha256"],
            ),
        )
        save(
            out / "finalization.json",
            dict(local_report_complete=True, cloud_status="uploaded", raw_records_uploaded=False),
        )
        print(json.dumps(dict(status="uploaded", run_id=run.id, run_url=run.url)))
    except Exception as error:
        save(
            out / "finalization.json",
            dict(
                local_report_complete=True,
                cloud_status="failed",
                error_type=type(error).__name__,
                raw_records_uploaded=False,
            ),
        )
        raise
    finally:
        if run is not None:
            run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    out = output_path(args.root)
    with (out / "archive-recovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish(args.root, args.review, args.publish)


if __name__ == "__main__":
    main()
