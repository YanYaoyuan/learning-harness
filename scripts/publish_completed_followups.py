#!/usr/bin/env python3
"""Publish explicit aggregate/figure whitelists for two completed follow-ups."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import time

from export_github_snapshot import checked
from learning_harness.bounded_queue import save
from learning_harness.common import file_sha


PHASES = {
    "nll": {
        "root": "runs/nll-scoring-diagnostic-v1",
        "out": "output/nll-scoring-diagnostic-v1",
        "pdf": "output/pdf/nll-scoring-diagnostic-v1/scoring-layout-and-selection.pdf",
        "png": "scoring-layout-and-selection.png",
        "review": "output/research-operations/20261113/nll-scoring-complete-local-check.json",
        "review_hash_key": "report_sha256",
        "artifact_prefix": "nll-scoring",
        "group": "nll-scoring-diagnostic",
        "name": "Completed read-only NLL scoring diagnostic",
    },
    "webshop": {
        "root": "runs/webshop-development-pilot-v1",
        "out": "output/webshop-development-pilot-v1",
        "pdf": "output/pdf/webshop-development-pilot-v1/base-calibration.pdf",
        "png": "base-calibration.png",
        "review": "output/research-operations/20261113/webshop-pilot-complete-local-check.json",
        "review_hash_key": "authoritative_report_sha256",
        "artifact_prefix": "webshop-calibration",
        "group": "webshop-development-pilot",
        "name": "Completed train-only WebShop base calibration",
    },
}


def read(path):
    return json.loads(Path(path).read_text())


def publish(kind, online):
    spec = PHASES[kind]
    root, out = Path(spec["root"]), Path(spec["out"])
    report = read(out / "aggregate-summary.json")
    review = read(spec["review"])
    if (
        read(root / "execution.json")["status"] != "completed"
        or report.get("complete") is not True
        or review.get("complete") is not True
        or review[spec["review_hash_key"]] != file_sha(out / "aggregate-summary.json")
        or report["freeze_sha256"] != file_sha(root / "freeze.json")
    ):
        raise ValueError("Completed, independently reviewed result required")
    files = {
        "aggregate-summary.json": out / "aggregate-summary.json",
        spec["png"]: out / spec["png"],
        Path(spec["pdf"]).name: Path(spec["pdf"]),
        "RESULTS.zh-CN.md": out / "RESULTS.zh-CN.md",
    }
    for name, path in files.items():
        if path.suffix in (".json", ".md"):
            checked(path.read_bytes(), str(path))
    hashes = {name: file_sha(path) for name, path in files.items()}
    prior = out / "archival.json"
    if prior.exists():
        if (
            read(prior).get("status") == "uploaded"
            and read(prior)["uploaded_sha256"] == hashes
        ):
            print(json.dumps(dict(kind=kind, status="already_uploaded_same_files")))
            return
        raise ValueError("Preserve differing previous publication evidence")
    if not online:
        print(
            json.dumps(dict(kind=kind, status="prepared", public_files_sha256=hashes))
        )
        return
    import wandb

    target = read("output/wandb/connection.json")
    host = target.get("base_url", target.get("host"))
    if host:
        os.environ["WANDB_BASE_URL"] = host
    prefix = report["freeze_sha256"][:12]
    run = wandb.init(
        entity=target["entity"],
        project=target["project"],
        id=prefix if kind == "nll" else prefix + "-calibration",
        name=spec["name"],
        group=spec["group"],
        job_type="reviewed-aggregate-results",
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
    try:
        run.summary.update(report)
        run.log({"figures/reviewed-results": wandb.Image(str(out / spec["png"]))})
        artifact = wandb.Artifact(
            spec["artifact_prefix"] + "-" + prefix + "-aggregates",
            type="reviewed-followup-aggregates",
        )
        for name, path in files.items():
            artifact.add_file(str(path), name=name)
        run.log_artifact(artifact).wait()
        if any(file_sha(files[name]) != sha for name, sha in hashes.items()):
            raise ValueError("Public files changed during upload")
        save(
            prior,
            dict(
                status="uploaded",
                checked_unix=time.time(),
                run_id=run.id,
                run_url=run.url,
                artifact_components=dict(
                    prefix=spec["artifact_prefix"],
                    freeze_prefix=prefix,
                    suffix="aggregates",
                ),
                uploaded_sha256=hashes,
                publisher_sha256=file_sha(Path(__file__)),
                independent_review_sha256=file_sha(spec["review"]),
                aggregate_only=True,
                raw_records_uploaded=False,
            ),
        )
        print(
            json.dumps(
                dict(kind=kind, status="uploaded", run_id=run.id, run_url=run.url)
            )
        )
    finally:
        run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    with Path("output/research-operations/followup-publication.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for kind in PHASES:
            publish(kind, args.publish)


if __name__ == "__main__":
    main()
