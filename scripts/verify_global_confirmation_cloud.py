#!/usr/bin/env python3
"""Byte-check the three approved confirmation cloud artifacts; W&B venv only."""

import json
import os
from pathlib import Path
import time

from learning_harness.common import file_sha


def main():
    import wandb

    out = Path("output/global-selection-confirmation-v1")
    archive = json.loads((out / "archival.json").read_text())
    connection = json.loads(Path("output/wandb/connection.json").read_text())
    host = connection.get("base_url") or connection.get("host")
    if host:
        os.environ["WANDB_BASE_URL"] = host
    api = wandb.Api(timeout=30)
    run = api.run(
        connection["entity"] + "/" + connection["project"] + "/" + archive["run_id"]
    )
    fingerprint = file_sha("runs/global-selection-confirmation-v1/freeze.json")[:12]
    name = "global-confirmation-" + fingerprint + "-final"
    artifact = next(
        a
        for a in run.logged_artifacts()
        if a.type == "research-aggregates" and a.name.split(":")[0] == name
    )
    folder = Path(artifact.download(root="/tmp/lh-confirmation-final-cloud"))
    if {p.name for p in folder.iterdir() if p.is_file()} != set(
        archive["uploaded_sha256"]
    ):
        raise ValueError("Unexpected cloud artifact contents")
    for filename, sha in archive["uploaded_sha256"].items():
        if file_sha(folder / filename) != sha or file_sha(out / filename) != sha:
            raise ValueError("Cloud/local bytes differ")
    receipt = dict(
        status="three_final_cloud_files_byte_verified",
        checked_unix=time.time(),
        artifact=artifact.name,
        files=archive["uploaded_sha256"],
        raw_records_uploaded=False,
    )
    (out / "cloud-verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(status=receipt["status"])))


if __name__ == "__main__":
    main()
