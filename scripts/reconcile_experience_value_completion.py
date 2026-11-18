#!/usr/bin/env python3
"""Resume read-only cloud checks after the original audit environment failed.

Run with .venv-wandb, not .venv-model. Preserve the failed audit and its
receipt; do not modify the frozen experiment or rerun any GPU workload.
"""

import fcntl
import json
import os
from pathlib import Path
import sys
import time

import wandb

import audit_experience_value_completion as original


ROOT = original.ROOT
OUT = original.OUT
RECEIPT = OUT / "cloud-reconciliation.json"


def main():
    lock = (ROOT / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if RECEIPT.exists():
        raise ValueError("Inspect the previous reconciliation before another attempt")
    original.validate()
    local = original.read(OUT / "local-audit.json")
    final_path = original.study.OUT / "aggregate-summary.json"
    final = original.read(final_path)
    if (
        local["status"] != "full_local_integrity_audit_passed"
        or local["models"] != 225
        or local["final_summary_sha256"] != original.file_sha(final_path)
        or local["parent_freeze_sha256"]
        != original.file_sha(original.study.ROOT / "freeze.json")
    ):
        raise ValueError("The completed local audit no longer matches")
    for name, sha in final["case_receipt_sha256"].items():
        path = original.study.ROOT / "cases" / name / "worker-result.json"
        if original.file_sha(path) != sha:
            raise ValueError("An audited worker receipt changed")
    state = dict(
        status="running", pid=os.getpid(), started_unix=time.time(),
        executable=sys.executable,
        source_sha256=original.file_sha(Path(__file__)),
        previous_state=original.read(ROOT / "state.json"),
        previous_recovery=original.read(OUT / "cloud-recovery.json"),
        verified_artifacts=0, native_student_runs_verified=0,
        gpu_work_started=False, raw_records_uploaded=False,
    )

    def progress(**values):
        state.update(values, checked_unix=time.time())
        original.save(RECEIPT, state)
        print(json.dumps(values), flush=True)

    progress()
    try:
        connection = original.read("output/wandb/connection.json")
        host = connection.get("base_url") or connection.get("host")
        if host:
            os.environ["WANDB_BASE_URL"] = host
        api = wandb.Api(timeout=30)
        project = connection["entity"] + "/" + connection["project"]
        archival = original.read(original.study.OUT / "archival.json")
        run = api.run(project + "/" + archival["run_id"])
        artifacts = {
            a.name.split(":")[0]: a
            for a in run.logged_artifacts() if a.type == "research-aggregates"
        }
        tracking = original.read(original.study.OUT / "interim-tracking.json")
        prefix = "experience-value-" + final["freeze_sha256"][:12]
        expected = {"final": (prefix + "-final", archival["uploaded_sha256"])}
        for node in original.NODES:
            item = tracking[str(node)]
            if item["status"] != "uploaded":
                raise ValueError("An interim artifact has not uploaded")
            expected[str(node)] = (
                prefix + f"-blocks-{node:02d}", item["uploaded_sha256"]
            )
        verified = {}
        for label, (name, hashes) in expected.items():
            artifact = artifacts[name]
            folder = Path(artifact.download(root="/tmp/lh-value-reconciled-" + label))
            if {p.name for p in folder.iterdir() if p.is_file()} != set(hashes):
                raise ValueError("Unexpected cloud artifact contents")
            for filename, sha in hashes.items():
                if original.file_sha(folder / filename) != sha:
                    raise ValueError("Cloud artifact bytes differ")
            verified[label] = dict(artifact=artifact.name, files=hashes)
            progress(verified_artifacts=len(verified))
        remote = {}
        for r in api.runs(
            project, filters={"group": "experience-value-development-v1/models"},
            per_page=100,
        ):
            remote[r.id] = r
            if len(remote) % 50 == 0:
                progress(remote_run_objects_loaded=len(remote))
        index = original.read(original.study.OUT / "native-training-wandb.json")
        count = 0
        for record in final["models"]:
            name = record["case"]["name"]
            if name == "base":
                continue
            item = index["runs"][name]
            if not item["finished"] or item["last_logged_step"] != 190:
                raise ValueError("An indexed native training run is incomplete")
            summary = remote[item["run_id"]].summary
            wanted = dict(
                training_presentations=768, evaluation_episodes=128,
                evaluation_success=record["mean_success"], termination_state="completed",
            )
            if any(summary.get(k) != value for k, value in wanted.items()):
                raise ValueError("A native training endpoint differs")
            count += 1
            if count % 16 == 0:
                progress(native_student_runs_verified=count)
        if count != 224:
            raise ValueError("Incomplete native student inventory")
        cloud = dict(
            status="all_final_and_interim_cloud_bytes_verified",
            checked_unix=time.time(), artifacts=verified,
            verified_files=sum(len(v["files"]) for v in verified.values()),
            native_student_runs_verified=count, raw_records_uploaded=False,
        )
        original.save(OUT / "cloud-audit.json", cloud)
        original.save(OUT / "handoff.json", dict(
            status="integrity_audited_next_analysis_ready", checked_unix=time.time(),
            local_audit_sha256=original.file_sha(OUT / "local-audit.json"),
            cloud_audit_sha256=original.file_sha(OUT / "cloud-audit.json"),
            models=225, verified_cloud_files=cloud["verified_files"],
            confirmation_evaluations=0, automatic_gpu_restart=False,
            raw_records_uploaded=False,
            required_next_steps=[
                "Analyze paired effects across eight initializations.",
                "Calibrate binary seed/task inference and power before confirmation.",
                "Keep overlapping schedules distinct from independent experience pools.",
                "Keep confirmation tasks sealed until a concrete protocol is frozen.",
            ],
        ))
        progress(status="completed", completed_unix=time.time())
        original.save(ROOT / "state.json", dict(
            status="integrity_audited_next_analysis_ready", checked_unix=time.time(),
            cloud_reconciliation_receipt=str(RECEIPT), automatic_gpu_restart=False,
        ))
    except BaseException as error:
        progress(status="failed", error_type=type(error).__name__)
        raise


if __name__ == "__main__":
    main()
