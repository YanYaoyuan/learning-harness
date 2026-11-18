#!/usr/bin/env python3
"""Wait for this study to finish, then independently audit local/cloud artifacts.

This observer is read-only with respect to the scientific study. It never starts
or retries GPU jobs, changes the matrix, opens confirmation tasks, or edits a
pre-existing report. Its own frozen receipt and findings live in a separate root.
"""

import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from learning_harness.bounded_queue import save
from learning_harness.common import file_sha
from learning_harness.experience_value_study import NODES, SOURCES, aggregate_cells
import run_experience_value_development as study
from supervise_research import process

ROOT = Path("runs/experience-value-completion-audit-v1")
OUT = Path("output/experience-value-completion-audit-v1")
SCRIPT = "scripts/audit_experience_value_completion.py"


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    parent = study.validate()
    if ROOT.exists():
        raise ValueError("Preserve the existing completion-audit preparation")
    ROOT.mkdir(exist_ok=False)
    paths = {SCRIPT, str(study.ROOT / "freeze.json"), "scripts/supervise_research.py"}
    save(
        ROOT / "freeze.json",
        dict(
            parent_root=str(study.ROOT),
            parent_freeze_sha256=file_sha(study.ROOT / "freeze.json"),
            expected_models=225,
            expected_students=224,
            expected_episodes=28800,
            expected_interim_blocks=list(NODES),
            wait_deadline_seconds=100 * 3600,
            automatic_gpu_restart=False,
            confirmation_evaluations=0,
            raw_records_uploaded=False,
            files={p: file_sha(p) for p in sorted(paths)},
            prepared_unix=time.time(),
            parent_scientific_protocol_sha256=hashlib.sha256(
                parent["protocol_text"].encode()
            ).hexdigest(),
        ),
    )
    print(
        json.dumps(
            dict(
                status="prepared",
                parent_freeze_sha256=file_sha(study.ROOT / "freeze.json"),
            )
        ),
        flush=True,
    )


def validate():
    own = read(ROOT / "freeze.json")
    for path, sha in own["files"].items():
        if file_sha(path) != sha:
            raise ValueError("Completion-audit input changed: " + path)
    return own, study.validate()


def local_audit(parent):
    ledger = read(study.ROOT / "execution.json")
    if (
        ledger["status"] != "completed"
        or ledger["open"]
        or len(ledger["cases"]) != 225
        or any(c["status"] != "completed" for c in ledger["cases"].values())
    ):
        raise ValueError("The full development matrix has not completed")
    records = []
    for index, case in enumerate(parent["cases"], 1):
        records.append(study.audit_result(case))
        save(
            ROOT / "state.json",
            dict(
                status="auditing_local",
                models_checked=index,
                planned_models=225,
                checked_unix=time.time(),
                automatic_gpu_restart=False,
            ),
        )
    base = next(r for r in records if r["case"]["name"] == "base")
    students = [r for r in records if r["case"]["name"] != "base"]
    initialization = defaultdict(set)
    for record in students:
        initialization[record["case"]["initialization_seed"]].add(
            record["training"]["initial_parameter_sha256"]
        )
    if len(initialization) != 8 or any(len(v) != 1 for v in initialization.values()):
        raise ValueError("Initial parameter pairing is incomplete or inconsistent")
    endpoint = aggregate_cells(students, list(range(1, 17)))
    final = read(study.OUT / "aggregate-summary.json")
    if (
        not final["complete"]
        or final["interim"]
        or final["endpoints"] != endpoint
        or final["baseline_success"] != base["mean_success"]
        or final["completed_episodes"] != 28800
    ):
        raise ValueError(
            "The final report does not match independently rebuilt endpoints"
        )
    snapshots = {}
    for block in NODES:
        target = study.OUT / "interim" / f"blocks-{block:02d}"
        report = read(target / "aggregate-summary.json")
        subset = [r for r in students if r["case"]["block"] <= block]
        if report["endpoints"] != aggregate_cells(subset, list(range(1, block + 1))):
            raise ValueError("Prespecified interim report changed")
        for name, sha in report["case_receipt_sha256"].items():
            if file_sha(study.ROOT / "cases" / name / "worker-result.json") != sha:
                raise ValueError("An interim case receipt changed")
        snapshots[str(block)] = dict(
            students=len(subset),
            summary_sha256=file_sha(target / "aggregate-summary.json"),
        )
    for name, sha in final["case_receipt_sha256"].items():
        if file_sha(study.ROOT / "cases" / name / "worker-result.json") != sha:
            raise ValueError("A final case receipt changed")
    per_initialization = {}
    for comparator in ("random", "high", "gradient_low"):
        differences = {}
        for seed in sorted(initialization):
            differences[str(seed)] = (
                sum(
                    endpoint["sources"][s]["gradient_high"]["per_initialization"][
                        str(seed)
                    ]
                    - endpoint["sources"][s][comparator]["per_initialization"][
                        str(seed)
                    ]
                    for s in SOURCES
                )
                / 2
            )
        per_initialization["gradient_high_minus_" + comparator] = differences
    usage = {
        k: sum(r["usage"][k] for r in records)
        for k in (
            "attempted_requests",
            "response_records",
            "error_records",
            "unknown_usage_requests",
            "known_input_tokens",
            "known_output_tokens",
        )
    }
    result = dict(
        status="full_local_integrity_audit_passed",
        checked_unix=time.time(),
        parent_freeze_sha256=file_sha(study.ROOT / "freeze.json"),
        models=225,
        students=224,
        episodes=28800,
        baseline_success=base["mean_success"],
        endpoints=endpoint,
        per_initialization_contrasts=per_initialization,
        usage=usage,
        snapshots=snapshots,
        accounted_gpu_seconds=ledger["gpu_seconds"],
        final_summary_sha256=file_sha(study.OUT / "aggregate-summary.json"),
        confirmation_evaluations=0,
        inferential_tests_performed=False,
        raw_records_uploaded=False,
    )
    save(OUT / "local-audit.json", result)
    return records, result


def cloud_audit(records):
    connection = read("output/wandb/connection.json")
    host = connection.get("base_url") or connection.get("host")
    if host:
        os.environ["WANDB_BASE_URL"] = host
    import wandb

    api = wandb.Api(timeout=30)
    archival = read(study.OUT / "archival.json")
    run = api.run(
        connection["entity"] + "/" + connection["project"] + "/" + archival["run_id"]
    )
    artifacts = {
        a.name.split(":")[0]: a
        for a in run.logged_artifacts()
        if a.type == "research-aggregates"
    }
    tracking = read(study.OUT / "interim-tracking.json")
    prefix = "experience-value-" + file_sha(study.ROOT / "freeze.json")[:12]
    expected = {"final": (prefix + "-final", archival["uploaded_sha256"])}
    for n in NODES:
        item = tracking[str(n)]
        if item["status"] != "uploaded":
            raise ValueError("An interim upload needs separate reconciliation")
        expected[str(n)] = (prefix + f"-blocks-{n:02d}", item["uploaded_sha256"])
    verified = {}
    for label, (name, hashes) in expected.items():
        artifact = artifacts[name]
        folder = Path(artifact.download(root="/tmp/lh-value-postrun-cloud-" + label))
        if {p.name for p in folder.iterdir() if p.is_file()} != set(hashes):
            raise ValueError("Unexpected cloud artifact contents")
        for filename, sha in hashes.items():
            if file_sha(folder / filename) != sha:
                raise ValueError("Cloud artifact bytes differ")
        verified[label] = dict(artifact=artifact.name, files=hashes)
    index = read(study.OUT / "native-training-wandb.json")
    remote = {
        r.id: r
        for r in api.runs(
            connection["entity"] + "/" + connection["project"],
            filters={"group": "experience-value-development-v1/models"},
            per_page=100,
        )
    }
    for record in records:
        name = record["case"]["name"]
        if name == "base":
            continue
        item = index["runs"][name]
        if not item["finished"] or item["last_logged_step"] != 190:
            raise ValueError("A native training run is incomplete")
        summary = remote[item["run_id"]].summary
        if (
            summary.get("training_presentations") != 768
            or summary.get("evaluation_episodes") != 128
            or summary.get("evaluation_success") != record["mean_success"]
            or summary.get("termination_state") != "completed"
        ):
            raise ValueError("A native cloud endpoint differs")
    result = dict(
        status="all_final_and_interim_cloud_bytes_verified",
        checked_unix=time.time(),
        artifacts=verified,
        verified_files=sum(len(v["files"]) for v in verified.values()),
        native_student_runs_verified=224,
        raw_records_uploaded=False,
    )
    save(OUT / "cloud-audit.json", result)
    return result


def run():
    own, parent = validate()
    lock = (ROOT / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROOT / "job.json").exists():
        raise ValueError("Inspect any previous completion-audit attempt")
    save(
        ROOT / "job.json",
        dict(
            pid=os.getpid(), argv=[sys.executable, *sys.argv], started_unix=time.time()
        ),
    )
    deadline = time.monotonic() + own["wait_deadline_seconds"]
    missing_polls = 0
    try:
        while True:
            ledger = read(study.ROOT / "execution.json")
            state = ledger["status"]
            if state in ("failed", "interrupted", "stopped_at_cap"):
                save(
                    ROOT / "state.json",
                    dict(
                        status="needs_parent_review",
                        parent_status=state,
                        checked_unix=time.time(),
                        automatic_gpu_restart=False,
                    ),
                )
                return
            if state == "completed" and (study.OUT / "archival.json").exists():
                if read(study.OUT / "archival.json").get("status") == "uploaded":
                    break
            if state not in ("completed", "evaluation_completed"):
                job = read(study.ROOT / "job.json")["pipeline"]
                observed = process(job["pid"], ["run_experience_value_development.py"])
                missing_polls = (
                    0
                    if observed["alive"] and observed["identity_verified"]
                    else missing_polls + 1
                )
                if missing_polls >= 2:
                    raise RuntimeError(
                        "Parent process is absent; retain the ledger for review"
                    )
            save(
                ROOT / "state.json",
                dict(
                    status="waiting_for_parent_and_archive",
                    parent_status=state,
                    completed_models=sum(
                        c["status"] == "completed" for c in ledger["cases"].values()
                    ),
                    checked_unix=time.time(),
                    automatic_gpu_restart=False,
                ),
            )
            if time.monotonic() > deadline:
                raise TimeoutError("Completion-audit observer deadline")
            time.sleep(30)
        records, local = local_audit(parent)
        save(
            ROOT / "state.json",
            dict(
                status="auditing_cloud",
                checked_unix=time.time(),
                automatic_gpu_restart=False,
            ),
        )
        cloud = cloud_audit(records)
        save(
            OUT / "handoff.json",
            dict(
                status="integrity_audited_next_analysis_ready",
                checked_unix=time.time(),
                local_audit_sha256=file_sha(OUT / "local-audit.json"),
                cloud_audit_sha256=file_sha(OUT / "cloud-audit.json"),
                models=local["models"],
                verified_cloud_files=cloud["verified_files"],
                confirmation_evaluations=0,
                required_next_steps=[
                    "Visually inspect the final scientific figures before paper delivery.",
                    "Analyze the prespecified paired effects across eight initializations; retain all negative outcomes.",
                    "Calibrate binary/seed/task inference and power before freezing any independent confirmation protocol.",
                    "Keep 28 overlapping schedule conditions distinct from independent collection pools or 224 independent corpora.",
                    "Do not open confirmation tasks or start another GPU experiment without its concrete protocol.",
                ],
                automatic_gpu_restart=False,
                raw_records_uploaded=False,
            ),
        )
        save(
            ROOT / "state.json",
            dict(
                status="integrity_audited_next_analysis_ready",
                checked_unix=time.time(),
                automatic_gpu_restart=False,
            ),
        )
    except BaseException as error:
        save(
            ROOT / "state.json",
            dict(
                status="audit_needs_attention",
                error_type=type(error).__name__,
                checked_unix=time.time(),
                automatic_gpu_restart=False,
            ),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        run()


if __name__ == "__main__":
    main()
