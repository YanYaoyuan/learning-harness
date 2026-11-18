#!/usr/bin/env python3
"""One frozen fresh-task replication, reusing 97 existing checkpoints."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import run_global_selection_baselines as previous
from learning_harness import benchmark
from learning_harness.adaptation import composition_revision
from learning_harness.batch_runner import BatchClient
from learning_harness.bounded_queue import run as bounded_run, save
from learning_harness.common import digest, file_sha, read_jsonl, write_jsonl
from learning_harness.experience_value_study import audit_episodes
from learning_harness.global_confirmation import matrix, validate_queries
from learning_harness.json_demo_executor import actor_messages
from learning_harness.webshop_requests import RequestJournal, summarize_requests
from learning_harness.webshop_serving import verify_base_model
from run_fresh_executor_calibration import JournalClient, validate_roles

ROOT = Path("runs/global-selection-confirmation-v1")
OUT = Path("output/global-selection-confirmation-v1")
SPEC = Path("docs/GLOBAL_CONFIRMATION_SPEC_2026-11-18.zh-CN.md")
SCRIPT = "scripts/run_global_confirmation.py"
engine, environment, read = previous.engine, previous.environment, previous.read
DATA = previous.parent.DATA
CAP = 30.5


def config(revision):
    result = previous.parent.config(revision)
    result.update(
        protocol_id="global-selection-confirmation-v1",
        manifest=str(ROOT / "queries.private.jsonl"),
    )
    return result


def validate():
    previous.validate()
    f = read(ROOT / "freeze.json")
    if (
        f["cases"] != matrix()
        or f["cap_gpu_hours"] != CAP
        or f["gpus"] != [0]
        or f["planned_episodes"] != 12416
        or f["protocol_text"] != SPEC.read_text()
        or f["official_tests_used"]
        or f["teacher_calls"] != 0
        or f["training_updates"] != 0
        or f["automatic_retry"]
        or f["inferential_tests_planned"]
        or f["allocation_seconds"] != 2000
        or f["minimum_allocation_seconds"] != 1100
    ):
        raise ValueError("Frozen confirmation scope changed")
    for path, sha in f["files"].items():
        if file_sha(path) != sha:
            raise ValueError("Confirmation input changed: " + path)
    rows = read_jsonl(ROOT / "queries.private.jsonl")
    validate_queries(
        rows,
        read_jsonl(DATA / "tasks/development.jsonl"),
        read_jsonl(DATA / "tasks/calibration.jsonl"),
    )
    expected = [
        dict(r, purpose="final_evaluation")
        for r in read_jsonl(DATA / "tasks/confirmation.jsonl")
    ]
    if rows != expected:
        raise ValueError("Confirmation panel differs from the sealed preparation")
    return f


def verify_reference(case):
    ref = read(ROOT / "references.private.json")[case["name"]]
    for path, sha in ref["files"].items():
        if file_sha(path) != sha:
            raise ValueError("Referenced checkpoint metadata changed")
    if case["name"] == "base":
        return ref, None, engine.REV
    point = Path(ref["checkpoint"])
    revision = composition_revision(point)
    if revision != ref["composition_revision"]:
        raise ValueError("Referenced model weights changed")
    return ref, point, revision


def freeze():
    if ROOT.exists():
        raise ValueError("Preserve any existing confirmation attempt")
    old = previous.validate()
    if read(previous.ROOT / "execution.json")["status"] != "completed":
        raise ValueError("Global baselines must complete before confirmation")
    audit = read("output/research-operations/20261118/global-final-local-audit.json")
    if audit["summary_sha256"] != file_sha(previous.OUT / "aggregate-summary.json"):
        raise ValueError("The complete development audit no longer matches")
    validate_roles()
    queries = [
        dict(r, purpose="final_evaluation")
        for r in read_jsonl(DATA / "tasks/confirmation.jsonl")
    ]
    validate_queries(
        queries,
        read_jsonl(DATA / "tasks/development.jsonl"),
        read_jsonl(DATA / "tasks/calibration.jsonl"),
    )
    references, disk_sizes = {}, []
    paths = set(old["files"]) | {
        SCRIPT,
        str(SPEC),
        "src/learning_harness/global_confirmation.py",
        "tests/test_global_confirmation.py",
        "scripts/report_global_confirmation.py",
        "scripts/track_global_confirmation.py",
        str(previous.ROOT / "freeze.json"),
        str(previous.OUT / "aggregate-summary.json"),
        "output/research-operations/20261118/global-final-local-audit.json",
        str(DATA / "preparation.json"),
        str(DATA / "preparation.private.json"),
    }
    for case in matrix():
        original_root = (
            previous.parent.ROOT
            if case["policy"] in ("base", "gradient_global")
            else previous.ROOT
        )
        folder = original_root / "cases" / case["name"]
        record = read(folder / "worker-result.json")
        if not record["complete"] or record["case"] != case:
            raise ValueError("A reference is not the expected completed model")
        files = {
            str(folder / "worker-result.json"): file_sha(folder / "worker-result.json")
        }
        ref = dict(
            source_folder=str(folder),
            training=record["training"],
            files=files,
            development_success=record["mean_success"],
            composition_revision=engine.REV,
            checkpoint=None,
        )
        if case["name"] != "base":
            point = folder / "checkpoint"
            revision = composition_revision(point)
            if revision != record["composition_revision"]:
                raise ValueError("A completed reference composition changed")
            ref.update(checkpoint=str(point), composition_revision=revision)
            for path in (
                point / "composition.json",
                folder / "A/experiment.json",
                folder / "A/seed-controls.json",
                folder / "A/adapter_config.json",
            ):
                files[str(path)] = file_sha(path)
            if (
                record["training"]["initial_parameter_sha256"]
                != old["initialization_anchors"][str(case["initialization_seed"])]
            ):
                raise ValueError("Reference initialization pairing changed")
        references[case["name"]] = ref
        disk_sizes.append(
            sum(
                (folder / p).stat().st_size
                for p in (
                    "evaluation/events.jsonl",
                    "evaluation/tasks.jsonl",
                    "model-requests.private",
                )
            )
        )
        paths.update(files)
    # Actual completed evaluations, with a 2x margin plus 2GiB reserve; no new weights.
    required = 2 * sum(disk_sizes) + 2 * 1024**3
    if shutil.disk_usage(ROOT.parent).free < required:
        raise ValueError(
            "Insufficient disk for twice the observed reference records plus reserve"
        )
    ROOT.mkdir()
    write_jsonl(ROOT / "queries.private.jsonl", queries)
    (ROOT / "queries.private.jsonl").chmod(0o600)
    save(ROOT / "references.private.json", references)
    (ROOT / "references.private.json").chmod(0o600)
    shutil.copyfile(previous.ROOT / "runtime-lock.json", ROOT / "runtime-lock.json")
    gate = dict(
        status="frozen",
        config_digests=[
            digest(config(r["composition_revision"])) for r in references.values()
        ],
        files={
            p: file_sha(p)
            for p in set(benchmark.SOURCE_FILES) | {str(ROOT / "queries.private.jsonl")}
        },
    )
    save(ROOT / "benchmark-gate.private.json", gate)
    for name in (
        "queries.private.jsonl",
        "references.private.json",
        "runtime-lock.json",
        "benchmark-gate.private.json",
    ):
        paths.add(str(ROOT / name))
    prior = read(previous.ROOT / "execution.json")["gpu_seconds"] / 3600
    if prior + CAP > 48:
        raise ValueError("Finishing stages exceed the 48GPU-hour cap")
    save(
        ROOT / "freeze.json",
        dict(
            name="Global experience selection: sealed fresh-task replication",
            protocol="global-selection-confirmation-v1",
            protocol_text=SPEC.read_text(),
            cases=matrix(),
            gpus=[0],
            cap_gpu_hours=CAP,
            planned_episodes=12416,
            allocation_seconds=2000,
            minimum_allocation_seconds=1100,
            training_updates=0,
            teacher_calls=0,
            official_tests_used=False,
            confirmation_tasks_used=True,
            automatic_retry=False,
            inferential_tests_planned=False,
            raw_records_uploaded=False,
            allowed_background_contexts=old["allowed_background_contexts"],
            prior_finishing_gpu_hours=prior,
            disk_preflight=dict(
                observed_reference_bytes=sum(disk_sizes),
                required_free_bytes=required,
                actual_free_bytes=shutil.disk_usage(ROOT).free,
            ),
            files={p: file_sha(p) for p in sorted(paths)},
            frozen_unix=time.time(),
        ),
    )
    validate()
    OUT.mkdir()
    save(
        OUT / "freeze-summary.json",
        dict(
            freeze_sha256=file_sha(ROOT / "freeze.json"),
            reused_checkpoints=97,
            tasks=128,
            episodes=12416,
            training_updates=0,
            cap_gpu_hours=CAP,
            prior_finishing_gpu_hours=prior,
            frozen_inputs=len(paths),
            analysis="Finite-panel paired descriptive replication; no p-values or confidence intervals.",
            raw_records_uploaded=False,
        ),
    )
    print(json.dumps(read(OUT / "freeze-summary.json")), flush=True)


def worker(name, gpu, seconds):
    from learning_harness.controlled_study import validate_reservation
    from run_action_controls import gpu_preflight

    frozen = validate()
    case = next(c for c in frozen["cases"] if c["name"] == name)
    if gpu != 0 or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Wrong physical GPU")
    deadline = time.monotonic() + 2
    while (
        name not in read(ROOT / "execution.json").get("open", {})
        and time.monotonic() < deadline
    ):
        time.sleep(0.02)
    validate_reservation(
        read(ROOT / "execution.json"),
        name,
        os.getpid(),
        gpu,
        seconds,
        file_sha(ROOT / "freeze.json"),
    )
    if shutil.disk_usage(ROOT).free < 2 * 1024**3:
        raise ValueError("Disk reserve reached; retain partial work and stop")
    folder = ROOT / "cases" / name
    if (folder / "model-requests.private").exists() or (folder / "evaluation").exists():
        raise ValueError("A previous attempt is retained; do not retry")
    save(
        folder / "gpu-preflight.json",
        gpu_preflight([0], frozen["allowed_background_contexts"]),
    )
    if previous.runtime() != read(ROOT / "runtime-lock.json"):
        raise ValueError("Frozen runtime differs")
    ref, point, revision = verify_reference(case)
    before = verify_base_model(Path(engine.MODEL), engine.REV)
    cfg = config(revision)
    save(folder / "evaluation-config.json", cfg)

    def actor(initial, history, memories, method, history_turns):
        if method != "json_direct" or memories:
            raise ValueError("Unexpected executor")
        return actor_messages(initial, history, history_turns)

    benchmark.actor_messages = actor
    save(folder / "progress.json", dict(phase="evaluation", reused_checkpoint=True))
    with engine.server(
        gpu, "confirmation-" + name, revision, folder / "server.log", point
    ) as env:
        previous_env = dict(os.environ)
        os.environ.update(env)
        try:
            with RequestJournal(folder / "model-requests.private") as journal:
                client = JournalClient(BatchClient(cfg["model"]), journal)
                rows = benchmark.run(
                    folder / "evaluation-config.json",
                    "data/alfworld",
                    folder / "evaluation",
                    freeze=ROOT / "benchmark-gate.private.json",
                    client=client,
                )
        finally:
            os.environ.clear()
            os.environ.update(previous_env)
    metric = audit_episodes(
        rows,
        read_jsonl(folder / "evaluation/events.jsonl"),
        summarize_requests(folder / "model-requests.private"),
        read_jsonl(ROOT / "queries.private.jsonl"),
    )
    after = verify_base_model(Path(engine.MODEL), engine.REV)
    if before != after or verify_reference(case)[2] != revision:
        raise ValueError("Served weights changed during evaluation")
    save(
        folder / "worker-result.json",
        dict(
            complete=True,
            case=case,
            freeze_sha256=file_sha(ROOT / "freeze.json"),
            **metric,
            training=ref["training"],
            reused_checkpoint=True,
            composition_revision=revision,
            model_identity=after,
            config_sha256=file_sha(folder / "evaluation-config.json"),
            results_sha256=file_sha(folder / "evaluation/tasks.jsonl"),
            events_sha256=file_sha(folder / "evaluation/events.jsonl"),
        ),
    )
    save(folder / "progress.json", dict(phase="completed", episodes=128))


def audit_result(case):
    folder = ROOT / "cases" / case["name"]
    result = read(folder / "worker-result.json")
    if (
        not result["complete"]
        or result["case"] != case
        or result["freeze_sha256"] != file_sha(ROOT / "freeze.json")
    ):
        raise ValueError("Completed confirmation identity differs")
    for key, path in (
        ("config_sha256", folder / "evaluation-config.json"),
        ("results_sha256", folder / "evaluation/tasks.jsonl"),
        ("events_sha256", folder / "evaluation/events.jsonl"),
    ):
        if result[key] != file_sha(path):
            raise ValueError("Confirmation result changed")
    metric = audit_episodes(
        read_jsonl(folder / "evaluation/tasks.jsonl"),
        read_jsonl(folder / "evaluation/events.jsonl"),
        summarize_requests(folder / "model-requests.private"),
        read_jsonl(ROOT / "queries.private.jsonl"),
    )
    if any(result[k] != value for k, value in metric.items()):
        raise ValueError("Reported confirmation differs from raw outcomes")
    ref, _, revision = verify_reference(case)
    if (
        result["training"] != ref["training"]
        or result["composition_revision"] != revision
    ):
        raise ValueError("Reference model identity differs")
    if read(folder / "evaluation-config.json") != config(revision):
        raise ValueError("Common executor changed")
    return result


def run():
    from run_action_controls import gpu_preflight

    frozen = validate()
    lock = (ROOT / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROOT / "execution.json").exists():
        raise ValueError("Inspect any previous confirmation attempt")
    save(
        ROOT / "gpu-preflight.json",
        gpu_preflight([0], frozen["allowed_background_contexts"]),
    )
    if shutil.disk_usage(ROOT).free < frozen["disk_preflight"]["required_free_bytes"]:
        raise ValueError("Disk preflight no longer holds")
    args = [
        str(Path(".venv-wandb/bin/python").absolute()),
        "scripts/track_global_confirmation.py",
    ]
    with (ROOT / "observer.log").open("x") as log:
        observer = subprocess.Popen(
            args,
            env=dict(environment(0), CUDA_VISIBLE_DEVICES=""),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    save(
        ROOT / "job.json",
        dict(
            pipeline=dict(
                pid=os.getpid(),
                argv=[sys.executable, *sys.argv],
                started_unix=time.time(),
            ),
            observer=dict(pid=observer.pid, argv=args, started_unix=time.time()),
        ),
    )

    def command(case, gpu, seconds):
        return [
            engine.CPU_PY,
            SCRIPT,
            "worker",
            "--case",
            case["name"],
            "--gpu",
            str(gpu),
            "--timeout",
            str(seconds),
        ]

    ledger = bounded_run(ROOT, frozen, command, environment)
    if ledger["status"] == "evaluation_completed":
        with (OUT / "report.log").open("x") as log:
            subprocess.run(
                [engine.MODEL_PY, "scripts/report_global_confirmation.py"],
                env=dict(
                    environment(0),
                    CUDA_VISIBLE_DEVICES="",
                    MPLCONFIGDIR="/tmp/lh-confirmation-mpl",
                ),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=900,
            )
        ledger["status"] = "completed"
        save(ROOT / "execution.json", ledger)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "validate", "run", "worker"))
    parser.add_argument("--case")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=2000)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "validate":
        print(json.dumps(dict(frozen_inputs=len(validate()["files"]))))
    elif args.command == "run":
        run()
    else:
        if args.case not in {c["name"] for c in matrix()} or args.timeout <= 0:
            raise ValueError("Explicit frozen case and positive timeout required")

        def stop(signum, frame):
            raise TimeoutError("Confirmation case deadline")

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGALRM, stop)
        signal.setitimer(signal.ITIMER_REAL, args.timeout)
        worker(args.case, args.gpu, args.timeout)


if __name__ == "__main__":
    main()
