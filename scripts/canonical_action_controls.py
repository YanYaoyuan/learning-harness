#!/usr/bin/env python3
"""Frozen canonical-score replication using the unchanged worker/report kernel."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import action_control_report as report
import action_control_worker as worker
from learning_harness.action_control_audit import audit_recipe, digest
from learning_harness.action_matched_sampling import action_matched_schedule
from learning_harness.bounded_queue import run as bounded_run, save
from learning_harness.common import file_sha, read_jsonl
from learning_harness.controlled_study import validate_plan

PARENT = Path("runs/variance-diagnostic-v1/stage4")
PROBE = Path("runs/nll-scoring-diagnostic-v1")
ROOT = Path("runs/variance-diagnostic-v1/stage5-canonical")
SPEC = Path("docs/CANONICAL_SELECTION_REPLICATION_SPEC_2026-11-13.zh-CN.md")
REPORT_SCRIPT = "scripts/canonical_action_controls.py"
OBSERVER_SCRIPT = "scripts/track_canonical_action_controls.py"
INTERIM_SCRIPT = "scripts/canonical_action_control_interim.py"
EXTRA_CODE = {
    REPORT_SCRIPT,
    OBSERVER_SCRIPT,
    INTERIM_SCRIPT,
    "tests/test_canonical_action_controls.py",
    str(SPEC),
}


def read(path):
    return json.loads(Path(path).read_text())


def aligned_scores(record, rows, data_sha):
    if record["data_sha256"] != data_sha or len(record["scores"]) != len(rows):
        raise ValueError("Canonical score/corpus identity mismatch")
    values = []
    for i, (row, score) in enumerate(zip(rows, record["scores"])):
        value = score["mean_target_nll"]
        if (
            score["index"] != i
            or score["task_id"] != row["task_id"]
            or score["content_sha256"] != digest_messages(row["messages"])
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("Canonical score row alignment or value mismatch")
        values.append(value)
    return values


def digest_messages(messages):
    import hashlib

    return hashlib.sha256(
        json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def load_recipe(path):
    path = Path(path)
    recipe = read(path)
    root = path.parent.parent.parent
    source, seed = recipe["source"], recipe["sampling_seed"]
    if source not in ("api", "generic") or root != ROOT:
        raise ValueError("Only this explicitly frozen source/study is supported")
    data = Path(f"runs/transfer-factorial-v1/corpora/full-{source}.jsonl")
    reference = (
        worker.parent.ROOT / "inputs/schedules" / f"random-{source}-s{seed}.json"
    )
    scores = root / "inputs/scores" / f"{source}.json"
    legacy = PARENT / "inputs/schedules" / path.name
    if (
        recipe["data"] != str(data)
        or recipe["data_sha256"] != file_sha(data)
        or recipe["reference_recipe"] != str(reference)
        or recipe["reference_recipe_sha256"] != file_sha(reference)
        or recipe["canonical_score_file"] != str(scores)
        or recipe["canonical_score_sha256"] != file_sha(scores)
        or recipe["legacy_recipe_sha256"] != file_sha(legacy)
        or recipe["score_sha256"]
        != (file_sha(scores) if recipe["policy"] == "min_nll" else None)
    ):
        raise ValueError("Canonical recipe provenance mismatch")
    rows = read_jsonl(data)
    score_record = read(scores)
    values = aligned_scores(score_record, rows, file_sha(data))
    if score_record["probe_freeze_sha256"] != file_sha(
        PROBE / "freeze.json"
    ) or score_record["probe_journal_sha256"] != file_sha(
        PROBE / "cases/base-model-readonly-scoring/forwards.private"
    ):
        raise ValueError("Canonical scores do not belong to the completed diagnostic")
    exposure = audit_recipe(recipe, rows, values, read(reference))
    if recipe["policy"] == "random" and recipe["indices"] != read(legacy)["indices"]:
        raise ValueError("Random control changed from the parent recipe")
    return recipe, exposure


def install_kernel_adapter():
    # In-process dependency injection only; the frozen source files are untouched.
    worker.load_recipe = load_recipe
    report.load_study = load_study


def load_study(root, require_completed_parent=True):
    root = Path(root)
    if root != ROOT:
        raise ValueError("Use the explicitly frozen canonical-replication root")
    worker.parent.load_freeze()
    frozen = read(root / "freeze.json")
    design = validate_plan(frozen)
    parent = read(PARENT / "freeze.json")
    required = (
        report.REQUIRED_CODE
        | EXTRA_CODE
        | {
            str(PARENT / "freeze.json"),
            str(PROBE / "freeze.json"),
            str(root / "a-lora.json"),
            str(PARENT / "a-lora.json"),
            str(PARENT / "freeze.json"),
            str(worker.parent.ROOT / "freeze.json"),
            str(root / "preparation.json"),
            str(root / "parent-review.json"),
            *[c["schedule"] for c in frozen["cases"]],
            *[str(root / "inputs/scores" / f"{s}.json") for s in ("api", "generic")],
        }
    )
    expected_cases = [
        dict(c, schedule=str(root / "inputs/schedules" / Path(c["schedule"]).name))
        for c in parent["cases"]
        if c["repeat"] == 0
    ]
    if (
        frozen["canonical_protocol"] != "canonical-action-selection-replication-v1"
        or frozen["cases"] != expected_cases
        or len(expected_cases) != 96
        or len(design["primary_seed_cells"]) != 24
        or frozen["planned_episodes"] != 6144
        or frozen["cap_gpu_hours"] != 18
        or frozen["allocation_seconds"] != 1800
        or frozen["minimum_allocation_seconds"] != 900
        or frozen["parent_stage3_freeze_sha256"]
        != file_sha(worker.parent.ROOT / "freeze.json")
        or frozen["parent_stage4_freeze_sha256"] != file_sha(PARENT / "freeze.json")
        or frozen["probe_freeze_sha256"] != file_sha(PROBE / "freeze.json")
        or frozen["report_script"] != REPORT_SCRIPT
        or frozen["observer_script"] != OBSERVER_SCRIPT
        or frozen["interim_script"] != INTERIM_SCRIPT
        or frozen["scientific_protocol_text"] != SPEC.read_text()
        or not required.issubset(frozen["files"])
    ):
        raise ValueError("Canonical study design/dependencies changed")
    for path, expected in frozen["files"].items():
        if file_sha(path) != expected:
            raise ValueError("Frozen canonical-study input changed: " + path)
    if read(root / "a-lora.json") != read(PARENT / "a-lora.json"):
        raise ValueError("Canonical replication changed optimizer/model configuration")
    if require_completed_parent:
        for directory in (PARENT, PROBE):
            ledger = read(directory / "execution.json")
            if ledger["status"] != "completed" or ledger.get("open"):
                raise ValueError(
                    "Parent studies must be complete, with no active reservation"
                )
    install_kernel_adapter()
    return frozen


def freeze(root):
    if root != ROOT or root.exists():
        raise ValueError("Use the new prescribed root; never overwrite preparation")
    original = report.load_study(PARENT)
    from run_nll_scoring_probe import validate, recover, CASE

    validate(PROBE)
    for directory in (PARENT, PROBE):
        if read(directory / "execution.json")["status"] != "completed":
            raise ValueError("Complete parent execution required")
    parent_report = Path("output/variance-diagnostic-v1/stage4/summary.json")
    parent_review = Path(
        "output/research-operations/20261113/stage4-complete-local-review.json"
    )
    review = read(parent_review)
    if (
        not read(parent_report)["complete"]
        or file_sha(parent_report) != review["parent_report_sha256"]
    ):
        raise ValueError("Complete parent review does not bind the current report")
    inputs = read(PROBE / "inputs.private")
    journal = PROBE / "cases" / CASE / "forwards.private"
    uniform = recover(journal)["uniform"]
    if set(uniform) != {r["content_sha256"] for r in inputs["entries"]}:
        raise ValueError("Require complete canonical scores")
    prepared = {}
    for source, keys in inputs["source_row_keys"].items():
        data = Path(f"runs/transfer-factorial-v1/corpora/full-{source}.jsonl")
        rows = read_jsonl(data)
        records = dict(
            data_sha256=file_sha(data),
            probe_freeze_sha256=file_sha(PROBE / "freeze.json"),
            probe_journal_sha256=file_sha(journal),
            scores=[
                dict(
                    index=i,
                    task_id=r["task_id"],
                    content_sha256=keys[i],
                    mean_target_nll=uniform[keys[i]],
                )
                for i, r in enumerate(rows)
            ],
        )
        values = aligned_scores(records, rows, file_sha(data))
        tasks = [r["task_id"] for r in rows]
        content = [digest_messages(r["messages"]) for r in rows]
        actions = []
        for row in rows:
            action = json.loads(row["messages"][-1]["content"])["action"]
            actions.append("go" if action.startswith("go to ") else action.split()[0])
        schedules = {}
        for seed in (101, 202, 303, 404):
            for policy in ("random", "min_nll"):
                name = f"{policy}-{source}-s{seed}.json"
                legacy = PARENT / "inputs/schedules" / name
                recipe = read(legacy)
                reference = read(recipe["reference_recipe"])
                indices = action_matched_schedule(
                    tasks, actions, content, reference["indices"], seed, policy, values
                )
                if policy == "random" and indices != recipe["indices"]:
                    raise ValueError("Random control changed")
                recipe.update(
                    indices=indices,
                    schedule_sha256=digest(indices),
                    canonical_score_file=str(root / "inputs/scores" / f"{source}.json"),
                    legacy_recipe_sha256=file_sha(legacy),
                    scoring_geometry="uniform_unpadded_explicit_positions_fixed_window_bf16",
                )
                audit_recipe(recipe, rows, values, reference)
                schedules[name] = recipe
        prepared[source] = (records, schedules)
    root.mkdir(parents=True, exist_ok=False)
    (root / "inputs/scores").mkdir(parents=True)
    (root / "inputs/schedules").mkdir()
    for source, (records, schedules) in prepared.items():
        score_file = root / "inputs/scores" / f"{source}.json"
        save(score_file, records)
        for name, recipe in schedules.items():
            recipe["canonical_score_sha256"] = file_sha(score_file)
            recipe["score_sha256"] = (
                file_sha(score_file) if recipe["policy"] == "min_nll" else None
            )
            save(root / "inputs/schedules" / name, recipe)
    (root / "a-lora.json").write_bytes((PARENT / "a-lora.json").read_bytes())
    save(
        root / "preparation.json",
        dict(
            cpu_only=True,
            new_model_calls=0,
            canonical_inputs=853,
            schedules=16,
            random_indices_unchanged=True,
            probe_journal_sha256=file_sha(journal),
        ),
    )
    save(
        root / "parent-review.json",
        dict(
            decision="run_canonical_selection_replication",
            complete_parent_reviewed=True,
            parent_report_sha256=file_sha(parent_report),
            parent_review_sha256=file_sha(parent_review),
            diagnostic_report_sha256=file_sha(
                Path("output/nll-scoring-diagnostic-v1/aggregate-summary.json")
            ),
        ),
    )
    cases = [
        dict(c, schedule=str(root / "inputs/schedules" / Path(c["schedule"]).name))
        for c in original["cases"]
        if c["repeat"] == 0
    ]
    fields = [
        "protocol",
        "teacher_calls",
        "official_tests_used",
        "primary_metric",
        "primary_contrast",
        "gpus",
        "cap_gpu_hours",
        "allocation_seconds",
        "minimum_allocation_seconds",
        "allowed_background_contexts",
        "initialization_levels",
        "sampling_levels",
        "interim_cell_milestones",
        "parent_stage3_freeze_sha256",
    ]
    frozen = {k: original[k] for k in fields}
    frozen.update(
        name="Canonical-scoring task/action selection replication",
        canonical_protocol="canonical-action-selection-replication-v1",
        cases=cases,
        planned_episodes=64 * len(cases),
        preparatory_gpu_seconds=0,
        report_script=REPORT_SCRIPT,
        observer_script=OBSERVER_SCRIPT,
        interim_script=INTERIM_SCRIPT,
        cloud_scope="aggregate_only",
        parent_stage4_freeze_sha256=file_sha(PARENT / "freeze.json"),
        probe_freeze_sha256=file_sha(PROBE / "freeze.json"),
        scientific_protocol_text=SPEC.read_text(),
        frozen_unix=time.time(),
        automatic_retry=False,
    )
    paths = (
        set(original["files"]) | set(read(PROBE / "freeze.json")["files"]) | EXTRA_CODE
    )
    paths |= {str(p) for p in root.rglob("*.json")}
    paths |= {
        str(PARENT / "freeze.json"),
        str(PROBE / "freeze.json"),
        str(journal),
        str(parent_report),
        str(parent_review),
        "output/nll-scoring-diagnostic-v1/aggregate-summary.json",
    }
    frozen["files"] = {p: file_sha(p) for p in sorted(paths)}
    validate_plan(frozen)
    save(root / "freeze.json", frozen)
    load_study(root)
    return dict(
        status="frozen_not_started",
        models=len(cases),
        episodes=frozen["planned_episodes"],
        freeze_sha256=file_sha(root / "freeze.json"),
    )


def run(root):
    from run_action_controls import disk_preflight, gpu_preflight
    from run_stage1_crossed_seeds import environment

    frozen = load_study(root)
    lock = (root / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "execution.json").exists():
        raise ValueError(
            "A prior attempt exists; inspect it rather than automatically rerun"
        )
    save(root / "storage-preflight.json", disk_preflight(root, frozen))
    save(
        root / "background-context-preflight.json",
        gpu_preflight(frozen["gpus"], frozen["allowed_background_contexts"]),
    )
    out = report.output_path(root)
    out.mkdir(parents=True, exist_ok=True)
    argv = [
        str(Path(".venv-wandb/bin/python").absolute()),
        OBSERVER_SCRIPT,
        "--root",
        str(root),
        "--out",
        str(out),
    ]
    env = dict(
        os.environ,
        PYTHONPATH=str(Path("src").absolute()),
        CUDA_VISIBLE_DEVICES="",
        LH_CONTROL_STUDY_ROOT=str(root),
    )
    for key in list(env):
        if key.startswith("LH_TEACHER_"):
            env.pop(key)
    with (root / "observer.log").open("x") as log:
        child = subprocess.Popen(
            argv,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    save(
        root / "job.json",
        dict(
            pipeline=dict(
                pid=os.getpid(),
                argv=[sys.executable, *sys.argv],
                started_unix=time.time(),
            ),
            observer=dict(pid=child.pid, argv=argv, started_unix=time.time()),
        ),
    )

    def command(case, gpu, seconds):
        return [
            worker.engine.MODEL_PY,
            REPORT_SCRIPT,
            "worker",
            "--root",
            str(root),
            "--case",
            case["name"],
            "--gpu",
            str(gpu),
            "--timeout",
            str(seconds),
        ]

    ledger = bounded_run(root, frozen, command, environment)
    if ledger["status"] == "evaluation_completed":
        with (out / "report.log").open("x") as log:
            subprocess.run(
                [worker.engine.MODEL_PY, REPORT_SCRIPT, "report", "--root", str(root)],
                env=dict(env, MPLCONFIGDIR="/tmp/lh-mpl"),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("freeze", "validate", "run", "worker", "report")
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--case")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.command == "freeze":
        print(json.dumps(freeze(args.root)), flush=True)
        return
    frozen = load_study(args.root)
    if args.command == "validate":
        for path in sorted({c["schedule"] for c in frozen["cases"]}):
            load_recipe(Path(path))
        print(
            json.dumps(
                dict(
                    status="validated",
                    models=len(frozen["cases"]),
                    training_started=False,
                )
            ),
            flush=True,
        )
    elif args.command == "run":
        run(args.root)
    elif args.command == "report":
        result = report.generate(args.root)
        print(
            json.dumps(dict(complete=result["complete"], models=result["cases"])),
            flush=True,
        )
    else:
        if args.case is None or args.gpu is None or args.timeout <= 0:
            parser.error("Worker needs case, physical GPU and positive timeout")

        def stop(signum, frame):
            raise worker.kernel.WorkerDeadline()

        signal.signal(signal.SIGALRM, stop)
        signal.signal(signal.SIGTERM, stop)
        signal.setitimer(signal.ITIMER_REAL, args.timeout)
        worker.run_case(args.root, args.case, args.gpu, args.timeout)
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
