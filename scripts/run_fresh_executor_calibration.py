#!/usr/bin/env python3
"""Bounded base-only calibration on sealed fresh train-derived task groups."""

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import run_api_harness as engine
from learning_harness.benchmark import SOURCE_FILES, run as benchmark_run
from learning_harness.batch_runner import BatchClient
from learning_harness.bounded_queue import run as bounded_run, save
from learning_harness.common import file_sha, read_jsonl
from learning_harness.fresh_tasks import COUNTS, FAMILIES, task_group
from learning_harness.webshop_requests import RequestJournal, summarize_requests
from learning_harness.webshop_serving import verify_base_model

ROOT = Path("runs/fresh-executor-calibration-v1")
DATA = Path("runs/fresh-experience-value-v1")
OUT = Path("output/fresh-executor-calibration-v1")
SPEC = Path("docs/FRESH_EXECUTOR_CALIBRATION_SPEC_2026-11-13.zh-CN.md")
PARENT = Path("runs/variance-diagnostic-v1/stage5-canonical")
SCRIPT = "scripts/run_fresh_executor_calibration.py"
TRACKER = "scripts/track_fresh_executor_calibration.py"
CODE = set(SOURCE_FILES) | {
    SCRIPT,
    TRACKER,
    "scripts/prepare_fresh_experience_tasks.py",
    str(SPEC),
    "src/learning_harness/fresh_tasks.py",
    "src/learning_harness/bounded_queue.py",
    "src/learning_harness/controlled_study.py",
    "src/learning_harness/webshop_requests.py",
    "src/learning_harness/webshop_serving.py",
    "src/learning_harness/webshop_factory.py",
    "src/learning_harness/idle_gpu_contexts.py",
    "src/learning_harness/adaptation.py",
    "scripts/run_action_controls.py",
    "scripts/action_control_report.py",
    "scripts/action_control_worker.py",
    "scripts/run_api_harness.py",
    "scripts/run_stage1_crossed_seeds.py",
    "scripts/run_stage0_repeat_eval.py",
    "scripts/supervise_research.py",
    "tests/test_fresh_tasks.py",
    "tests/test_fresh_executor_calibration.py",
}


def read(path):
    return json.loads(Path(path).read_text())


def environment(gpu):
    from run_stage1_crossed_seeds import environment as inherited_environment

    return inherited_environment(gpu)


class JournalClient:
    def __init__(self, client, journal):
        self.client, self.journal, self.next_call = client, journal, 0
        self.model = client.model

    def complete_many(self, requests, max_tokens, temperature, revision):
        calls = list(range(self.next_call, self.next_call + len(requests)))
        self.next_call += len(requests)
        for call in calls:
            self.journal(dict(kind="attempt", turn=call, max_tokens=max_tokens))
        try:
            response = self.client.complete_many(
                requests, max_tokens, temperature, revision
            )
        except BaseException as error:
            for call in calls:
                self.journal(
                    dict(kind="error", turn=call, error_type=type(error).__name__)
                )
            raise
        for call, row in zip(calls, response["results"]):
            self.journal(
                dict(
                    kind="response",
                    turn=call,
                    response={k: row[k] for k in ("input_tokens", "output_tokens")},
                )
            )
        return response


def validate_roles():
    prepared = read(DATA / "preparation.json")
    if (
        prepared["status"] != "fresh_roles_prepared_not_evaluated"
        or prepared["new_model_calls"] != 0
    ):
        raise ValueError("Unexpected preparation history")
    excluded = read(DATA / "preparation.private.json")
    prior_groups, prior_ids = (
        set(excluded["excluded_groups"]),
        set(excluded["excluded_ids"]),
    )
    owners = {}
    for role, count in COUNTS.items():
        path = DATA / "tasks" / (role + ".jsonl")
        rows = read_jsonl(path)
        if (
            file_sha(path) != prepared["roles"][role]["sha256"]
            or len(rows) != 2 * count
        ):
            raise ValueError("Fresh-role identity/count changed")
        for row in rows:
            group = task_group(row)
            if (
                row["official_split"] != "train"
                or row["fresh_role"] != role
                or row["partition"] != "fresh_" + role
                or group in owners
                or group in prior_groups
                or row["task_id"] in prior_ids
            ):
                raise ValueError("Fresh-role leakage or role mismatch")
            owners[group] = role
        if Counter(r["task_type"] for r in rows) != Counter(
            {f: count for f in FAMILIES}
        ):
            raise ValueError("Family allocation changed")
    return prepared


def config(method):
    return dict(
        protocol_id="fresh-executor-base-calibration-v1",
        manifest=str(DATA / "tasks/calibration.jsonl"),
        method=method,
        revision=engine.REV,
        max_turns=40,
        max_calls=40,
        max_trials=1,
        max_new_tokens=128,
        history_turns=12,
        batch_size=8,
        temperature=0.0,
        seeds=[11],
        model=dict(
            base_url_env="LH_STUDENT_BASE_URL",
            model_env="LH_STUDENT_MODEL",
            api_key_env="LH_STUDENT_API_KEY",
        ),
    )


def validate(root=ROOT):
    if root != ROOT:
        raise ValueError("Only the declared fresh-calibration root is supported")
    frozen = read(root / "freeze.json")
    expected_cases = [
        dict(name=m, method=m, gpu=0) for m in ("json_direct", "react_retry")
    ]
    if (
        frozen["cases"] != expected_cases
        or frozen["gpus"] != [0]
        or frozen["planned_episodes"] != 64
        or frozen["cap_gpu_hours"] != 1
        or frozen["allocation_seconds"] != 1750
        or frozen["minimum_allocation_seconds"] != 700
        or frozen["teacher_calls"] != 0
        or frozen["official_tests_used"]
        or frozen["automatic_retry"]
        or frozen["protocol_text"] != SPEC.read_text()
        or not CODE.issubset(frozen["files"])
    ):
        raise ValueError("Frozen calibration scope changed")
    for path, expected in frozen["files"].items():
        if file_sha(path) != expected:
            raise ValueError("Calibration input changed: " + path)
    for method in ("json_direct", "react_retry"):
        if read(root / "configs" / (method + ".json")) != config(method):
            raise ValueError("Executor configuration changed")
    validate_roles()
    return frozen


def freeze():
    if ROOT.exists():
        raise ValueError("Never overwrite a prior calibration freeze or attempt")
    validate_roles()
    parent = read(PARENT / "freeze.json")
    ledger = read(PARENT / "execution.json")
    if ledger["status"] != "completed" or ledger.get("open"):
        raise ValueError("Previous controlled study must be complete")
    ROOT.mkdir(parents=True, exist_ok=False)
    for method in ("json_direct", "react_retry"):
        save(ROOT / "configs" / (method + ".json"), config(method))
    paths = CODE | {
        str(DATA / "preparation.json"),
        str(DATA / "preparation.private.json"),
        str(DATA / "public-reset-signatures.private.json"),
        str(PARENT / "freeze.json"),
        "data/models/Qwen3-4B-Instruct-2507/download-manifest.json",
    }
    paths |= {str(DATA / "tasks" / (role + ".jsonl")) for role in COUNTS}
    paths |= {str(p) for p in (ROOT / "configs").glob("*.json")}
    paths |= {
        str(Path("data/alfworld") / row["game_relpath"])
        for role in COUNTS
        for row in read_jsonl(DATA / "tasks" / (role + ".jsonl"))
    }
    frozen = dict(
        name="Fresh placement/cleaning base executor calibration",
        protocol="fresh-executor-calibration-v1",
        cases=[dict(name=m, method=m, gpu=0) for m in ("json_direct", "react_retry")],
        gpus=[0],
        planned_episodes=64,
        cap_gpu_hours=1,
        allocation_seconds=1750,
        minimum_allocation_seconds=700,
        allowed_background_contexts=parent["allowed_background_contexts"],
        teacher_calls=0,
        official_tests_used=False,
        automatic_retry=False,
        raw_records_uploaded=False,
        parent_freeze_sha256=file_sha(PARENT / "freeze.json"),
        protocol_text=SPEC.read_text(),
        preparation_sha256=file_sha(DATA / "preparation.json"),
        files={p: file_sha(p) for p in sorted(paths)},
        frozen_unix=time.time(),
    )
    save(ROOT / "freeze.json", frozen)
    validate()
    print(
        json.dumps(
            dict(
                status="frozen_not_started",
                episodes=64,
                inputs=len(paths),
                freeze_sha256=file_sha(ROOT / "freeze.json"),
            )
        ),
        flush=True,
    )


def worker(case, gpu, seconds):
    from learning_harness.controlled_study import validate_reservation

    frozen = validate()
    if gpu != 0 or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Only the frozen physical GPU is allowed")
    until = time.monotonic() + 2
    while True:
        ledger = read(ROOT / "execution.json")
        if case in ledger.get("open", {}) or time.monotonic() >= until:
            break
        time.sleep(0.02)
    validate_reservation(
        ledger, case, os.getpid(), gpu, seconds, file_sha(ROOT / "freeze.json")
    )
    plan = next(c for c in frozen["cases"] if c["name"] == case)
    folder = ROOT / "cases" / case
    if (folder / "model-requests.private").exists():
        raise ValueError("Prior partial requests retained; do not automatically retry")
    before = verify_base_model(Path(engine.MODEL), engine.REV)
    with engine.server(
        gpu, "fresh-base-calibration", engine.REV, folder / "server.log"
    ) as env:
        previous = dict(os.environ)
        os.environ.update(env)
        try:
            with RequestJournal(folder / "model-requests.private") as journal:
                client = JournalClient(BatchClient(config(case)["model"]), journal)
                results = benchmark_run(
                    ROOT / "configs" / (case + ".json"),
                    "data/alfworld",
                    folder / "evaluation",
                    client=client,
                )
        finally:
            os.environ.clear()
            os.environ.update(previous)
    after = verify_base_model(Path(engine.MODEL), engine.REV)
    usage = summarize_requests(folder / "model-requests.private")
    if before != after or not usage["complete_usage"] or usage["error_records"]:
        raise ValueError("Model identity or complete request accounting failed")
    tasks = read_jsonl(DATA / "tasks/calibration.jsonl")
    expected = {r["task_id"] for r in tasks}
    if (
        len(results) != 32
        or {r["task_id"] for r in results} != expected
        or any(
            r["seed"] != 11 or len(r["trials"]) != 1 or r["calls"] > 40 for r in results
        )
        or sum(r["calls"] for r in results) != usage["attempted_requests"]
        or sum(r["input_tokens"] for r in results) != usage["known_input_tokens"]
        or sum(r["output_tokens"] for r in results) != usage["known_output_tokens"]
    ):
        raise ValueError("Raw episodes and request usage disagree")
    events = read_jsonl(folder / "evaluation/events.jsonl")
    record = dict(
        complete=True,
        case=plan,
        freeze_sha256=file_sha(ROOT / "freeze.json"),
        episodes=32,
        successes=sum(r["success"] for r in results),
        mean_success=sum(r["success"] for r in results) / 32,
        families={
            f: dict(
                tasks=16,
                successes=sum(r["success"] for r in results if r["task_type"] == f),
            )
            for f in FAMILIES
        },
        invalid_format_decisions=sum(bool(r.get("invalid_format")) for r in events),
        thought_decisions=sum(bool(r.get("thought")) for r in events),
        usage=usage,
        model_identity=after,
        results_sha256=file_sha(folder / "evaluation/tasks.jsonl"),
        events_sha256=file_sha(folder / "evaluation/events.jsonl"),
    )
    save(folder / "worker-result.json", record)


def audit_result(folder, result, case, tasks):
    """Recompute every reported count from local episodes, events and requests."""
    rows = read_jsonl(folder / "evaluation/tasks.jsonl")
    events = read_jsonl(folder / "evaluation/events.jsonl")
    usage = summarize_requests(folder / "model-requests.private")
    expected = {r["task_id"]: r for r in tasks}
    if (
        not result["complete"]
        or result["case"] != case
        or len(rows) != 32
        or len(expected) != 32
        or {r["task_id"] for r in rows} != set(expected)
        or result["results_sha256"] != file_sha(folder / "evaluation/tasks.jsonl")
        or result["events_sha256"] != file_sha(folder / "evaluation/events.jsonl")
        or not usage["complete_usage"]
        or usage["error_records"]
    ):
        raise ValueError("Raw calibration identity or request accounting failed")
    for row in rows:
        own = [e for e in events if e["run_id"] == row["run_id"]]
        task = expected[row["task_id"]]
        if (
            row["seed"] != 11
            or row["method"] != case["method"]
            or row["task_type"] != task["task_type"]
            or row["task_group"] != task["task_group"]
            or len(row["trials"]) != 1
            or not 1 <= row["calls"] <= 40
            or len(own) != row["calls"]
            or any(
                e["task_id"] != row["task_id"]
                or e["seed"] != 11
                or e["method"] != case["method"]
                or e["trial"] != 1
                or e["phase"] != "act"
                for e in own
            )
            or [e["call"] for e in own] != list(range(1, row["calls"] + 1))
            or row["success"] != own[-1]["success"]
            or any(
                sum(e[k] for e in own) != row[k]
                for k in ("input_tokens", "output_tokens")
            )
        ):
            raise ValueError("Episode/event calibration accounting failed")
    if (
        len(events) != sum(r["calls"] for r in rows)
        or len(events) != usage["attempted_requests"]
        or sum(r["input_tokens"] for r in rows) != usage["known_input_tokens"]
        or sum(r["output_tokens"] for r in rows) != usage["known_output_tokens"]
    ):
        raise ValueError("Calibration usage totals disagree")
    derived = dict(
        episodes=32,
        successes=sum(r["success"] for r in rows),
        mean_success=sum(r["success"] for r in rows) / 32,
        families={
            f: dict(
                tasks=16,
                successes=sum(r["success"] for r in rows if r["task_type"] == f),
            )
            for f in FAMILIES
        },
        invalid_format_decisions=sum(bool(e.get("invalid_format")) for e in events),
        thought_decisions=sum(bool(e.get("thought")) for e in events),
        usage=usage,
    )
    if any(result[k] != value for k, value in derived.items()):
        raise ValueError("Reported calibration counts disagree with raw records")
    return derived


def report():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frozen = validate()
    groups = {}
    for case in frozen["cases"]:
        folder = ROOT / "cases" / case["name"]
        result = read(folder / "worker-result.json")
        if result["freeze_sha256"] != file_sha(ROOT / "freeze.json"):
            raise ValueError("Completed calibration result changed")
        groups[case["method"]] = audit_result(
            folder, result, case, read_jsonl(DATA / "tasks/calibration.jsonl")
        )
    result = dict(
        complete=True,
        freeze_sha256=file_sha(ROOT / "freeze.json"),
        calibration_only=True,
        groups=groups,
        completed_episodes=64,
        teacher_calls=0,
        official_tests_used=False,
        development_query_evaluations=0,
        confirmation_query_evaluations=0,
        raw_records_uploaded=False,
        scope="Fresh train-derived placement/cleaning base calibration; no SFT or selection effect.",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / "aggregate-summary.json", result)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    values = [100 * groups[m]["mean_success"] for m in ("json_direct", "react_retry")]
    ax.bar(range(2), values, color=["#147D92", "#C85B39"], width=0.55)
    for i, value in enumerate(values):
        ax.text(i, value + 2, f"{value:.1f}%", ha="center")
    ax.set(
        xticks=range(2),
        xticklabels=["Common JSON", "Common ReAct"],
        ylim=(0, 105),
        ylabel="Success (%)",
        title="Fresh-task base calibration / 32 shared tasks",
    )
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUT / "executor-calibration.png", dpi=180)
    plt.close(fig)
    lines = [
        "# 新任务基础学生执行方式校准",
        "",
        "32个独立放置/清洗任务、两执行方式，共64回合。属于校准，开发/确认集未评测，无SFT效应结论。",
        "",
        "| 执行方式 | 成功/32 | 成功率 | 请求数 |",
        "|---|---:|---:|---:|",
    ]
    for method, g in groups.items():
        lines.append(
            f"| {method} | {g['successes']}/32 | {g['mean_success']:.4%} | {g['usage']['attempted_requests']} |"
        )
    (OUT / "RESULTS.zh-CN.md").write_text("\n".join(lines) + "\n")


def run():
    from run_action_controls import gpu_preflight

    frozen = validate()
    lock = (ROOT / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROOT / "execution.json").exists():
        raise ValueError("Prior attempt exists; inspect before rerun")
    save(
        ROOT / "gpu-preflight.json",
        gpu_preflight([0], frozen["allowed_background_contexts"]),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    args = [str(Path(".venv-wandb/bin/python").absolute()), TRACKER]
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
                [engine.MODEL_PY, SCRIPT, "report"],
                env=dict(
                    environment(0),
                    CUDA_VISIBLE_DEVICES="",
                    MPLCONFIGDIR="/tmp/lh-fresh-cal-mpl",
                ),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=120,
            )
        ledger["status"] = "completed"
        save(ROOT / "execution.json", ledger)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("freeze", "validate", "run", "worker", "report"))
    p.add_argument("--case")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--timeout", type=float, default=1750)
    a = p.parse_args()
    if a.command == "freeze":
        freeze()
    elif a.command == "validate":
        f = validate()
        print(
            json.dumps(
                dict(status="validated", episodes=64, frozen_inputs=len(f["files"]))
            )
        )
    elif a.command == "run":
        run()
    elif a.command == "report":
        report()
    else:
        if a.case not in ("json_direct", "react_retry") or a.timeout <= 0:
            p.error("Explicit supported case and positive timeout required")

        def stop(signum, frame):
            raise TimeoutError("Fresh calibration deadline or stop signal")

        signal.signal(signal.SIGALRM, stop)
        signal.signal(signal.SIGTERM, stop)
        signal.setitimer(signal.ITIMER_REAL, a.timeout)
        worker(a.case, a.gpu, a.timeout)


if __name__ == "__main__":
    main()
