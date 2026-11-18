#!/usr/bin/env python3
"""Reuse the unchanged calibration worker with an explicit JSON-example actor."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import run_fresh_executor_calibration as original
import run_optimizer_aware_selection_fp32 as predecessor
from learning_harness import benchmark
from learning_harness.bounded_queue import run as bounded_run, save
from learning_harness.common import file_sha, read_jsonl
from learning_harness.json_demo_executor import actor_messages

ROOT = Path("runs/json-examples-calibration-v1")
OUT = Path("output/json-examples-calibration-v1")
PREVIOUS = Path("runs/fresh-executor-calibration-v1")
SPEC = Path("docs/JSON_PUBLIC_EXAMPLES_CALIBRATION_SPEC_2026-11-13.zh-CN.md")
SCRIPT = "scripts/run_json_examples_calibration.py"
CASE = "json_public_examples"
ORIGINAL_CONFIG = original.config
CODE = {
    SCRIPT,
    "scripts/track_json_examples_calibration.py",
    str(SPEC),
    "src/learning_harness/json_demo_executor.py",
    "tests/test_json_demo_executor.py",
    "tests/test_json_examples_calibration.py",
    "src/learning_harness/public_demo_support.py",
    "scripts/run_optimizer_aware_selection_fp32.py",
}


def read(p):
    return json.loads(Path(p).read_text())


def config(name=CASE):
    if name != CASE:
        raise ValueError("Only the JSON public-example condition is registered")
    return dict(ORIGINAL_CONFIG("json_direct"), executor="json_public_examples_v1")


def case():
    return dict(
        name=CASE, method="json_direct", executor="json_public_examples_v1", gpu=0
    )


def validate():
    frozen = read(ROOT / "freeze.json")
    if (
        frozen["cases"] != [case()]
        or frozen["planned_episodes"] != 32
        or frozen["gpus"] != [0]
        or frozen["cap_gpu_hours"] != 0.5
        or frozen["allocation_seconds"] != 1750
        or frozen["minimum_allocation_seconds"] != 700
        or frozen["teacher_calls"] != 0
        or frozen["official_tests_used"]
        or frozen["automatic_retry"]
        or frozen["protocol_text"] != SPEC.read_text()
        or not CODE <= set(frozen["files"])
    ):
        raise ValueError("JSON example calibration scope changed")
    for p, sha in frozen["files"].items():
        if file_sha(p) != sha:
            raise ValueError("Calibration input changed: " + p)
    if read(ROOT / "configs" / (CASE + ".json")) != config():
        raise ValueError("Config changed")
    original.validate_roles()
    return frozen


def freeze():
    if ROOT.exists():
        raise ValueError("Preserve prior calibration freeze")
    original.validate()
    predecessor.validate()
    if read(PREVIOUS / "execution.json")["status"] != "completed":
        raise ValueError("Previous base calibration incomplete")
    paths = (
        set(read(PREVIOUS / "freeze.json")["files"])
        | CODE
        | {str(PREVIOUS / "freeze.json"), str(predecessor.ROOT / "freeze.json")}
    )
    for m in ("json_direct", "react_retry"):
        p = PREVIOUS / "cases" / m
        paths.update(
            str(v)
            for v in (
                p / "worker-result.json",
                p / "evaluation/tasks.jsonl",
                p / "evaluation/events.jsonl",
                p / "model-requests.private",
            )
        )
    for p in paths:
        if not Path(p).is_file():
            raise ValueError("Missing dependency: " + p)
    ROOT.mkdir(exist_ok=False)
    save(ROOT / "configs" / (CASE + ".json"), config())
    paths.add(str(ROOT / "configs" / (CASE + ".json")))
    frozen = dict(
        name="Public examples with a matched JSON action interface",
        cases=[case()],
        gpus=[0],
        planned_episodes=32,
        cap_gpu_hours=0.5,
        allocation_seconds=1750,
        minimum_allocation_seconds=700,
        allowed_background_contexts=read(PREVIOUS / "freeze.json")[
            "allowed_background_contexts"
        ],
        protocol_text=SPEC.read_text(),
        teacher_calls=0,
        official_tests_used=False,
        automatic_retry=False,
        raw_records_uploaded=False,
        files={p: file_sha(p) for p in sorted(paths)},
        frozen_unix=time.time(),
    )
    save(ROOT / "freeze.json", frozen)
    validate()
    print(
        json.dumps(
            dict(
                status="frozen_waiting_for_gpu",
                inputs=len(paths),
                planned_episodes=32,
                freeze_sha256=file_sha(ROOT / "freeze.json"),
            )
        ),
        flush=True,
    )


def bind_worker():
    # Dependency injection is confined to this child. All historical source bytes
    # and the direct JSON parser remain unchanged; only actor messages differ.
    def actor(initial, history, memories, method, history_turns):
        if method != "json_direct" or memories:
            raise ValueError("Unexpected actor or memory")
        return actor_messages(initial, history, history_turns)

    benchmark.actor_messages = actor
    original.ROOT = ROOT
    original.config = config
    original.validate = validate


def report():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    validate()
    tasks = read_jsonl(original.DATA / "tasks/calibration.jsonl")
    groups = {}
    for name, folder, plan, freeze_path in (
        (
            "json_direct",
            PREVIOUS / "cases/json_direct",
            dict(name="json_direct", method="json_direct", gpu=0),
            PREVIOUS / "freeze.json",
        ),
        (
            "react_retry",
            PREVIOUS / "cases/react_retry",
            dict(name="react_retry", method="react_retry", gpu=0),
            PREVIOUS / "freeze.json",
        ),
        (CASE, ROOT / "cases" / CASE, case(), ROOT / "freeze.json"),
    ):
        result = read(folder / "worker-result.json")
        if result["freeze_sha256"] != file_sha(freeze_path):
            raise ValueError("Prior/new result identity changed")
        groups[name] = original.audit_result(folder, result, plan, tasks)
    aggregate = dict(
        complete=True,
        freeze_sha256=file_sha(ROOT / "freeze.json"),
        calibration_only=True,
        new_episodes=32,
        previous_episodes=64,
        groups=groups,
        teacher_calls=0,
        official_tests_used=False,
        development_query_evaluations=0,
        confirmation_query_evaluations=0,
        raw_records_uploaded=False,
        scope="Format-compatible public-example base calibration; no learned student or isolated thought-effect claim.",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / "aggregate-summary.json", aggregate)
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    values = [
        100 * groups[m]["mean_success"] for m in ("json_direct", CASE, "react_retry")
    ]
    ax.bar(range(3), values, color=["#147D92", "#6B8E5B", "#C85B39"], width=0.6)
    for i, value in enumerate(values):
        ax.text(i, value + 2, f"{value:.1f}%", ha="center")
    ax.set(
        xticks=range(3),
        xticklabels=["JSON", "JSON + public examples", "ReAct"],
        ylim=(0, 105),
        ylabel="Success (%)",
        title="Same 32 calibration tasks / base model",
    )
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUT / "json-interface-calibration.png", dpi=180)
    plt.close(fig)
    lines = [
        "# JSON兼容公共示例校准",
        "",
        "新增32回合，开发/确认集仍未评测。前轮JSON/ReAct作为同任务历史校准坐标；没有最终学生训练收益结论。",
        "",
        "| 条件 | 成功/32 | 请求数 |",
        "|---|---:|---:|",
    ]
    for name, g in groups.items():
        lines.append(
            f"| {name} | {g['successes']}/32 | {g['usage']['attempted_requests']} |"
        )
    lines.extend(
        [
            "",
            "保持JSON指令和解析，只增加公共示例，便于后续JSON动作监督学生采用共同部署接口。与ReAct的比较还涉及思考与格式差异，不是单因素机制实验。",
            "",
            "![校准图](json-interface-calibration.png)",
        ]
    )
    (OUT / "RESULTS.zh-CN.md").write_text("\n".join(lines) + "\n")


def run():
    from run_action_controls import gpu_preflight

    frozen = validate()
    lock = (ROOT / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROOT / "execution.json").exists():
        raise ValueError("Prior attempt exists; no automatic retry")
    save(
        ROOT / "gpu-preflight.json",
        gpu_preflight([0], frozen["allowed_background_contexts"]),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    args = [
        str(Path(".venv-wandb/bin/python").absolute()),
        "scripts/track_json_examples_calibration.py",
    ]
    with (ROOT / "observer.log").open("x") as log:
        observer = subprocess.Popen(
            args,
            env=dict(original.environment(0), CUDA_VISIBLE_DEVICES=""),
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
            engine_cpu(),
            SCRIPT,
            "worker",
            "--case",
            CASE,
            "--timeout",
            str(seconds),
        ]

    ledger = bounded_run(ROOT, frozen, command, original.environment)
    if ledger["status"] == "evaluation_completed":
        with (OUT / "report.log").open("x") as log:
            subprocess.run(
                [original.engine.MODEL_PY, SCRIPT, "report"],
                env=dict(
                    original.environment(0),
                    CUDA_VISIBLE_DEVICES="",
                    MPLCONFIGDIR="/tmp/lh-json-examples-mpl",
                ),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )
        ledger["status"] = "completed"
        save(ROOT / "execution.json", ledger)


def engine_cpu():
    return original.engine.CPU_PY


def dispatch():
    validate()
    lock = (ROOT / "dispatch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = ROOT / "dispatch.json"
    if path.exists():
        raise ValueError("Previous dispatch retained")
    save(
        path,
        dict(
            status="waiting_for_feature_completion",
            pid=os.getpid(),
            started_unix=time.time(),
            upstream=str(predecessor.ROOT),
            automatic_retry=False,
        ),
    )
    save(
        ROOT / "job.json",
        dict(
            dispatcher=dict(
                pid=os.getpid(),
                argv=[sys.executable, *sys.argv],
                started_unix=time.time(),
            )
        ),
    )
    deadline = time.monotonic() + 7800
    while True:
        ledger = read(predecessor.ROOT / "execution.json")
        state = ledger["status"]
        if state == "completed":
            predecessor.validate()
            result = read(
                predecessor.ROOT / "cases" / predecessor.CASE / "worker-result.json"
            )
            if not result["complete"] or result["freeze_sha256"] != file_sha(
                predecessor.ROOT / "freeze.json"
            ):
                raise ValueError("Upstream completion identity mismatch")
            save(
                path,
                dict(
                    status="upstream_complete_starting",
                    pid=os.getpid(),
                    upstream=str(predecessor.ROOT),
                    automatic_retry=False,
                ),
            )
            run()
            save(
                path,
                dict(
                    status="finished",
                    execution_status=read(ROOT / "execution.json")["status"],
                    automatic_retry=False,
                ),
            )
            return
        if state in ("failed", "interrupted", "stopped_at_cap"):
            save(
                path,
                dict(
                    status="stopped_due_to_upstream",
                    upstream_status=state,
                    automatic_retry=False,
                ),
            )
            return
        if time.monotonic() > deadline:
            raise TimeoutError("Queue deadline; do not restart upstream")
        time.sleep(30)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command", choices=("freeze", "validate", "run", "dispatch", "worker", "report")
    )
    p.add_argument("--case")
    p.add_argument("--timeout", type=float, default=1750)
    a = p.parse_args()
    if a.command == "freeze":
        freeze()
    elif a.command == "validate":
        print(json.dumps(dict(inputs=len(validate()["files"]))))
    elif a.command == "run":
        run()
    elif a.command == "dispatch":
        dispatch()
    elif a.command == "report":
        report()
    else:
        if a.case != CASE or a.timeout <= 0:
            raise ValueError("Explicit registered case and timeout required")

        def stop(signum, frame):
            raise TimeoutError("Calibration deadline")

        signal.signal(signal.SIGALRM, stop)
        signal.signal(signal.SIGTERM, stop)
        signal.setitimer(signal.ITIMER_REAL, a.timeout)
        bind_worker()
        original.worker(CASE, 0, a.timeout)


if __name__ == "__main__":
    main()
