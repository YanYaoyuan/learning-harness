#!/usr/bin/env python3
"""Bounded, read-only re-evaluation of two stored Stage 3 low-NLL checkpoints."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

import run_stage3_task_controls as parent
import stage1_worker as kernel
from run_action_controls import gpu_preflight
from run_stage1_crossed_seeds import environment
from learning_harness.adaptation import composition_revision
from learning_harness.bounded_queue import run, save
from learning_harness.common import file_sha, read_jsonl
from learning_harness.controlled_study import audit_episodes, validate_reservation

ROOT = Path("runs/variance-diagnostic-v1/stage3-reliability-check")
OUT = Path("output/variance-diagnostic-v1/stage3-reliability-check")


def freeze():
    if (ROOT / "freeze.json").exists():
        raise ValueError("Existing diagnostic freeze must be preserved")
    original = parent.load_freeze()
    if (
        json.loads((parent.ROOT / "execution.json").read_text())["status"]
        != "completed"
        or not json.loads((parent.OUT / "summary.json").read_text())["complete"]
    ):
        raise ValueError("Complete the parent study before this diagnostic")
    contexts = json.loads(
        Path("output/research-operations/20261113/background-contexts.json").read_text()
    )
    gpu_preflight([0], contexts)
    cases, paths = [], set(original["files"])
    paths |= {
        str(parent.ROOT / "freeze.json"),
        str(parent.OUT / "summary.json"),
        "scripts/repeat_stage3_checkpoints.py",
        "docs/STAGE3_REPEAT_RELIABILITY_2026-11-13.zh-CN.md",
        "scripts/run_action_controls.py",
        "src/learning_harness/idle_gpu_contexts.py",
        "src/learning_harness/controlled_study.py",
    }
    for repeat in (0, 1):
        folder = parent.ROOT / "cases" / f"task_matched_min_nll-api-i101-s101-r{repeat}"
        result = json.loads((folder / "worker-result.json").read_text())
        if not result["complete"] or result["case"]["gpu"] != 0:
            raise ValueError("Expected completed same-GPU checkpoints")
        cases.append(
            dict(
                name=f"stored-r{repeat}",
                training_repeat=repeat,
                source="api",
                gpu=0,
                checkpoint=str(folder / "checkpoint"),
                historical_result=str(folder / "worker-result.json"),
            )
        )
        paths |= {
            str(folder / "worker-result.json"),
            str(folder / "A/adapter_model.safetensors"),
            str(folder / "checkpoint/composition.json"),
            str(folder / "checkpoint/B/adapter_model.safetensors"),
        }
        for family in ("cool", "heat"):
            paths |= {
                str(folder / "evaluation" / family / name)
                for name in (
                    "config.json",
                    "episodes.jsonl",
                    "run.json",
                    "summary.json",
                )
            }
    ROOT.mkdir(parents=True, exist_ok=True)
    save(ROOT / "background-contexts.json", contexts)
    paths.add(str(ROOT / "background-contexts.json"))
    value = dict(
        protocol="stage3-fixed-checkpoint-repeat-v1",
        name="Stage 3 / fixed-checkpoint reliability check",
        cases=cases,
        gpus=[0],
        cap_gpu_hours=0.8,
        allocation_seconds=1200,
        minimum_allocation_seconds=600,
        planned_episodes=128,
        teacher_calls=0,
        new_training=False,
        official_tests_used=False,
        selected_after_observing_results=True,
        allowed_background_contexts=contexts,
        frozen_unix=time.time(),
        files={p: file_sha(p) for p in sorted(paths)},
    )
    save(ROOT / "freeze.json", value)
    return load()


def load():
    parent.load_freeze()
    value = json.loads((ROOT / "freeze.json").read_text())
    for name, expected in value["files"].items():
        if file_sha(name) != expected:
            raise ValueError("Diagnostic frozen input changed: " + name)
    return value


def worker(name, seconds):
    frozen = load()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Diagnostic worker must use its frozen GPU 0 reservation")
    deadline = time.monotonic() + 2
    while True:
        ledger = json.loads((ROOT / "execution.json").read_text())
        if name in ledger.get("open", {}) or time.monotonic() >= deadline:
            break
        time.sleep(0.02)
    validate_reservation(
        ledger, name, os.getpid(), 0, seconds, file_sha(ROOT / "freeze.json")
    )
    case = next(c for c in frozen["cases"] if c["name"] == name)
    point = Path(case["checkpoint"])
    before = composition_revision(point)
    old = json.loads(Path(case["historical_result"]).read_text())
    if before != old["composition_revision"]:
        raise ValueError("Checkpoint no longer matches historical evaluation")
    scores = {}
    for family in ("cool", "heat"):
        directory = ROOT / "cases" / name / "evaluation" / family
        result = kernel.evaluate(point, directory, family, "api", 0)
        rate = audit_episodes(
            read_jsonl(directory / "episodes.jsonl"),
            read_jsonl(kernel.OLD / family / "b_query.jsonl"),
        )
        historical = Path(case["historical_result"]).parent / "evaluation" / family
        if result["metrics"]["none"]["success_rate"] != rate or json.loads(
            (directory / "config.json").read_text()
        ) != json.loads((historical / "config.json").read_text()):
            raise ValueError("Raw repeated outcomes or configuration differ")
        scores[family] = rate
    if composition_revision(point) != before:
        raise ValueError("Read-only re-evaluation changed checkpoint")
    mean = sum(scores.values()) / 2
    save(
        ROOT / "cases" / name / "worker-result.json",
        dict(
            complete=True,
            case=case,
            freeze_sha256=file_sha(ROOT / "freeze.json"),
            historical_mean_success=old["mean_success"],
            mean_success=mean,
            difference_pp=100 * (mean - old["mean_success"]),
            success=scores,
            historical_success=old["success"],
            checkpoint_unchanged=True,
            new_training=False,
            new_episodes=64,
            teacher_calls=0,
        ),
    )


def report(ledger):
    frozen = load()
    records = [
        json.loads((ROOT / "cases" / c["name"] / "worker-result.json").read_text())
        for c in frozen["cases"]
    ]
    if not all(
        r["complete"]
        and r["checkpoint_unchanged"]
        and r["freeze_sha256"] == file_sha(ROOT / "freeze.json")
        for r in records
    ):
        raise ValueError("Incomplete diagnostic cannot enter report")
    OUT.mkdir(parents=True, exist_ok=True)
    summary = dict(
        complete=True,
        cases=2,
        actual_evaluation_episodes=128,
        freeze_sha256=file_sha(ROOT / "freeze.json"),
        results=records,
        gpu_hours=ledger["gpu_seconds"] / 3600,
        new_training=False,
        teacher_calls=0,
        official_tests_used=False,
        selected_after_observing_results=True,
        inference_scope="One as-is repeat of two stored models, not a population noise estimate; original study means unchanged",
    )
    save(OUT / "summary.json", summary)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    upper = max(
        35,
        max(
            100 * r[k]
            for r in records
            for k in ("historical_mean_success", "mean_success")
        )
        + 10,
    )
    for x, r in enumerate(records):
        values = [100 * r["historical_mean_success"], 100 * r["mean_success"]]
        ax.bar([x - 0.18, x + 0.18], values, width=0.32, color=["#2879ad", "#d46a30"])
        for offset, value in zip((-0.18, 0.18), values):
            ax.text(x + offset, value + 1, f"{value:.2f}%", ha="center", fontsize=9)
    ax.set(
        xticks=[0, 1],
        xticklabels=["Stored training r0", "Stored training r1"],
        ylabel="Same 64-task success (%)",
        ylim=(0, upper),
        title="AS-IS CHECKPOINT RE-EVALUATION — no retraining",
    )
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#2879ad"),
            plt.Rectangle((0, 0), 1, 1, color="#d46a30"),
        ],
        labels=["Historical evaluation", "Fresh evaluation"],
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(OUT / ("checkpoint-repeats." + extension), dpi=180)
    plt.close(fig)
    ledger["status"] = "completed"
    save(ROOT / "execution.json", ledger)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--freeze", action="store_true")
    modes.add_argument("--worker")
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    if args.freeze:
        print(
            json.dumps(
                dict(
                    status="frozen",
                    cases=len(freeze()["cases"]),
                    training_started=False,
                )
            ),
            flush=True,
        )
        return
    if args.worker:

        def stop(signum, frame):
            raise kernel.WorkerDeadline()

        signal.signal(signal.SIGALRM, stop)
        signal.signal(signal.SIGTERM, stop)
        signal.setitimer(signal.ITIMER_REAL, args.timeout)
        worker(args.worker, args.timeout)
        signal.setitimer(signal.ITIMER_REAL, 0)
        return
    frozen = load()
    lock = (ROOT / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    checked = gpu_preflight([0], frozen["allowed_background_contexts"])
    save(ROOT / "background-preflight.json", checked)
    save(
        ROOT / "job.json",
        dict(
            pipeline=dict(
                pid=os.getpid(),
                argv=[sys.executable, *sys.argv],
                started_unix=time.time(),
            )
        ),
    )

    def command(case, gpu, seconds):
        return [
            kernel.engine.MODEL_PY,
            "scripts/repeat_stage3_checkpoints.py",
            "--worker",
            case["name"],
            "--timeout",
            str(seconds),
        ]

    ledger = run(ROOT, frozen, command, environment)
    if ledger["status"] == "evaluation_completed":
        report(ledger)


if __name__ == "__main__":
    main()
