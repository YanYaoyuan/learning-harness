#!/usr/bin/env python3
"""Freeze and launch matched-compute full-pass versus subset-replay diagnostics."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import run_api_harness as engine
import run_stage1_crossed_seeds as previous
from learning_harness.bounded_queue import run, save
from learning_harness.common import file_sha
from train_lora import fixed_schedule
from train_lora_sampling_intervention import schedule

ROOT = Path("runs/variance-diagnostic-v1/stage2")
OUT = Path("output/variance-diagnostic-v1/stage2")
ARMS = ("full_pass", "subset_replay")
FILES = ["docs/VARIANCE_DIAGNOSTIC_STAGE2_SPEC.md", "scripts/run_stage2_sampling.py",
    "scripts/stage2_worker.py", "scripts/stage2_sampling_report.py", "scripts/train_lora_sampling_intervention.py",
    "scripts/track_stage2.py", "scripts/track_stage1.py", "scripts/upload_research_results.py",
    "src/learning_harness/bounded_queue.py", "runs/variance-diagnostic-v1/stage1/freeze.json",
    "output/variance-diagnostic-v1/stage1/summary.json"]


def plan():
    cases = []
    for case in previous.plan():
        parity = (previous.SEEDS.index(case["initialization_seed"]) + previous.SEEDS.index(case["sampling_seed"])) % 2
        for arm in (ARMS if parity == 0 else ARMS[::-1]):
            cases.append(dict(case, name=f"{arm}-{case['name']}", arm=arm))
    return cases


def smoke():
    if (ROOT / "freeze.json").exists():
        raise ValueError("Smoke belongs before freeze")
    result_path = ROOT / "smoke/result.json"
    if result_path.exists():
        if not json.loads(result_path.read_text()).get("passed"):
            raise ValueError("Prior smoke failed; inspect before retry")
        return
    previous.load_freeze()
    cfg = json.loads((previous.ROOT / "a-lora.json").read_text())
    cfg["max_steps"] = 1
    save(ROOT / "smoke/config.json", cfg)
    started = time.monotonic()
    result = dict(passed=False, teacher_calls=0, task_evaluations=0, controls=[])
    try:
        for arm in ARMS:
            directory = ROOT / "smoke" / arm
            engine.command([engine.MODEL_PY, "scripts/train_lora_sampling_intervention.py", "--sampling-arm", arm,
                "--data", "runs/transfer-factorial-v1/corpora/full-api.jsonl", "--model", engine.MODEL,
                "--revision", engine.REV, "--config", str(ROOT / "smoke/config.json"), "--out", str(directory),
                "--initialization-seed", "101", "--sampling-seed", "101"],
                previous.environment(1), ROOT / "smoke" / f"{arm}.log", timeout=180)
            ctrl = json.loads((directory / "seed-controls.json").read_text())
            if not ctrl.get("completed") or ctrl.get("sampling_arm") != arm:
                raise ValueError("Real GPU policy wrapper did not complete")
            result["controls"].append(ctrl)
        a, b = result["controls"]
        if a["initial_parameter_sha256"] != b["initial_parameter_sha256"] or a["schedule_sha256"] != b["schedule_sha256"]:
            raise ValueError("Policy wrapper changed initialization or shared one-update prefix")
        result["schedule_audits"] = []
        for length in (721, 732):
            for seed in previous.SEEDS:
                prefix = fixed_schedule(length, 256, seed)
                full = schedule(fixed_schedule, length, 768, seed, "full_pass")
                replay = schedule(fixed_schedule, length, 768, seed, "subset_replay")
                if full[:256] != prefix or replay[:256] != prefix or len(set(full)) != length or len(set(replay)) != 256:
                    raise ValueError("Main-budget membership or shared prefix changed")
                result["schedule_audits"].append(dict(length=length, seed=seed,
                    full_unique=len(set(full)), replay_unique=len(set(replay)),
                    full_sha256=hashlib.sha256(json.dumps(full).encode()).hexdigest(),
                    replay_sha256=hashlib.sha256(json.dumps(replay).encode()).hexdigest()))
        result["passed"] = True
    finally:
        result["gpu_seconds"] = time.monotonic() - started
        save(result_path, result)


def freeze():
    if (ROOT / "freeze.json").exists():
        raise ValueError("Stage 2 freeze is never overwritten")
    old = previous.load_freeze()
    result = json.loads((ROOT / "smoke/result.json").read_text())
    if not result["passed"]:
        raise ValueError("Policy smoke did not pass")
    prior = json.loads(Path("output/variance-diagnostic-v1/stage1/summary.json").read_text())
    if not prior["complete"] or prior["cases"] != 34:
        raise ValueError("Stage 1 report incomplete")
    config = json.loads((previous.ROOT / "a-lora.json").read_text())
    config["max_steps"] = 192
    save(ROOT / "a-lora.json", config)
    weights = (ROOT / "smoke/full_pass/adapter_model.safetensors").stat().st_size
    required = 2 * weights * len(plan()) + 2 * 1024 ** 3
    if shutil.disk_usage(ROOT).free < required:
        raise ValueError("Need all A/B checkpoints plus 2 GiB records reserve")
    files = set(FILES + list(old["files"]) + [str(ROOT / "a-lora.json"), str(ROOT / "smoke/result.json")])
    for case in previous.plan():
        files.add(str(previous.ROOT / "cases" / case["name"] / "worker-result.json"))
    save(ROOT / "freeze.json", dict(protocol="variance-diagnostic-v1-stage2", frozen_unix=time.time(),
        cases=plan(), files={p: file_sha(p) for p in sorted(files)}, gpus=[0, 1], cap_gpu_hours=14,
        allocation_seconds=1800, minimum_allocation_seconds=900, presentations=768, optimizer_updates=192,
        planned_episodes=4352, teacher_calls=0, official_tests_used=False,
        primary_comparison="full_pass minus subset_replay, same 768 presentations and 192 updates",
        old_256_reference="historical dose reference; not a compute-matched policy comparison"))


def load_freeze():
    value = json.loads((ROOT / "freeze.json").read_text())
    if value["cases"] != plan() or value["cap_gpu_hours"] != 14:
        raise ValueError("Stage 2 design changed")
    for path, expected in value["files"].items():
        if file_sha(path) != expected:
            raise ValueError("Stage 2 frozen input changed: " + path)
    previous.load_freeze()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    controller_lock = (ROOT / "controller.lock").open("a")
    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.smoke:
        smoke()
        return
    if args.freeze:
        freeze()
        return
    frozen = load_freeze()
    for gpu in frozen["gpus"]:
        if not previous.stage0.port_is_free(gpu):
            raise ValueError("Existing model server occupies GPU " + str(gpu))
    def command(case, gpu, seconds):
        return [engine.MODEL_PY, "scripts/stage2_worker.py", "--case", case["name"],
                "--gpu", str(gpu), "--timeout", str(seconds)]
    ledger = run(ROOT, frozen, command, previous.environment,
                 json.loads((ROOT / "smoke/result.json").read_text())["gpu_seconds"])
    if ledger["status"] == "evaluation_completed":
        OUT.mkdir(parents=True, exist_ok=True)
        try:
            with (OUT / "report.log").open("w") as log:
                subprocess.run([engine.MODEL_PY, "scripts/stage2_sampling_report.py"], env=previous.environment(0),
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        except Exception:
            ledger["report_failed"] = True
            save(ROOT / "execution.json", ledger)
            raise
        ledger["status"] = "completed"
        save(ROOT / "execution.json", ledger)


if __name__ == "__main__":
    main()
