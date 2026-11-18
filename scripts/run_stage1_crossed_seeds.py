#!/usr/bin/env python3
"""Freeze and run the bounded 4×4 seed diagnostic, with two counterbalanced GPU queues."""
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time

import run_api_harness as engine
import run_stage0_repeat_eval as stage0
from learning_harness.common import file_sha

ROOT = Path("runs/variance-diagnostic-v1/stage1")
OUT = Path("output/variance-diagnostic-v1/stage1")
OLD = Path("runs/transfer-factorial-v1")
SEEDS = (101, 202, 303, 404)
FILES = ["docs/VARIANCE_DIAGNOSTIC_STAGE1_SPEC.md", "scripts/run_stage1_crossed_seeds.py",
         "scripts/stage1_worker.py", "scripts/stage1_variance_report.py", "scripts/train_lora_split_seeds.py",
         "scripts/train_lora.py", "scripts/run_api_harness.py", "scripts/run_stage0_repeat_eval.py",
         "scripts/hf_batch_server.py", "src/learning_harness/adaptation.py", "src/learning_harness/batch_runner.py",
         "src/learning_harness/analysis.py", "src/learning_harness/common.py",
         "runs/variance-diagnostic-v1/stage0/result.json", "runs/variance-diagnostic-v1/seed-control-smoke-v1/result.json"]


def plan():
    cases = []
    for i, init in enumerate(SEEDS):
        for j, sample in enumerate(SEEDS):
            for source in ("api", "generic"):
                gpu = (i + j) % 2 if source == "api" else 1 - (i + j) % 2
                cases.append(dict(name=f"{source}-i{init}-s{sample}-r0", source=source,
                    initialization_seed=init, sampling_seed=sample, repeat=0, gpu=gpu))
    for source in ("api", "generic"):
        cases.append(dict(name=f"{source}-i101-s101-r1", source=source, initialization_seed=101,
                          sampling_seed=101, repeat=1, gpu=0 if source == "api" else 1))
    return cases


def environment(gpu):
    env = engine.environment(gpu)
    env["WORLD_SIZE"] = "1"
    for name in ("RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(name, None)
    for name in ("NO_PROXY", "no_proxy"):
        env[name] = ",".join(v for v in (env.get(name, ""), "127.0.0.1", "localhost", "::1") if v)
    return env


def allocation_seconds(cap, spent, reserved, maximum=1200, minimum=600):
    available = cap - spent - reserved
    return min(maximum, available) if available >= minimum else 0


def check_inputs():
    historical_freeze = json.loads(Path("runs/variance-diagnostic-v1/stage0/freeze.json").read_text())
    stage0.check_historical_hashes(historical_freeze["documented_hash_exceptions"])
    control = json.loads(Path("runs/variance-diagnostic-v1/seed-control-smoke-v1/result.json").read_text())
    if not control["passed"]:
        raise ValueError("Split-seed GPU control check did not pass")
    if not json.loads(Path("runs/variance-diagnostic-v1/stage0/result.json").read_text())["decision"] == "Stage 1 in full":
        raise ValueError("Stage 0 did not choose the full grid")
    return historical_freeze


def freeze():
    path = ROOT / "freeze.json"
    if path.exists():
        raise ValueError("Stage 1 freeze is never overwritten")
    historical_freeze = check_inputs()
    smoke = json.loads((ROOT / "pipeline-smoke/worker-result.json").read_text())
    if not smoke["complete"] or not smoke["smoke"]:
        raise ValueError("End-to-end pipeline smoke is missing")
    checkpoint_bytes = (Path("runs/variance-diagnostic-v1/seed-control-smoke-v1/api-init101-sample101/adapter_model.safetensors").stat().st_size
                        + (ROOT / "pipeline-smoke/checkpoint/B/adapter_model.safetensors").stat().st_size)
    required_bytes = checkpoint_bytes * len(plan()) + 2 * 1024 ** 3
    available_bytes = shutil.disk_usage(ROOT).free
    if available_bytes < required_bytes:
        raise ValueError(f"Need {required_bytes} bytes for all checkpoints plus 2 GiB records reserve; only {available_bytes} available")
    config = json.loads((OLD / "a-lora.json").read_text())
    stage0.save(ROOT / "a-lora.json", config)
    paths = FILES + [str(ROOT / "a-lora.json"), str(ROOT / "pipeline-smoke/worker-result.json"),
                     "runs/api-harness-v1/train.jsonl"]
    for source in ("api", "generic"):
        paths.extend([str(OLD / "corpora" / f"full-{source}.jsonl"), str(OLD / "corpora" / f"full-{source}.jsonl.meta.json")])
    for family in ("cool", "heat"):
        paths.extend([str(OLD / family / n) for n in ("b_query.jsonl", "diagnostic.jsonl")])
        for source in ("api", "generic"):
            paths.append(str(OLD / family / "evaluation/main" / f"full-{source}-seed1101/B/step-0000/config.json"))
    ref = OLD / "heat/trajectories/full-api-seed1101/step-0000"
    paths.extend([str(ref / "composition.json"), str(ref / "B/adapter_config.json"), str(ref / "B/adapter_model.safetensors")])
    stage0.save(path, dict(protocol="variance-diagnostic-v1-stage1", frozen_unix=time.time(),
        cases=plan(), files={p: file_sha(p) for p in paths}, cap_gpu_hours=6.0, allocation_seconds=1200,
        minimum_allocation_seconds=600, gpus=[0, 1], teacher_calls=0, official_tests_used=False,
        storage_check=dict(available_bytes=available_bytes, required_bytes=required_bytes,
                           per_case_checkpoint_bytes=checkpoint_bytes, records_reserve_bytes=2 * 1024 ** 3),
        historical_hash_exceptions=historical_freeze["documented_hash_exceptions"]))


def load_freeze():
    value = json.loads((ROOT / "freeze.json").read_text())
    if value["cases"] != plan() or value["cap_gpu_hours"] != 6.0:
        raise ValueError("Frozen design changed")
    for name, expected in value["files"].items():
        if file_sha(name) != expected:
            raise ValueError("Stage 1 frozen file changed: " + name)
    stage0.check_historical_hashes(value["historical_hash_exceptions"])
    return value


def run():
    frozen = load_freeze()
    for gpu in frozen["gpus"]:
        if not stage0.port_is_free(gpu):
            raise ValueError("An existing model server occupies GPU " + str(gpu))
    ledger_path = ROOT / "execution.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
    else:
        smoke_seconds = json.loads((ROOT / "pipeline-smoke-timing.json").read_text())["seconds"]
        control_seconds = json.loads(Path("runs/variance-diagnostic-v1/seed-control-smoke-v1/result.json").read_text())["total_gpu_hours"] * 3600
        ledger = dict(status="new", gpu_seconds=smoke_seconds + control_seconds, preparatory_seconds=smoke_seconds + control_seconds,
                      cases={}, open={}, teacher_calls=0, official_tests_used=False)
    for case, attempt in list(ledger["open"].items()):
        try:
            os.kill(attempt["pid"], 0)
        except ProcessLookupError:
            ledger["gpu_seconds"] += attempt["allocation"] + 20
            ledger["cases"][case] = dict(status="unaccounted_stop_charged_conservatively", seconds=attempt["allocation"] + 20)
            del ledger["open"][case]
        else:
            raise ValueError("A recorded worker is still alive; no duplicate run")
    ledger["status"] = "running"
    ledger.pop("requested_stop", None)
    ledger["freeze_sha256"] = file_sha(ROOT / "freeze.json")
    stage0.save(ledger_path, ledger)
    lock, stop = threading.RLock(), threading.Event()

    def interrupt(signum, frame):
        stop.set()
        with lock:
            ledger["requested_stop"] = signum
            stage0.save(ledger_path, ledger)
            for attempt in ledger["open"].values():
                try:
                    os.killpg(attempt["pid"], signal.SIGTERM)
                except ProcessLookupError:
                    pass
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)

    def queue(gpu):
        for case in [c for c in frozen["cases"] if c["gpu"] == gpu]:
            if stop.is_set():
                return
            directory = ROOT / "cases" / case["name"]
            completed = directory / "worker-result.json"
            if completed.exists() and json.loads(completed.read_text()).get("complete"):
                continue
            with lock:
                allocation = allocation_seconds(frozen["cap_gpu_hours"] * 3600, ledger["gpu_seconds"],
                    sum(a["allocation"] for a in ledger["open"].values()),
                    frozen["allocation_seconds"], frozen["minimum_allocation_seconds"])
                if allocation == 0:
                    ledger["cap_prevented_start"] = True
                    stage0.save(ledger_path, ledger)
                    stop.set()
                    return
                directory.mkdir(parents=True, exist_ok=True)
                started = time.monotonic()
                log = (directory / f"worker-{time.time_ns()}.log").open("x")
                try:
                    process = subprocess.Popen([engine.MODEL_PY, "scripts/stage1_worker.py", "--case", case["name"],
                        "--gpu", str(gpu), "--timeout", str(allocation)], env=environment(gpu),
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    log.close()
                    stop.set()
                    raise
                ledger["open"][case["name"]] = dict(pid=process.pid, gpu=gpu, allocation=allocation, started_unix=time.time())
                stage0.save(ledger_path, ledger)
            outcome = "failed"
            try:
                code = process.wait(timeout=allocation + 2)
                if code == 0 and completed.exists() and json.loads(completed.read_text()).get("complete"):
                    outcome = "completed"
                else:
                    stop.set()
            except subprocess.TimeoutExpired:
                outcome = "timed_out"
                stop.set()
            except BaseException:
                stop.set()
                raise
            finally:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                log.close()
                seconds = time.monotonic() - started
                with lock:
                    ledger["gpu_seconds"] += seconds
                    ledger["cases"][case["name"]] = dict(status=outcome, seconds=seconds, gpu=gpu)
                    ledger["open"].pop(case["name"], None)
                    stage0.save(ledger_path, ledger)
                print(json.dumps(dict(case=case["name"], status=outcome, seconds=seconds)), flush=True)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(queue, gpu) for gpu in frozen["gpus"]]
            for future in futures:
                future.result()
    except BaseException:
        ledger["status"] = "failed"
        stage0.save(ledger_path, ledger)
        raise
    finished = sum((ROOT / "cases" / c["name"] / "worker-result.json").exists() for c in frozen["cases"])
    ledger["status"] = "interrupted" if ledger.get("requested_stop") else "evaluation_completed" if finished == 34 else "stopped_at_cap" if ledger.get("cap_prevented_start") else "failed"
    stage0.save(ledger_path, ledger)
    if finished != 34 or ledger.get("requested_stop"):
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "report.log").open("w") as handle:
        subprocess.run([engine.MODEL_PY, "scripts/stage1_variance_report.py"], env=environment(0), stdout=handle, stderr=subprocess.STDOUT, check=True)
    ledger["status"] = "completed"
    stage0.save(ledger_path, ledger)


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
        if (ROOT / "freeze.json").exists():
            raise ValueError("Smoke belongs before freeze")
        check_inputs()
        started = time.monotonic()
        timing_path = ROOT / "pipeline-smoke-timing.json"
        prior_seconds = json.loads(timing_path.read_text())["seconds"] if timing_path.exists() else 0
        try:
            with (ROOT / f"pipeline-smoke-{time.time_ns()}.log").open("x") as handle:
                subprocess.run([engine.MODEL_PY, "scripts/stage1_worker.py", "--smoke", "--case", "pipeline-smoke", "--gpu", "1", "--timeout", "600"],
                               env=environment(1), stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=625)
        finally:
            stage0.save(timing_path, dict(seconds=prior_seconds + time.monotonic() - started))
    elif args.freeze:
        freeze()
    else:
        run()


if __name__ == "__main__":
    main()
