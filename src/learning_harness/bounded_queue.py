"""Durable two-GPU job accounting with reserved budgets and process-group cleanup."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from learning_harness.common import file_sha


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def allocation(cap, spent, reserved, maximum, minimum):
    available = cap - spent - reserved
    return min(maximum, available) if available >= minimum else 0


def run(root, frozen, command, environment, preparatory_seconds=0):
    """Caller holds a controller flock; result files are written only by successful workers."""
    root = Path(root)
    path = root / "execution.json"
    ledger = json.loads(path.read_text()) if path.exists() else dict(
        gpu_seconds=preparatory_seconds, preparatory_seconds=preparatory_seconds,
        cases={}, open={}, teacher_calls=0, official_tests_used=False)
    for name, item in list(ledger["open"].items()):
        try:
            os.kill(item["pid"], 0)
        except ProcessLookupError:
            seconds = item["allocation"] + 20
            ledger["gpu_seconds"] += seconds
            ledger["cases"][name] = dict(status="unaccounted_stop_charged_conservatively", seconds=seconds)
            del ledger["open"][name]
        else:
            raise ValueError("Recorded worker still alive; no duplicate launch")
    ledger.update(status="running", freeze_sha256=file_sha(root / "freeze.json"))
    ledger.pop("requested_stop", None)
    lock, stop = threading.RLock(), threading.Event()
    save(path, ledger)

    def interrupt(signum, frame):
        stop.set()
        with lock:
            ledger["requested_stop"] = signum
            save(path, ledger)
            for item in ledger["open"].values():
                try:
                    os.killpg(item["pid"], signal.SIGTERM)
                except ProcessLookupError:
                    pass
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)

    def queue(gpu):
        for case in (c for c in frozen["cases"] if c["gpu"] == gpu):
            if stop.is_set():
                return
            directory = root / "cases" / case["name"]
            result = directory / "worker-result.json"
            if result.exists() and json.loads(result.read_text()).get("complete"):
                continue
            with lock:
                if stop.is_set():
                    return
                seconds = allocation(frozen["cap_gpu_hours"] * 3600, ledger["gpu_seconds"],
                    sum(v["allocation"] for v in ledger["open"].values()),
                    frozen["allocation_seconds"], frozen["minimum_allocation_seconds"])
                if not seconds:
                    ledger["cap_prevented_start"] = True
                    stop.set()
                    save(path, ledger)
                    return
                directory.mkdir(parents=True, exist_ok=True)
                log = (directory / f"worker-{time.time_ns()}.log").open("x")
                started = time.monotonic()
                try:
                    process = subprocess.Popen(command(case, gpu, seconds), env=environment(gpu),
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    log.close()
                    stop.set()
                    raise
                ledger["open"][case["name"]] = dict(pid=process.pid, gpu=gpu, allocation=seconds, started_unix=time.time())
                save(path, ledger)
            outcome = "failed"
            try:
                code = process.wait(timeout=seconds + 2)
                if code == 0 and result.exists() and json.loads(result.read_text()).get("complete"):
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
                elapsed = time.monotonic() - started
                with lock:
                    ledger["gpu_seconds"] += elapsed
                    ledger["cases"][case["name"]] = dict(status=outcome, seconds=elapsed, gpu=gpu)
                    ledger["open"].pop(case["name"], None)
                    save(path, ledger)
                print(json.dumps(dict(case=case["name"], status=outcome, seconds=elapsed)), flush=True)
    try:
        with ThreadPoolExecutor(max_workers=len(frozen["gpus"])) as pool:
            futures = [pool.submit(queue, gpu) for gpu in frozen["gpus"]]
            for future in as_completed(futures):
                future.result()
    except BaseException:
        ledger["status"] = "failed"
        save(path, ledger)
        raise
    complete = sum((root / "cases" / c["name"] / "worker-result.json").exists() and
                   json.loads((root / "cases" / c["name"] / "worker-result.json").read_text()).get("complete", False)
                   for c in frozen["cases"])
    ledger["completed_cases"] = complete
    ledger["status"] = ("interrupted" if ledger.get("requested_stop") else
                        "evaluation_completed" if complete == len(frozen["cases"]) else
                        "stopped_at_cap" if ledger.get("cap_prevented_start") else "failed")
    save(path, ledger)
    return ledger
