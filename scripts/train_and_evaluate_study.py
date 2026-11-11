#!/usr/bin/env python3
"""Run six bounded local training/evaluation jobs on two explicitly assigned GPUs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import queue
import random
import subprocess
import threading
import time
import urllib.request

ROOT = Path("runs/dual5090-v1")
BASE = "data/models/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[101])
    args = parser.parse_args()
    selected = json.loads((ROOT / "selection.json").read_text())["arm"]
    jobs = [(arm, method, seed) for seed in args.seeds for arm in ["none", selected]
            for method in ["target_sft", "mixed_sft", "context_distill"]]
    random.Random(20261109).shuffle(jobs)
    gpu_pool = queue.Queue()
    for gpu in [0, 1]:
        gpu_pool.put(gpu)
    progress, lock = {}, threading.Lock()

    def save(name, status, **details):
        with lock:
            progress[name] = dict(status=status, **details)
            temporary = ROOT / "job-progress.json.part"
            temporary.write_text(json.dumps(progress, indent=2) + "\n")
            temporary.replace(ROOT / "job-progress.json")
            print(name, status, flush=True)

    def run(job):
        arm, method, seed = job
        name = f"{arm}-{method}-seed{seed}"
        gpu = gpu_pool.get()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(Path("src").resolve()),
                   ALFWORLD_DATA=str(Path("data/alfworld").resolve()))
        checkpoint = ROOT / "checkpoints" / name
        logs = ROOT / "training-logs"
        logs.mkdir(parents=True, exist_ok=True)
        server = None
        started = time.monotonic()
        try:
            if not (checkpoint / "experiment.json").exists():
                save(name, "training", gpu=gpu)
                with (logs / f"{name}.log").open("w") as log:
                    subprocess.run([str(Path(".venv-model/bin/python").absolute()), "scripts/train_lora.py",
                        "--model", BASE, "--revision", REVISION, "--data", str(ROOT / "corpora" / f"{arm}.jsonl"),
                        "--method", method, "--seed", str(seed), "--config", str(ROOT / "lora.json"),
                        "--out", str(checkpoint)], env=env, stdout=log, stderr=subprocess.STDOUT,
                        check=True, timeout=2700)
            evaluation = ROOT / "trained-eval" / name
            evaluation.mkdir(parents=True, exist_ok=True)
            if (evaluation / "summary.json").exists():
                save(name, "completed", gpu=gpu, elapsed_seconds=time.monotonic() - started)
                return
            save(name, "evaluating", gpu=gpu)
            adapter_hash = hashlib.sha256((checkpoint / "adapter_model.safetensors").read_bytes()).hexdigest()
            served_revision = REVISION + "/adapter-sha256:" + adapter_hash
            port = 19280 + gpu
            config = json.loads(Path("configs/probe.json").read_text())
            config.update(protocol_id="dual5090-discovery-v1-probe", manifest=str(ROOT / "probe.jsonl"),
                          max_steps=40, max_student_output_tokens=5120)
            config_path = evaluation / "config.json"
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            env.update(LH_STUDENT_BASE_URL=f"http://127.0.0.1:{port}/v1", LH_STUDENT_MODEL=name,
                       LH_STUDENT_REVISION=served_revision)
            with (evaluation / "server.log").open("w") as server_log:
                server = subprocess.Popen([str(Path(".venv-model/bin/python").absolute()), "scripts/hf_server.py",
                    "--model", BASE, "--adapter", str(checkpoint), "--served-model-name", name,
                    "--port", str(port)], env=env, stdout=server_log, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + 240
                while True:
                    if server.poll() is not None:
                        raise RuntimeError("Evaluation server exited")
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                            health = json.load(response)
                        if health.get("revision") == served_revision:
                            break
                    except OSError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("Evaluation server startup")
                    time.sleep(1)
                with (evaluation / "rollout.log").open("w") as log:
                    subprocess.run([str(Path(".venv/bin/python").absolute()), "-m", "learning_harness", "rollout",
                        "--config", str(config_path), "--evaluation", "--out", str(evaluation)],
                        env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=2700)
                subprocess.run([str(Path(".venv/bin/python").absolute()), "-m", "learning_harness", "summarize",
                    str(evaluation)], env=env, stdout=subprocess.DEVNULL, check=True)
            save(name, "completed", gpu=gpu, elapsed_seconds=time.monotonic() - started)
        except Exception as exc:
            save(name, "failed", gpu=gpu, error_type=type(exc).__name__, elapsed_seconds=time.monotonic() - started)
            raise
        finally:
            if server is not None:
                server.terminate()
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
            gpu_pool.put(gpu)

    failed = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failed.append(dict(job=futures[future], error_type=type(exc).__name__))
    if failed:
        raise RuntimeError(f"{len(failed)} jobs failed; inspect job-progress.json")
    print("All local discovery jobs completed.", flush=True)


if __name__ == "__main__":
    main()
