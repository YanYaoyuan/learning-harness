#!/usr/bin/env python3
"""Two independent GPUs: frozen job list, local LoRA and batched H0 evaluation."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import urllib.request

BASE = "data/models/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/dual5090-v2")
    args = parser.parse_args()
    root = Path(args.root)
    for path, expected in json.loads((root / "inputs.sha256.json").read_text()).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen input changed: {path}")
    jobs = json.loads((root / "jobs.json").read_text())
    pool_gpus = queue.Queue()
    for gpu in (0, 1):
        pool_gpus.put(gpu)
    progress, lock = {}, threading.Lock()

    def save(name, status, **fields):
        with lock:
            progress[name] = dict(status=status, **fields)
            tmp = root / "job-progress.json.part"
            tmp.write_text(json.dumps(progress, indent=2) + "\n")
            tmp.replace(root / "job-progress.json")
            print(name, status, flush=True)

    def run(job):
        name = job["name"]
        gpu = pool_gpus.get()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                   PYTHONPATH=str(Path("src").absolute()), ALFWORLD_DATA=str(Path("data/alfworld").absolute()))
        checkpoint = root / "checkpoints" / name
        evaluation = root / "evaluation" / name
        evaluation.mkdir(parents=True, exist_ok=True)
        server = None
        started = time.monotonic()
        try:
            revision = REVISION
            if job.get("data"):
                if not (checkpoint / "experiment.json").exists():
                    save(name, "training", gpu=gpu)
                    with (evaluation / "training.log").open("w") as log:
                        subprocess.run([str(Path(".venv-model/bin/python").absolute()), "scripts/train_lora.py",
                            "--data", job["data"], "--model", BASE, "--revision", REVISION,
                            "--method", job.get("method", "target_sft"), "--seed", str(job["seed"]),
                            "--config", str(root / "lora.json"), "--out", str(checkpoint)], env=env,
                            stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
                training = json.loads((checkpoint / "experiment.json").read_text())
                if training["training_data_sha256"] != hashlib.sha256(Path(job["data"]).read_bytes()).hexdigest():
                    raise ValueError("Completed checkpoint uses a different corpus")
                if training["training_seed"] != job["seed"]:
                    raise ValueError("Training seed changed")
                if training["config"] != json.loads((root / "lora.json").read_text()):
                    raise ValueError("Completed checkpoint uses a different training config")
                revision += "/adapter-sha256:" + hashlib.sha256((checkpoint / "adapter_model.safetensors").read_bytes()).hexdigest()
            summary = evaluation / "summary.json"
            if summary.exists() and json.loads(summary.read_text())["complete"]:
                metadata = json.loads((evaluation / "run.json").read_text())
                if metadata["revision"] != revision:
                    raise ValueError("Evaluation checkpoint changed")
                save(name, "completed", gpu=gpu, elapsed_seconds=time.monotonic() - started)
                return
            save(name, "evaluating", gpu=gpu)
            config = json.loads(Path("configs/probe.json").read_text())
            config.update(protocol_id="dual5090-v2-probe", manifest=str(root / "probe.jsonl"),
                          batch_size=8, max_steps=40, max_student_output_tokens=5120)
            cfg = evaluation / "config.json"
            cfg.write_text(json.dumps(config, indent=2) + "\n")
            port = 19480 + gpu
            env.update(LH_STUDENT_BASE_URL=f"http://127.0.0.1:{port}/v1", LH_STUDENT_MODEL=name,
                       LH_STUDENT_REVISION=revision)
            command = [str(Path(".venv-model/bin/python").absolute()), "scripts/hf_batch_server.py",
                       "--model", BASE, "--served-model-name", name, "--port", str(port)]
            if job.get("data"):
                command += ["--adapter", str(checkpoint)]
            with (evaluation / "server.log").open("w") as server_log:
                server = subprocess.Popen(command, env=env, stdout=server_log, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + 180
                while True:
                    if server.poll() is not None:
                        raise RuntimeError("Local inference server exited")
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                            health = json.load(response)
                        if health["revision"] == revision:
                            break
                    except OSError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("Local inference server startup")
                    time.sleep(1)
                with (evaluation / "rollout.log").open("w") as log:
                    subprocess.run([str(Path(".venv/bin/python").absolute()), "-m", "learning_harness.batch_runner",
                        "--config", str(cfg), "--evaluation", "--out", str(evaluation)], env=env,
                        stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
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
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
            pool_gpus.put(gpu)

    failures = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(run, job): job["name"] for job in jobs}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failures.append((futures[future], type(exc).__name__))
    if failures:
        raise RuntimeError(f"Jobs failed: {failures}")
    print("All frozen jobs completed", flush=True)


if __name__ == "__main__":
    main()
