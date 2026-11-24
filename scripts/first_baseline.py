#!/usr/bin/env python3
"""Run the four-task B0 check using verified local model weights."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-download", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=3)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--out", default="runs/first-baseline")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    model_dir = Path("data/models/Qwen3-4B-Instruct-2507")
    receipt = model_dir / "download-manifest.json"
    deadline = time.monotonic() + args.timeout_hours * 3600
    if args.wait_for_download:
        print("Waiting for verified model download; no paid API calls.", flush=True)
        while not receipt.exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Model download did not complete within the stated timeout")
            time.sleep(5)
    else:
        subprocess.run(["/usr/bin/python3", "scripts/download_student.py"], check=True)
    manifest = json.loads(receipt.read_text())
    if manifest["revision"] != "cdbee75f17c01a7cc42f958dc650907174af0554":
        raise ValueError("Unexpected model revision")
    environment = dict(os.environ)
    environment.update(PYTHONPATH=str(root / "src"), ALFWORLD_DATA=str(root / "data/alfworld"),
                       LH_STUDENT_BASE_URL=f"http://127.0.0.1:{args.port}/v1",
                       LH_STUDENT_MODEL="Qwen/Qwen3-4B-Instruct-2507",
                       LH_STUDENT_REVISION=manifest["revision"])
    server_command = [str(root / ".venv-model/bin/python"), "scripts/hf_server.py",
                      "--model", str(model_dir), "--served-model-name", environment["LH_STUDENT_MODEL"],
                      "--port", str(args.port)]
    with (output / "server.log").open("w") as server_log:
        server = subprocess.Popen(server_command, env=environment, stdout=server_log, stderr=subprocess.STDOUT)
        try:
            ready_deadline = time.monotonic() + 300
            while True:
                if server.poll() is not None:
                    raise RuntimeError("Local model server exited; inspect server.log")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2) as response:
                        ready = json.load(response)
                    if ready.get("revision") == manifest["revision"]:
                        break
                except OSError:
                    pass
                if time.monotonic() > ready_deadline:
                    raise TimeoutError("Model server startup timeout")
                time.sleep(2)
            print("Model ready; running four development tasks under H0.", flush=True)
            base_command = [str(root / ".venv/bin/python"), "-m", "learning_harness"]
            with (output / "rollout.log").open("w") as log:
                subprocess.run(base_command + ["rollout", "--config", "configs/smoke_model.json", "--out", str(output)],
                               env=environment, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
            subprocess.run(base_command + ["summarize", str(output)], env=environment, check=True)
            print("Four-task B0 check finished. This is not the final benchmark.", flush=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
