#!/usr/bin/env python3
"""Four one-update training checks; inspect control hashes, never task performance."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path("runs/variance-diagnostic-v1/seed-control-smoke-v1")
CASES = (("generic", 101, 101), ("generic", 101, 202), ("generic", 202, 101), ("api", 101, 101))


def main():
    if ROOT.exists():
        raise ValueError("Smoke directory exists; retain it and inspect before another attempt")
    ROOT.mkdir(parents=True)
    config = json.loads(Path("runs/transfer-factorial-v1/a-lora.json").read_text())
    config["max_steps"] = 1
    config_path = ROOT / "one-update-config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    plan = dict(purpose="Verify seed controls, not learner performance", gpu=1, updates=1,
                cases=CASES, timeout_seconds_per_training=180, teacher_calls=0,
                wrapper_sha256=hashlib.sha256(Path("scripts/train_lora_split_seeds.py").read_bytes()).hexdigest())
    (ROOT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", PYTHONPATH=str(Path("src").absolute()))
    for key in list(env):
        if key.startswith("LH_TEACHER_"):
            env.pop(key)
    controls, timings = [], []
    started = time.monotonic()
    try:
        for source, initialization, sampling in CASES:
            name = f"{source}-init{initialization}-sample{sampling}"
            out = ROOT / name
            command = [str(Path(".venv-model/bin/python").absolute()), "scripts/train_lora_split_seeds.py",
                "--data", f"runs/transfer-factorial-v1/corpora/full-{source}.jsonl",
                "--model", "data/models/Qwen3-4B-Instruct-2507",
                "--revision", "cdbee75f17c01a7cc42f958dc650907174af0554",
                "--config", str(config_path), "--out", str(out),
                "--initialization-seed", str(initialization), "--sampling-seed", str(sampling)]
            begin = time.monotonic()
            with (ROOT / f"{name}.log").open("x") as handle:
                subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=180)
            timings.append(dict(case=name, seconds=time.monotonic() - begin))
            controls.append(json.loads((out / "seed-controls.json").read_text()))
        initial = [c["initial_parameter_sha256"] for c in controls]
        orders = [c["schedule_sha256"] for c in controls]
        checks = dict(same_initialization_across_sampling=initial[0] == initial[1],
                      same_initialization_across_sources=initial[0] == initial[3],
                      changed_initialization_changes_weights=initial[0] != initial[2],
                      same_sampling_across_initialization=orders[0] == orders[2],
                      changed_sampling_changes_order=orders[0] != orders[1],
                      all_training_completed=all(c["completed"] for c in controls))
        result = dict(complete=True, passed=all(checks.values()), checks=checks, timings=timings,
                      total_gpu_hours=(time.monotonic() - started) / 3600, task_evaluations=0, teacher_calls=0)
        (ROOT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        if not result["passed"]:
            raise ValueError("Split-seed control failed")
        print(json.dumps(result), flush=True)
    except BaseException as error:
        (ROOT / "failure.json").write_text(json.dumps(dict(error_type=type(error).__name__,
            seconds=time.monotonic() - started, completed_cases=len(controls)), indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
