#!/usr/bin/env python3
"""Stage 0 of the variance diagnostic: repeat post-A evaluations of stored checkpoints.

Specification: docs/VARIANCE_DIAGNOSTIC_STAGE0_SPEC.md. No training, no teacher calls, no
official test sets. Writes only under runs/variance-diagnostic-v1/stage0/.

Run from the repository root on one GPU, with no other experiment using that GPU:

    PYTHONPATH=src .venv-model/bin/python scripts/run_stage0_repeat_eval.py --gpu 0

Re-running resumes: completed evaluations are verified and skipped.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from pathlib import Path

import run_api_harness as engine
from learning_harness.adaptation import composition_revision
from learning_harness.analysis import summarize
from learning_harness.common import file_sha, read_jsonl, write_jsonl

SOURCE_ROOT = Path("runs/transfer-factorial-v1")
ROOT = Path("runs/variance-diagnostic-v1/stage0")
FAMILIES = ["cool", "heat"]
SOURCES = ["api", "generic"]
B_SEEDS = [1101, 1202, 1303]
REPLICATES = [1, 2, 3]
DETERMINISM_CELLS = [("heat", "api", 1101), ("heat", "generic", 1101)]
PERMUTATION_SEED = 20261111
SECONDS_RESERVED_PER_RUN = 300


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def verify_frozen() -> None:
    """The historical study's inputs and execution code must be byte-identical."""
    for name in ("inputs.sha256.json", "curricula.sha256.json", "execution-code.sha256.json"):
        path = SOURCE_ROOT / name
        if path.exists():
            for target, expected in json.loads(path.read_text()).items():
                if file_sha(target) != expected:
                    raise ValueError(f"Frozen input or code changed since the historical run: {target}")


def historical(family, source, seed) -> Path:
    return SOURCE_ROOT / family / "evaluation" / "main" / f"full-{source}-seed{seed}" / "B" / "step-0000"


def checkpoint(family, source, seed) -> Path:
    return SOURCE_ROOT / family / "trajectories" / f"full-{source}-seed{seed}" / "step-0000"


def permuted_manifest(family, replicate) -> Path:
    """Same task records as the historical query manifest, in a fixed shuffled order."""
    original = read_jsonl(SOURCE_ROOT / family / "b_query.jsonl")
    rows = list(original)
    random.Random(PERMUTATION_SEED * 100 + 10 * FAMILIES.index(family) + replicate).shuffle(rows)
    if rows == original:
        raise ValueError("Permutation left the task order unchanged")
    path = ROOT / "manifests" / f"{family}-rep{replicate}.jsonl"
    if path.exists():
        if read_jsonl(path) != rows:
            raise ValueError(f"Existing manifest differs from the specified permutation: {path}")
    else:
        write_jsonl(path, rows)
    return path


def port_is_free(gpu) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{19780 + gpu}/health", timeout=2):
            return False
    except OSError:
        return True


def evaluate(gpu, directory: Path, config: dict, family, source, seed) -> float:
    """Run one evaluation exactly as the historical pipeline did; return seconds spent."""
    point = checkpoint(family, source, seed)
    revision = composition_revision(point)  # verifies the A and B adapter hashes
    recorded = json.loads((historical(family, source, seed) / "run.json").read_text())["revision"]
    if revision != recorded:
        raise ValueError(f"Checkpoint revision differs from the historical evaluation: {point}")
    config_path = directory / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError(f"Existing configuration differs: {config_path}")
    else:
        save(config_path, config)
    summary = directory / "summary.json"
    if summary.exists() and json.loads(summary.read_text())["complete"]:
        if json.loads((directory / "run.json").read_text())["revision"] != revision:
            raise ValueError(f"Completed run has a different revision: {directory}")
        return 0.0
    started = time.monotonic()
    alias = json.loads((historical(family, source, seed) / "run.json").read_text())["model"]
    with engine.server(gpu, alias, revision, directory / "server.log", point) as env:
        engine.command([engine.CPU_PY, "-m", "learning_harness.batch_runner", "--config", str(config_path),
                        "--evaluation", "--out", str(directory)], env, directory / "rollout.log")
    result = summarize(directory)
    if not result["complete"] or result["metrics"]["none"]["error_count"]:
        raise RuntimeError(f"Evaluation incomplete or errored; not counted as task failure: {directory}")
    return time.monotonic() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--cap-gpu-hours", type=float, default=4.0,
                        help="Hard cap on cumulative evaluation time across invocations")
    args = parser.parse_args()

    verify_frozen()
    if not port_is_free(args.gpu):
        raise SystemExit("A model server is already listening for this GPU; do not run concurrently")
    ledger_path = ROOT / "execution.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(
        gpu_seconds=0.0, runs=[], status="running", teacher_calls=0, official_tests_used=False)
    if ledger.get("gpu") not in (None, args.gpu):
        raise SystemExit("Stage 0 uses one GPU throughout; a different GPU was used before")
    ledger.update(gpu=args.gpu, cap_gpu_hours=args.cap_gpu_hours, status="running")

    # Replicate-major order, so that a stop leaves whole replicates. Replicate 0 is the
    # descriptive determinism check: an unmodified rerun of two cells, run last.
    plan = [(replicate, family, source, seed) for replicate in REPLICATES
            for family in FAMILIES for seed in B_SEEDS for source in SOURCES]
    plan += [(0, family, source, seed) for family, source, seed in DETERMINISM_CELLS]

    for replicate, family, source, seed in plan:
        base = json.loads((historical(family, source, seed) / "config.json").read_text())
        if replicate == 0:
            directory = ROOT / "determinism" / family / f"full-{source}-seed{seed}"
            config = base
        else:
            directory = ROOT / f"rep{replicate}" / family / f"full-{source}-seed{seed}"
            config = dict(base, manifest=str(permuted_manifest(family, replicate)))
        if ledger["gpu_seconds"] + SECONDS_RESERVED_PER_RUN > args.cap_gpu_hours * 3600:
            ledger["status"] = "stopped_at_cap"
            save(ledger_path, ledger)
            print("GPU-time cap reached; stopping. Complete replicates remain usable.", flush=True)
            return
        try:
            seconds = evaluate(args.gpu, directory, config, family, source, seed)
        except Exception as error:
            ledger["status"] = "failed"
            ledger["error"] = dict(type=type(error).__name__, directory=str(directory))
            save(ledger_path, ledger)
            raise
        if seconds:
            ledger["gpu_seconds"] += seconds
            ledger["runs"].append(dict(directory=str(directory), seconds=seconds))
        save(ledger_path, ledger)
        print(f"done {directory} ({seconds:.0f}s, cumulative {ledger['gpu_seconds'] / 3600:.2f} GPU-h)", flush=True)

    ledger["status"] = "completed"
    save(ledger_path, ledger)


if __name__ == "__main__":
    main()
