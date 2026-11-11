#!/usr/bin/env python3
"""Stage 0 of the variance diagnostic: repeat post-A evaluations of stored checkpoints.

Specification: docs/VARIANCE_DIAGNOSTIC_STAGE0_SPEC.md. No training, no teacher calls, no
official test sets. Writes only under runs/variance-diagnostic-v1/stage0/.

Three modes, all run from the repository root with no other experiment on the chosen GPU:

    # 1. One evaluation to check the plumbing. Allowed before freezing; never in the estimator.
    PYTHONPATH=src .venv-model/bin/python scripts/run_stage0_repeat_eval.py --smoke --gpu 0

    # 2. Freeze the specification, code hashes and execution choices. Refuses to overwrite.
    PYTHONPATH=src .venv-model/bin/python scripts/run_stage0_repeat_eval.py --freeze --gpu 0 \
        --cap-gpu-hours 4 --hash-exception "scripts/transfer_factorial_report.py=<reason>"

    # 3. Run Stage 0. Takes every choice from freeze.json and validates it first.
    PYTHONPATH=src .venv-model/bin/python scripts/run_stage0_repeat_eval.py

Re-running mode 3 resumes: completed evaluations are verified and skipped. All wall time of
every attempt, including the smoke check and failed, interrupted or timed-out runs, is
charged against the cap.
"""
from __future__ import annotations

import argparse
import json
import random
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import run_api_harness as engine
from learning_harness.adaptation import composition_revision
from learning_harness.analysis import summarize
from learning_harness.common import file_sha, read_jsonl, write_jsonl

SOURCE_ROOT = Path("runs/transfer-factorial-v1")
ROOT = Path("runs/variance-diagnostic-v1/stage0")
FREEZE = ROOT / "freeze.json"
LEDGER = ROOT / "execution.json"
FROZEN_FILES = ["docs/VARIANCE_DIAGNOSTIC_STAGE0_SPEC.md", "scripts/run_stage0_repeat_eval.py",
                "scripts/stage0_variance_share.py"]
HISTORICAL_HASH_LISTS = ["inputs.sha256.json", "curricula.sha256.json", "execution-code.sha256.json"]
FAMILIES = ["cool", "heat"]
SOURCES = ["api", "generic"]
B_SEEDS = [1101, 1202, 1303]
REPLICATES = [1, 2, 3]
DETERMINISM_CELLS = [("heat", "api", 1101), ("heat", "generic", 1101)]
SMOKE_CELL = ("heat", "api", 1101)
PERMUTATION_SEED = 20261111
MINIMUM_SECONDS_TO_START = 300
DEFAULT_CAP_GPU_HOURS = 4.0


class Terminated(BaseException):
    """Raised from the SIGTERM handler so that accounting and server shutdown still run."""


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


# ---------------------------------------------------------------------------------------
# Frozen-input validation


def is_reporting_script(path: str) -> bool:
    """Only report, plot and tracking scripts may carry a documented hash exception.

    Everything else in the historical hash lists can affect what an evaluation computes.
    """
    name = Path(path).name
    return path.startswith("scripts/") and ("report" in name or name.startswith(("plot_", "track_")))


def historical_hash_mismatches() -> dict:
    """Return {path: (recorded, current)} for every historical frozen file that changed."""
    mismatches = {}
    for name in HISTORICAL_HASH_LISTS:
        path = SOURCE_ROOT / name
        if path.exists():
            for target, recorded in json.loads(path.read_text()).items():
                current = file_sha(target)
                if current != recorded:
                    mismatches[target] = (recorded, current)
    return mismatches


def check_historical_hashes(documented: list | None) -> list:
    """Raise unless every mismatch is a documented exception on a reporting script.

    documented=None is the pre-freeze smoke mode: mismatches on reporting scripts are
    tolerated and returned so they can be recorded; anything else still raises.
    """
    mismatches = historical_hash_mismatches()
    blocking = [path for path in mismatches if not is_reporting_script(path)]
    if blocking:
        raise ValueError(f"Frozen input or execution code changed since the historical run: {blocking}")
    if documented is None:
        return sorted(mismatches)
    known = {entry["path"]: entry for entry in documented}
    for path, (recorded, current) in mismatches.items():
        entry = known.get(path)
        if entry is None:
            raise ValueError(f"Undocumented hash mismatch: {path}")
        if entry["recorded_sha256"] != recorded or entry["current_sha256"] != current:
            raise ValueError(f"File changed again after its exception was documented: {path}")
    stale = sorted(set(known) - set(mismatches))
    if stale:
        raise ValueError(f"Documented exception no longer applies: {stale}")
    return sorted(mismatches)


def write_freeze(args) -> None:
    if FREEZE.exists():
        raise SystemExit(f"{FREEZE} already exists; a freeze is never overwritten")
    if args.gpu is None:
        raise SystemExit("--freeze needs --gpu")
    if args.include_historical and not args.attestation:
        raise SystemExit("--include-historical needs --attestation stating that the model environment, "
                         "GPU driver and GPU model are unchanged since the historical run")
    reasons = {}
    for item in args.hash_exception:
        path, separator, reason = item.partition("=")
        if not separator or not reason.strip():
            raise SystemExit("--hash-exception takes PATH=REASON")
        reasons[path] = reason.strip()
    mismatches = historical_hash_mismatches()
    exceptions = []
    for path, (recorded, current) in mismatches.items():
        if not is_reporting_script(path):
            raise SystemExit(f"Changed file can affect evaluation and cannot be excepted: {path}")
        if path not in reasons:
            raise SystemExit(f"Hash mismatch needs --hash-exception with its provenance: {path}")
        exceptions.append(dict(path=path, recorded_sha256=recorded, current_sha256=current,
                               reason=reasons.pop(path)))
    if reasons:
        raise SystemExit(f"Exception given for a file that matches its recorded hash: {sorted(reasons)}")
    save(FREEZE, dict(
        frozen_unix=time.time(),
        files={path: file_sha(path) for path in FROZEN_FILES},
        gpu=args.gpu,
        cap_gpu_hours=args.cap_gpu_hours,
        include_historical=bool(args.include_historical),
        environment_attestation=args.attestation,
        documented_hash_exceptions=exceptions,
        replicates=REPLICATES,
        permutation_seed=PERMUTATION_SEED,
        determinism_cells=DETERMINISM_CELLS,
    ))
    print(f"Wrote {FREEZE}")


def load_freeze(args) -> dict:
    if not FREEZE.exists():
        raise SystemExit(f"{FREEZE} is missing. Run --smoke, then --freeze, before Stage 0.")
    freeze = json.loads(FREEZE.read_text())
    for path, recorded in freeze["files"].items():
        if file_sha(path) != recorded:
            raise SystemExit(f"Changed after freezing: {path}")
    if sorted(freeze["files"]) != sorted(FROZEN_FILES):
        raise SystemExit("freeze.json does not cover the required files")
    if freeze["replicates"] != REPLICATES or freeze["permutation_seed"] != PERMUTATION_SEED:
        raise SystemExit("Replicate design differs from the frozen one")
    if freeze["include_historical"] and not freeze["environment_attestation"]:
        raise SystemExit("Frozen choice includes the historical replicate without an attestation")
    if args.gpu is not None and args.gpu != freeze["gpu"]:
        raise SystemExit(f"Frozen GPU is {freeze['gpu']}")
    if args.cap_gpu_hours is not None and args.cap_gpu_hours != freeze["cap_gpu_hours"]:
        raise SystemExit("The cap is frozen and is not changed at run time")
    check_historical_hashes(freeze["documented_hash_exceptions"])
    return freeze


# ---------------------------------------------------------------------------------------
# Time accounting


def load_ledger() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return dict(gpu_seconds=0.0, attempts=[], open=None, status="new", teacher_calls=0,
                official_tests_used=False)


def charge(ledger: dict, directory, seconds: float, outcome: str) -> None:
    ledger["gpu_seconds"] += max(0.0, seconds)
    ledger["attempts"].append(dict(directory=str(directory), seconds=max(0.0, seconds), outcome=outcome))
    ledger["open"] = None
    save(LEDGER, ledger)


def recover_open_attempt(ledger: dict) -> None:
    """Charge an attempt that ended without accounting (kill -9, power loss).

    Its duration is taken as start time to the newest file modification in its directory.
    """
    attempt = ledger.get("open")
    if not attempt:
        return
    directory = Path(attempt["directory"])
    newest = max((p.stat().st_mtime for p in directory.rglob("*") if p.is_file()), default=attempt["started_unix"])
    charge(ledger, directory, newest - attempt["started_unix"], "recovered_after_unaccounted_stop")


def remaining_seconds(ledger: dict, cap_gpu_hours: float) -> float:
    return cap_gpu_hours * 3600 - ledger["gpu_seconds"]


# ---------------------------------------------------------------------------------------
# Evaluation


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


def is_complete(directory: Path, revision: str) -> bool:
    summary = directory / "summary.json"
    if not (summary.exists() and json.loads(summary.read_text())["complete"]):
        return False
    if json.loads((directory / "run.json").read_text())["revision"] != revision:
        raise ValueError(f"Completed run has a different revision: {directory}")
    return True


def attempt(ledger: dict, cap_gpu_hours: float, gpu: int, directory: Path, config: dict, cell) -> str:
    """Run one evaluation as the historical pipeline did.

    Returns 'skipped', 'completed' or 'stopped_at_cap'. Any other ending raises after the
    elapsed time has been charged.
    """
    family, source, seed = cell
    point = checkpoint(family, source, seed)
    revision = composition_revision(point)  # verifies the A and B adapter hashes
    recorded = json.loads((historical(family, source, seed) / "run.json").read_text())
    if revision != recorded["revision"]:
        raise ValueError(f"Checkpoint revision differs from the historical evaluation: {point}")
    config_path = directory / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError(f"Existing configuration differs: {config_path}")
    else:
        save(config_path, config)
    if is_complete(directory, revision):
        return "skipped"
    if remaining_seconds(ledger, cap_gpu_hours) < MINIMUM_SECONDS_TO_START:
        return "stopped_at_cap"

    started = time.time()
    ledger["open"] = dict(directory=str(directory), started_unix=started)
    save(LEDGER, ledger)
    outcome = "failed"
    try:
        with engine.server(gpu, recorded["model"], revision, directory / "server.log", point) as env:
            # The rollout may use only what is left of the cap after server start-up.
            budget = remaining_seconds(ledger, cap_gpu_hours) - (time.time() - started)
            if budget <= 0:
                outcome = "stopped_at_cap"
                return outcome
            engine.command([engine.CPU_PY, "-m", "learning_harness.batch_runner", "--config", str(config_path),
                            "--evaluation", "--out", str(directory)], env, directory / "rollout.log",
                           timeout=budget)
        result = summarize(directory)
        if not result["complete"] or result["metrics"]["none"]["error_count"]:
            raise RuntimeError(f"Evaluation incomplete or errored; not counted as task failure: {directory}")
        outcome = "completed"
        return outcome
    except subprocess.TimeoutExpired:
        outcome = "stopped_at_cap"
        return outcome
    except (KeyboardInterrupt, Terminated):
        outcome = "interrupted"
        raise
    finally:
        charge(ledger, directory, time.time() - started, outcome)


def plan_for(mode: str) -> list:
    """(label, directory, replicate, cell) in execution order."""
    if mode == "smoke":
        family, source, seed = SMOKE_CELL
        return [("smoke", ROOT / "smoke" / family / f"full-{source}-seed{seed}", 0, SMOKE_CELL)]
    # Replicate-major order, so that a stop leaves whole replicates. Replicate 0 is the
    # descriptive determinism check: an unmodified rerun of two cells, run last.
    plan = [(f"rep{r}", ROOT / f"rep{r}" / f / f"full-{k}-seed{s}", r, (f, k, s))
            for r in REPLICATES for f in FAMILIES for s in B_SEEDS for k in SOURCES]
    plan += [("determinism", ROOT / "determinism" / f / f"full-{k}-seed{s}", 0, (f, k, s))
             for f, k, s in DETERMINISM_CELLS]
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="One unmodified evaluation of one cell")
    mode.add_argument("--freeze", action="store_true", help="Write freeze.json and exit")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--cap-gpu-hours", type=float)
    parser.add_argument("--include-historical", action="store_true",
                        help="With --freeze: use the historical evaluation as a fourth replicate")
    parser.add_argument("--attestation", help="With --freeze --include-historical: environment attestation")
    parser.add_argument("--hash-exception", action="append", default=[], metavar="PATH=REASON",
                        help="With --freeze: document a changed reporting script")
    args = parser.parse_args()

    if args.freeze:
        if args.cap_gpu_hours is None:
            args.cap_gpu_hours = DEFAULT_CAP_GPU_HOURS
        write_freeze(args)
        return

    ledger = load_ledger()
    recover_open_attempt(ledger)
    if args.smoke:
        if args.gpu is None:
            raise SystemExit("--smoke needs --gpu")
        if FREEZE.exists():
            raise SystemExit("Stage 0 is frozen; the smoke check belongs before the freeze")
        gpu = args.gpu
        cap = args.cap_gpu_hours if args.cap_gpu_hours is not None else DEFAULT_CAP_GPU_HOURS
        ledger["undocumented_reporting_script_mismatches"] = check_historical_hashes(None)
    else:
        freeze = load_freeze(args)
        gpu, cap = freeze["gpu"], freeze["cap_gpu_hours"]
        ledger["freeze_sha256"] = file_sha(FREEZE)
    if ledger.get("gpu") not in (None, gpu):
        raise SystemExit("Stage 0 uses one GPU throughout; a different GPU was used before")
    if not port_is_free(gpu):
        raise SystemExit("A model server is already listening for this GPU; do not run concurrently")
    ledger.update(gpu=gpu, cap_gpu_hours=cap, status="running")
    save(LEDGER, ledger)

    def on_sigterm(signum, frame):
        raise Terminated()
    signal.signal(signal.SIGTERM, on_sigterm)

    final_status = "smoke_completed" if args.smoke else "completed"
    try:
        for label, directory, replicate, cell in plan_for("smoke" if args.smoke else "full"):
            family, source, seed = cell
            base = json.loads((historical(family, source, seed) / "config.json").read_text())
            config = base if replicate == 0 else dict(base, manifest=str(permuted_manifest(family, replicate)))
            outcome = attempt(ledger, cap, gpu, directory, config, cell)
            print(f"{outcome:15s} {directory}  cumulative {ledger['gpu_seconds'] / 3600:.2f} of {cap:g} GPU-h",
                  flush=True)
            if outcome == "stopped_at_cap":
                final_status = "stopped_at_cap"
                print("GPU-time cap reached. Only complete replicates enter the estimator.", flush=True)
                break
    except (KeyboardInterrupt, Terminated):
        ledger["status"] = "interrupted"
        save(LEDGER, ledger)
        raise SystemExit("Interrupted; elapsed time was charged. Re-run to resume.")
    except Exception as error:
        ledger["status"] = "failed"
        ledger["error"] = dict(type=type(error).__name__, message=str(error)[:500])
        save(LEDGER, ledger)
        raise
    ledger["status"] = final_status
    save(LEDGER, ledger)


if __name__ == "__main__":
    main()
