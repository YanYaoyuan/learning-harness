#!/usr/bin/env python3
"""Candidate CPU audit and future frozen task/action-controlled ALFWorld worker."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import time

import run_api_harness as engine
import run_stage3_task_controls as parent
import stage1_worker as kernel
from run_stage1_crossed_seeds import environment
from learning_harness.action_control_audit import audit_recipe, digest
from learning_harness.adaptation import composition_revision
from learning_harness.bounded_queue import save
from learning_harness.common import file_sha, read_jsonl
from learning_harness.controlled_study import validate_plan, validate_reservation


def load_recipe(path):
    recipe = json.loads(path.read_text())
    source, seed = recipe["source"], recipe["sampling_seed"]
    if source not in ("api", "generic"):
        raise ValueError("Only existing frozen source pools supported")
    data = Path(f"runs/transfer-factorial-v1/corpora/full-{source}.jsonl")
    reference_path = parent.ROOT / "inputs/schedules" / f"random-{source}-s{seed}.json"
    scores_path = parent.ROOT / "base-nll" / f"{source}.json"
    if (
        Path(recipe["data"]) != data
        or recipe["data_sha256"] != file_sha(data)
        or Path(recipe["reference_recipe"]) != reference_path
        or recipe["reference_recipe_sha256"] != file_sha(reference_path)
    ):
        raise ValueError("Candidate changed frozen data or random-reference identity")
    expected_score = file_sha(scores_path) if recipe["policy"] == "min_nll" else None
    if recipe["score_sha256"] != expected_score:
        raise ValueError("NLL-selection score provenance differs")
    rows = read_jsonl(data)
    scored = json.loads(scores_path.read_text())
    if scored["data_sha256"] != file_sha(data) or len(scored["scores"]) != len(rows):
        raise ValueError("Score/corpus identity mismatch")
    for index, (row, score) in enumerate(zip(rows, scored["scores"])):
        if score["index"] != index or score["task_id"] != row["task_id"]:
            raise ValueError("NLL row alignment mismatch")
    reference = json.loads(reference_path.read_text())
    exposure = audit_recipe(
        recipe, rows, [s["mean_target_nll"] for s in scored["scores"]], reference
    )
    return recipe, exposure


def run_case(root, name, gpu, seconds=None):
    # Refuse execution before the current experiment and its local report finish.
    if (
        json.loads((parent.ROOT / "execution.json").read_text())["status"]
        != "completed"
        or not json.loads((parent.OUT / "summary.json").read_text())["complete"]
    ):
        raise ValueError("Stage 3 must finish before this follow-up can train")
    frozen = json.loads((root / "freeze.json").read_text())
    validate_plan(frozen)
    ledger_path = root / "execution.json"
    # Popen precedes the controller's atomic reservation write by a few ms.
    deadline = time.monotonic() + 2
    while True:
        ledger = json.loads(ledger_path.read_text())
        if name in ledger.get("open", {}) or time.monotonic() >= deadline:
            break
        time.sleep(0.02)
    validate_reservation(
        ledger, name, os.getpid(), gpu, seconds, file_sha(root / "freeze.json")
    )
    if (
        frozen["parent_stage3_freeze_sha256"] != file_sha(parent.ROOT / "freeze.json")
        or frozen["teacher_calls"] != 0
        or frozen["official_tests_used"]
    ):
        raise ValueError("Unapproved parent, teacher or evaluation scope")
    for path, expected in frozen["files"].items():
        if file_sha(path) != expected:
            raise ValueError("Follow-up frozen input changed: " + path)
    case = next(c for c in frozen["cases"] if c["name"] == name)
    required = {
        case["schedule"],
        str(root / "a-lora.json"),
        str(parent.ROOT / "freeze.json"),
        "scripts/action_control_worker.py",
        "src/learning_harness/action_control_audit.py",
        "src/learning_harness/action_matched_sampling.py",
        "src/learning_harness/controlled_study.py",
    }
    if not required.issubset(frozen["files"]):
        raise ValueError(
            "Worker, controls, recipe and optimizer config must all be frozen"
        )
    if case["gpu"] != gpu or os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise ValueError("Frozen physical GPU and worker visibility must agree")
    recipe_path = Path(case["schedule"])
    recipe, exposure = load_recipe(recipe_path)
    if (
        recipe.get("candidate_only", True)
        or recipe["source"] != case["source"]
        or recipe["policy"] != case["policy"]
        or recipe["sampling_seed"] != case["sampling_seed"]
    ):
        raise ValueError("Formalized recipe and frozen case must agree")
    config = json.loads((root / "a-lora.json").read_text())
    if config != json.loads((parent.ROOT / "a-lora.json").read_text()):
        raise ValueError(
            "Follow-up must preserve the parent optimizer/model configuration"
        )
    if config["max_steps"] * config["global_batch_size"] != recipe["presentations"]:
        raise ValueError("Optimizer budget differs from exact matched schedule")
    for key in ("NO_PROXY", "no_proxy"):
        os.environ[key] = ",".join(
            filter(None, [os.environ.get(key, ""), "127.0.0.1,localhost,::1"])
        )
    directory = root / "cases" / name
    adapter = directory / "A"
    control_path = adapter / "seed-controls.json"
    if not control_path.exists() or not json.loads(control_path.read_text()).get(
        "completed"
    ):
        if adapter.exists() and any(adapter.iterdir()):
            raise ValueError("Partial training retained; inspect before resuming")
        engine.command(
            [
                engine.MODEL_PY,
                "scripts/train_lora_fixed_schedule.py",
                "--schedule",
                str(recipe_path),
                "--data",
                recipe["data"],
                "--model",
                engine.MODEL,
                "--revision",
                engine.REV,
                "--config",
                str(root / "a-lora.json"),
                "--out",
                str(adapter),
                "--initialization-seed",
                str(case["initialization_seed"]),
                "--sampling-seed",
                str(case["sampling_seed"]),
            ],
            environment(gpu),
            directory / "training.log",
        )
    control = json.loads(control_path.read_text())
    metadata = json.loads((adapter / "experiment.json").read_text())
    if (
        not control["completed"]
        or control["selection_policy"] != case["policy"]
        or control["initialization_seed"] != case["initialization_seed"]
        or control["sampling_seed"] != case["sampling_seed"]
        or control["external_schedule_sha256"] != file_sha(recipe_path)
        or control["schedule_sha256"] != digest(recipe["indices"])
        or control["trained_adapter_sha256"]
        != file_sha(adapter / "adapter_model.safetensors")
        or metadata["training_data_sha256"] != recipe["data_sha256"]
        or metadata["config"] != config
        or metadata["processed_examples_rank0"] != recipe["presentations"]
        or metadata["used_examples"] != len(read_jsonl(recipe["data"]))
    ):
        raise ValueError("Actual training does not satisfy the frozen intervention")
    types = {
        row["task_id"]: row["task_type"]
        for row in read_jsonl("runs/api-harness-v1/train.jsonl")
    }
    rows = read_jsonl(recipe["data"])
    exposure.update(
        input_tokens=metadata["input_tokens_rank0"],
        target_tokens=metadata["target_tokens_rank0"],
        task_type_presentations=dict(
            Counter(types[rows[i]["task_id"]] for i in recipe["indices"])
        ),
    )
    point = directory / "checkpoint"
    kernel.make_composition(adapter, point)
    scores = {
        f: kernel.evaluate(point, directory / "evaluation" / f, f, case["source"], gpu)[
            "metrics"
        ]["none"]["success_rate"]
        for f in kernel.FAMILIES
    }
    nll = kernel.score_nll(point, directory / "diagnostic-nll.json")
    save(
        directory / "worker-result.json",
        dict(
            complete=True,
            smoke=False,
            case=case,
            success=scores,
            freeze_sha256=file_sha(root / "freeze.json"),
            mean_success=sum(scores.values()) / 2,
            mean_nll=sum(nll[f]["task_mean_nll"] for f in kernel.FAMILIES) / 2,
            control=control,
            exposure=exposure,
            composition_revision=composition_revision(point),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", type=Path, nargs="+")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    parent.load_freeze()
    if args.validate_only:
        values = []
        for path in args.validate_only:
            recipe, exposure = load_recipe(path)
            values.append(
                dict(
                    path=str(path),
                    candidate_only=recipe.get("candidate_only", True),
                    controls=exposure,
                )
            )
        print(
            json.dumps(
                dict(
                    cpu_only=True,
                    training_started=False,
                    validated=len(values),
                    results=values,
                )
            ),
            flush=True,
        )
        return
    if args.root is None or args.case is None or args.gpu is None:
        parser.error(
            "Provide --validate-only, or a formally frozen --root, --case and --gpu"
        )
    if args.timeout <= 0:
        parser.error("Positive worker timeout required")

    def stop(signum, frame):
        raise kernel.WorkerDeadline()

    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.setitimer(signal.ITIMER_REAL, args.timeout)
    run_case(args.root, args.case, args.gpu, args.timeout)
    signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
