#!/usr/bin/env python3
"""Prepare/freeze/run a bounded train-only NLL measurement diagnostic."""

import argparse
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from learning_harness.bounded_queue import run as bounded_run, save
from learning_harness.common import file_sha
from learning_harness.nll_scoring_probe import (
    make_batch,
    masked_nll,
    prepare_inputs,
    selection_changes,
    summarize,
    trials_for,
    validate_encoded,
)
from learning_harness.webshop_pilot import (
    PARENT,
    PARENT_SHA,
    REVISION,
    require_parent_completed,
    subprocess_environment,
)
from learning_harness.webshop_requests import RequestJournal, summarize_requests
from learning_harness.webshop_serving import model_runtime, verify_base_model

MODEL = Path("data/models/Qwen3-4B-Instruct-2507")
PROTOCOL = Path("docs/NLL_SCORING_DIAGNOSTIC_SPEC_2026-11-13.zh-CN.md")
CODE = [
    "scripts/run_nll_scoring_probe.py",
    "src/learning_harness/nll_scoring_probe.py",
    "scripts/train_lora.py",
    "scripts/score_training_nll.py",
    "src/learning_harness/common.py",
    "src/learning_harness/bounded_queue.py",
    "src/learning_harness/action_matched_sampling.py",
    "src/learning_harness/task_matched_sampling.py",
    "src/learning_harness/webshop_requests.py",
    "src/learning_harness/webshop_pilot.py",
    "src/learning_harness/webshop_serving.py",
    "src/learning_harness/webshop_factory.py",
    "src/learning_harness/webshop_public.py",
    "src/learning_harness/idle_gpu_contexts.py",
    "scripts/run_action_controls.py",
    "scripts/run_stage0_repeat_eval.py",
    "scripts/action_control_report.py",
    "tests/test_nll_scoring_probe.py",
    "tests/test_nll_scoring_probe_math.py",
]
CASE = "base-model-readonly-scoring"


def prepare(root):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Token preparation must explicitly hide CUDA")
    if (root / "inputs.private").exists() or (root / "freeze.json").exists():
        raise ValueError("Preserve earlier private inputs/freeze")
    from transformers import AutoTokenizer
    from train_lora import encode_example

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    payload = prepare_inputs(tokenizer, encode_example)
    with (root / "inputs.private").open("x") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    os.chmod(root / "inputs.private", 0o600)
    save(
        root / "preparation.json",
        dict(
            status="cpu_tokens_prepared",
            prepared_unix=time.time(),
            gpu_used=False,
            unique_inputs=len(payload["entries"]),
            sentinel_inputs=len(payload["stress_keys"]),
            replay_batches=len(payload["replay_batches"]),
            inputs_sha256=file_sha(root / "inputs.private"),
            model_runtime=model_runtime(),
            code_sha256={p: file_sha(p) for p in CODE},
        ),
    )


def freeze(root, gpu):
    if (root / "freeze.json").exists() or (root / "execution.json").exists():
        raise ValueError("Never overwrite a freeze or previous attempt")
    python = str(Path(".venv-model/bin/python").absolute())
    if not (root / "preparation.json").exists():
        with (root / "cpu-preparation.log").open("x") as log:
            subprocess.run(
                [python, __file__, "prepare", "--root", str(root)],
                env=subprocess_environment(),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )
    prepared = json.loads((root / "preparation.json").read_text())
    if (
        prepared["status"] != "cpu_tokens_prepared"
        or prepared["inputs_sha256"] != file_sha(root / "inputs.private")
        or prepared["code_sha256"] != {p: file_sha(p) for p in CODE}
    ):
        raise ValueError("CPU preparation/code identity changed")
    payload = json.loads((root / "inputs.private").read_text())
    parent = json.loads((PARENT / "freeze.json").read_text())
    if file_sha(PARENT / "freeze.json") != PARENT_SHA:
        raise ValueError("Parent identity changed")
    files = dict(payload["input_files"])
    paths = [
        *CODE,
        str(PROTOCOL),
        str(root / "inputs.private"),
        str(root / "preparation.json"),
        str(PARENT / "freeze.json"),
        str(MODEL / "download-manifest.json"),
    ]
    paths += [
        str(p)
        for p in MODEL.iterdir()
        if p.suffix == ".json" and "safetensors" not in p.name
    ]
    for stage in ("stage3", "stage4"):
        paths += [
            str(p)
            for p in Path(f"runs/variance-diagnostic-v1/{stage}/inputs/schedules").glob(
                "*.json"
            )
        ]
    for path in paths:
        files[path] = file_sha(path)
    frozen = dict(
        name="Read-only NLL scoring geometry/precision diagnostic",
        parent_freeze_sha256=PARENT_SHA,
        model_revision=REVISION,
        model_directory=str(MODEL),
        model_python=python,
        model_runtime=prepared["model_runtime"],
        gpu=gpu,
        gpus=[gpu],
        cases=[dict(name=CASE, gpu=gpu)],
        cap_gpu_hours=600 / 3600,
        allocation_seconds=576,
        minimum_allocation_seconds=576,
        worker_soft_stop_seconds=560,
        sentinel_inputs=32,
        fp32_direct_checks=4,
        repeat_tolerance=1e-6,
        direct_loss_tolerance=1e-4,
        allowed_background_contexts=parent["allowed_background_contexts"],
        automatic_retry=False,
        teacher_calls=0,
        environment_episodes=0,
        official_test_evaluations=0,
        optimizer_updates=0,
        adapters=None,
        protocol_text=PROTOCOL.read_text(),
        protocol_sha256=file_sha(PROTOCOL),
        files=files,
        frozen_unix=time.time(),
    )
    with (root / "freeze.json").open("x") as handle:
        handle.write(json.dumps(frozen, indent=2) + "\n")
    validate(root)
    print(
        json.dumps(
            dict(
                status="frozen_not_executed",
                freeze_sha256=file_sha(root / "freeze.json"),
                unique_inputs=prepared["unique_inputs"],
                sentinel_inputs=32,
            )
        ),
        flush=True,
    )


def validate(root):
    frozen = json.loads((root / "freeze.json").read_text())
    if (
        frozen["parent_freeze_sha256"] != PARENT_SHA
        or frozen["model_revision"] != REVISION
        or frozen["model_directory"] != str(MODEL)
        or frozen["gpus"] != [frozen["gpu"]]
        or type(frozen["gpu"]) is not int
        or frozen["gpu"] not in (0, 1)
        or frozen["cases"] != [dict(name=CASE, gpu=frozen["gpu"])]
        or frozen["cap_gpu_hours"] != 600 / 3600
        or frozen["allocation_seconds"] != 576
        or frozen["minimum_allocation_seconds"] != 576
        or frozen["worker_soft_stop_seconds"] != 560
        or frozen["sentinel_inputs"] != 32
        or frozen["fp32_direct_checks"] != 4
        or frozen["repeat_tolerance"] != 1e-6
        or frozen["direct_loss_tolerance"] != 1e-4
        or frozen["teacher_calls"] != 0
        or frozen["environment_episodes"] != 0
        or frozen["official_test_evaluations"] != 0
        or frozen["optimizer_updates"] != 0
        or frozen["adapters"] is not None
        or frozen["automatic_retry"] is not False
        or frozen["protocol_text"] != PROTOCOL.read_text()
        or frozen["protocol_sha256"] != file_sha(PROTOCOL)
        or not set(CODE + [str(PROTOCOL), str(root / "inputs.private")]).issubset(
            frozen["files"]
        )
    ):
        raise ValueError("Frozen scoring scope changed")
    for path, expected in frozen["files"].items():
        if file_sha(path) != expected:
            raise ValueError("Frozen scoring input changed: " + path)
    return frozen


def recover(journal_path):
    state = dict(
        stress=[],
        fp32=[],
        direct_checks=[],
        uniform={},
        legacy=[],
        padded_input_positions=0,
    )
    if not journal_path.exists():
        return state
    for raw in journal_path.read_bytes().splitlines(keepends=True):
        if not raw.endswith(b"\n"):
            break
        event = json.loads(raw)
        if event["kind"] != "response":
            continue
        response = event["response"]
        positions = response["padded_input_positions"]
        if type(positions) is not int or positions < response["input_tokens"]:
            raise ValueError("Missing/invalid materialized input-position count")
        state["padded_input_positions"] += positions
        phase = response["phase"]
        if phase == "uniform":
            for row in response["scores"]:
                key = row["content_sha256"]
                if key in state["uniform"]:
                    raise ValueError("Uniform scoring retried a content identity")
                state["uniform"][key] = row["nll"]
        else:
            state[phase].extend(response["scores"])
    return state


def build_report(root, inputs):
    directory = root / "cases" / CASE
    journal = directory / "forwards.private"
    state = recover(journal)
    metrics = summarize(state["stress"], state["fp32"], state["direct_checks"])
    expected_trials = sum(
        t["feasible"]
        for e in inputs["entries"]
        if e["content_sha256"] in inputs["stress_keys"]
        for t in trials_for(e)
    )
    expected_replay = sum(
        len(inputs["source_row_keys"][r["source"]][r["start"] : r["start"] + 8])
        for r in inputs["replay_batches"]
    )
    expected_pairs = {
        (e["content_sha256"], t["name"])
        for e in inputs["entries"]
        if e["content_sha256"] in inputs["stress_keys"]
        for t in trials_for(e)
        if t["feasible"]
    }
    actual_pairs = {(r["content_sha256"], r["condition"]) for r in state["stress"]}
    expected_legacy = {
        (f"original_batch_{r['source']}_{r['start']}", slot, key)
        for r in inputs["replay_batches"]
        for slot, key in enumerate(
            inputs["source_row_keys"][r["source"]][r["start"] : r["start"] + 8]
        )
    }
    actual_legacy = {
        (r["condition"], r["row_slot"], r["content_sha256"]) for r in state["legacy"]
    }
    complete = (
        actual_pairs == expected_pairs
        and len(state["stress"]) == expected_trials
        and {r["content_sha256"] for r in state["fp32"]} == set(inputs["stress_keys"])
        and len(state["direct_checks"]) == 4
        and {r["content_sha256"] for r in state["direct_checks"]}
        == set(inputs["stress_keys"][:4])
        and len(state["legacy"]) == expected_replay
        and actual_legacy == expected_legacy
    )
    full = len(state["uniform"]) == len(inputs["entries"])
    usage = summarize_requests(journal)
    result = dict(
        complete=complete,
        freeze_sha256=file_sha(root / "freeze.json"),
        primary_observations_complete=complete,
        metrics=metrics,
        stress_scores=len(state["stress"]),
        expected_stress_scores=expected_trials,
        predeclared_infeasible_padding_trials=32 * 8 - expected_trials,
        legacy_replay_observations=len(state["legacy"]),
        expected_legacy_replay_observations=expected_replay,
        legacy_replay_mean_absolute_difference=(
            sum(r["cached_absolute_difference"] for r in state["legacy"])
            / len(state["legacy"])
        )
        if state["legacy"]
        else None,
        legacy_replay_max_absolute_difference=max(
            (r["cached_absolute_difference"] for r in state["legacy"]), default=None
        ),
        uniform_unique_scored=len(state["uniform"]),
        uniform_unique_planned=len(inputs["entries"]),
        uniform_complete=full,
        selection_changes=[],
        forward_attempts=usage["attempted_requests"],
        completed_forwards=usage["response_records"],
        unknown_usage_forwards=usage["unknown_usage_requests"],
        known_input_tokens=usage["known_input_tokens"],
        known_padded_input_positions=state["padded_input_positions"],
        forward_journal_sha256=file_sha(journal) if journal.exists() else None,
        generated_tokens=0,
        teacher_calls=0,
        environment_episodes=0,
        official_test_evaluations=0,
        optimizer_updates=0,
        raw_records_uploaded=False,
        scope="Train-only instrument diagnostic; sentinel stress sample is not representative. No new SFT or policy-performance estimate.",
    )
    if complete and full and metrics["optional_rescoring_gate_passed"]:
        from train_lora import fixed_schedule

        result["selection_changes"] = selection_changes(
            inputs, state["uniform"], fixed_schedule
        )
    return result


def worker(root, gpu, timeout):
    started = time.monotonic()
    frozen = validate(root)
    require_parent_completed()
    if gpu != frozen["gpu"] or os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise ValueError("Frozen GPU and visibility disagree")
    ledger = json.loads((root / "execution.json").read_text())
    reservation = ledger.get("open", {}).get(CASE)
    # The queue starts the child immediately before recording its reservation.
    until = time.monotonic() + 2
    while not reservation and time.monotonic() < until:
        time.sleep(0.02)
        ledger = json.loads((root / "execution.json").read_text())
        reservation = ledger.get("open", {}).get(CASE)
    if (
        not reservation
        or reservation["pid"] != os.getpid()
        or reservation["gpu"] != gpu
        or reservation["allocation"] != timeout
        or timeout != 576
        or ledger["freeze_sha256"] != file_sha(root / "freeze.json")
    ):
        raise ValueError("Missing or changed bounded queue reservation")
    directory = root / "cases" / CASE
    inputs = json.loads((root / "inputs.private").read_text())
    entries = {r["content_sha256"]: r for r in inputs["entries"]}
    sentinels = [entries[k] for k in inputs["stress_keys"]]
    deadline = started + frozen["worker_soft_stop_seconds"]

    def expired(*_):
        raise TimeoutError("Scoring diagnostic soft deadline reached")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(max(1, int(deadline - time.monotonic())))
    base_identity = verify_base_model(MODEL, REVISION)
    identity = dict(base_identity)
    if model_runtime() != frozen["model_runtime"]:
        raise ValueError("Model runtime differs from the frozen preparation")
    import torch
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Exactly one dedicated visible CUDA device required")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    identity.update(
        runtime=model_runtime(),
        physical_gpu=gpu,
        pid=os.getpid(),
        fp32_matmul_precision="highest",
        tf32=False,
        device=torch.cuda.get_device_name(0),
    )
    save(directory / "model-identity.json", identity)
    model = None
    with RequestJournal(directory / "forwards.private") as journal:
        turn = 0

        def load(dtype):
            loaded, info = AutoModelForCausalLM.from_pretrained(
                MODEL,
                revision=REVISION,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=dtype,
                attn_implementation="sdpa",
                output_loading_info=True,
            )
            if any(
                info.get(k)
                for k in (
                    "missing_keys",
                    "unexpected_keys",
                    "mismatched_keys",
                    "error_msgs",
                )
            ):
                raise ValueError("Base weights did not load completely")
            if loaded.dtype != dtype:
                raise ValueError("Requested model precision was not loaded")
            return loaded.to("cuda:0").eval().requires_grad_(False)

        def forward(
            phase,
            rows,
            condition,
            *,
            padding=0,
            explicit=True,
            window=None,
            legacy=False,
            hf=False,
            cached_values=None,
        ):
            nonlocal turn
            if time.monotonic() >= deadline:
                expired()
            batch = make_batch(
                [r["encoded"] for r in rows], inputs["pad_token_id"], padding, explicit
            )
            journal(
                dict(
                    kind="attempt",
                    turn=turn,
                    phase=phase,
                    condition=condition,
                    dtype=str(model.dtype),
                    content_sha256s=[r["content_sha256"] for r in rows],
                )
            )
            try:
                tensors = {
                    k: torch.tensor(v, device="cuda:0", dtype=torch.long)
                    for k, v in batch.items()
                    if k != "labels"
                }
                kwargs = {} if legacy else dict(use_cache=False)
                if hf:
                    tensors["labels"] = torch.tensor(
                        batch["labels"], device="cuda:0", dtype=torch.long
                    )
                else:
                    kwargs["logits_to_keep"] = window
                with torch.inference_mode():
                    output = model(**tensors, **kwargs)
                    values = (
                        [float(output.loss)]
                        if hf
                        else masked_nll(torch, output.logits, batch["labels"])
                    )
                records = [
                    dict(
                        content_sha256=row["content_sha256"],
                        condition=condition,
                        row_slot=slot,
                        nll=value,
                    )
                    for slot, (row, value) in enumerate(zip(rows, values))
                ]
                if any(not math.isfinite(value) or value < 0 for value in values):
                    raise ValueError("Model returned a nonfinite/negative loss")
                if hf:
                    reference = recover(directory / "forwards.private")["fp32"]
                    known = next(
                        r["nll"]
                        for r in reference
                        if r["content_sha256"] == rows[0]["content_sha256"]
                    )
                    records[0]["absolute_difference"] = abs(values[0] - known)
                if cached_values is not None:
                    for record, cached in zip(records, cached_values):
                        record["cached_absolute_difference"] = abs(
                            record["nll"] - cached
                        )
                response = dict(
                    phase=phase,
                    scores=records,
                    input_tokens=sum(len(r["encoded"]["input_ids"]) for r in rows),
                    padded_input_positions=sum(len(r) for r in batch["input_ids"]),
                    output_tokens=0,
                )
                journal(dict(kind="response", turn=turn, response=response))
                turn += 1
                return values
            except BaseException as error:
                journal(dict(kind="error", turn=turn, error_type=type(error).__name__))
                raise

        try:
            # Independent loads preserve original FP32 buffers; no dtype round trip.
            model = load(torch.float32)
            for row in sentinels:
                forward(
                    "fp32",
                    [row],
                    "fp32_reference",
                    window=inputs["fixed_logits_window"],
                )
            for row in sentinels[:4]:
                forward("direct_checks", [row], "hf_full_labels_loss", hf=True)
            model = None
            gc.collect()
            torch.cuda.empty_cache()
            model = load(torch.bfloat16)
            for row in sentinels:
                for trial in trials_for(row):
                    if trial["feasible"]:
                        window = (
                            inputs["fixed_logits_window"]
                            if trial["window"] == "fixed"
                            else 1 + validate_encoded(row["encoded"])
                        )
                        forward(
                            "stress",
                            [row],
                            trial["name"],
                            padding=trial["padding"],
                            explicit=trial["explicit_positions"],
                            window=window,
                        )
            for replay in inputs["replay_batches"]:
                keys = inputs["source_row_keys"][replay["source"]][
                    replay["start"] : replay["start"] + 8
                ]
                rows = [entries[k] for k in keys]
                cached = [
                    next(
                        o["cached_nll"]
                        for o in row["observations"]
                        if o["source"] == replay["source"]
                        and o["index"] == replay["start"] + slot
                    )
                    for slot, row in enumerate(rows)
                ]
                forward(
                    "legacy",
                    rows,
                    f"original_batch_{replay['source']}_{replay['start']}",
                    explicit=False,
                    legacy=True,
                    window=1 + max(validate_encoded(r["encoded"]) for r in rows),
                    cached_values=cached,
                )
            state = recover(directory / "forwards.private")
            gates = summarize(state["stress"], state["fp32"], state["direct_checks"])
            if gates["optional_rescoring_gate_passed"]:
                for row in inputs["entries"]:
                    if deadline - time.monotonic() < 20:
                        break
                    forward(
                        "uniform",
                        [row],
                        "uniform_canonical",
                        window=inputs["fixed_logits_window"],
                    )
            if verify_base_model(MODEL, REVISION) != base_identity:
                raise ValueError("Base-model files changed")
            result = build_report(root, inputs)
            result["base_model_files_after_verified"] = True
            save(directory / "worker-result.json", result)
        finally:
            signal.alarm(0)
            model = None
            gc.collect()
            torch.cuda.empty_cache()


def run(root):
    frozen = validate(root)
    parent = require_parent_completed()
    from action_control_report import load_study
    from run_action_controls import gpu_preflight

    load_study(PARENT)
    if (root / "execution.json").exists():
        raise ValueError("Prior attempt preserved; no automatic restart")
    save(root / "parent-review.json", parent)
    save(
        root / "gpu-preflight.json",
        dict(
            allowed=gpu_preflight(
                [frozen["gpu"]], frozen["allowed_background_contexts"]
            )
        ),
    )
    save(
        root / "job.json",
        dict(
            pipeline=dict(
                pid=os.getpid(),
                argv=[frozen["model_python"], __file__, "run", "--root", str(root)],
            )
        ),
    )

    def command(case, gpu, seconds):
        return [
            frozen["model_python"],
            __file__,
            "worker",
            "--root",
            str(root),
            "--gpu",
            str(gpu),
            "--timeout",
            str(seconds),
            "--case",
            CASE,
        ]

    ledger = bounded_run(root, frozen, command, subprocess_environment)
    validate(root)
    inputs = json.loads((root / "inputs.private").read_text())
    result = build_report(root, inputs)
    worker_path = root / "cases" / CASE / "worker-result.json"
    attestation = json.loads(worker_path.read_text()) if worker_path.exists() else {}
    result["base_model_files_after_verified"] = attestation.get(
        "base_model_files_after_verified", False
    ) and attestation.get("freeze_sha256") == file_sha(root / "freeze.json")
    result["complete"] = (
        result["complete"] and result["base_model_files_after_verified"]
    )
    result.update(
        execution_status=ledger["status"],
        gpu_seconds=ledger["gpu_seconds"],
        cap_exceeded=ledger["gpu_seconds"] > 600,
    )
    out = Path("output") / root.relative_to("runs")
    save(out / "aggregate-summary.json", result)
    if (
        ledger["status"] == "evaluation_completed"
        and result["complete"]
        and not result["cap_exceeded"]
    ):
        ledger["status"] = "completed"
        save(root / "execution.json", ledger)
    else:
        raise RuntimeError(
            "Scoring run incomplete; preserve partial journal and charged time"
        )
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["prepare", "freeze", "validate", "run", "worker"]
    )
    parser.add_argument(
        "--root", type=Path, default=Path("runs/nll-scoring-diagnostic-v1")
    )
    parser.add_argument("--gpu", type=int, choices=[0, 1])
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--case")
    args = parser.parse_args()
    args.root.resolve().relative_to(Path("runs").resolve())
    if args.mode in ("freeze", "worker") and args.gpu is None:
        raise ValueError("GPU is fixed at freeze and checked at worker launch")
    if args.mode not in ("freeze", "worker") and args.gpu is not None:
        raise ValueError("No GPU override after freeze")
    if (args.mode == "worker") != (args.timeout is not None):
        raise ValueError("Only a reserved worker receives a timeout")
    if (args.mode == "worker" and args.case != CASE) or (
        args.mode != "worker" and args.case is not None
    ):
        raise ValueError("Only the fixed reserved scoring case may execute")
    args.root.mkdir(parents=True, exist_ok=True)
    if args.mode == "worker":
        worker(args.root, args.gpu, args.timeout)
        return
    lock_name = "preparation.lock" if args.mode == "prepare" else "controller.lock"
    with (args.root / lock_name).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == "prepare":
            prepare(args.root)
        elif args.mode == "freeze":
            freeze(args.root, args.gpu)
        elif args.mode == "validate":
            validate(args.root)
            print("Frozen scoring inputs verified; no GPU executed")
        else:
            run(args.root)


if __name__ == "__main__":
    main()
