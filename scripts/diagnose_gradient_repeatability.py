#!/usr/bin/env python3
"""Separate, recorded numerical diagnostic on the failed pilot's saved checkpoint."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

import run_api_harness as engine
import run_optimizer_aware_selection as parent
from learning_harness.bounded_queue import run as bounded_run, save
from learning_harness.common import file_sha, read_jsonl
from learning_harness.webshop_requests import RequestJournal, summarize_requests

ROOT = Path("runs/gradient-repeatability-diagnostic-v1")
OUT = Path("output/gradient-repeatability-diagnostic-v1")
CASE = "saved-checkpoint-numerics"
SCRIPT = "scripts/diagnose_gradient_repeatability.py"
CHECKPOINT = parent.ROOT / "cases" / parent.CASE / "checkpoints/step-0008"
MODES = ("bf16_checkpointing", "bf16_eval", "fp32_math_checkpointing")


def read(p):
    return json.loads(Path(p).read_text())


def validate():
    f = read(ROOT / "freeze.json")
    for p, sha in f["files"].items():
        if file_sha(p) != sha:
            raise ValueError("Diagnostic input changed: " + p)
    if f["modes"] != list(MODES) or f["cases"] != [dict(name=CASE, gpu=0)]:
        raise ValueError("Diagnostic scope changed")
    return f


def freeze():
    if ROOT.exists():
        raise ValueError("Preserve prior diagnostic")
    old, _ = parent.validate()
    if read(parent.ROOT / "execution.json")["status"] != "failed":
        raise ValueError("Inspect original pilot before diagnostic")
    paths = set(old["files"]) | {
        SCRIPT,
        str(parent.ROOT / "freeze.json"),
        str(CHECKPOINT / "adapter_config.json"),
        str(CHECKPOINT / "adapter_model.safetensors"),
    }
    f = dict(
        name="Saved-checkpoint gradient numerical diagnostic",
        cases=[dict(name=CASE, gpu=0)],
        gpus=[0],
        cap_gpu_hours=0.5,
        allocation_seconds=1750,
        minimum_allocation_seconds=300,
        allowed_background_contexts=old["allowed_background_contexts"],
        modes=list(MODES),
        definition="Two fixed train-only inputs, three repeats per precision/mode; no updates. Compare original FP32 reduction against FP64 cosine/L2 and elementwise identity. Thresholds unchanged: cosine>=.9999, relative L2<=.001.",
        decision_rule="If gradients pass FP64 but fail original reduction, fix metric reductions; if real BF16 differences exceed thresholds and FP32 passes, use separately frozen FP32 scorer; otherwise stop and diagnose.",
        files={p: file_sha(p) for p in sorted(paths)},
        automatic_retry=False,
        teacher_calls=0,
        official_tests_used=False,
        frozen_unix=time.time(),
    )
    save(ROOT / "freeze.json", f)
    print(
        json.dumps(
            dict(
                status="frozen",
                inputs=len(paths),
                freeze_sha256=file_sha(ROOT / "freeze.json"),
            )
        )
    )


def worker(seconds):
    from learning_harness.controlled_study import validate_reservation
    from learning_harness.gradient_value import flat_gradient, lora_parameters, unit
    from learning_harness.adaptation import parameter_hash

    f = validate()
    until = time.monotonic() + 2
    while True:
        ledger = read(ROOT / "execution.json")
        if CASE in ledger.get("open", {}) or time.monotonic() >= until:
            break
        time.sleep(0.02)
    validate_reservation(
        ledger, CASE, os.getpid(), 0, seconds, file_sha(ROOT / "freeze.json")
    )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Wrong GPU")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import PeftModel
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from train_lora import encode_example

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(parent.SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        engine.MODEL, local_files_only=True, trust_remote_code=False
    )
    rows = [
        read_jsonl(parent.SUPPORT)[0],
        read_jsonl(parent.ROOT / "pool.private.jsonl")[0],
    ]
    inputs = [encode_example(tokenizer, r, 2048) for r in rows]
    model = AutoModelForCausalLM.from_pretrained(
        engine.MODEL,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
    ).to("cuda")
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, str(CHECKPOINT), is_trainable=True)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    pairs = lora_parameters(model)
    before = parameter_hash(model, "default")
    records = []
    directory = ROOT / "cases" / CASE
    serial = 0
    with RequestJournal(directory / "forwards.private") as journal:
        for mode in f["modes"]:
            if mode == "fp32_math_checkpointing":
                model.float()
                torch.use_deterministic_algorithms(True)
            model.eval() if mode == "bf16_eval" else model.train()
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.eval()
            for index, row in enumerate(inputs):
                first = None
                for repeat in range(3):
                    model.zero_grad(set_to_none=True)
                    call, serial = serial, serial + 1
                    journal(
                        dict(
                            kind="attempt",
                            turn=call,
                            mode=mode,
                            index=index,
                            repeat=repeat,
                        )
                    )
                    try:
                        batch = {
                            k: torch.tensor([v], device="cuda", dtype=torch.long)
                            for k, v in row.items()
                        }
                        if mode == "fp32_math_checkpointing":
                            with sdpa_kernel(SDPBackend.MATH):
                                loss = model(**batch).loss
                                loss.backward()
                        else:
                            loss = model(**batch).loss
                            loss.backward()
                        vector = flat_gradient(pairs)
                        journal(
                            dict(
                                kind="response",
                                turn=call,
                                response=dict(
                                    input_tokens=len(row["input_ids"]), output_tokens=0
                                ),
                            )
                        )
                    except BaseException as e:
                        journal(
                            dict(kind="error", turn=call, error_type=type(e).__name__)
                        )
                        raise
                    norm64 = torch.linalg.vector_norm(vector.double()).item()
                    if first is None:
                        first = vector
                    cosine32 = torch.dot(unit(first), unit(vector)).item()
                    a, b = first.double(), vector.double()
                    cosine64 = (
                        torch.dot(a, b)
                        / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
                    ).item()
                    relative64 = (
                        torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(a)
                    ).item()
                    record = dict(
                        mode=mode,
                        input_index=index,
                        repeat=repeat,
                        loss=loss.item(),
                        norm_fp64=norm64,
                        cosine_original_fp32=cosine32,
                        cosine_fp64=cosine64,
                        relative_l2_fp64=relative64,
                        elementwise_equal=torch.equal(first, vector),
                        pass_original=cosine32 >= 0.9999 and relative64 <= 0.001,
                        pass_fp64=cosine64 >= 0.9999 and relative64 <= 0.001,
                    )
                    records.append(record)
                    print(json.dumps(record), flush=True)
                    save(
                        directory / "progress.json",
                        dict(
                            phase="diagnostic",
                            completed_forwards=serial,
                            total_forwards=18,
                        ),
                    )
                    del a, b
    unchanged = before == parameter_hash(model, "default")
    if not unchanged:
        raise ValueError("Diagnostic changed LoRA weights")
    usage = summarize_requests(directory / "forwards.private")
    if usage["attempted_requests"] != 18 or not usage["complete_usage"]:
        raise ValueError("Incomplete diagnostic")
    result = dict(
        complete=True,
        freeze_sha256=file_sha(ROOT / "freeze.json"),
        records=records,
        usage=usage,
        weights_unchanged=unchanged,
        training_updates=0,
        query_evaluations=0,
    )
    save(directory / "worker-result.json", result)
    save(OUT / "aggregate-summary.json", result)


def run():
    from run_action_controls import gpu_preflight

    f = validate()
    lock = (ROOT / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROOT / "execution.json").exists():
        raise ValueError("Prior diagnostic attempt retained")
    save(
        ROOT / "gpu-preflight.json",
        gpu_preflight([0], f["allowed_background_contexts"]),
    )
    save(
        ROOT / "job.json",
        dict(
            pipeline=dict(
                pid=os.getpid(),
                argv=[sys.executable, *sys.argv],
                started_unix=time.time(),
            )
        ),
    )

    def env(gpu):
        return dict(parent.environment(gpu), CUBLAS_WORKSPACE_CONFIG=":4096:8")

    def command(case, gpu, seconds):
        return [
            engine.MODEL_PY,
            SCRIPT,
            "worker",
            "--case",
            CASE,
            "--timeout",
            str(seconds),
        ]

    ledger = bounded_run(ROOT, f, command, env)
    if ledger["status"] == "evaluation_completed":
        ledger["status"] = "completed"
        save(ROOT / "execution.json", ledger)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("freeze", "run", "worker"))
    p.add_argument("--case")
    p.add_argument("--timeout", type=float, default=1750)
    a = p.parse_args()
    if a.command == "freeze":
        freeze()
    elif a.command == "run":
        run()
    else:
        if a.case != CASE:
            raise ValueError("Explicit case required")

        def stop(signum, frame):
            raise TimeoutError("Diagnostic deadline")

        signal.signal(signal.SIGALRM, stop)
        signal.signal(signal.SIGTERM, stop)
        signal.setitimer(signal.ITIMER_REAL, a.timeout)
        worker(a.timeout)


if __name__ == "__main__":
    main()
