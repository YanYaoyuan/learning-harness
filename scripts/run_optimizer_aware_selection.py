#!/usr/bin/env python3
"""Bounded, frozen warmup and exact optimizer-aware LoRA feature pilot."""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import run_api_harness as engine
from learning_harness.bounded_queue import run as bounded_run, save
from learning_harness.common import file_sha, read_jsonl, write_jsonl
from learning_harness.gradient_value import (
    adam_feature,
    combine_scores,
    flat_gradient,
    lora_parameters,
    optimizer_moments,
    streaming_target_means,
    unit,
)
from learning_harness.webshop_requests import RequestJournal, summarize_requests
from train_lora import encode_example, fixed_schedule

ROOT = Path("runs/optimizer-aware-selection-v1")
OUT = Path("output/optimizer-aware-selection-v1")
SPEC = Path("docs/OPTIMIZER_AWARE_SELECTION_PILOT_SPEC_2026-11-13.zh-CN.md")
SCRIPT = "scripts/run_optimizer_aware_selection.py"
CASE = "warmup-and-exact-features"
SEED = 20261113
SUPPORT = Path("runs/public-gradient-support-v1/support.private.jsonl")
PARENT = Path("runs/variance-diagnostic-v1/stage5-canonical")
CODE = {
    SCRIPT,
    str(SPEC),
    "src/learning_harness/gradient_value.py",
    "tests/test_gradient_value.py",
    "scripts/report_optimizer_aware_selection.py",
    "scripts/track_optimizer_aware_selection.py",
    "scripts/train_lora.py",
    "scripts/canonical_action_controls.py",
    "scripts/run_api_harness.py",
    "scripts/run_stage1_crossed_seeds.py",
    "scripts/run_action_controls.py",
    "scripts/run_stage0_repeat_eval.py",
    "src/learning_harness/common.py",
    "src/learning_harness/bounded_queue.py",
    "src/learning_harness/controlled_study.py",
    "src/learning_harness/adaptation.py",
    "src/learning_harness/webshop_requests.py",
    "src/learning_harness/webshop_serving.py",
    "src/learning_harness/webshop_factory.py",
    "src/learning_harness/idle_gpu_contexts.py",
    "src/learning_harness/action_matched_sampling.py",
    "src/learning_harness/action_control_audit.py",
    "third_party/LESS-reference/SOURCE.json",
    "third_party/LESS-reference/LICENSE",
    "third_party/LESS-reference/less/data_selection/collect_grad_reps.py",
    "third_party/LESS-reference/less/data_selection/matching.py",
}


def read(path):
    return json.loads(Path(path).read_text())


def message_sha(row):
    return hashlib.sha256(
        json.dumps(row["messages"], sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def environment(gpu):
    from run_stage1_crossed_seeds import environment as inherited

    return dict(
        inherited(gpu),
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false",
    )


def validate():
    frozen = read(ROOT / "freeze.json")
    if (
        frozen["cases"] != [dict(name=CASE, gpu=0)]
        or frozen["gpus"] != [0]
        or frozen["cap_gpu_hours"] != 2
        or frozen["allocation_seconds"] != 7100
        or frozen["minimum_allocation_seconds"] != 1800
        or frozen["teacher_calls"] != 0
        or frozen["official_tests_used"]
        or frozen["automatic_retry"]
        or frozen["protocol_text"] != SPEC.read_text()
        or not CODE <= set(frozen["files"])
    ):
        raise ValueError("Optimizer-aware pilot scope changed")
    for path, expected in frozen["files"].items():
        if file_sha(path) != expected:
            raise ValueError("Frozen pilot input changed: " + path)
    plan = read(ROOT / "plan.private.json")
    if (
        plan["seed"] != SEED
        or plan["checkpoints"] != [8, 16, 24, 32]
        or plan["unique_candidates"] != 853
        or plan["support_examples"] != 40
        or len(plan["warmup_subset"]) != 43
        or len(plan["warmup_schedule"]) != 128
        or set(plan["warmup_schedule"]) != set(plan["warmup_subset"])
    ):
        raise ValueError("Warmup or scoring plan changed")
    return frozen, plan


def freeze():
    if ROOT.exists():
        raise ValueError("Preserve any prior preparation or attempt")
    if read(PARENT / "execution.json")["status"] != "completed":
        raise ValueError("Prior canonical experiment must finish")
    from canonical_action_controls import aligned_scores

    sources, pool, lookup, paths = {}, [], {}, set(CODE)
    paths.update(p for p in read(PARENT / "freeze.json")["files"] if p.endswith(".py"))
    for source in ("api", "generic"):
        path = Path(f"runs/transfer-factorial-v1/corpora/full-{source}.jsonl")
        scores = PARENT / "inputs/scores" / (source + ".json")
        rows = read_jsonl(path)
        aligned_scores(read(scores), rows, file_sha(path))
        mapping = []
        for row in rows:
            if row["partition"] != "train":
                raise ValueError("Only old train data may enter candidate pool")
            key = message_sha(row)
            if key not in lookup:
                lookup[key] = len(pool)
                pool.append(dict(row, content_sha256=key))
            mapping.append(lookup[key])
        sources[source] = dict(
            data=str(path),
            data_sha256=file_sha(path),
            unique_indices=mapping,
            score_file=str(scores),
        )
        paths.update((str(path), str(scores)))
        paths.update(
            str(p) for p in (PARENT / "inputs/schedules").glob(f"*-{source}-s*.json")
        )
    support = read_jsonl(SUPPORT)
    preparation = Path("output/public-gradient-support-v1/preparation-summary.json")
    if (
        len(pool) != 853
        or len(support) != 40
        or file_sha(SUPPORT) != read(preparation)["support_sha256"]
        or any(
            r["source"] != "public_react_demonstration" or r["partition"] != "train"
            for r in support
        )
    ):
        raise ValueError("Expected fixed canonical pool and public support")
    subset = sorted(
        range(len(pool)),
        key=lambda i: hashlib.sha256(
            f"warmup:{SEED}:{pool[i]['content_sha256']}".encode()
        ).hexdigest(),
    )[:43]
    schedule = [subset[i] for i in fixed_schedule(43, 128, SEED)]
    # Check all code exists before creating the never-overwritten study root.
    for p in paths:
        if not Path(p).is_file():
            raise ValueError("Missing dependency: " + p)
    ROOT.mkdir(exist_ok=False)
    write_jsonl(ROOT / "pool.private.jsonl", pool)
    os.chmod(ROOT / "pool.private.jsonl", 0o600)
    plan = dict(
        seed=SEED,
        model=engine.MODEL,
        revision=engine.REV,
        unique_candidates=853,
        support_examples=40,
        sources=sources,
        warmup_subset=subset,
        warmup_schedule=schedule,
        checkpoints=[8, 16, 24, 32],
        checkpoint_weights=[0.25] * 4,
        config=dict(
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lr=1e-4,
            betas=[0.9, 0.999],
            eps=1e-8,
            weight_decay=0,
            max_sequence_length=2048,
            global_batch_size=4,
            steps=32,
            max_grad_norm=1.0,
        ),
    )
    save(ROOT / "plan.private.json", plan)
    os.chmod(ROOT / "plan.private.json", 0o600)
    paths.update(
        (
            str(ROOT / "plan.private.json"),
            str(ROOT / "pool.private.jsonl"),
            str(SUPPORT),
            str(preparation),
            str(PARENT / "freeze.json"),
            "runs/fresh-executor-calibration-v1/freeze.json",
            "data/models/Qwen3-4B-Instruct-2507/download-manifest.json",
        )
    )
    frozen = dict(
        name="Optimizer-aware exact LoRA selection pilot",
        protocol="optimizer-aware-selection-v1",
        cases=[dict(name=CASE, gpu=0)],
        gpus=[0],
        cap_gpu_hours=2,
        allocation_seconds=7100,
        minimum_allocation_seconds=1800,
        allowed_background_contexts=read(PARENT / "freeze.json")[
            "allowed_background_contexts"
        ],
        teacher_calls=0,
        official_tests_used=False,
        automatic_retry=False,
        raw_records_uploaded=False,
        protocol_text=SPEC.read_text(),
        files={p: file_sha(p) for p in sorted(paths)},
        frozen_unix=time.time(),
    )
    save(ROOT / "freeze.json", frozen)
    validate()
    print(
        json.dumps(
            dict(
                status="frozen_not_started",
                inputs=len(paths),
                candidates=853,
                support=40,
                freeze_sha256=file_sha(ROOT / "freeze.json"),
            )
        ),
        flush=True,
    )


def worker(gpu, seconds):
    from learning_harness.controlled_study import validate_reservation
    from learning_harness.adaptation import parameter_hash
    from learning_harness.webshop_serving import verify_base_model, model_runtime

    frozen, plan = validate()
    if gpu != 0 or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Wrong physical GPU")
    until = time.monotonic() + 2
    while True:
        ledger = read(ROOT / "execution.json")
        if CASE in ledger.get("open", {}) or time.monotonic() >= until:
            break
        time.sleep(0.02)
    validate_reservation(
        ledger, CASE, os.getpid(), gpu, seconds, file_sha(ROOT / "freeze.json")
    )
    directory = ROOT / "cases" / CASE
    if (directory / "forwards.private").exists():
        raise ValueError("A prior attempt exists; inspect instead of retrying")
    identity_before = verify_base_model(Path(engine.MODEL), engine.REV)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import LoraConfig, get_peft_model

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(SEED)
    config = plan["config"]
    tokenizer = AutoTokenizer.from_pretrained(
        engine.MODEL, local_files_only=True, trust_remote_code=False
    )
    pool, support = read_jsonl(ROOT / "pool.private.jsonl"), read_jsonl(SUPPORT)
    candidates = [encode_example(tokenizer, r, 2048) for r in pool]
    targets = [encode_example(tokenizer, r, 2048) for r in support]
    if any(r is None for r in candidates + targets):
        raise ValueError("Scoring would silently drop an overlength example")
    model = AutoModelForCausalLM.from_pretrained(
        engine.MODEL,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
    ).to("cuda")
    model.config.use_cache = False
    if getattr(model.config, "attention_dropout", 0) != 0:
        raise ValueError("Scoring requires deterministic zero attention dropout")
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=config["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    pairs = lora_parameters(model)
    if any(
        isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()
    ):
        raise ValueError("Unexpected training-mode BatchNorm")
    optimizer = torch.optim.AdamW(
        [p for _, p in pairs],
        lr=config["lr"],
        betas=tuple(config["betas"]),
        eps=config["eps"],
        weight_decay=0.0,
    )
    save(
        directory / "runtime.json",
        dict(
            model_runtime(),
            gpu_name=torch.cuda.get_device_name(0),
            trainable_parameters=sum(p.numel() for _, p in pairs),
            parameter_order=[dict(name=n, shape=list(p.shape)) for n, p in pairs],
        ),
    )
    serial, checkpoints = 0, []
    save(
        directory / "progress.json",
        dict(phase="warmup", updates=0, completed_checkpoints=0, candidates_scored=0),
    )
    with RequestJournal(directory / "forwards.private") as journal:

        def forward(row, phase, step, index, scale=1.0, clear=True, gradient=True):
            nonlocal serial
            call, serial = serial, serial + 1
            if clear:
                model.zero_grad(set_to_none=True)
            journal(
                dict(
                    kind="attempt",
                    turn=call,
                    phase=phase,
                    checkpoint=step,
                    index=index,
                    input_tokens=len(row["input_ids"]),
                    target_tokens=sum(v != -100 for v in row["labels"]),
                )
            )
            try:
                batch = {
                    k: torch.tensor([v], dtype=torch.long, device="cuda")
                    for k, v in row.items()
                }
                loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss")
                (loss * scale).backward()
                vector = flat_gradient(pairs) if gradient else None
                value = loss.detach().item()
                journal(
                    dict(
                        kind="response",
                        turn=call,
                        response=dict(
                            input_tokens=len(row["input_ids"]), output_tokens=0
                        ),
                        loss=value,
                    )
                )
                return vector, value
            except BaseException as error:
                journal(dict(kind="error", turn=call, error_type=type(error).__name__))
                raise

        def repeat_check(row, phase, step, index, first):
            repeat, _ = forward(row, phase, step, index)
            cosine = torch.dot(unit(first), unit(repeat)).item()
            relative = (
                torch.linalg.vector_norm(first - repeat)
                / torch.linalg.vector_norm(first)
            ).item()
            if cosine < 0.9999 or relative > 0.001:
                raise ValueError("Real-model gradient repeatability failed")
            return dict(cosine=cosine, relative_l2_error=relative)

        for step in range(1, 33):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for index in plan["warmup_schedule"][(step - 1) * 4 : step * 4]:
                _, loss = forward(
                    candidates[index],
                    "warmup",
                    step,
                    index,
                    scale=0.25,
                    clear=False,
                    gradient=False,
                )
                losses.append(loss)
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in pairs], 1.0)
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite warmup gradient norm")
            optimizer.step()
            with (directory / "warmup.jsonl").open("a") as log:
                log.write(
                    json.dumps(
                        dict(
                            step=step,
                            loss=sum(losses) / 4,
                            grad_norm=float(norm),
                            presentations=step * 4,
                        )
                    )
                    + "\n"
                )
            save(
                directory / "progress.json",
                dict(
                    phase="warmup",
                    updates=step,
                    completed_checkpoints=len(checkpoints),
                    candidates_scored=0,
                    last_warmup_loss=sum(losses) / 4,
                ),
            )
            if step not in plan["checkpoints"]:
                continue
            checkpoint = directory / "checkpoints" / f"step-{step:04d}"
            model.save_pretrained(checkpoint, safe_serialization=True)
            torch.save(optimizer.state_dict(), checkpoint / "optimizer.private.pt")
            os.chmod(checkpoint / "optimizer.private.pt", 0o600)
            before = parameter_hash(model, "default")
            first, second = optimizer_moments(optimizer, pairs)
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
            # Preserve activation checkpointing while disabling all stochastic dropout.
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.eval()
            diagnostic = {}

            def support_gradients():
                for i, (row, meta) in enumerate(zip(targets, support)):
                    vector, _ = forward(row, "support", step, i)
                    if i == 0:
                        diagnostic["support"] = repeat_check(
                            row, "support_repeat", step, i, vector
                        )
                    yield (
                        (
                            "place"
                            if meta["demo_key"].startswith("react_put")
                            else "clean"
                        ),
                        vector,
                    )

            means, counts = streaming_target_means(support_gradients())
            if counts != {"place": 18, "clean": 22}:
                raise ValueError("Public support allocation changed")
            values = []
            for i, row in enumerate(candidates):
                gradient, _ = forward(row, "candidate", step, i)
                if i == 0:
                    diagnostic["candidate"] = repeat_check(
                        row, "candidate_repeat", step, i, gradient
                    )
                feature = unit(adam_feature(gradient, first, second))
                scores = [
                    torch.dot(feature, means[k]).item() for k in ("place", "clean")
                ]
                if not all(math.isfinite(v) and abs(v) <= 1.00001 for v in scores):
                    raise ValueError("Invalid cosine feature score")
                values.append(scores)
                if (i + 1) % 32 == 0 or i + 1 == len(candidates):
                    save(
                        directory / "progress.json",
                        dict(
                            phase="features",
                            updates=step,
                            completed_checkpoints=len(checkpoints),
                            candidates_scored=i + 1,
                            planned_candidates=853,
                        ),
                    )
            after_first, after_second = optimizer_moments(optimizer, pairs)
            if (
                before != parameter_hash(model, "default")
                or not torch.equal(first, after_first)
                or not torch.equal(second, after_second)
                or not torch.equal(cpu_rng, torch.get_rng_state())
                or not torch.equal(cuda_rng, torch.cuda.get_rng_state())
            ):
                raise ValueError(
                    "Scoring changed weights, optimizer state or random stream"
                )
            record = dict(
                step=step,
                scores=values,
                support_counts=counts,
                repeatability=diagnostic,
                adapter_sha256=file_sha(checkpoint / "adapter_model.safetensors"),
                optimizer_sha256=file_sha(checkpoint / "optimizer.private.pt"),
                parameter_hash=before,
                scoring_state_unchanged=True,
            )
            save(checkpoint / "features.private.json", record)
            os.chmod(checkpoint / "features.private.json", 0o600)
            checkpoints.append(record)
            # Free checkpoint-local state before the next optimizer updates.
            del first, second, after_first, after_second, means, gradient, feature
    usage = summarize_requests(directory / "forwards.private")
    if (
        usage["attempted_requests"] != 3708
        or not usage["complete_usage"]
        or usage["error_records"]
    ):
        raise ValueError(
            "Expected 128 warmup plus 4*(40 support+853 candidate+2 repeat) forwards"
        )
    final = combine_scores(
        [c["scores"] for c in checkpoints], plan["checkpoint_weights"]
    )
    save(
        directory / "scores.private.json",
        dict(
            freeze_sha256=file_sha(ROOT / "freeze.json"),
            content_sha256=[r["content_sha256"] for r in pool],
            scores=final,
            checkpoints=[c["step"] for c in checkpoints],
        ),
    )
    os.chmod(directory / "scores.private.json", 0o600)
    identity_after = verify_base_model(Path(engine.MODEL), engine.REV)
    if identity_after != identity_before:
        raise ValueError("Base-model files changed")
    save(
        directory / "worker-result.json",
        dict(
            complete=True,
            freeze_sha256=file_sha(ROOT / "freeze.json"),
            unique_candidates=853,
            support_examples=40,
            checkpoints=4,
            warmup_updates=32,
            warmup_presentations=128,
            usage=usage,
            model_identity=identity_after,
            scores_sha256=file_sha(directory / "scores.private.json"),
            checkpoint_receipts={
                str(c["step"]): file_sha(
                    directory
                    / "checkpoints"
                    / f"step-{c['step']:04d}"
                    / "features.private.json"
                )
                for c in checkpoints
            },
            teacher_calls=0,
            query_evaluations=0,
        ),
    )
    save(
        directory / "progress.json",
        dict(
            phase="completed",
            updates=32,
            completed_checkpoints=4,
            candidates_scored=853,
        ),
    )


def run():
    from run_action_controls import gpu_preflight

    frozen, _ = validate()
    lock = (ROOT / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROOT / "execution.json").exists():
        raise ValueError("Prior execution exists; no automatic retry")
    save(
        ROOT / "gpu-preflight.json",
        gpu_preflight([0], frozen["allowed_background_contexts"]),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    args = [
        str(Path(".venv-wandb/bin/python").absolute()),
        "scripts/track_optimizer_aware_selection.py",
    ]
    with (ROOT / "observer.log").open("x") as log:
        observer = subprocess.Popen(
            args,
            env=dict(environment(0), CUDA_VISIBLE_DEVICES=""),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    save(
        ROOT / "job.json",
        dict(
            pipeline=dict(
                pid=os.getpid(),
                argv=[sys.executable, *sys.argv],
                started_unix=time.time(),
            ),
            observer=dict(pid=observer.pid, argv=args, started_unix=time.time()),
        ),
    )

    def command(case, gpu, seconds):
        return [
            engine.MODEL_PY,
            SCRIPT,
            "worker",
            "--case",
            case["name"],
            "--gpu",
            str(gpu),
            "--timeout",
            str(seconds),
        ]

    ledger = bounded_run(ROOT, frozen, command, environment)
    if ledger["status"] == "evaluation_completed":
        with (OUT / "report.log").open("x") as log:
            subprocess.run(
                [engine.MODEL_PY, "scripts/report_optimizer_aware_selection.py"],
                env=dict(
                    environment(0),
                    CUDA_VISIBLE_DEVICES="",
                    MPLCONFIGDIR="/tmp/lh-gradient-value-mpl",
                ),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )
        ledger["status"] = "completed"
        save(ROOT / "execution.json", ledger)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "validate", "run", "worker"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--case")
    parser.add_argument("--timeout", type=float, default=7100)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "validate":
        frozen, plan = validate()
        print(
            json.dumps(
                dict(inputs=len(frozen["files"]), candidates=plan["unique_candidates"])
            )
        )
    elif args.command == "run":
        run()
    else:
        if args.timeout <= 0 or args.case != CASE:
            parser.error("Explicit frozen case and positive timeout required")

        def stop(signum, frame):
            raise TimeoutError("Pilot deadline or requested stop")

        signal.signal(signal.SIGALRM, stop)
        signal.signal(signal.SIGTERM, stop)
        signal.setitimer(signal.ITIMER_REAL, args.timeout)
        worker(args.gpu, args.timeout)


if __name__ == "__main__":
    main()
