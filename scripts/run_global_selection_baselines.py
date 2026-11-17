#!/usr/bin/env python3
"""Missing equal-freedom global baselines; completed parent remains immutable."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import run_experience_value_development as parent
from learning_harness import benchmark
from learning_harness.adaptation import composition_revision
from learning_harness.batch_runner import BatchClient
from learning_harness.bounded_queue import run as bounded_run, save
from learning_harness.common import file_sha, read_jsonl
from learning_harness.experience_value_study import audit_episodes
from learning_harness.global_experience_selection import (
    POLICIES,
    global_schedule,
    matrix,
)
from learning_harness.json_demo_executor import actor_messages
from learning_harness.webshop_requests import RequestJournal, summarize_requests
from run_fresh_executor_calibration import JournalClient

ROOT = Path("runs/global-selection-baselines-v1")
OUT = Path("output/global-selection-baselines-v1")
SPEC = Path("docs/GLOBAL_SELECTION_BASELINES_SPEC_2026-11-17.zh-CN.md")
SCRIPT = "scripts/run_global_selection_baselines.py"
engine, ZERO, SHARED_ZERO = parent.engine, parent.ZERO, parent.SHARED_ZERO
environment, runtime, read = parent.environment, parent.runtime, parent.read
SOURCES, SAMPLING = parent.SOURCES, parent.SAMPLING


def config(revision):
    result = parent.config(revision)
    result.update(
        protocol_id="global-selection-baselines-v1",
        manifest=str(ROOT / "queries.private.jsonl"),
    )
    return result


def recipe_path(source, policy, sampling):
    return ROOT / "recipes" / f"{policy}-{source}-s{sampling}.private.json"


def audit_recipe(source, policy, sampling):
    recipe = read(recipe_path(source, policy, sampling))
    rows, nll, gradient, meta = parent.scores_for(source)
    if (
        recipe["indices"] != global_schedule(rows, sampling, policy, nll)
        or recipe["data_sha256"] != file_sha(meta["data"])
        or recipe["data"] != meta["data"]
        or recipe["policy"] != policy
        or recipe["sampling_seed"] != sampling
        or recipe["presentations"] != 768
        or recipe["distinct_support"] != 256
    ):
        raise ValueError("Global baseline recipe changed")
    return recipe


def validate():
    parent.validate()
    f = read(ROOT / "freeze.json")
    if (
        f["cases"] != matrix()
        or f["gpus"] != [0]
        or f["cap_gpu_hours"] != 24
        or f["planned_episodes"] != 8192
        or f["protocol_text"] != SPEC.read_text()
        or f["teacher_calls"] != 0
        or f["confirmation_tasks_used"]
        or f["official_tests_used"]
        or f["automatic_retry"]
        or f["allocation_seconds"] != 2000
        or f["minimum_allocation_seconds"] != 1100
    ):
        raise ValueError("Global baseline scope changed")
    for path, sha in f["files"].items():
        if file_sha(path) != sha:
            raise ValueError("Frozen baseline input changed: " + path)
    if read(ROOT / "a-lora.json") != read(parent.ROOT / "a-lora.json"):
        raise ValueError("Student configuration changed")
    return f


def freeze():
    if ROOT.exists():
        raise ValueError("Preserve any previous baseline preparation")
    f = parent.validate()
    if read(parent.ROOT / "execution.json")["status"] != "completed":
        raise ValueError("The parent study must be complete")
    final = read(parent.OUT / "aggregate-summary.json")
    local = read("output/experience-value-completion-audit-v1/local-audit.json")
    if local["final_summary_sha256"] != file_sha(parent.OUT / "aggregate-summary.json"):
        raise ValueError("Parent report no longer matches the complete audit")
    code = {
        SCRIPT,
        str(SPEC),
        "src/learning_harness/global_experience_selection.py",
        "tests/test_global_experience_selection.py",
        "scripts/report_global_selection_baselines.py",
        "scripts/track_global_selection_baselines.py",
        "scripts/track_global_selection_training.py",
        "src/learning_harness/training_log_metrics.py",
    }
    paths = (
        set(f["files"])
        | code
        | {
            str(parent.ROOT / "freeze.json"),
            str(parent.OUT / "aggregate-summary.json"),
            "output/experience-value-completion-audit-v1/local-audit.json",
        }
    )
    from transformers import AutoTokenizer
    from train_lora import encode_example

    tokenizer = AutoTokenizer.from_pretrained(
        engine.MODEL, local_files_only=True, trust_remote_code=False
    )
    recipes, anchors = [], {}
    for source in SOURCES:
        rows, nll, gradient, meta = parent.scores_for(source)
        encoded = [encode_example(tokenizer, r, 2048) for r in rows]
        if any(r is None for r in encoded):
            raise ValueError("Eligible content unexpectedly dropped")
        for sampling in SAMPLING:
            reference = parent.read(
                parent.recipe_path(source, "gradient_global", sampling)
            )
            if (
                global_schedule(rows, sampling, "gradient_global", gradient)
                != reference["indices"]
            ):
                raise ValueError(
                    "The completed global gradient replay is not exactly reproduced"
                )
            for policy in POLICIES:
                indices = global_schedule(rows, sampling, policy, nll)
                recipes.append(
                    (
                        recipe_path(source, policy, sampling),
                        dict(
                            source=source,
                            policy=policy,
                            sampling_seed=sampling,
                            data=meta["data"],
                            data_sha256=file_sha(meta["data"]),
                            indices=indices,
                            presentations=768,
                            distinct_support=256,
                            coverage_matched=False,
                            input_tokens=sum(
                                len(encoded[i]["input_ids"]) for i in indices
                            ),
                            target_tokens=sum(
                                sum(v != -100 for v in encoded[i]["labels"])
                                for i in indices
                            ),
                        ),
                    )
                )
    reused = []
    for record in final["models"]:
        case = record["case"]
        if case["policy"] != "gradient_global" and case["name"] != "base":
            continue
        path = parent.ROOT / "cases" / case["name"] / "worker-result.json"
        if file_sha(path) != final["case_receipt_sha256"][case["name"]]:
            raise ValueError("Reused parent receipt changed")
        paths.add(str(path))
        reused.append(case)
        if case["name"] != "base":
            key = str(case["initialization_seed"])
            value = record["training"]["initial_parameter_sha256"]
            if key in anchors and anchors[key] != value:
                raise ValueError("Parent initialization pairing differs")
            anchors[key] = value
    if len(reused) != 33 or len(anchors) != 8:
        raise ValueError("Incomplete parent reference matrix")
    ROOT.mkdir()
    for name in ("a-lora.json", "runtime-lock.json", "queries.private.jsonl"):
        shutil.copyfile(parent.ROOT / name, ROOT / name)
        paths.add(str(ROOT / name))
    (ROOT / "queries.private.jsonl").chmod(0o600)
    for path, recipe in recipes:
        save(path, recipe)
        path.chmod(0o600)
        paths.add(str(path))
    save(
        ROOT / "freeze.json",
        dict(
            name="Global experience selection: equal-freedom missing baselines",
            protocol="global-selection-baselines-v1",
            protocol_text=SPEC.read_text(),
            cases=matrix(),
            gpus=[0],
            planned_models=64,
            planned_episodes=8192,
            cap_gpu_hours=24,
            allocation_seconds=2000,
            minimum_allocation_seconds=1100,
            preparatory_seconds=0,
            allowed_background_contexts=f["allowed_background_contexts"],
            teacher_calls=0,
            confirmation_tasks_used=False,
            official_tests_used=False,
            automatic_retry=False,
            raw_records_uploaded=False,
            parent_freeze_sha256=file_sha(parent.ROOT / "freeze.json"),
            reused_cases=reused,
            initialization_anchors=anchors,
            frozen_unix=time.time(),
            files={p: file_sha(p) for p in sorted(paths)},
        ),
    )
    validate()
    OUT.mkdir()
    save(
        OUT / "freeze-summary.json",
        dict(
            freeze_sha256=file_sha(ROOT / "freeze.json"),
            new_students=64,
            reused_students=32,
            reused_base=1,
            new_episodes=8192,
            frozen_inputs=len(paths),
            cap_gpu_hours=24,
            confirmation_evaluations=0,
            raw_records_uploaded=False,
        ),
    )
    print(json.dumps(read(OUT / "freeze-summary.json")), flush=True)


def training_audit(folder, case, recipe):
    result = inherited_training_audit(folder, case, recipe)
    expected = read(ROOT / "freeze.json")["initialization_anchors"][
        str(case["initialization_seed"])
    ]
    if result["initial_parameter_sha256"] != expected:
        raise ValueError(
            "Initial weights differ from the existing paired gradient student"
        )
    return result


def run():
    from run_action_controls import gpu_preflight

    f = validate()
    lock = (ROOT / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROOT / "execution.json").exists():
        raise ValueError("Inspect any previous execution before resuming")
    if shutil.disk_usage(ROOT).free < 12 * 1024**3:
        raise ValueError("At least 12GiB free workspace required")
    save(
        ROOT / "gpu-preflight.json",
        gpu_preflight([0], f["allowed_background_contexts"]),
    )
    jobs = dict(
        pipeline=dict(
            pid=os.getpid(), argv=[sys.executable, *sys.argv], started_unix=time.time()
        )
    )
    for name, filename in (
        ("observer", "track_global_selection_baselines.py"),
        ("training_telemetry", "track_global_selection_training.py"),
    ):
        args = [str(Path(".venv-wandb/bin/python").absolute()), "scripts/" + filename]
        with (ROOT / (name + ".log")).open("x") as log:
            process = subprocess.Popen(
                args,
                env=dict(environment(0), CUDA_VISIBLE_DEVICES=""),
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        jobs[name] = dict(pid=process.pid, argv=args, started_unix=time.time())
    save(ROOT / "job.json", jobs)

    def command(case, gpu, seconds):
        return [
            engine.CPU_PY,
            SCRIPT,
            "worker",
            "--case",
            case["name"],
            "--gpu",
            str(gpu),
            "--timeout",
            str(seconds),
        ]

    ledger = bounded_run(ROOT, f, command, environment)
    if ledger["status"] == "evaluation_completed":
        with (OUT / "report.log").open("x") as log:
            subprocess.run(
                [engine.MODEL_PY, "scripts/report_global_selection_baselines.py"],
                env=dict(
                    environment(0),
                    CUDA_VISIBLE_DEVICES="",
                    MPLCONFIGDIR="/tmp/lh-global-mpl",
                ),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=900,
            )
        ledger["status"] = "completed"
        save(ROOT / "execution.json", ledger)


def inherited_training_audit(folder, case, recipe):
    adapter = folder / "A"
    meta, control = (
        read(adapter / "experiment.json"),
        read(adapter / "seed-controls.json"),
    )
    if (
        not control["completed"]
        or control["initialization_seed"] != case["initialization_seed"]
        or control["sampling_seed"] != case["sampling_seed"]
        or control["selection_policy"] != case["policy"]
        or control["external_schedule_sha256"]
        != file_sha(recipe_path(case["source"], case["policy"], case["sampling_seed"]))
        or control["trained_adapter_sha256"]
        != file_sha(adapter / "adapter_model.safetensors")
        or meta["training_data_sha256"] != recipe["data_sha256"]
        or meta["config"] != read(ROOT / "a-lora.json")
        or meta["training_seed"] != case["initialization_seed"]
        or meta["processed_examples_rank0"] != 768
        or meta["input_tokens_rank0"] != recipe["input_tokens"]
        or meta["target_tokens_rank0"] != recipe["target_tokens"]
        or meta["used_examples"] != len(read_jsonl(recipe["data"]))
        or meta["overlength_dropped"] != 0
    ):
        raise ValueError("Actual SFT identity/budget differs from frozen recipe")
    return dict(
        initial_parameter_sha256=control["initial_parameter_sha256"],
        adapter_sha256=control["trained_adapter_sha256"],
        input_tokens=meta["input_tokens_rank0"],
        target_tokens=meta["target_tokens_rank0"],
        presentations=768,
        optimizer_steps=192,
        experiment_sha256=file_sha(adapter / "experiment.json"),
        control_sha256=file_sha(adapter / "seed-controls.json"),
    )


def make_composition(folder):
    point = folder / "checkpoint"
    if point.exists():
        raise ValueError("Prior composition retained")
    template = read(ZERO / "composition.json")
    point.mkdir()
    (point / "B").symlink_to(SHARED_ZERO.absolute(), target_is_directory=True)
    metadata = dict(
        base_model=engine.MODEL,
        base_revision=engine.REV,
        lora_config=template["lora_config"],
        a_adapter=str((folder / "A").absolute()),
        a_adapter_sha256=file_sha(folder / "A/adapter_model.safetensors"),
        b_adapter_sha256=file_sha(SHARED_ZERO / "adapter_model.safetensors"),
        optimizer_step=0,
        semantics="Cold-start student A plus shared immutable zero B; no B update and no merging",
    )
    save(point / "composition.json", metadata)
    return point, composition_revision(point)


def worker(name, gpu, seconds):
    from learning_harness.controlled_study import validate_reservation
    from learning_harness.webshop_serving import verify_base_model

    f = validate()
    case = next(c for c in f["cases"] if c["name"] == name)
    if gpu != 0 or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Wrong reserved GPU")
    until = time.monotonic() + 2
    while True:
        ledger = read(ROOT / "execution.json")
        if name in ledger.get("open", {}) or time.monotonic() >= until:
            break
        time.sleep(0.02)
    validate_reservation(
        ledger, name, os.getpid(), gpu, seconds, file_sha(ROOT / "freeze.json")
    )
    from run_action_controls import gpu_preflight

    save(
        ROOT / "cases" / name / "gpu-preflight.json",
        gpu_preflight([0], f["allowed_background_contexts"]),
    )
    if runtime() != read(ROOT / "runtime-lock.json"):
        raise ValueError("The frozen model or environment package versions changed")
    folder = ROOT / "cases" / name
    if (folder / "model-requests.private").exists() or (folder / "A").exists():
        raise ValueError("Prior partial work retained; do not retry")
    before = verify_base_model(Path(engine.MODEL), engine.REV)
    save(
        folder / "progress.json",
        dict(phase="training" if name != "base" else "loading_base"),
    )
    training, point, revision = None, None, engine.REV
    if name != "base":
        recipe = audit_recipe(case["source"], case["policy"], case["sampling_seed"])
        path = recipe_path(case["source"], case["policy"], case["sampling_seed"])
        engine.command(
            [
                engine.MODEL_PY,
                "scripts/train_lora_fixed_schedule.py",
                "--schedule",
                str(path),
                "--data",
                recipe["data"],
                "--model",
                engine.MODEL,
                "--revision",
                engine.REV,
                "--config",
                str(ROOT / "a-lora.json"),
                "--out",
                str(folder / "A"),
                "--initialization-seed",
                str(case["initialization_seed"]),
                "--sampling-seed",
                str(case["sampling_seed"]),
            ],
            environment(gpu),
            folder / "training.log",
            timeout=seconds,
        )
        training = training_audit(folder, case, recipe)
        point, revision = make_composition(folder)
    cfg = config(revision)
    save(folder / "evaluation-config.json", cfg)

    def actor(initial, history, memories, method, history_turns):
        if method != "json_direct" or memories:
            raise ValueError("Unexpected actor interface")
        return actor_messages(initial, history, history_turns)

    benchmark.actor_messages = actor
    save(
        folder / "progress.json",
        dict(phase="evaluation", training_complete=name != "base"),
    )
    with engine.server(
        gpu, "value-" + name, revision, folder / "server.log", point
    ) as env:
        previous = dict(os.environ)
        os.environ.update(env)
        try:
            with RequestJournal(folder / "model-requests.private") as journal:
                client = JournalClient(BatchClient(cfg["model"]), journal)
                rows = benchmark.run(
                    folder / "evaluation-config.json",
                    "data/alfworld",
                    folder / "evaluation",
                    client=client,
                )
        finally:
            os.environ.clear()
            os.environ.update(previous)
    after = verify_base_model(Path(engine.MODEL), engine.REV)
    if before != after:
        raise ValueError("Base-model identity changed")
    usage = summarize_requests(folder / "model-requests.private")
    metric = audit_episodes(
        rows,
        read_jsonl(folder / "evaluation/events.jsonl"),
        usage,
        read_jsonl(ROOT / "queries.private.jsonl"),
    )
    if training is not None and training != training_audit(folder, case, recipe):
        raise ValueError("Training artifact changed during evaluation")
    if point is not None and composition_revision(point) != revision:
        raise ValueError("Served composition changed")
    save(
        folder / "worker-result.json",
        dict(
            complete=True,
            case=case,
            freeze_sha256=file_sha(ROOT / "freeze.json"),
            **metric,
            training=training,
            composition_revision=revision,
            model_identity=after,
            config_sha256=file_sha(folder / "evaluation-config.json"),
            results_sha256=file_sha(folder / "evaluation/tasks.jsonl"),
            events_sha256=file_sha(folder / "evaluation/events.jsonl"),
        ),
    )
    save(folder / "progress.json", dict(phase="completed", episodes=128))


def audit_result(case):
    folder = ROOT / "cases" / case["name"]
    result = read(folder / "worker-result.json")
    if (
        not result["complete"]
        or result["case"] != case
        or result["freeze_sha256"] != file_sha(ROOT / "freeze.json")
    ):
        raise ValueError("Completed case identity differs")
    for key, path in (
        ("config_sha256", folder / "evaluation-config.json"),
        ("results_sha256", folder / "evaluation/tasks.jsonl"),
        ("events_sha256", folder / "evaluation/events.jsonl"),
    ):
        if result[key] != file_sha(path):
            raise ValueError("Completed artifact changed")
    metric = audit_episodes(
        read_jsonl(folder / "evaluation/tasks.jsonl"),
        read_jsonl(folder / "evaluation/events.jsonl"),
        summarize_requests(folder / "model-requests.private"),
        read_jsonl(ROOT / "queries.private.jsonl"),
    )
    if any(result[k] != v for k, v in metric.items()):
        raise ValueError("Stored metric differs from raw outcomes")
    if case["name"] != "base":
        recipe = audit_recipe(case["source"], case["policy"], case["sampling_seed"])
        if result["training"] != training_audit(folder, case, recipe):
            raise ValueError("Stored SFT budget differs")
        point = folder / "checkpoint"
        metadata = read(point / "composition.json")
        if (
            metadata["a_adapter_sha256"]
            != file_sha(folder / "A/adapter_model.safetensors")
            or metadata["b_adapter_sha256"]
            != file_sha(SHARED_ZERO / "adapter_model.safetensors")
            or composition_revision(point) != result["composition_revision"]
        ):
            raise ValueError("Model composition differs")
    elif result["training"] is not None or result["composition_revision"] != engine.REV:
        raise ValueError("The baseline must be an untrained base model")
    if read(folder / "evaluation-config.json") != config(
        result["composition_revision"]
    ):
        raise ValueError("Evaluation config differs")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "validate", "run", "worker"))
    parser.add_argument("--case")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=2000)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "validate":
        print(json.dumps(dict(frozen_inputs=len(validate()["files"]))))
    elif args.command == "run":
        run()
    else:
        if args.timeout <= 0 or args.case not in {c["name"] for c in matrix()}:
            raise ValueError(
                "An explicit frozen case and positive timeout are required"
            )

        def stop(signum, frame):
            raise TimeoutError("Global baseline case deadline")

        signal.signal(signal.SIGALRM, stop)
        signal.signal(signal.SIGTERM, stop)
        signal.setitimer(signal.ITIMER_REAL, args.timeout)
        worker(args.case, args.gpu, args.timeout)


if __name__ == "__main__":
    main()
