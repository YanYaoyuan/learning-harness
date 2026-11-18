#!/usr/bin/env python3
"""Freeze and execute the full fresh-task student learning development matrix."""

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

import run_api_harness as engine
import run_optimizer_aware_selection_fp32 as features
import run_json_examples_calibration as calibration
from learning_harness import benchmark
from learning_harness.adaptation import composition_revision
from learning_harness.batch_runner import BatchClient
from learning_harness.bounded_queue import run as bounded_run, save
from learning_harness.common import file_sha, read_jsonl, write_jsonl
from learning_harness.experience_value_study import (
    INITIALIZATIONS,
    NODES,
    POLICIES,
    SAMPLING,
    SOURCES,
    audit_episodes,
    clean_zero_branch,
    matrix,
    select,
    validate_queries,
)
from learning_harness.json_demo_executor import actor_messages
from learning_harness.webshop_requests import RequestJournal, summarize_requests
from run_fresh_executor_calibration import JournalClient

ROOT = Path("runs/experience-value-development-v1")
OUT = Path("output/experience-value-development-v1")
DATA = Path("runs/fresh-experience-value-v1")
PARENT = Path("runs/variance-diagnostic-v1/stage5-canonical")
ZERO = Path("runs/transfer-factorial-v1/heat/trajectories/full-api-seed1101/step-0000")
SHARED_ZERO = ROOT / "shared-zero-B"
SPEC = Path("docs/EXPERIENCE_VALUE_DEVELOPMENT_SPEC_2026-11-14.zh-CN.md")
SCRIPT = "scripts/run_experience_value_development.py"
CODE = {
    SCRIPT,
    str(SPEC),
    "src/learning_harness/experience_value_study.py",
    "tests/test_experience_value_study.py",
    "scripts/report_experience_value_development.py",
    "scripts/track_experience_value_development.py",
    "scripts/train_lora_fixed_schedule.py",
    "scripts/train_lora_split_seeds.py",
    "scripts/train_lora.py",
    "src/learning_harness/nll_range_sampling.py",
}


def read(p):
    return json.loads(Path(p).read_text())


def environment(gpu):
    from run_stage1_crossed_seeds import environment as original

    return dict(original(gpu), TOKENIZERS_PARALLELISM="false")


def runtime():
    code = "import importlib.metadata as m,json,sys; print(json.dumps(dict(python=sys.version,packages={n:m.version(n) for n in sys.argv[1:]})))"
    return {
        label: json.loads(
            subprocess.check_output(
                [python, "-c", code, *packages],
                env=dict(environment(0), CUDA_VISIBLE_DEVICES=""),
                text=True,
                timeout=30,
            )
        )
        for label, python, packages in (
            (
                "model",
                engine.MODEL_PY,
                (
                    "torch",
                    "transformers",
                    "peft",
                    "accelerate",
                    "tokenizers",
                    "safetensors",
                ),
            ),
            ("environment", engine.CPU_PY, ("textworld", "alfworld")),
        )
    }


def scores_for(source):
    plan = read(features.ROOT / "plan.private.json")
    meta = plan["sources"][source]
    rows = read_jsonl(meta["data"])
    from canonical_action_controls import aligned_scores

    nll = aligned_scores(read(meta["score_file"]), rows, file_sha(meta["data"]))
    score = read(features.ROOT / "cases" / features.CASE / "scores.private.json")
    gradient = [score["scores"][i] for i in meta["unique_indices"]]
    return rows, nll, gradient, meta


def recipe_path(source, policy, sampling):
    return ROOT / "recipes" / f"{policy}-{source}-s{sampling}.private.json"


def config(revision):
    return dict(
        protocol_id="experience-value-development-v1",
        manifest=str(ROOT / "queries.private.jsonl"),
        method="json_direct",
        executor="json_public_examples_v1",
        revision=revision,
        max_turns=40,
        max_calls=40,
        max_trials=1,
        max_new_tokens=128,
        history_turns=12,
        batch_size=8,
        temperature=0.0,
        seeds=[11],
        model=dict(
            base_url_env="LH_STUDENT_BASE_URL",
            model_env="LH_STUDENT_MODEL",
            api_key_env="LH_STUDENT_API_KEY",
        ),
    )


def validate():
    f = read(ROOT / "freeze.json")
    if (
        f["cases"] != matrix()
        or len(f["cases"]) != 225
        or f["gpus"] != [0]
        or f["planned_episodes"] != 28800
        or f["cap_gpu_hours"] != 96
        or f["allocation_seconds"] != 2000
        or f["minimum_allocation_seconds"] != 1100
        or f["interim_complete_blocks"] != list(NODES)
        or f["teacher_calls"] != 0
        or f["official_tests_used"]
        or f["confirmation_tasks_used"]
        or f["automatic_retry"]
        or f["protocol_text"] != SPEC.read_text()
        or f["inferential_tests_planned"]
        or not CODE <= set(f["files"])
    ):
        raise ValueError("Frozen development matrix changed")
    for p, sha in f["files"].items():
        if file_sha(p) != sha:
            raise ValueError("Development input changed: " + p)
    queries = read_jsonl(ROOT / "queries.private.jsonl")
    validate_queries(
        queries,
        read_jsonl(DATA / "tasks/calibration.jsonl"),
        read_jsonl(DATA / "tasks/confirmation.jsonl"),
    )
    if read(ROOT / "a-lora.json") != read(PARENT / "a-lora.json"):
        raise ValueError("Final SFT differs from the preserved trainer configuration")
    return f


def audit_recipe(source, policy, sampling):
    recipe = read(recipe_path(source, policy, sampling))
    rows, nll, gradient, meta = scores_for(source)
    reference = read(PARENT / "inputs/schedules" / f"random-{source}-s{sampling}.json")[
        "indices"
    ]
    expected = select(rows, nll, gradient, reference, sampling, policy)
    if (
        recipe["indices"] != expected
        or recipe["data"] != meta["data"]
        or recipe["data_sha256"] != file_sha(meta["data"])
        or recipe["source"] != source
        or recipe["policy"] != policy
        or recipe["sampling_seed"] != sampling
    ):
        raise ValueError("Frozen selector cannot be regenerated")
    return recipe


def freeze():
    if ROOT.exists():
        raise ValueError("Preserve any existing study preparation")
    calibration.validate()
    features.validate()
    for parent in (calibration.ROOT, features.ROOT):
        ledger = read(parent / "execution.json")
        if ledger["status"] != "completed" or ledger["open"]:
            raise ValueError("Prior phases incomplete")
    public = read(
        "output/research-operations/20261114/completed-followups-local-check.json"
    )
    cloud = read(
        "output/research-operations/20261114/completed-followups-cloud-check.json"
    )
    if (
        public["status"] != "completed_outputs_independently_recomputed"
        or cloud["status"] != "all_six_aggregate_files_byte_verified"
    ):
        raise ValueError("Independent completed-phase reviews missing")
    queries = read_jsonl(DATA / "tasks/development.jsonl")
    validate_queries(
        queries,
        read_jsonl(DATA / "tasks/calibration.jsonl"),
        read_jsonl(DATA / "tasks/confirmation.jsonl"),
    )
    # A new study must never silently resume after any development evaluation.
    for p in DATA.glob("**/episodes.jsonl"):
        if p.stat().st_size:
            raise ValueError("Unexpected development evaluation history")
    paths = (
        set(read(calibration.ROOT / "freeze.json")["files"])
        | set(read(features.ROOT / "freeze.json")["files"])
        | CODE
    )
    paths.update(
        (
            str(calibration.ROOT / "freeze.json"),
            str(features.ROOT / "freeze.json"),
            str(features.ROOT / "plan.private.json"),
            str(features.ROOT / "cases" / features.CASE / "scores.private.json"),
            str(PARENT / "a-lora.json"),
            str(ZERO / "composition.json"),
            str(ZERO / "B/adapter_model.safetensors"),
            str(ZERO / "B/adapter_config.json"),
            "output/research-operations/20261114/completed-followups-local-check.json",
            "output/research-operations/20261114/completed-followups-cloud-check.json",
        )
    )
    for p in paths:
        if not Path(p).is_file():
            raise ValueError("Missing frozen dependency: " + p)
    from transformers import AutoTokenizer
    from train_lora import encode_example

    tokenizer = AutoTokenizer.from_pretrained(
        engine.MODEL, local_files_only=True, trust_remote_code=False
    )
    preparations = []
    for source in SOURCES:
        rows, nll, gradient, meta = scores_for(source)
        from learning_harness.fresh_tasks import task_group

        protected_groups = {
            r["task_group"]
            for role in ("calibration", "development", "confirmation")
            for r in read_jsonl(DATA / "tasks" / (role + ".jsonl"))
        }
        if any(
            task_group(r) is None or task_group(r) in protected_groups for r in rows
        ):
            raise ValueError("Training content overlaps a protected fresh task group")
        encoded = [encode_example(tokenizer, r, 2048) for r in rows]
        if any(r is None for r in encoded):
            raise ValueError("Previously eligible training content would be dropped")
        paths.update((meta["data"], meta["data"] + ".meta.json", meta["score_file"]))
        for sampling in SAMPLING:
            reference_path = (
                PARENT / "inputs/schedules" / f"random-{source}-s{sampling}.json"
            )
            paths.add(str(reference_path))
            reference = read(reference_path)["indices"]
            for policy in POLICIES:
                indices = select(rows, nll, gradient, reference, sampling, policy)
                if policy in ("random", "low", "middle", "high"):
                    historical = (
                        Path("runs/nll-range-controls-v1/candidate-schedules")
                        / f"{policy}-{source}-s{sampling}.json"
                    )
                    if read(historical)["indices"] != indices:
                        raise ValueError("Prepared NLL control changed")
                    paths.add(str(historical))
                if policy in ("gradient_high", "gradient_global"):
                    kind = "matched" if policy == "gradient_high" else "global"
                    historical = (
                        features.ROOT
                        / "candidate-schedules"
                        / f"gradient-{kind}-{source}-s{sampling}.json"
                    )
                    if read(historical)["indices"] != indices:
                        raise ValueError("Prepared gradient baseline changed")
                    paths.add(str(historical))
                recipe = dict(
                    status="frozen_training_input",
                    source=source,
                    policy=policy,
                    sampling_seed=sampling,
                    data=meta["data"],
                    data_sha256=file_sha(meta["data"]),
                    indices=indices,
                    presentations=768,
                    distinct_support=256,
                    coverage_matched=policy != "gradient_global",
                    input_tokens=sum(len(encoded[i]["input_ids"]) for i in indices),
                    target_tokens=sum(
                        sum(v != -100 for v in encoded[i]["labels"]) for i in indices
                    ),
                    feature_freeze_sha256=file_sha(features.ROOT / "freeze.json"),
                )
                preparations.append((recipe_path(source, policy, sampling), recipe))
    template = read(ZERO / "composition.json")
    if not template["initialization"]["exact_zero_b_logit_check"]:
        raise ValueError("The shared B branch was not validated as zero")
    training_config = read(PARENT / "a-lora.json")
    if any(
        template["lora_config"][k] != training_config[k]
        for k in ("lora_r", "lora_alpha", "lora_dropout", "target_modules")
    ):
        raise ValueError("Shared zero adapter configuration differs from final SFT")
    from safetensors import safe_open

    with safe_open(
        ZERO / "B/adapter_model.safetensors", framework="pt", device="cpu"
    ) as tensors:
        active_zero, foreign = clean_zero_branch(
            {k: tensors.get_tensor(k) for k in tensors.keys()}
        )
    locked_runtime = runtime()
    ROOT.mkdir(exist_ok=False)
    save(ROOT / "runtime-lock.json", locked_runtime)
    from safetensors.torch import save_file

    SHARED_ZERO.mkdir()
    save_file(active_zero, str(SHARED_ZERO / "adapter_model.safetensors"))
    shutil.copyfile(ZERO / "B/adapter_config.json", SHARED_ZERO / "adapter_config.json")
    save(
        SHARED_ZERO / "source.json",
        dict(
            source_sha256=file_sha(ZERO / "B/adapter_model.safetensors"),
            active_tensors=len(active_zero),
            removed_foreign_named_adapter_keys=foreign,
            semantics="Exact active zero branch; foreign A export keys excluded; source bytes preserved.",
        ),
    )
    paths.update(str(p) for p in SHARED_ZERO.iterdir())
    write_jsonl(ROOT / "queries.private.jsonl", queries)
    (ROOT / "queries.private.jsonl").chmod(0o600)
    save(ROOT / "a-lora.json", read(PARENT / "a-lora.json"))
    for path, recipe in preparations:
        save(path, recipe)
        path.chmod(0o600)
        paths.add(str(path))
    paths.update(
        (
            str(ROOT / "queries.private.jsonl"),
            str(ROOT / "a-lora.json"),
            str(ROOT / "runtime-lock.json"),
        )
    )
    paths.update(
        str(p / "execution.json")
        for p in (
            features.ROOT,
            calibration.ROOT,
            Path("runs/fresh-executor-calibration-v1"),
            Path("runs/nll-scoring-diagnostic-v1"),
        )
    )
    prior = (
        read(features.ROOT / "execution.json")["gpu_seconds"]
        + read(calibration.ROOT / "execution.json")["gpu_seconds"]
        + read("runs/fresh-executor-calibration-v1/execution.json")["gpu_seconds"]
        + read("runs/nll-scoring-diagnostic-v1/execution.json")["gpu_seconds"]
    )
    f = dict(
        name="Fresh-task student experience-value development",
        protocol="experience-value-development-v1",
        cases=matrix(),
        gpus=[0],
        planned_students=224,
        planned_models=225,
        planned_episodes=28800,
        cap_gpu_hours=96,
        allocation_seconds=2000,
        minimum_allocation_seconds=1100,
        preparatory_seconds=prior,
        interim_complete_blocks=list(NODES),
        inferential_tests_planned=False,
        allowed_background_contexts=read(calibration.ROOT / "freeze.json")[
            "allowed_background_contexts"
        ],
        teacher_calls=0,
        official_tests_used=False,
        confirmation_tasks_used=False,
        automatic_retry=False,
        raw_records_uploaded=False,
        protocol_text=SPEC.read_text(),
        files={p: file_sha(p) for p in sorted(paths)},
        frozen_unix=time.time(),
    )
    save(ROOT / "freeze.json", f)
    validate()
    OUT.mkdir(parents=True, exist_ok=True)
    save(
        OUT / "freeze-summary.json",
        dict(
            freeze_sha256=file_sha(ROOT / "freeze.json"),
            students=224,
            models=225,
            planned_episodes=28800,
            development_tasks=128,
            confirmation_evaluations=0,
            initializations=list(INITIALIZATIONS),
            sampling_seeds=list(SAMPLING),
            policies=list(POLICIES),
            cap_gpu_hours=96,
            preparatory_seconds=prior,
            frozen_inputs=len(paths),
            raw_records_uploaded=False,
        ),
    )
    print(
        json.dumps(
            dict(
                status="frozen_not_started",
                inputs=len(paths),
                students=224,
                planned_episodes=28800,
                freeze_sha256=file_sha(ROOT / "freeze.json"),
            )
        ),
        flush=True,
    )


def training_audit(folder, case, recipe):
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


def run():
    from run_action_controls import gpu_preflight

    f = validate()
    lock = (ROOT / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROOT / "execution.json").exists():
        raise ValueError("Inspect any prior execution before resuming")
    if shutil.disk_usage(ROOT).free < 24 * 1024**3:
        raise ValueError("At least24GiB free workspace required")
    save(
        ROOT / "gpu-preflight.json",
        gpu_preflight([0], f["allowed_background_contexts"]),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    args = [
        str(Path(".venv-wandb/bin/python").absolute()),
        "scripts/track_experience_value_development.py",
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

    ledger = bounded_run(
        ROOT, f, command, environment, preparatory_seconds=f["preparatory_seconds"]
    )
    if ledger["status"] == "evaluation_completed":
        with (OUT / "report.log").open("x") as log:
            subprocess.run(
                [engine.MODEL_PY, "scripts/report_experience_value_development.py"],
                env=dict(
                    environment(0),
                    CUDA_VISIBLE_DEVICES="",
                    MPLCONFIGDIR="/tmp/lh-value-development-mpl",
                ),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
            )
        ledger["status"] = "completed"
        save(ROOT / "execution.json", ledger)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("freeze", "validate", "run", "worker"))
    p.add_argument("--case")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--timeout", type=float, default=2000)
    a = p.parse_args()
    if a.command == "freeze":
        freeze()
    elif a.command == "validate":
        print(json.dumps(dict(frozen_inputs=len(validate()["files"]))))
    elif a.command == "run":
        run()
    else:
        if a.timeout <= 0 or a.case not in {c["name"] for c in matrix()}:
            raise ValueError("Explicit frozen case and positive timeout required")

        def stop(signum, frame):
            raise TimeoutError("Development case deadline")

        signal.signal(signal.SIGALRM, stop)
        signal.signal(signal.SIGTERM, stop)
        signal.setitimer(signal.ITIMER_REAL, a.timeout)
        worker(a.case, a.gpu, a.timeout)


if __name__ == "__main__":
    main()
