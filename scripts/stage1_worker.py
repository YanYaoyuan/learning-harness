#!/usr/bin/env python3
"""One controlled A adapter, two fixed query evaluations, and read-only NLL scoring."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import signal
import subprocess

import run_api_harness as engine
from run_stage0_repeat_eval import save
from learning_harness.adaptation import composition_revision
from learning_harness.analysis import summarize
from learning_harness.common import file_sha, read_jsonl, write_jsonl

ROOT = Path("runs/variance-diagnostic-v1/stage1")
OLD = Path("runs/transfer-factorial-v1")
FAMILIES = ("cool", "heat")


class WorkerDeadline(BaseException):
    pass


def make_composition(adapter, point):
    template_path = OLD / "heat/trajectories/full-api-seed1101/step-0000"
    template = json.loads((template_path / "composition.json").read_text())
    if not template["initialization"]["exact_zero_b_logit_check"]:
        raise ValueError("Reference B branch has no verified zero check")
    if point.exists():
        revision = composition_revision(point)
        existing = json.loads((point / "composition.json").read_text())
        if existing["a_adapter_sha256"] != file_sha(adapter / "adapter_model.safetensors"):
            raise ValueError("Existing composition has different A weights")
        return revision
    point.mkdir(parents=True)
    shutil.copytree(template_path / "B", point / "B")
    metadata = dict(base_model=engine.MODEL, base_revision=engine.REV,
        lora_config=template["lora_config"], a_adapter=str(adapter.absolute()),
        a_adapter_sha256=file_sha(adapter / "adapter_model.safetensors"),
        b_adapter_sha256=file_sha(point / "B/adapter_model.safetensors"), optimizer_step=0,
        semantics="Controlled A plus the same immutable zero B branch; no B optimization",
        zero_b_reference=str(template_path), zero_b_reference_verified=True)
    save(point / "composition.json", metadata)
    return composition_revision(point)


def evaluate(point, directory, family, source, gpu, smoke=False):
    historical = OLD / family / "evaluation/main" / f"full-{source}-seed1101/B/step-0000"
    config = json.loads((historical / "config.json").read_text())
    if smoke:
        manifest = directory / "calibration.jsonl"
        write_jsonl(manifest, read_jsonl(OLD / family / "b_calibration.jsonl")[:4])
    else:
        manifest = OLD / family / "b_query.jsonl"
    config.update(manifest=str(manifest), protocol_id="variance-diagnostic-v1-stage1")
    path = directory / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Evaluation configuration changed")
    save(path, config)
    revision = composition_revision(point)
    if (directory / "summary.json").exists():
        summary = json.loads((directory / "summary.json").read_text())
        if summary["complete"]:
            if summary["metrics"]["none"]["error_count"] or summary["recorded"] != (4 if smoke else 32):
                raise ValueError("Cached evaluation is errored or has the wrong task count")
            if json.loads((directory / "run.json").read_text())["revision"] != revision:
                raise ValueError("Completed evaluation has different weights")
            return summary
    with engine.server(gpu, "stage1-" + point.parent.name, revision, directory / "server.log", point) as env:
        engine.command([engine.CPU_PY, "-m", "learning_harness.batch_runner", "--config", str(path),
                        "--evaluation", "--out", str(directory)], env, directory / "rollout.log")
    summary = summarize(directory)
    if not summary["complete"] or summary["metrics"]["none"]["error_count"]:
        raise ValueError("Evaluation errored or is incomplete")
    return summary


def score_nll(point, output):
    import torch
    from transformers import AutoTokenizer
    from train_lora import encode_example
    from learning_harness.adaptation import load_composition
    before_a = file_sha(Path(json.loads((point / "composition.json").read_text())["a_adapter"]) / "adapter_model.safetensors")
    before_b = file_sha(point / "B/adapter_model.safetensors")
    if output.exists():
        result = json.loads(output.read_text())
        if result["composition_revision"] != composition_revision(point):
            raise ValueError("NLL weights changed")
        if any(result[f]["data_sha256"] != file_sha(OLD / f / "diagnostic.jsonl") for f in FAMILIES):
            raise ValueError("NLL diagnostic changed")
        return result
    tokenizer = AutoTokenizer.from_pretrained(engine.MODEL, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, revision = load_composition(point, engine.MODEL, engine.REV)
    model.eval()
    result = dict(composition_revision=revision, used_for_updates=False, model_selection=False,
                  budget_rule_used=True, no_live_trainer=True)
    for family in FAMILIES:
        examples = []
        data = OLD / family / "diagnostic.jsonl"
        for row in read_jsonl(data):
            if row["partition"] != "dev" or row["adaptation_role"] != "b_diagnostic":
                raise ValueError("Diagnostic role changed")
            encoded = encode_example(tokenizer, dict(row, partition="train"), 2048)
            if encoded is None:
                raise ValueError("Frozen diagnostic eligibility changed")
            examples.append((row["task_id"], encoded))
        grouped = defaultdict(list)
        with torch.inference_mode():
            for offset in range(0, len(examples), 8):
                batch = examples[offset:offset + 8]
                length = max(len(row["input_ids"]) for _, row in batch)
                ids = torch.tensor([[tokenizer.pad_token_id] * (length - len(row["input_ids"])) + row["input_ids"] for _, row in batch], device="cuda:0")
                mask = torch.tensor([[0] * (length - len(row["input_ids"])) + row["attention_mask"] for _, row in batch], device="cuda:0")
                targets = [[v for v in row["labels"] if v != -100] for _, row in batch]
                logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=max(map(len, targets)) + 1).logits
                for (task, _), values, target in zip(batch, logits, targets):
                    loss = torch.nn.functional.cross_entropy(values[-len(target) - 1:-1].float(), torch.tensor(target, device="cuda:0"), reduction="mean")
                    grouped[task].append(float(loss))
        per_task = {task: sum(values) / len(values) for task, values in grouped.items()}
        result[family] = dict(task_mean_nll=sum(per_task.values()) / len(per_task), per_task_nll=per_task,
                              examples=len(examples), tasks=len(per_task), data_sha256=file_sha(data))
    if before_a != file_sha(Path(json.loads((point / "composition.json").read_text())["a_adapter"]) / "adapter_model.safetensors") or before_b != file_sha(point / "B/adapter_model.safetensors"):
        raise ValueError("Diagnostic modified checkpoint files")
    result["checkpoint_unchanged"] = True
    save(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    def timeout(signum, frame):
        raise WorkerDeadline()
    signal.signal(signal.SIGALRM, timeout)
    signal.signal(signal.SIGTERM, timeout)
    signal.setitimer(signal.ITIMER_REAL, args.timeout)
    if args.smoke:
        case = dict(name="pipeline-smoke", source="api", initialization_seed=101, sampling_seed=101)
        directory = ROOT / "pipeline-smoke"
        adapter = Path("runs/variance-diagnostic-v1/seed-control-smoke-v1/api-init101-sample101")
    else:
        freeze = json.loads((ROOT / "freeze.json").read_text())
        for path, expected in freeze["files"].items():
            if file_sha(path) != expected:
                raise ValueError("Frozen Stage 1 input changed: " + path)
        case = next(c for c in freeze["cases"] if c["name"] == args.case)
        if case["gpu"] != args.gpu:
            raise ValueError("GPU assignment differs from the frozen plan")
        directory = ROOT / "cases" / args.case
        adapter = directory / "A"
        control = adapter / "seed-controls.json"
        if not control.exists() or not json.loads(control.read_text()).get("completed"):
            if adapter.exists() and any(adapter.iterdir()):
                raise ValueError("Partial training retained; inspect before resuming")
            env = engine.environment(args.gpu)
            for name in ("NO_PROXY", "no_proxy"):
                env[name] = ",".join(v for v in (env.get(name, ""), "127.0.0.1", "localhost", "::1") if v)
            engine.command([engine.MODEL_PY, "scripts/train_lora_split_seeds.py", "--data", str(OLD / "corpora" / f"full-{case['source']}.jsonl"),
                "--model", engine.MODEL, "--revision", engine.REV, "--config", str(ROOT / "a-lora.json"),
                "--out", str(adapter), "--initialization-seed", str(case["initialization_seed"]),
                "--sampling-seed", str(case["sampling_seed"])], env, directory / "training.log")
    point = directory / "checkpoint"
    control = json.loads((adapter / "seed-controls.json").read_text())
    metadata = json.loads((adapter / "experiment.json").read_text())
    corpus_path = OLD / "corpora" / f"full-{case['source']}.jsonl"
    if (not control["completed"] or control["initialization_seed"] != case["initialization_seed"]
            or control["sampling_seed"] != case["sampling_seed"]
            or control["trained_adapter_sha256"] != file_sha(adapter / "adapter_model.safetensors")
            or metadata["training_data_sha256"] != file_sha(corpus_path)):
        raise ValueError("Cached training does not match this case")
    if not args.smoke and metadata["config"] != json.loads((ROOT / "a-lora.json").read_text()):
        raise ValueError("Training configuration changed")
    make_composition(adapter, point)
    scores = {}
    for family in FAMILIES:
        summary = evaluate(point, directory / "evaluation" / family, family, case["source"], args.gpu, args.smoke)
        scores[family] = summary["metrics"]["none"]["success_rate"]
    nll = score_nll(point, directory / "diagnostic-nll.json")
    from train_lora import fixed_schedule
    corpus = read_jsonl(corpus_path)
    if metadata["used_examples"] != len(corpus):
        raise ValueError("Eligibility changed; exposure requires the actual filtered row mapping")
    schedule = fixed_schedule(len(corpus), metadata["processed_examples_rank0"], case["sampling_seed"])
    types = {r["task_id"]: r["task_type"] for r in read_jsonl("runs/api-harness-v1/train.jsonl")}
    exposure = dict(processed_examples=len(schedule), unique_examples=len(set(schedule)),
        unique_tasks=len({corpus[i]["task_id"] for i in schedule}),
        task_type_presentations=dict(Counter(types[corpus[i]["task_id"]] for i in schedule)),
        input_tokens=metadata["input_tokens_rank0"], target_tokens=metadata["target_tokens_rank0"])
    save(directory / "worker-result.json", dict(complete=True, case=case, success=scores,
        mean_success=sum(scores.values()) / 2, mean_nll=sum(nll[f]["task_mean_nll"] for f in FAMILIES) / 2,
        control=control, exposure=exposure, smoke=args.smoke,
        composition_revision=composition_revision(point)))
    signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
