#!/usr/bin/env python3
"""Fixed-compute full-pool versus subset-replay sampling intervention."""
import argparse
from collections import Counter
import json
from pathlib import Path
import signal

import run_api_harness as engine
import stage1_worker as core
from run_stage1_crossed_seeds import environment
from train_lora import fixed_schedule
from train_lora_sampling_intervention import schedule
from learning_harness.bounded_queue import save
from learning_harness.common import file_sha, read_jsonl
from learning_harness.adaptation import composition_revision

ROOT = Path("runs/variance-diagnostic-v1/stage2")
OLD = Path("runs/transfer-factorial-v1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    def timeout(signum, frame):
        raise core.WorkerDeadline()
    signal.signal(signal.SIGALRM, timeout)
    signal.signal(signal.SIGTERM, timeout)
    signal.setitimer(signal.ITIMER_REAL, args.timeout)
    freeze = json.loads((ROOT / "freeze.json").read_text())
    for path, expected in freeze["files"].items():
        if file_sha(path) != expected:
            raise ValueError("Stage 2 frozen file changed: " + path)
    case = next(c for c in freeze["cases"] if c["name"] == args.case)
    if case["gpu"] != args.gpu:
        raise ValueError("Frozen GPU assignment changed")
    directory = ROOT / "cases" / case["name"]
    adapter = directory / "A"
    control_path = adapter / "seed-controls.json"
    corpus_path = OLD / "corpora" / f"full-{case['source']}.jsonl"
    if not control_path.exists() or not json.loads(control_path.read_text()).get("completed"):
        if adapter.exists() and any(adapter.iterdir()):
            raise ValueError("Partial training retained; inspect before resume")
        engine.command([engine.MODEL_PY, "scripts/train_lora_sampling_intervention.py",
            "--sampling-arm", case["arm"], "--data", str(corpus_path), "--model", engine.MODEL,
            "--revision", engine.REV, "--config", str(ROOT / "a-lora.json"), "--out", str(adapter),
            "--initialization-seed", str(case["initialization_seed"]),
            "--sampling-seed", str(case["sampling_seed"])], environment(args.gpu), directory / "training.log")
    ctrl = json.loads(control_path.read_text())
    metadata = json.loads((adapter / "experiment.json").read_text())
    if (not ctrl["completed"] or ctrl.get("sampling_arm") != case["arm"]
            or ctrl["initialization_seed"] != case["initialization_seed"]
            or ctrl["sampling_seed"] != case["sampling_seed"]
            or ctrl["trained_adapter_sha256"] != file_sha(adapter / "adapter_model.safetensors")
            or metadata["training_data_sha256"] != file_sha(corpus_path)
            or metadata["config"] != json.loads((ROOT / "a-lora.json").read_text())):
        raise ValueError("Training does not match the frozen case")
    corpus = read_jsonl(corpus_path)
    if metadata["used_examples"] != len(corpus) or metadata["processed_examples_rank0"] != 768:
        raise ValueError("Eligibility or actual presentation budget changed")
    indices = schedule(fixed_schedule, len(corpus), 768, case["sampling_seed"], case["arm"])
    import hashlib
    if hashlib.sha256(json.dumps(indices).encode()).hexdigest() != ctrl["schedule_sha256"]:
        raise ValueError("Realized schedule differs from intervention")
    counts = Counter(indices)
    if case["arm"] == "full_pass" and len(counts) != len(corpus):
        raise ValueError("Incomplete full-pool pass")
    if case["arm"] == "subset_replay" and (len(counts) != 256 or set(counts.values()) != {3}):
        raise ValueError("Replay membership/frequencies changed")
    types = {r["task_id"]: r["task_type"] for r in read_jsonl("runs/api-harness-v1/train.jsonl")}
    exposure = dict(processed_examples=768, unique_examples=len(counts),
        unique_tasks=len({corpus[i]["task_id"] for i in indices}),
        task_type_presentations=dict(Counter(types[corpus[i]["task_id"]] for i in indices)),
        input_tokens=metadata["input_tokens_rank0"], target_tokens=metadata["target_tokens_rank0"],
        minimum_repetitions=min(counts.values()), maximum_repetitions=max(counts.values()))
    point = directory / "checkpoint"
    core.make_composition(adapter, point)
    scores = {family: core.evaluate(point, directory / "evaluation" / family, family,
              case["source"], args.gpu)["metrics"]["none"]["success_rate"] for family in core.FAMILIES}
    nll = core.score_nll(point, directory / "diagnostic-nll.json")
    save(directory / "worker-result.json", dict(complete=True, smoke=False, case=case, success=scores,
        mean_success=sum(scores.values()) / 2, mean_nll=sum(nll[f]["task_mean_nll"] for f in core.FAMILIES) / 2,
        control=ctrl, exposure=exposure, composition_revision=composition_revision(point)))
    signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
