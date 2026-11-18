#!/usr/bin/env python3
"""Controlled target-SFT invocation of the byte-frozen trainer with separate seeds.

The original trainer is loaded in this child process only. Its schedule function is bound
to the sampling seed; its --seed remains the initialization/dropout seed. No frozen file
is edited. Only fixed-budget target-SFT is supported.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def schedule_for_sampling_seed(original, sampling_seed):
    def schedule(length, budget, ignored_initialization_seed):
        return original(length, budget, sampling_seed)
    return schedule


def validate_config(config):
    if not config.get("fixed_sample_budget"):
        raise ValueError("Split-seed control requires the frozen sequential fixed-budget sampler")
    if config["max_steps"] <= 0 or config["global_batch_size"] <= 0:
        raise ValueError("Positive update and sample budgets required")


def training_arguments(args):
    return ["train_lora.py", "--data", args.data, "--model", args.model, "--revision", args.revision,
            "--config", args.config, "--seed", str(args.initialization_seed),
            "--out", args.out, "--method", "target_sft"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "model", "revision", "config", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--initialization-seed", type=int, required=True)
    parser.add_argument("--sampling-seed", type=int, required=True)
    args = parser.parse_args()
    validate_config(json.loads(Path(args.config).read_text()))
    trainer_path = Path("scripts/train_lora.py")
    sha = hashlib.sha256(trainer_path.read_bytes()).hexdigest()
    historical = json.loads(Path("runs/source-comparison-v1/inputs.sha256.json").read_text())
    if historical[str(trainer_path)] != sha:
        raise ValueError("The historical trainer changed; do not silently change the diagnostic")
    spec = importlib.util.spec_from_file_location("frozen_trainer_for_split_seed_diagnostic", trainer_path)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    trainer.fixed_schedule = schedule_for_sampling_seed(trainer.fixed_schedule, args.sampling_seed)
    import peft
    from learning_harness.adaptation import parameter_hash
    original = peft.get_peft_model
    control = dict(initialization_seed=args.initialization_seed, sampling_seed=args.sampling_seed,
        initialization_factor="LoRA initialization plus dropout stream", trainer_sha256=sha,
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), completed=False)
    output = Path(args.out)

    def capture_initial_parameters(*positional, **keywords):
        model = original(*positional, **keywords)
        name = keywords.get("adapter_name", "default")
        count = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n and f".{name}." in n)
        if not count:
            raise ValueError("No LoRA parameters were captured")
        control.update(initial_parameter_sha256=parameter_hash(model, name), initial_parameter_count=count)
        output.mkdir(parents=True, exist_ok=True)
        (output / "seed-controls.json").write_text(json.dumps(control, indent=2) + "\n")
        return model

    old_argv = sys.argv
    peft.get_peft_model = capture_initial_parameters
    try:
        sys.argv = training_arguments(args)
        trainer.main()
        metadata = json.loads((output / "experiment.json").read_text())
        expected = trainer.fixed_schedule(metadata["used_examples"], metadata["processed_examples_rank0"],
                                          args.initialization_seed)
        schedule_sha = hashlib.sha256(json.dumps(expected).encode()).hexdigest()
        if schedule_sha != metadata["schedule_sha256"] or metadata["training_seed"] != args.initialization_seed:
            raise ValueError("Actual training does not match the split-seed controls")
        control.update(completed=True, schedule_sha256=schedule_sha,
                       trained_adapter_sha256=hashlib.sha256((output / "adapter_model.safetensors").read_bytes()).hexdigest())
        (output / "seed-controls.json").write_text(json.dumps(control, indent=2) + "\n")
    finally:
        sys.argv = old_argv
        peft.get_peft_model = original


if __name__ == "__main__":
    main()
