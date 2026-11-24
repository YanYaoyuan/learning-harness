#!/usr/bin/env python3
"""Check the LoRA training entry point with a small randomly initialized model."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def main():
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    from safetensors.torch import load_file
    root = Path("runs/stack-check")
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507",
                    revision="cdbee75f17c01a7cc42f958dc650907174af0554", local_files_only=True)
    model_dir = root / "random-model"
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokenizer), hidden_size=64,
        intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, tie_word_embeddings=True))
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    rows = [dict(partition="train", messages=[
        dict(role="user", content=f"Validation fixture {i}. Return a JSON action."),
        dict(role="assistant", content='{"action": "look"}')]) for i in range(4)]
    data = root / "synthetic.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    Path(str(data) + ".meta.json").write_text(json.dumps(dict(validation_only=True, research_result=False)))
    config = json.loads(Path("configs/lora.json").read_text())
    config.update(max_steps=2, global_batch_size=2, max_sequence_length=256, lora_r=4, lora_alpha=8)
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config))
    command = [sys.executable, "scripts/train_lora.py", "--model", str(model_dir), "--revision",
               "random-model-validation-only", "--data", str(data), "--config", str(config_path),
               "--out", str(root / "adapter")]
    with (root / "training.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    weights = load_file(root / "adapter/adapter_model.safetensors")
    updated = any("lora_B" in key and bool(value.abs().sum() > 0) for key, value in weights.items())
    if not updated:
        raise ValueError("LoRA weights were not updated")
    result = dict(kind="synthetic_training_stack_validation", research_result=False,
                  actual_train_entrypoint_passed=True, optimizer_steps=2, lora_updated=True)
    (root / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
