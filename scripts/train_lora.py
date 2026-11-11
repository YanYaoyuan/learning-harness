#!/usr/bin/env python3
"""Bounded rejection-SFT, mixed-context and local context-distillation baselines."""
from __future__ import annotations

import argparse
import json
import os
import hashlib
import random
from pathlib import Path


def encode_example(tokenizer, row, max_length):
    if row.get("partition") != "train":
        raise ValueError("Only train-partition examples may update weights")
    messages = row["messages"]
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Expected one final assistant target")
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if full[:len(prefix)] != prefix:
        raise ValueError("Chat-template boundary mismatch; refusing to guess loss mask")
    if len(full) > max_length:
        return None
    if len(full) == len(prefix):
        raise ValueError("Empty assistant target")
    return dict(input_ids=full, attention_mask=[1] * len(full),
                labels=[-100] * len(prefix) + full[len(prefix):])


def fixed_schedule(length, budget, seed):
    """Full shuffled passes followed by a partial pass; exactly budget samples."""
    if length <= 0 or budget <= 0:
        raise ValueError("Positive dataset length and sample budget required")
    rng = random.Random(seed)
    indices = []
    while len(indices) < budget:
        order = list(range(length))
        rng.shuffle(order)
        indices.extend(order[:budget - len(indices)])
    return indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--revision", required=True, help="Actual HF commit or immutable local model identifier")
    parser.add_argument("--config", default="configs/lora.json")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", choices=["target_sft", "mixed_sft", "context_distill"], default="target_sft")
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training requested but this session has no CUDA device")
    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty checkpoint directory")
    config = json.loads(Path(args.config).read_text())
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if config["global_batch_size"] % world_size:
        raise ValueError("Global batch size must be divisible by GPU process count")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    encoded = []
    mixture_count = 0
    # A fixed, balanced assignment changes the context view, not corpus membership.
    ordered = sorted(range(len(rows)), key=lambda i: hashlib.sha256(
        f"{args.seed}:{i}".encode()).hexdigest())
    hinted_indices = set(ordered[:len(ordered) // 2])
    for index, row in enumerate(rows):
        base = encode_example(tokenizer, row, config["max_sequence_length"])
        teacher = None
        if "teacher_messages" in row:
            teacher = encode_example(tokenizer, dict(row, messages=row["teacher_messages"]), config["max_sequence_length"])
            # Enforce the same length-based eligibility for all three methods.
            if teacher is None:
                continue
            if base is not None and [x for x in base["labels"] if x != -100] != [x for x in teacher["labels"] if x != -100]:
                raise ValueError("Teacher and student completion tokens must align exactly")
        if base is None:
            continue
        if args.method != "target_sft" and teacher is None:
            raise ValueError("This method requires teacher_messages in the training corpus")
        if args.method == "mixed_sft" and index in hinted_indices:
            base = teacher
            mixture_count += 1
        if args.method == "context_distill":
            base = dict(base, teacher_input_ids=teacher["input_ids"],
                        teacher_attention_mask=teacher["attention_mask"],
                        target_length=sum(v != -100 for v in base["labels"]))
        encoded.append(base)
    examples = encoded
    if len(examples) < config["global_batch_size"]:
        raise ValueError("Too few complete examples; collect more successful train rollouts")
    eligible_examples = len(examples)
    schedule = None
    if config.get("fixed_sample_budget", False):
        schedule = fixed_schedule(len(examples), config["max_steps"] * config["global_batch_size"], args.seed)
        examples = [examples[index] for index in schedule]
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                                                torch_dtype=torch.bfloat16, trust_remote_code=False)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=config["lora_r"], lora_alpha=config["lora_alpha"],
                          lora_dropout=config["lora_dropout"], target_modules=config["target_modules"],
                          task_type="CAUSAL_LM", bias="none"))
    model.enable_input_require_grads()
    frozen_teacher = None
    if args.method == "context_distill":
        if world_size != 1:
            raise ValueError("Discovery context distillation runs as one process per GPU")
        frozen_teacher = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
            torch_dtype=torch.bfloat16, trust_remote_code=False).to("cuda:0").eval()
        for parameter in frozen_teacher.parameters():
            parameter.requires_grad_(False)
        frozen_teacher.config.use_cache = False

    def collate(batch):
        length = max(len(row["input_ids"]) for row in batch)
        output_batch = {key: torch.tensor([row[key] + [pad] * (length - len(row[key])) for row in batch])
                       for key, pad in (("input_ids", tokenizer.pad_token_id), ("attention_mask", 0), ("labels", -100))}
        if args.method == "context_distill":
            teacher_length = max(len(row["teacher_input_ids"]) for row in batch)
            for key, pad in (("teacher_input_ids", tokenizer.pad_token_id), ("teacher_attention_mask", 0)):
                output_batch[key] = torch.tensor([row[key] + [pad] * (teacher_length - len(row[key])) for row in batch])
            output_batch["target_length"] = torch.tensor([row["target_length"] for row in batch])
        return output_batch

    class AccountingTrainer(Trainer):
        input_tokens_seen = 0
        target_tokens_seen = 0
        teacher_input_tokens_seen = 0
        examples_seen = 0

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # All methods use the same per-example loss and accumulation rule.
            # Custom KL does not consume Trainer's global-token normalization.
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            self.input_tokens_seen += int(inputs["attention_mask"].sum().item())
            self.examples_seen += int(inputs["input_ids"].shape[0])
            self.target_tokens_seen += int((inputs["labels"] != -100).sum().item())
            if frozen_teacher is not None:
                import torch.nn.functional as F
                # Microbatch size one keeps variable prefix offsets exact.
                target_length = int(inputs.pop("target_length")[0])
                teacher_ids = inputs.pop("teacher_input_ids")
                teacher_mask = inputs.pop("teacher_attention_mask")
                self.teacher_input_tokens_seen += int(teacher_mask.sum().item())
                with torch.no_grad():
                    teacher_logits = frozen_teacher(input_ids=teacher_ids, attention_mask=teacher_mask,
                        logits_to_keep=target_length + 1).logits[:, :-1].float()
                outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                logits_to_keep=target_length + 1)
                student_logits = outputs.logits[:, :-1].float()
                targets = inputs["labels"][:, -target_length:]
                ce = F.cross_entropy(student_logits.reshape(-1, student_logits.shape[-1]), targets.reshape(-1))
                kl = F.kl_div(F.log_softmax(student_logits, dim=-1),
                              F.softmax(teacher_logits, dim=-1), reduction="sum") / targets.numel()
                weight = config.get("distill_kl_weight", 0.5)
                loss = (1 - weight) * ce + weight * kl
                return (loss, outputs) if return_outputs else loss
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch=num_items_in_batch)

    arguments = TrainingArguments(
        output_dir=args.out, max_steps=config["max_steps"], per_device_train_batch_size=1,
        gradient_accumulation_steps=config["global_batch_size"] // world_size,
        learning_rate=config["learning_rate"], warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"], bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, seed=args.seed,
        data_seed=args.seed, logging_steps=10, save_strategy="no", report_to=[],
        remove_unused_columns=False, ddp_find_unused_parameters=False,
    )
    trainer = AccountingTrainer(model=model, args=arguments, train_dataset=examples, data_collator=collate)
    trainer.train()
    trainer.save_model(args.out)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.out)
        (output / "experiment.json").write_text(json.dumps(dict(
            baseline=args.method, model=args.model, revision=args.revision,
            training_seed=args.seed, world_size=world_size, config=config,
            training_data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
            input_examples=len(rows), used_examples=eligible_examples, overlength_dropped=len(rows) - eligible_examples,
            processed_examples_rank0=trainer.examples_seen,
            scheduled_examples=len(examples) if schedule is not None else None,
            unique_scheduled_examples=len(set(schedule)) if schedule is not None else None,
            schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest() if schedule is not None else None,
            hinted_examples=mixture_count, teacher_model="same frozen base with public strategy" if frozen_teacher is not None else None,
            input_tokens_rank0=trainer.input_tokens_seen, target_tokens_rank0=trainer.target_tokens_seen,
            teacher_input_tokens_rank0=trainer.teacher_input_tokens_seen,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            source_export_meta=json.loads(Path(args.data + ".meta.json").read_text()),
        ), indent=2) + "\n")


if __name__ == "__main__":
    main()
