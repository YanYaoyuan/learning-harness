#!/usr/bin/env python3
"""Local batched inference with independent per-request sampling streams."""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path


def make_sampler(torch, seeds, temperature, top_p):
    from transformers import LogitsProcessor, TemperatureLogitsWarper, TopPLogitsWarper

    class IndependentSampler(LogitsProcessor):
        def __init__(self):
            self.generators = None

        def __call__(self, input_ids, scores):
            if self.generators is None:
                self.generators = [torch.Generator(device=scores.device).manual_seed(seed) for seed in seeds]
            scores = TemperatureLogitsWarper(temperature)(input_ids, scores)
            scores = TopPLogitsWarper(top_p)(input_ids, scores)
            probabilities = torch.softmax(scores.float(), dim=-1)
            chosen = torch.stack([torch.multinomial(p, 1, generator=g) for p, g in zip(probabilities, self.generators)])
            return torch.full_like(scores, -float("inf")).scatter_(1, chosen, 0)

    return IndependentSampler()


def completion_length(ids, eos_ids):
    for index, token in enumerate(ids):
        if token in eos_ids:
            return index + 1
    return len(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--revision", default="cdbee75f17c01a7cc42f958dc650907174af0554")
    adapters = parser.add_mutually_exclusive_group()
    adapters.add_argument("--adapter", type=Path)
    adapters.add_argument("--composition", type=Path)
    parser.add_argument("--served-model-name", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--port", type=int, default=19380)
    parser.add_argument("--max-batch", type=int, default=8)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    revision = args.revision
    if args.composition:
        from learning_harness.adaptation import load_composition
        model,revision=load_composition(args.composition,args.model,args.revision)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=False).to("cuda:0").eval()
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False).eval()
            revision += "/adapter-sha256:" + hashlib.sha256((args.adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, code, value):
            payload = json.dumps(value).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path != "/health":
                self.send_json(404, {"error": "not found"})
                return
            self.send_json(200, dict(ready=True, model=args.served_model_name, revision=revision,
                                    implementation="batch-independent-sampling-v1", max_batch=args.max_batch))

        def do_POST(self):
            if self.path != "/v1/batch/chat/completions":
                self.send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 2_000_000:
                    raise ValueError("Request size")
                body = json.loads(self.rfile.read(length))
                requests = body["requests"]
                if body["model"] != args.served_model_name or not 0 < len(requests) <= args.max_batch:
                    raise ValueError("Model or batch size")
                max_tokens = int(body["max_tokens"])
                temperature = float(body["temperature"])
                if not 0 < max_tokens <= 128 or not 0 <= temperature <= 2:
                    raise ValueError("Generation budget")
                rendered = [tokenizer.apply_chat_template(r["messages"], tokenize=False,
                            add_generation_prompt=True) for r in requests]
                tokens = tokenizer(rendered, add_special_tokens=False, padding=True, return_tensors="pt")
                if tokens["input_ids"].shape[1] + max_tokens > 8192:
                    raise ValueError("Context budget")
                prompt_counts = tokens["attention_mask"].sum(dim=1).tolist()
                tokens = {key: value.to(model.device) for key, value in tokens.items()}
                kwargs = dict(max_new_tokens=max_tokens, do_sample=False,
                              pad_token_id=tokenizer.pad_token_id)
                if temperature > 0:
                    kwargs["logits_processor"] = [make_sampler(torch, [int(r["seed"]) for r in requests], temperature, 0.9)]
                with torch.inference_mode():
                    generated = model.generate(**tokens, **kwargs)
                completions = generated[:, tokens["input_ids"].shape[1]:].tolist()
                answers = []
                for prompt_count, ids in zip(prompt_counts, completions):
                    count = completion_length(ids, eos_ids)
                    answers.append(dict(text=tokenizer.decode(ids[:count], skip_special_tokens=True),
                        input_tokens=prompt_count, output_tokens=count, response_model=args.served_model_name))
                self.send_json(200, dict(results=answers, revision=revision,
                    implementation="batch-independent-sampling-v1",
                    padded_prompt_tokens=int(tokens["input_ids"].numel())))
            except Exception as exc:
                self.send_json(400, {"error": type(exc).__name__})

        def log_message(self, *_):
            pass

    print(json.dumps(dict(ready=True, revision=revision, port=args.port)), flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
