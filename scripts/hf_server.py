#!/usr/bin/env python3
"""Small serial local inference server for the pilot; no vLLM dependency."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--revision", default="cdbee75f17c01a7cc42f958dc650907174af0554")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--served-model-name")
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device visible")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                                                torch_dtype=torch.bfloat16, trust_remote_code=False,
                                                attn_implementation="sdpa").to(f"cuda:{args.gpu}").eval()
    served_name = args.served_model_name or args.model
    served_revision = args.revision
    if args.adapter:
        from peft import PeftModel
        adapter_hash = hashlib.sha256((args.adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False).eval()
        served_revision += "/adapter-sha256:" + adapter_hash

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, value):
            payload = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path in ("/health", "/v1/models"):
                self.send_json(200, dict(model=served_name, revision=served_revision, ready=True))
            else:
                self.send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 256_000:
                    raise ValueError("request size")
                body = json.loads(self.rfile.read(length))
                if body["model"] != served_name:
                    raise ValueError("model mismatch")
                budget = int(body.get("max_tokens", 128))
                if not 0 < budget <= 1024:
                    raise ValueError("output budget")
                tokens = tokenizer.apply_chat_template(body["messages"], add_generation_prompt=True,
                                                        tokenize=True, return_tensors="pt", return_dict=True)
                tokens = {key: value.to(model.device) for key, value in tokens.items()}
                prompt_length = tokens["input_ids"].shape[-1]
                if prompt_length + budget > args.context:
                    raise ValueError("context budget")
                torch.manual_seed(int(body.get("seed", 0)))
                temperature = float(body.get("temperature", 0))
                kwargs = dict(max_new_tokens=budget, do_sample=temperature > 0,
                              pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
                if temperature > 0:
                    kwargs.update(temperature=temperature, top_p=0.9)
                with torch.inference_mode():
                    output = model.generate(**tokens, **kwargs)
                completion = output[0, prompt_length:]
                content = tokenizer.decode(completion, skip_special_tokens=True)
                self.send_json(200, dict(model=served_name,
                    choices=[dict(index=0, message=dict(role="assistant", content=content),
                                  finish_reason="length" if len(completion) == budget else "stop")],
                    usage=dict(prompt_tokens=prompt_length, completion_tokens=len(completion),
                               total_tokens=prompt_length + len(completion))))
            except Exception as exc:
                # No prompts, request bodies, or private keys in error logs.
                self.send_json(400, dict(error=type(exc).__name__))

        def log_message(self, *_):
            pass

    print(json.dumps(dict(ready=True, host="127.0.0.1", port=args.port, model=served_name,
                          revision=served_revision, gpu=args.gpu)), flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
