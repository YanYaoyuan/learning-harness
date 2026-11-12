"""Dedicated base-only serial serving for the WebShop newline-stop protocol."""

import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import uuid

from .webshop_factory import stream_sha

IMPLEMENTATION = "webshop-base-line-stop-v1"


def model_runtime():
    return dict(
        python=sys.version,
        packages={
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "tokenizers", "safetensors")
        },
    )


def verify_base_model(directory, revision):
    directory = Path(directory)
    manifest = json.loads((directory / "download-manifest.json").read_text())
    if (
        manifest["repo"] != "Qwen/Qwen3-4B-Instruct-2507"
        or manifest["revision"] != revision
    ):
        raise ValueError("Wrong base-model provenance")
    entries = manifest["files"]
    names = [entry["path"] for entry in entries]
    if len(set(names)) != len(names) or any(Path(n).name != n for n in names):
        raise ValueError("Invalid base-model manifest paths")
    weights = {n for n in names if n.endswith(".safetensors")}
    required = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    }
    if not weights or not required.issubset(names):
        raise ValueError("Incomplete base-model manifest")
    actual_weights = {p.name for p in directory.glob("*.safetensors")}
    if (
        actual_weights != weights
        or list(directory.glob("*.bin"))
        or any(directory.glob("adapter*"))
    ):
        raise ValueError("Unexpected model or adapter weights")
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    if set(index["weight_map"].values()) != weights:
        raise ValueError("Weight index differs from the manifest")
    for entry in entries:
        path = directory / entry["path"]
        if path.stat().st_size != entry["bytes"] or stream_sha(path) != entry["sha256"]:
            raise ValueError("Base-model bytes differ: " + entry["path"])
    return dict(
        model=manifest["repo"],
        revision=revision,
        adapters=None,
        manifest_sha256=stream_sha(directory / "download-manifest.json"),
        weights_sha256=hashlib.sha256(
            json.dumps(
                {
                    entry["path"]: entry["sha256"]
                    for entry in entries
                    if entry["path"] in weights
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    )


def validate_request(body, alias):
    allowed = {"model", "messages", "max_tokens", "temperature", "seed", "stop"}
    if not isinstance(body, dict) or set(body) - allowed or body.get("model") != alias:
        raise ValueError("Unsupported request or wrong model")
    budget = body.get("max_tokens")
    if type(budget) is not int or not 0 < budget <= 100:
        raise ValueError("Output budget must be an integer in [1,100]")
    temperature = body.get("temperature")
    if (
        type(temperature) not in (float, int)
        or not math.isfinite(temperature)
        or temperature != 0
        or body.get("stop") != ["\n"]
    ):
        raise ValueError("This protocol requires greedy decoding and newline stop")
    if type(body.get("seed")) is not int:
        raise ValueError("Explicit integer seed required")
    messages = body.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or [m.get("role") for m in messages if isinstance(m, dict)]
        != ["system", "user"]
        or any(
            set(m) != {"role", "content"} or not isinstance(m["content"], str)
            for m in messages
        )
    ):
        raise ValueError("Expected explicit system and user text messages")
    if len(messages[1]["content"]) > 6400:
        raise ValueError("User prompt exceeds protocol character budget")
    return messages, budget


def handler_class(engine):
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
                self.send_json(200, engine.health)
            else:
                self.send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 128000:
                    raise ValueError("Request size")
                body = json.loads(self.rfile.read(length))
                messages, budget = validate_request(body, engine.health["model"])
                result = engine.complete(messages, budget, body["seed"])
                self.send_json(
                    200,
                    dict(
                        id="webshop-" + uuid.uuid4().hex,
                        model=engine.health["model"],
                        choices=[
                            dict(
                                index=0,
                                message=dict(role="assistant", content=result["text"]),
                                finish_reason=result["finish_reason"],
                            )
                        ],
                        usage=dict(
                            prompt_tokens=result["input_tokens"],
                            completion_tokens=result["output_tokens"],
                            total_tokens=result["input_tokens"]
                            + result["output_tokens"],
                        ),
                    ),
                )
            except Exception as error:
                # Requests, goals, response bodies and exception text stay out of logs.
                self.send_json(400, dict(error=type(error).__name__))

        def log_message(self, *_):
            pass

    return Handler


def line_stop_criterion(tokenizer, prompt_length, base_class):
    class StopAtNewline(base_class):
        def __call__(self, input_ids, scores, **kwargs):
            return "\n" in tokenizer.decode(
                input_ids[0, prompt_length:], skip_special_tokens=True
            )

    return StopAtNewline()


class BaseEngine:
    def __init__(self, directory, revision, alias):
        identity = verify_base_model(directory, revision)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Exactly one dedicated visible CUDA GPU is required")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                directory,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .to("cuda:0")
            .eval()
        )
        self.health = dict(
            identity,
            base_model=identity["model"],
            model=alias,
            ready=True,
            implementation=IMPLEMENTATION,
            pid=os.getpid(),
            context_tokens=8192,
            max_output_tokens=100,
            stop=["\n"],
            dtype="bfloat16",
            attention="sdpa",
            runtime=model_runtime(),
        )

    def complete(self, messages, budget, seed):
        from transformers import StoppingCriteria, StoppingCriteriaList

        tokens = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        tokens = {key: value.to(self.model.device) for key, value in tokens.items()}
        length = tokens["input_ids"].shape[-1]
        if length + budget > 8192:
            raise ValueError("Context token budget exceeded")
        self.torch.manual_seed(seed)
        criterion = line_stop_criterion(self.tokenizer, length, StoppingCriteria)
        with self.torch.inference_mode():
            output = self.model.generate(
                **tokens,
                max_new_tokens=budget,
                do_sample=False,
                stopping_criteria=StoppingCriteriaList([criterion]),
                pad_token_id=self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id,
            )
        completion = output[0, length:]
        decoded = self.tokenizer.decode(completion, skip_special_tokens=True)
        stopped = "\n" in decoded or len(completion) < budget
        # Count every generated token, including a newline-spanning final token.
        return dict(
            text=decoded.split("\n", 1)[0],
            input_tokens=length,
            output_tokens=len(completion),
            finish_reason="stop" if stopped else "length",
        )


def serve(engine, port):
    HTTPServer(("127.0.0.1", port), handler_class(engine)).serve_forever()
