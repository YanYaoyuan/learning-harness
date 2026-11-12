import hashlib
from http.server import HTTPServer
import json
from threading import Thread
import urllib.request

import pytest

from learning_harness.webshop_agent import LineStopClient
from learning_harness.webshop_serving import (
    handler_class,
    line_stop_criterion,
    validate_request,
    verify_base_model,
)


def request():
    return dict(
        model="base-only",
        messages=[
            dict(role="system", content="system"),
            dict(role="user", content="public task"),
        ],
        max_tokens=100,
        temperature=0,
        seed=11,
        stop=["\n"],
    )


@pytest.mark.parametrize(
    "change",
    [
        {"stop": []},
        {"temperature": 0.1},
        {"max_tokens": 101},
        {"max_tokens": True},
        {"adapter": "some-weights"},
        {"model": "adapter"},
        {"seed": None},
        {"messages": [None, {}]},
    ],
)
def test_server_refuses_protocol_drift(change):
    with pytest.raises(ValueError):
        validate_request(dict(request(), **change), "base-only")


def test_line_stop_decodes_completion_only_including_multitoken_context():
    class Tokens:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, index):
            row, columns = index
            assert row == 0
            return self.values[columns]

    class Tokenizer:
        def decode(self, values, **kwargs):
            return "".join(values)

    criterion = line_stop_criterion(Tokenizer(), 2, object)
    assert not criterion(Tokens(["prompt\n", "prompt", "think[", "plan]"]), None)
    assert criterion(Tokens(["prompt\n", "prompt", "think[", "plan]\nextra"]), None)


def test_real_http_client_sends_stop_and_preserves_usage(monkeypatch):
    class Engine:
        health = dict(model="base-only", adapters=None, ready=True, revision="revision")
        seen = []

        def complete(self, messages, budget, seed):
            self.seen.append((messages, budget, seed))
            return dict(
                text="think[plan]",
                input_tokens=17,
                output_tokens=5,
                finish_reason="stop",
            )

    engine = Engine()
    server = HTTPServer(("127.0.0.1", 0), handler_class(engine))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost,::1")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost,::1")
    monkeypatch.setenv("TEST_WEBSHOP_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("TEST_WEBSHOP_MODEL", "base-only")
    monkeypatch.delenv("TEST_WEBSHOP_KEY", raising=False)
    try:
        client = LineStopClient(
            dict(
                base_url_env="TEST_WEBSHOP_URL",
                model_env="TEST_WEBSHOP_MODEL",
                api_key_env="TEST_WEBSHOP_KEY",
            )
        )
        response = client.complete(request()["messages"], max_tokens=100, seed=11)
        assert response["text"] == "think[plan]" and response["output_tokens"] == 5
        assert engine.seen[0][1:] == (100, 11)
        with urllib.request.urlopen(
            client.base_url.removesuffix("/v1") + "/health"
        ) as reply:
            assert json.load(reply)["adapters"] is None
        with pytest.raises(RuntimeError, match="HTTP 400"):
            client.complete(request()["messages"], max_tokens=101, seed=11)
        assert len(engine.seen) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_base_manifest_checks_actual_bytes_and_rejects_adapters(tmp_path):
    files = {
        "config.json": "{}",
        "generation_config.json": "{}",
        "tokenizer.json": "{}",
        "tokenizer_config.json": "{}",
        "model.safetensors.index.json": json.dumps(
            {"weight_map": {"x": "model.safetensors"}}
        ),
        "model.safetensors": "fixed-base-bytes",
    }
    for name, content in files.items():
        (tmp_path / name).write_text(content)
    manifest = dict(
        repo="Qwen/Qwen3-4B-Instruct-2507",
        revision="revision",
        files=[
            dict(
                path=name,
                bytes=len(content.encode()),
                sha256=hashlib.sha256(content.encode()).hexdigest(),
            )
            for name, content in files.items()
        ],
    )
    (tmp_path / "download-manifest.json").write_text(json.dumps(manifest))
    identity = verify_base_model(tmp_path, "revision")
    assert identity["adapters"] is None and len(identity["weights_sha256"]) == 64
    (tmp_path / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="Unexpected"):
        verify_base_model(tmp_path, "revision")
    (tmp_path / "adapter_config.json").unlink()
    (tmp_path / "model.safetensors").write_text("altered-base-data")
    with pytest.raises(ValueError, match="bytes differ"):
        verify_base_model(tmp_path, "revision")
