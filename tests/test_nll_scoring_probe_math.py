"""Run separately in the model runtime with CUDA_VISIBLE_DEVICES='' for CPU math."""

import pytest

from learning_harness.nll_scoring_probe import make_batch, masked_nll

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("padding", [0, 5])
@pytest.mark.parametrize("window", [3, None])
def test_target_nll_matches_independent_shifted_full_labels_loss(padding, window):
    torch.set_num_threads(1)
    rows = [
        dict(input_ids=[1, 2, 3, 4], attention_mask=[1] * 4, labels=[-100, -100, 3, 4]),
        dict(input_ids=[5, 6, 7], attention_mask=[1] * 3, labels=[-100, -100, 7]),
    ]
    batch = make_batch(rows, 0, padding, True)
    generator = torch.Generator(device="cpu").manual_seed(123)
    logits = torch.randn(
        2, len(batch["labels"][0]), 8, generator=generator, device="cpu"
    )
    actual = masked_nll(
        torch, logits[:, -window:] if window else logits, batch["labels"]
    )
    expected = [
        float(
            torch.nn.functional.cross_entropy(
                logits[i, :-1],
                torch.tensor(labels[1:], device="cpu"),
                ignore_index=-100,
            )
        )
        for i, labels in enumerate(batch["labels"])
    ]
    assert actual == pytest.approx(expected, abs=1e-6)
    assert not torch.cuda.is_initialized()


def test_empty_or_insufficient_prediction_windows_fail_on_real_tensors():
    logits = torch.zeros(1, 2, 8, device="cpu")
    with pytest.raises(ValueError, match="omitted"):
        masked_nll(torch, logits, [[-100, -100, 3, 4]])


@pytest.mark.parametrize("direct_loss_bias", [0.0, 0.01])
def test_complete_scoring_worker_on_explicit_cpu_standin(
    tmp_path, monkeypatch, direct_loss_bias
):
    """Actual Torch loss/journal/phase ordering; no CUDA, model weights or data."""
    import importlib.util
    import json
    import math
    import os
    from pathlib import Path

    transformers = pytest.importorskip("transformers")
    monkeypatch.syspath_prepend(str(Path("scripts").absolute()))
    from learning_harness.common import file_sha

    spec = importlib.util.spec_from_file_location(
        "probe_worker_cpu_test", "scripts/run_nll_scoring_probe.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    keys = [f"{i:064x}" for i in range(32)]
    encoded = dict(
        input_ids=[1, 2, 3, 4], attention_mask=[1] * 4, labels=[-100, -100, 3, 4]
    )
    entries = [
        dict(
            content_sha256=key,
            encoded=encoded,
            observations=[dict(source="api", index=i, cached_nll=math.log(8))],
        )
        for i, key in enumerate(keys)
    ]
    inputs = dict(
        entries=entries,
        stress_keys=keys,
        source_row_keys={"api": keys},
        replay_batches=[dict(source="api", start=0)],
        fixed_logits_window=3,
        pad_token_id=0,
    )
    (tmp_path / "inputs.private").write_text(json.dumps(inputs))
    (tmp_path / "freeze.json").write_text("{}")
    (tmp_path / "execution.json").write_text(
        json.dumps(
            dict(
                freeze_sha256=file_sha(tmp_path / "freeze.json"),
                open={runner.CASE: dict(pid=os.getpid(), gpu=1, allocation=576)},
            )
        )
    )
    directory = tmp_path / "cases" / runner.CASE
    directory.mkdir(parents=True)
    monkeypatch.setattr(
        runner,
        "validate",
        lambda root: dict(
            gpu=1, model_runtime={"cpu": True}, worker_soft_stop_seconds=560
        ),
    )
    monkeypatch.setattr(runner, "require_parent_completed", lambda: {})
    monkeypatch.setattr(
        runner, "verify_base_model", lambda *args: {"manifest": "explicit-test-fixture"}
    )
    monkeypatch.setattr(runner, "model_runtime", lambda: {"cpu": True})
    monkeypatch.setattr(runner, "selection_changes", lambda *args: [dict(standin=True)])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda *args: "EXPLICIT_CPU_STANDIN"
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    original_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        if kwargs.get("device") == "cuda:0":
            kwargs["device"] = "cpu"
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    loads = []

    class Model:
        def __init__(self, dtype):
            self.dtype = dtype

        def to(self, device):
            assert device == "cuda:0"
            return self

        def eval(self):
            return self

        def requires_grad_(self, enabled):
            assert not enabled
            return self

        def __call__(self, input_ids, labels=None, logits_to_keep=0, **kwargs):
            from types import SimpleNamespace

            length = input_ids.shape[1]
            width = min(length, logits_to_keep) if logits_to_keep else length
            logits = torch.zeros(
                input_ids.shape[0], width, 8, dtype=self.dtype, device="cpu"
            )
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].float().reshape(-1, 8),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
                return SimpleNamespace(loss=loss + direct_loss_bias)
            return SimpleNamespace(logits=logits)

    def load(*args, **kwargs):
        dtype = kwargs["torch_dtype"]
        assert kwargs["local_files_only"] and not kwargs["trust_remote_code"]
        assert kwargs["output_loading_info"]
        loads.append(dtype)
        return Model(dtype), dict(
            missing_keys=[], unexpected_keys=[], mismatched_keys=[], error_msgs=[]
        )

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load)
    runner.worker(tmp_path, 1, 576)
    result = json.loads((directory / "worker-result.json").read_text())
    assert loads == [torch.float32, torch.bfloat16]
    assert result["complete"] and result["base_model_files_after_verified"]
    assert result["stress_scores"] == 256 and result["legacy_replay_observations"] == 8
    assert result["metrics"]["optional_rescoring_gate_passed"] == (
        direct_loss_bias == 0
    )
    assert result["uniform_unique_scored"] == (32 if direct_loss_bias == 0 else 0)
    assert result["forward_attempts"] == result["completed_forwards"]
    assert result["known_padded_input_positions"] > result["known_input_tokens"] > 0
    assert result["generated_tokens"] == result["teacher_calls"] == 0
    assert not torch.cuda.is_initialized()
