from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from learning_harness.analysis import export_sft, summarize
from learning_harness.common import digest, file_sha, read_jsonl, write_jsonl
from learning_harness.manifests import CORE_TYPES, prepare
from learning_harness.runner import rollout, validate_protocol
from learning_harness.support import SUPPORT_PREFIX, canonicalize_action, injection_step, messages_for, parse_action
from learning_harness.teacher import BudgetLedger
from learning_harness.teacher import build_teacher
from learning_harness.api import ChatClient


def configuration(tmp_path, rows, arms=None):
    manifest = tmp_path / "tasks.jsonl"
    write_jsonl(manifest, rows)
    config = json.loads(Path("configs/pilot.json").read_text())
    config.update(manifest=str(manifest), teacher_bank=None, arms=arms or ["none"], rollout_seeds=[11], max_steps=10)
    return config


def test_no_support_on_heldout_or_evaluation(tmp_path):
    row = dict(task_id="test", purpose="final_evaluation", partition="test_unseen")
    config = configuration(tmp_path, [row], ["early"])
    with pytest.raises(ValueError, match="H0"):
        validate_protocol(config, [row])
    config["arms"] = ["none"]
    config["teacher_bank"] = "a-bank.jsonl"
    with pytest.raises(ValueError, match="H0"):
        validate_protocol(config, [row])


def test_support_is_deterministic_and_expires():
    config = json.loads(Path("configs/pilot.json").read_text())["support"]
    first = injection_step("random_timing", "task-1", 11, config)
    assert first == injection_step("random_timing", "task-1", 11, config)
    assert 1 <= first <= 8
    with_hint = messages_for("observation", [], "advice")
    without_hint = messages_for("observation", [])
    assert any(m["content"].startswith(SUPPORT_PREFIX) for m in with_hint)
    assert not any(m["content"].startswith(SUPPORT_PREFIX) for m in without_hint)
    assert parse_action('{"action": "look"}') == ("look", False)
    assert parse_action('{"action": ["look", "inventory"]}')[1]
    assert canonicalize_action("put apple 1 in/on countertop 1") == "move apple 1 to countertop 1"
    assert canonicalize_action("put apple 1 on countertop 1") == "move apple 1 to countertop 1"
    assert canonicalize_action("take apple 1 from countertop 1") == "take apple 1 from countertop 1"


def test_budget_fails_before_an_excess_request(tmp_path):
    settings = dict(input_cny_per_million=10, output_cny_per_million=10,
                    max_calls=1, max_cost_cny=1)
    ledger = BudgetLedger(tmp_path / "ledger.json", settings)
    token = ledger.reserve([{"role": "user", "content": "hello"}], 100)
    ledger.settle(token, dict(input_tokens=10, output_tokens=10))
    with pytest.raises(ValueError, match="exhausted"):
        ledger.reserve([], 100)
    assert len(json.loads((tmp_path / "ledger.json").read_text())["calls"]) == 1


def test_manifest_keeps_task_groups_together(tmp_path):
    data_root = tmp_path / "data"
    for split in ("train", "valid_seen", "valid_unseen"):
        for task_type in CORE_TYPES:
            for group in range(5):
                for trial in range(2):
                    folder = data_root / "json_2.1.1" / split / f"{task_type}-{group}" / f"trial-{trial}"
                    folder.mkdir(parents=True)
                    (folder / "traj_data.json").write_text(json.dumps({"task_type": task_type}))
                    (folder / "game.tw-pddl").write_text('{"solvable": true}')
    prepare(data_root, tmp_path / "manifests", per_type=1)
    owners = {}
    for name in ("pilot", "dev", "probe", "train"):
        for row in read_jsonl(tmp_path / "manifests" / f"{name}.jsonl"):
            assert owners.setdefault(row["task_group"], name) == name
    assert len(read_jsonl(tmp_path / "manifests/pilot.jsonl")) == 4


class FakeEnvironment:
    """Used only for orchestration invariants, never reported as ALFWorld scores."""
    def __init__(self, game, seed, max_steps):
        self.turn = 0

    def reset(self):
        return "public observation"

    def step(self, action):
        self.turn += 1
        return "public next observation", self.turn == 2, self.turn == 2

    def close(self):
        pass


class FakeClient:
    model = "test-only"

    def complete(self, messages, **kwargs):
        return dict(text='{"action": "look"}', input_tokens=20, output_tokens=5, response_model="test-only")


def test_runner_resume_and_training_export(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "game").write_text("fixture")
    row = dict(task_id="one", purpose="development", partition="train", task_group="group",
               task_type="pick_and_place_simple", game_relpath="game", game_sha256=file_sha(root / "game"))
    config = configuration(tmp_path, [row], ["none", "early"])
    bank = tmp_path / "bank.jsonl"
    write_jsonl(bank, [dict(task_id="one", hint="use public observations", public_reset_sha256=digest("public observation"))])
    config["teacher_bank"] = str(bank)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "run"
    for _ in range(2):
        rollout(config_path, root, output, client=FakeClient(), environment_factory=FakeEnvironment)
    assert len(read_jsonl(output / "episodes.jsonl")) == 2
    assert summarize(output, bootstrap=100)["complete"]
    exported = tmp_path / "sft.jsonl"
    export_sft(output, "early", exported)
    assert len(read_jsonl(exported)) == 2
    assert all(not m["content"].startswith(SUPPORT_PREFIX)
               for row in read_jsonl(exported) for m in row["messages"])
    # A partial crashed attempt must not contaminate a later completed episode.
    steps = read_jsonl(output / "steps.jsonl")
    orphan = dict(next(r for r in steps if r["arm"] == "early"), attempt_id="crashed-attempt")
    write_jsonl(output / "steps.jsonl", [orphan] + steps)
    export_sft(output, "early", exported)
    assert len(read_jsonl(exported)) == 2
    config["temperature"] = 0.3
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="different config"):
        rollout(config_path, root, output, client=FakeClient(), environment_factory=FakeEnvironment)


def test_development_episodes_cannot_be_exported_for_training(tmp_path):
    (tmp_path / "run.json").write_text('{"fingerprint": "fixture"}')
    write_jsonl(tmp_path / "episodes.jsonl", [dict(task_id="dev", rollout_seed=1, arm="none",
                                                    partition="dev", success=True, status="completed")])
    write_jsonl(tmp_path / "steps.jsonl", [])
    with pytest.raises(ValueError, match="No successful training"):
        export_sft(tmp_path, "none", tmp_path / "forbidden.jsonl")


def test_training_masks_only_final_assistant_target():
    spec = importlib.util.spec_from_file_location("train_lora", "scripts/train_lora.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Tokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            result = []
            for message in messages:
                result += [10 if message["role"] == "user" else 20] + list(message["content"].encode())
            if add_generation_prompt:
                result += [20]
            return result

    row = dict(partition="train", messages=[dict(role="user", content="prompt"), dict(role="assistant", content="look")])
    encoded = module.encode_example(Tokenizer(), row, 100)
    assert sum(label != -100 for label in encoded["labels"]) == 4
    row["partition"] = "probe"
    with pytest.raises(ValueError, match="Only train"):
        module.encode_example(Tokenizer(), row, 100)


def test_probe_and_final_tasks_cannot_trigger_teacher_calls(tmp_path):
    for partition, purpose in (("probe", "development"), ("test_unseen", "final_evaluation")):
        manifest = tmp_path / "tasks.jsonl"
        write_jsonl(manifest, [dict(task_id="reserved", partition=partition, purpose=purpose)])
        with pytest.raises(ValueError, match="restricted"):
            build_teacher(manifest, tmp_path, "nonexistent-config.json", tmp_path / "bank.jsonl", tmp_path / "ledger.json")
    assert not (tmp_path / "bank.jsonl").exists()


def test_endpoint_requires_real_usage_and_hides_error_body(monkeypatch):
    import io
    import urllib.error
    import urllib.request
    monkeypatch.setenv("TEST_BASE", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("TEST_MODEL", "test-only")
    client = ChatClient(dict(base_url_env="TEST_BASE", model_env="TEST_MODEL", api_key_env="TEST_KEY"))
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: io.StringIO(json.dumps(
        {"choices": [{"message": {"content": "hello"}}]})))
    with pytest.raises(ValueError, match="actual"):
        client.complete([])

    def failed_request(*a, **k):
        raise urllib.error.HTTPError("http://local", 401, "private-body", {}, io.BytesIO(b"test-secret"))
    monkeypatch.setattr(urllib.request, "urlopen", failed_request)
    with pytest.raises(RuntimeError) as error:
        client.complete([])
    assert str(error.value) == "Model endpoint returned HTTP 401"
