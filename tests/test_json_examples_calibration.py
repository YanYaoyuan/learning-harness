import importlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path("scripts").absolute()))


def test_worker_binding_changes_only_actor_and_local_configuration(monkeypatch):
    runner = importlib.import_module("run_json_examples_calibration")
    original = runner.original
    for key in ("ROOT", "config", "validate"):
        monkeypatch.setattr(original, key, getattr(original, key))
    monkeypatch.setattr(
        runner.benchmark, "actor_messages", runner.benchmark.actor_messages
    )
    old_worker = original.worker
    runner.bind_worker()
    assert original.worker is old_worker
    assert original.ROOT == runner.ROOT
    config = original.config(runner.CASE)
    assert config["method"] == "json_direct"
    assert config["executor"] == "json_public_examples_v1"
    assert config["max_trials"] == 1 and config["max_calls"] == 40
    initial = "Public room. Your task is to: clean an apple and put it on table."
    messages = runner.benchmark.actor_messages(initial, [], [], "json_direct", 12)
    assert "Two public examples" in messages[0]["content"]
    assert runner.benchmark.parse_line('{"action":"look"}', "json_direct") == (
        "look",
        False,
        False,
    )
    with pytest.raises(ValueError, match="Unexpected"):
        runner.benchmark.actor_messages(initial, [], ["memory"], "json_direct", 12)
