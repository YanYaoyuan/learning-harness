import json

import pytest

from learning_harness.json_demo_executor import actor_messages, examples
from learning_harness.support import SYSTEM_PROMPT, messages_for


@pytest.mark.parametrize("family", ["put", "clean"])
def test_examples_keep_actions_but_remove_thoughts_and_preserve_json(family):
    text = examples(family)
    actions = [
        json.loads(line[2:])["action"]
        for line in text.splitlines()
        if line.startswith("> ")
    ]
    assert actions and "think:" not in text
    assert any(action.startswith("move ") for action in actions)
    assert all(not action.startswith("put ") for action in actions)


def test_current_json_history_and_instruction_are_unchanged():
    initial = "Public room. Your task is to: clean some apple and put it in sidetable."
    history = [dict(response='{"action":"look"}', observation="You see apple 1.")]
    messages = actor_messages(initial, history)
    baseline = messages_for(initial, history)
    assert messages[0]["content"].startswith(SYSTEM_PROMPT)
    assert messages[1:] == baseline[1:]
    assert "clean apple" in messages[0]["content"]
    assert all("TRAINING SUPPORT" not in m["content"] for m in messages)


def test_unregistered_family_is_not_silently_routed():
    with pytest.raises(ValueError):
        actor_messages("Your task is to: heat an apple", [])
