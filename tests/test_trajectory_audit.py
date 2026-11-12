import json

import pytest

from learning_harness.trajectory_audit import episode_behavior


def fixture(actions, observations, success=False):
    events = []
    for i, (action, observation) in enumerate(zip(actions, observations)):
        invalid = action is None
        raw = "__invalid_action__" if invalid else action
        canonical = raw.replace("put apple 1 on shelf 1", "move apple 1 to shelf 1")
        events.append(
            dict(
                attempt_id="run",
                task_id="task",
                arm="none",
                rollout_seed=11,
                step=i,
                response="not JSON" if invalid else json.dumps(dict(action=action)),
                raw_action=raw,
                action=canonical,
                invalid_action=invalid,
                observation=observation,
                success=success and i == len(actions) - 1,
                support_active=False,
                input_tokens=10,
                output_tokens=4,
            )
        )
    episode = dict(
        attempt_id="run",
        task_id="task",
        arm="none",
        rollout_seed=11,
        status="completed",
        support_injected=False,
        steps=len(events),
        invalid_actions=sum(e["invalid_action"] for e in events),
        success=success,
        input_tokens=10 * len(events),
        output_tokens=4 * len(events),
    )
    return episode, events


def test_format_failure_and_literal_environment_no_effect_are_separate():
    episode, events = fixture(
        ["go to cabinet 1", None], ["Nothing happens.", "Nothing happens."]
    )
    result = episode_behavior(episode, events, 40)
    assert result["format_error_fraction"] == 0.5
    assert result["literal_no_effect_fraction"] == 1
    assert result["any_format_error"] == 1


def test_public_put_alias_is_counted_as_a_placement_attempt():
    episode, events = fixture(
        ["take apple 1 from table 1", "put apple 1 on shelf 1"],
        ["You pick up the apple.", "You put the apple on the shelf."],
        True,
    )
    result = episode_behavior(episode, events, 40)
    assert result["take_attempt"] == result["move_attempt"] == result["success"] == 1
    assert result["format_error_fraction"] == 0


@pytest.mark.parametrize("change", ["missing", "format", "tokens", "identity"])
def test_trace_inconsistency_is_rejected(change):
    episode, events = fixture(["look", "look"], ["A room.", "A room."])
    if change == "missing":
        events.pop()
    elif change == "format":
        events[0]["invalid_action"] = True
    elif change == "tokens":
        events[0]["input_tokens"] = 99
    else:
        events[0]["attempt_id"] = "other"
    with pytest.raises(ValueError):
        episode_behavior(episode, events, 40)
