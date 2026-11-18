"""Observable trajectory diagnostics; never infer hidden environment validity."""

from collections import Counter

from .support import canonicalize_action, parse_action


def episode_behavior(episode, events, call_limit):
    events = sorted(events, key=lambda e: e["step"])
    count = len(events)
    if (
        not count
        or count != episode["steps"]
        or [e["step"] for e in events] != list(range(count))
    ):
        raise ValueError("Missing, duplicated or unordered episode steps")
    if (
        count > call_limit
        or episode["status"] != "completed"
        or episode["support_injected"]
    ):
        raise ValueError("Wrong cap, incomplete episode or remaining assistance")
    format_errors, no_effect, actions, families = 0, 0, [], []
    for event in events:
        if any(
            event[k] != episode[k]
            for k in ("attempt_id", "task_id", "arm", "rollout_seed")
        ):
            raise ValueError("Step belongs to a different episode")
        if event["support_active"]:
            raise ValueError("Assistance was present in a withdrawal evaluation")
        raw, invalid = parse_action(event["response"])
        if (
            raw != event["raw_action"]
            or canonicalize_action(raw) != event["action"]
            or invalid != event["invalid_action"]
        ):
            raise ValueError(
                "Logged action or format flag disagrees with protocol parser"
            )
        format_errors += invalid
        no_effect += event["observation"].strip() == "Nothing happens."
        action = event["action"]
        actions.append(action)
        families.append("go" if action.startswith("go to ") else action.split()[0])
    if (
        format_errors != episode["invalid_actions"]
        or bool(events[-1]["success"]) != episode["success"]
        or sum(e["input_tokens"] for e in events) != episode["input_tokens"]
        or sum(e["output_tokens"] for e in events) != episode["output_tokens"]
    ):
        raise ValueError("Episode aggregate differs from step records")
    return dict(
        success=int(episode["success"]),
        steps=count,
        take_attempt=int("take" in families),
        move_attempt=int("move" in families),
        any_format_error=int(format_errors > 0),
        format_error_fraction=format_errors / count,
        literal_no_effect_fraction=no_effect / count,
        hit_call_limit=int(count == call_limit),
        exact_command_repetition_fraction=1 - len(set(actions)) / count,
        action_family_counts=dict(Counter(families)),
    )
