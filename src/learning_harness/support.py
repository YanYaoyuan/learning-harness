from __future__ import annotations

import random

from .common import digest

ARMS = {"none", "early", "random_timing", "delayed"}
SUPPORT_PREFIX = "[TRAINING SUPPORT] "

SYSTEM_PROMPT = """You act in a text household environment. Use only the public observations.
Return exactly one JSON object: {"action": "one environment command"}.
Available command forms include: look; inventory; go to RECEPTACLE; open RECEPTACLE;
close RECEPTACLE; take OBJECT from RECEPTACLE; move OBJECT to RECEPTACLE;
clean OBJECT with RECEPTACLE; heat OBJECT with RECEPTACLE; cool OBJECT with RECEPTACLE;
use OBJECT; examine OBJECT. Use names and numbers actually observed.
To inspect a receptacle named in the room overview, first use 'go to RECEPTACLE'.
'look' does not travel. Do not use 'examine RECEPTACLE' as navigation.
For placement, the canonical command is 'move OBJECT to RECEPTACLE', for example:
{"action": "move apple 1 to countertop 1"}.
An invalid action uses one turn. Discover missing information with environment actions.
Do not output multiple actions, Markdown, or a final answer instead of an action."""


def injection_step(arm, task_id, seed, config):
    if arm not in ARMS:
        raise ValueError(f"Unknown support arm: {arm}")
    if arm == "none":
        return None
    if arm == "early":
        return 0
    if arm == "delayed":
        return int(config["delayed_step"])
    generator = random.Random(int(digest([task_id, seed, "support-timing"]), 16))
    return generator.randint(int(config["random_first_step"]), int(config["random_last_step"]))


def messages_for(initial, history, hint=None, history_turns=12):
    # Support is reconstructed every call and expires explicitly; it is not
    # accidentally retained in an ever-growing chat transcript.
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": initial}]
    if hint:
        messages.append({"role": "user", "content": SUPPORT_PREFIX + hint})
    for event in history[-history_turns:]:
        messages += [{"role": "assistant", "content": event["response"]},
                     {"role": "user", "content": event["observation"]}]
    return messages


def parse_action(response):
    import json
    try:
        value = json.loads(response)
        action = value["action"]
        if not isinstance(action, str) or not action.strip() or "\n" in action or len(action) > 256:
            raise ValueError
        return action.strip(), False
    except (ValueError, KeyError, TypeError):
        return "__invalid_action__", True


def canonicalize_action(action):
    """Public lexical compatibility only; never consults environment state."""
    import re
    match = re.fullmatch(r"put (.+?) (?:in/on|in|on) (.+)", action)
    return f"move {match.group(1)} to {match.group(2)}" if match else action

