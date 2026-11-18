"""Public-example executor preserving the student's JSON action interface."""

from functools import lru_cache
import json
import re

from .benchmark import DEMOS, public_family
from .public_demo_support import support_from_demo
from .support import canonicalize_action, messages_for


@lru_cache(maxsize=2)
def examples(family):
    if family not in ("put", "clean"):
        raise ValueError("This executor is registered only for placement and cleaning")
    source = json.loads(DEMOS.read_text())
    result = []
    for key in (f"react_{family}_1", f"react_{family}_0"):
        text = source[key]
        support_from_demo(text, key)  # Fail closed on an unrecognized source format.
        parts = re.split(r"(?m)^> ", text)
        lines = [parts[0].strip()]
        for block in parts[1:]:
            action, _, observation = block.partition("\n")
            if action.startswith("think:"):
                continue
            response = json.dumps(dict(action=canonicalize_action(action.strip())))
            lines.extend(["> " + response, observation.strip()])
        result.append("\n".join(lines))
    return "\n\n".join(result)


def actor_messages(initial, history, history_turns=12):
    messages = messages_for(initial, history, history_turns=history_turns)
    messages[0]["content"] += (
        "\nTwo public examples follow. Return one JSON action at a time using the current observations.\n"
        + examples(public_family(initial))
    )
    return messages
