"""Convert public action demonstrations into independent JSON target support."""

import json
import re

from .support import canonicalize_action, messages_for


def support_from_demo(text, key):
    blocks = re.split(r"(?m)^> ", text)
    initial = blocks[0].strip()
    if "Your task is to:" not in initial or len(blocks) < 2:
        raise ValueError("Unrecognized public demonstration")
    history, rows = [], []
    for block in blocks[1:]:
        action, separator, observation = block.partition("\n")
        if action.startswith("think:"):
            if observation.strip() not in ("", "OK."):
                raise ValueError("Thought pair carries an unexpected observation")
            continue
        if not separator or not action.strip() or not observation.strip():
            raise ValueError("Incomplete public action/observation pair")
        action = canonicalize_action(action.strip())
        response = json.dumps(dict(action=action))
        messages = messages_for(initial, history, history_turns=12)
        messages.append(dict(role="assistant", content=response))
        rows.append(
            dict(
                task_id="public-react:" + key,
                partition="train",
                source="public_react_demonstration",
                messages=messages,
                demo_key=key,
                action_index=len(history),
            )
        )
        history.append(dict(response=response, observation=observation.strip()))
    if not rows:
        raise ValueError("Public demonstration has no environment actions")
    return rows
