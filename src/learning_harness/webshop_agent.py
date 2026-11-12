"""Budgeted chat port of public one-shot prompts on the native public interface."""

import re

from .api import ChatClient

SYSTEM = (
    "You interact with a shopping simulator. Return one action on one line: "
    "search[words], click[displayed control], or think[reasoning]. "
    "Choose search words yourself and click only displayed controls."
)


class LineStopClient(ChatClient):
    """Extend the frozen generic client with this protocol's fixed newline stop."""

    def __init__(self, settings):
        super().__init__(settings)
        self.extra_body = dict(self.extra_body, stop=["\n"])


def public_text(view):
    observation = view.observation.replace(" [SEP] ", "\n")
    controls = "\n[Search]" if view.can_search else ""
    if view.clickables:
        controls += "\nDisplayed controls: " + " ".join(
            "[" + s + "]" for s in view.clickables
        )
    return observation + controls


def parse_decision(text):
    if not isinstance(text, str) or not text.strip():
        return None
    line = text.strip().splitlines()[0].strip()
    line = re.sub(r"^Action:\s*", "", line, flags=re.IGNORECASE)
    match = re.fullmatch(r"(search|click|think)\[([^\r\n]*)\]", line)
    if match is None or not match.group(2).strip():
        return None
    return match.group(1), match.group(2).strip()


def run_episode(
    client,
    episode,
    prefix,
    *,
    max_calls=14,
    max_new_tokens=100,
    max_context_characters=6400,
    seed=11,
    on_event=None,
    on_request=None,
):
    """Only returned aggregate fields are suitable for cloud publication.

    Events contain task text/model responses and must remain local. The caller
    provides a dedicated base-model client after the previous GPU study ends.
    """
    if any(
        type(x) is not int or x <= 0
        for x in [max_calls, max_new_tokens, max_context_characters]
    ):
        raise ValueError(
            "Positive integer decision, output and context budgets required"
        )
    if not isinstance(prefix, str) or len(prefix) + 128 >= max_context_characters:
        raise ValueError("Prompt prefix leaves insufficient bounded history")
    calls = input_tokens = output_tokens = physical = thoughts = invalid_format = (
        invalid_control
    ) = 0
    done = False
    score = 0.0
    try:
        view = episode.reset()
        history = public_text(view) + "\n\nAction:"
        for turn in range(max_calls):
            body = prefix + history[-(max_context_characters - len(prefix)) :]
            messages = [
                dict(role="system", content=SYSTEM),
                dict(role="user", content=body),
            ]
            # Persist the attempt before the HTTP request. A lost response is an
            # unknown-cost attempt, never an unbilled zero or an automatic retry.
            if on_request:
                on_request(dict(kind="attempt", turn=turn, messages=messages))
            calls += 1
            try:
                response = client.complete(
                    messages,
                    max_tokens=max_new_tokens,
                    temperature=0.0,
                    seed=seed + turn,
                )
            except BaseException as error:
                if on_request:
                    on_request(
                        dict(kind="error", turn=turn, error_type=type(error).__name__)
                    )
                raise
            if on_request:
                on_request(dict(kind="response", turn=turn, response=response))
            if (
                any(
                    type(response.get(k)) is not int or response[k] < 0
                    for k in ["input_tokens", "output_tokens"]
                )
                or response["output_tokens"] > max_new_tokens
            ):
                raise ValueError("Endpoint usage missing or output budget exceeded")
            input_tokens += response["input_tokens"]
            output_tokens += response["output_tokens"]
            decision = parse_decision(response["text"])
            executed = False
            feedback = "Invalid action!"
            wire = (
                response["text"].strip().splitlines()[0]
                if response["text"].strip()
                else ""
            )
            if decision is None:
                invalid_format += 1
            else:
                kind, argument = decision
                wire = f"{kind}[{argument}]"
                legal = (
                    kind == "think"
                    or (kind == "search" and view.can_search)
                    or (
                        kind == "click"
                        and argument.lower() in {s.lower() for s in view.clickables}
                    )
                )
                if not legal:
                    invalid_control += 1
                elif kind == "think":
                    thoughts += 1
                    feedback = "OK."
                else:
                    transition = episode.step(wire)
                    physical += 1
                    executed = True
                    view = transition.view
                    score, done = transition.reward, transition.done
                    feedback = public_text(view)
            if on_event:
                on_event(
                    dict(
                        turn=turn,
                        messages=messages,
                        response=response["text"],
                        action=wire,
                        physical_action_executed=executed,
                        observation=feedback,
                        reward=score,
                        done=done,
                        input_tokens=response["input_tokens"],
                        output_tokens=response["output_tokens"],
                    )
                )
            if done:
                break
            history += f" {wire}\nObservation: {feedback}\n\nAction:"
        return dict(
            status="completed" if done else "decision_budget",
            success=done and score == 1.0,
            score=score,
            model_calls=calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            physical_actions=physical,
            thought_decisions=thoughts,
            invalid_format_decisions=invalid_format,
            invalid_control_decisions=invalid_control,
        )
    finally:
        episode.close()
