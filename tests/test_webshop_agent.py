import pytest

from learning_harness.webshop_agent import run_episode
from learning_harness.webshop_public import PublicView, Transition


class Episode:
    def __init__(self, large=False):
        self.steps = []
        self.closed = False
        self.large = large

    def reset(self):
        return PublicView(
            "Public task." + ("x" * 20000 if self.large else ""), True, ()
        )

    def step(self, action):
        self.steps.append(action)
        if action == "search[shirt]":
            return Transition(
                PublicView("Public results.", False, ("product-a",)), 0.0, False
            )
        if action == "click[product-a]":
            return Transition(
                PublicView("Public item.", False, ("buy now",)), 0.0, False
            )
        assert action == "click[Buy Now]"
        return Transition(PublicView("Episode finished.", False, ()), 1.0, True)

    def close(self):
        self.closed = True


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, messages, **kwargs):
        self.requests.append((messages, kwargs))
        return dict(text=next(self.responses), input_tokens=5, output_tokens=3)


def test_thought_is_paid_and_only_physical_actions_reach_environment():
    episode = Episode()
    client = Client(
        ["think[plan]", "search[shirt]", "click[product-a]", "Action: click[Buy Now]"]
    )
    events = []
    result = run_episode(
        client, episode, "Published example.\n", on_event=events.append
    )
    assert result["success"] and result["score"] == 1
    assert (
        result["model_calls"] == 4
        and result["physical_actions"] == 3
        and result["thought_decisions"] == 1
    )
    assert result["input_tokens"] == 20 and result["output_tokens"] == 12
    assert (
        events[0]["observation"] == "OK." and not events[0]["physical_action_executed"]
    )
    assert episode.closed


def test_invalid_and_invisible_actions_consume_calls_without_oracle_repair():
    episode = Episode()
    client = Client(
        ["narrative instead of action", "click[unseen target]", "think[wait]"]
    )
    result = run_episode(client, episode, "Example.", max_calls=3)
    assert (
        result["invalid_format_decisions"] == 1
        and result["invalid_control_decisions"] == 1
    )
    assert result["model_calls"] == 3 and result["output_tokens"] == 9
    assert not episode.steps and not result["success"] and episode.closed


def test_budget_does_not_request_an_unused_completion():
    episode = Episode()
    client = Client(["think[first]", "think[second]", "unrequested"])
    result = run_episode(client, episode, "Example.", max_calls=2)
    assert result["status"] == "decision_budget" and len(client.requests) == 2
    assert next(client.responses) == "unrequested"


def test_character_budget_preserves_prefix_and_includes_actual_usage():
    episode = Episode(large=True)
    client = Client(["think[plan]"])
    prefix = "Fixed public example.\n"
    run_episode(client, episode, prefix, max_calls=1, max_context_characters=500)
    prompt = client.requests[0][0][1]["content"]
    assert (
        len(prompt) == 500 and prompt.startswith(prefix) and prompt.endswith("Action:")
    )


def test_endpoint_overrun_is_rejected_and_episode_is_closed():
    episode = Episode()

    class Overrun(Client):
        def complete(self, *args, **kwargs):
            return dict(text="think[plan]", input_tokens=5, output_tokens=101)

    with pytest.raises(ValueError):
        run_episode(Overrun([]), episode, "Example.", max_new_tokens=100)
    assert episode.closed
