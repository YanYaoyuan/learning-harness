from dataclasses import asdict

import pytest

from learning_harness.webshop_public import PublicWebShop


class Environment:
    def __init__(self):
        self.resets = []
        self.steps = []
        self.reward = 0.5
        self.done = False

    @property
    def server(self):
        raise AssertionError("Policy adapter accessed hidden environment state")

    def get_valid_actions(self):
        raise AssertionError("Privileged baseline action generator was used")

    def get_available_actions(self):
        return dict(
            has_search_bar=True,
            clickables=["search", "visible button"],
            hidden_query="DO_NOT_FORWARD",
            attributes=["DO_NOT_FORWARD"],
        )

    def reset(self, session):
        self.resets.append(session)
        return "Public instruction.", dict(hidden_goal="DO_NOT_FORWARD")

    def step(self, action):
        self.steps.append(action)
        observation = "DO_NOT_FORWARD_TERMINAL_DEBUG" if self.done else "Public page."
        return observation, self.reward, self.done, dict(hidden_goal="DO_NOT_FORWARD")


def test_policy_input_uses_only_public_observation_and_controls():
    env = Environment()
    adapter = PublicWebShop(env, [2])
    view = adapter.reset(2)
    assert asdict(view) == dict(
        observation="Public instruction.",
        can_search=True,
        clickables=("visible button",),
    )
    transition = adapter.step("search[words chosen by policy]")
    assert set(asdict(transition.view)) == {"observation", "can_search", "clickables"}
    assert "DO_NOT_FORWARD" not in str(asdict(transition.view))
    assert env.steps == ["search[words chosen by policy]"]


def test_unpermitted_goal_is_rejected_before_backend_reset():
    env = Environment()
    adapter = PublicWebShop(env, [2])
    with pytest.raises(ValueError):
        adapter.reset(0)
    with pytest.raises(ValueError):
        adapter.reset(True)
    assert env.resets == []


def test_terminal_debug_is_excluded_and_further_steps_are_rejected():
    env = Environment()
    adapter = PublicWebShop(env, [2])
    adapter.reset(2)
    env.done = True
    transition = adapter.step("click[buy now]")
    assert asdict(transition.view) == dict(
        observation="Episode finished.", can_search=False, clickables=()
    )
    assert not transition.success
    with pytest.raises(ValueError):
        adapter.step("click[buy now]")
    assert len(env.steps) == 1


def test_full_terminal_reward_is_success_but_partial_reward_is_not():
    env = Environment()
    adapter = PublicWebShop(env, [2])
    for reward, expected in [(0.5, False), (1.0, True)]:
        adapter.reset(2)
        env.reward, env.done = reward, True
        assert adapter.step("click[buy now]").success is expected


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), 1.1, -0.1])
def test_invalid_reward_is_rejected(reward):
    env = Environment()
    adapter = PublicWebShop(env, [2])
    adapter.reset(2)
    env.reward = reward
    with pytest.raises(ValueError):
        adapter.step("search[public words]")
