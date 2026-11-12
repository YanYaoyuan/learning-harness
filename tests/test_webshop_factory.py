from dataclasses import asdict
from types import SimpleNamespace

import pytest

from learning_harness.common import digest
from learning_harness.webshop_factory import GoalPoolGate, SplitEnvironment


def fixture():
    goals = [
        dict(instruction_text=f"Public request {i}", asin=str(i)) for i in range(1503)
    ]
    records = [
        dict(
            global_index=i,
            split="test" if i < 500 else "eval" if i < 1500 else "train",
            goal_sha256=digest(g),
            visible_instruction_sha256=digest(g["instruction_text"]),
        )
        for i, g in enumerate(goals)
    ]
    return goals, records


def restricted():
    goals, records = fixture()
    gate = GoalPoolGate(records, "train")
    selected = [g for i, g in enumerate(goals) if gate(i, g)]
    return SimpleNamespace(goals=selected, user_sessions={}), gate


class AutoResetEnvironment:
    constructions = 0

    def __init__(self, observation_mode, server, get_image, session_prefix):
        type(self).constructions += 1
        self.server, self.prefix = server, session_prefix
        self.reset(0)

    def reset(self, session):
        self.session = self.prefix + str(session)
        self.server.user_sessions[self.session] = dict(goal=self.server.goals[session])
        return self.server.goals[session]["instruction_text"], None

    def get_available_actions(self):
        return dict(has_search_bar=True, clickables=["search"])

    def step(self, action):
        return "Done debug.", 0.0, True, dict(secret="never forward")

    def close(self):
        pass


def test_filter_precedes_automatic_reset_and_global_mapping_is_exact():
    server, gate = restricted()
    factory = SplitEnvironment(server, AutoResetEnvironment, gate, {})
    episode = factory.episode(1502)
    assert all(int(s["goal"]["asin"]) >= 1500 for s in server.user_sessions.values())
    view = episode.reset()
    assert asdict(view)["observation"] == "Public request 1502"
    assert all(int(s["goal"]["asin"]) >= 1500 for s in server.user_sessions.values())
    assert set(asdict(view)) == {"observation", "can_search", "clickables"}
    episode.close()
    assert not server.user_sessions


def test_unpermitted_global_index_never_constructs_environment():
    server, gate = restricted()
    factory = SplitEnvironment(server, AutoResetEnvironment, gate, {})
    before = AutoResetEnvironment.constructions
    with pytest.raises(ValueError):
        factory.episode(0)
    assert AutoResetEnvironment.constructions == before and not server.user_sessions


def test_identity_mismatch_stops_before_filter_acceptance():
    goals, records = fixture()
    gate = GoalPoolGate(records, "train")
    with pytest.raises(ValueError):
        gate(0, dict(goals[0], instruction_text="Changed request"))
    assert gate.checked == 0


def test_incomplete_global_traversal_cannot_construct_pool():
    goals, records = fixture()
    gate = GoalPoolGate(records, "train")
    gate(0, goals[0])
    with pytest.raises(ValueError):
        SplitEnvironment(
            SimpleNamespace(goals=[], user_sessions={}), AutoResetEnvironment, gate, {}
        )


def test_closing_one_episode_keeps_other_sessions():
    server, gate = restricted()
    factory = SplitEnvironment(server, AutoResetEnvironment, gate, {})
    first = factory.episode(1500)
    first.reset()
    first_sessions = set(server.user_sessions)
    second = factory.episode(1501)
    second.reset()
    second_sessions = set(server.user_sessions) - first_sessions
    first.close()
    assert second_sessions and set(server.user_sessions) == second_sessions
    second.close()
    assert not server.user_sessions


def test_filtered_pool_order_is_verified_before_environment_construction():
    server, gate = restricted()
    server.goals.reverse()
    with pytest.raises(ValueError):
        SplitEnvironment(server, AutoResetEnvironment, gate, {})
