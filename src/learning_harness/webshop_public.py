"""Explicit policy-input boundary for an already configured WebShop env."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PublicView:
    observation: str
    can_search: bool
    clickables: tuple[str, ...]


@dataclass(frozen=True)
class Transition:
    view: PublicView
    reward: float
    done: bool

    @property
    def success(self):
        return self.done and self.reward == 1.0


class PublicWebShop:
    """Expose text and DOM controls; never ask a baseline for goal-based actions.

    The caller must construct the underlying env with the permitted goal pool
    before passing it here. This is a policy-data boundary, not a Python sandbox.
    Only ``view`` is sent to a policy; reward/done remain runner-side fields.
    """

    def __init__(self, environment, allowed_goal_indices):
        permitted = tuple(allowed_goal_indices)
        if (
            not permitted
            or len(set(permitted)) != len(permitted)
            or any(type(i) is not int or i < 0 for i in permitted)
        ):
            raise ValueError(
                "A nonempty explicit set of permitted goal indices is required"
            )
        self._environment = environment
        self._permitted = frozenset(permitted)
        self._done = True

    def _view(self, observation):
        if not isinstance(observation, str):
            raise ValueError("Expected an upstream text observation")
        controls = self._environment.get_available_actions()
        if type(controls["has_search_bar"]) is not bool or not isinstance(
            controls["clickables"], list
        ):
            raise ValueError("Unexpected public DOM control schema")
        if any(not isinstance(c, str) for c in controls["clickables"]):
            raise ValueError("Public button labels must be strings")
        return PublicView(
            observation=observation,
            can_search=controls["has_search_bar"],
            clickables=tuple(c for c in controls["clickables"] if c != "search"),
        )

    def reset(self, goal_index):
        if type(goal_index) is not int or goal_index not in self._permitted:
            raise ValueError("Goal is outside this runner's permitted pool")
        observation, _ = self._environment.reset(session=goal_index)
        self._done = False
        return self._view(observation)

    def step(self, action):
        if self._done:
            raise ValueError("Reset a permitted episode before requesting a step")
        if not isinstance(action, str):
            raise ValueError("Upstream actions must be strings")
        observation, reward, done, _ = self._environment.step(action)
        if (
            type(done) is not bool
            or not math.isfinite(float(reward))
            or not 0 <= float(reward) <= 1
        ):
            raise ValueError("Unexpected upstream done flag or bounded reward")
        self._done = done
        # Terminal debug HTML is not needed for another decision and can carry
        # hidden fields in upstream versions. Keep it outside the policy input.
        view = (
            PublicView("Episode finished.", False, ())
            if done
            else self._view(observation)
        )
        return Transition(view=view, reward=float(reward), done=done)
