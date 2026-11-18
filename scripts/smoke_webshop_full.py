#!/usr/bin/env python3
"""One predetermined train-only mechanics episode on the complete catalog."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import time

from learning_harness.bounded_queue import save
from learning_harness.webshop_factory import build_full_environment, stream_sha


def main():
    out = Path("output/webshop-preparation/20261113/full-environment")
    receipt = out / "mechanics.json"
    if receipt.exists():
        raise ValueError("Preserve the existing complete-catalog mechanics attempt")
    out.mkdir(parents=True, exist_ok=True)
    state = dict(
        status="constructing_train_backend",
        started_unix=time.time(),
        pid=os.getpid(),
        code_sha256={
            p: stream_sha(p)
            for p in [
                "scripts/smoke_webshop_full.py",
                "src/learning_harness/webshop_factory.py",
                "src/learning_harness/webshop_public.py",
            ]
        },
        split="train",
        global_goal_index=1500,
        model_calls=0,
        gpu_used=False,
        official_test_evaluations=0,
        eval_episodes=0,
        full_benchmark_ready=False,
    )
    save(receipt, state)
    os.nice(10)
    episode = None
    try:
        factory = build_full_environment(
            "output/webshop-preparation/20261113/goal-identities", "train"
        )
        state["backend"] = factory.metadata
        state["status"] = "public_train_mechanics"
        save(receipt, state)
        episode = factory.episode(1500)
        view = episode.reset()
        fields = list(asdict(view))
        steps = []
        transition = episode.step("search[shirt]")
        steps.append(
            dict(action_type="search", reward=transition.reward, done=transition.done)
        )
        items = [
            s for s in transition.view.clickables if re.fullmatch("[a-z0-9]{10}", s)
        ]
        if not items:
            raise ValueError("No public product results in complete catalog")
        transition = episode.step("click[" + items[0] + "]")
        steps.append(
            dict(
                action_type="click_public_product",
                reward=transition.reward,
                done=transition.done,
            )
        )
        transition = episode.step("click[buy now]")
        steps.append(
            dict(
                action_type="click_buy", reward=transition.reward, done=transition.done
            )
        )
        if not transition.done or transition.view.observation != "Episode finished.":
            raise ValueError("Missing terminal state or terminal debug leaked")
        episode.close()
        episode = None
        # Pool/session metadata stays backend-side; no per-goal text is exported.
        if factory._server.user_sessions:
            raise ValueError("Closed episode left user sessions behind")
        state.update(
            status="full_train_environment_mechanics_passed",
            finished_unix=time.time(),
            training_mechanics_episodes=1,
            public_view_fields=fields,
            terminal_debug_excluded=True,
            backend_sessions_after_close=0,
            all_global_goal_fingerprints_verified=factory.metadata[
                "verified_global_goals"
            ],
            steps=steps,
            note="Predetermined public actions, not a model-performance result or formal benchmark evaluation.",
        )
        save(receipt, state)
        print(json.dumps(state), flush=True)
    except BaseException as error:
        state.update(
            status="failed", error_type=type(error).__name__, stopped_unix=time.time()
        )
        save(receipt, state)
        raise
    finally:
        if episode is not None:
            episode.close()


if __name__ == "__main__":
    main()
