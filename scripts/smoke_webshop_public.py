#!/usr/bin/env python3
"""Exercise the policy boundary on synthetic preview goals without model calls."""

import builtins
from dataclasses import asdict
import json
from pathlib import Path
import random
import re
import sys
import time
from unittest.mock import patch

from learning_harness.bounded_queue import save
from learning_harness.common import file_sha
from learning_harness.webshop_public import PublicWebShop


def main():
    out = Path("output/webshop-preparation/20261113/public-preview")
    receipt = out / "mechanics.json"
    if receipt.exists():
        raise ValueError("Keep the previous mechanics attempt")
    out.mkdir(parents=True, exist_ok=True)
    empty = out / "empty-human-annotations.json"
    with empty.open("x") as handle:
        handle.write("{}\n")
    reference = json.loads(
        Path(
            "output/webshop-preparation/20261112/original-reference/reference.json"
        ).read_text()
    )
    source = Path(reference["source_path"])
    code = [
        source / p
        for p in [
            "web_agent_site/engine/engine.py",
            "web_agent_site/engine/goal.py",
            "web_agent_site/envs/web_agent_text_env.py",
        ]
    ]
    before = {str(p.relative_to(source)): file_sha(p) for p in code}
    sys.path[:0] = [str(source), reference["nlp_pythonpath"]]
    import numpy as np
    import torch
    import web_agent_site.engine.engine as engine
    from web_agent_site.engine.goal import nlp
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

    if torch.cuda.is_available() or nlp.meta["name"] != "core_web_lg":
        raise ValueError("Wrong CPU/reward reference profile")
    original_human_path = Path(engine.HUMAN_ATTR_PATH).resolve()
    engine.HUMAN_ATTR_PATH = str(empty.absolute())
    original_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        if (
            isinstance(file, (str, bytes, Path))
            and Path(file).resolve() == original_human_path
        ):
            raise AssertionError("Original human annotations must stay unopened")
        return original_open(file, *args, **kwargs)

    state = dict(
        status="running",
        started_unix=time.time(),
        model_calls=0,
        gpu_used=False,
        human_goals_used=False,
        human_annotation_file_opened=False,
        runtime_human_annotation_path="empty synthetic-preview fixture",
        official_test_evaluations=0,
        full_benchmark_ready=False,
    )
    save(receipt, state)
    try:
        random.seed(11)
        np.random.seed(11)
        torch.manual_seed(11)
        with patch("builtins.open", guarded_open):
            env = WebAgentTextEnv(
                observation_mode="text",
                num_products=1000,
                human_goals=False,
                get_image=False,
            )
            public = PublicWebShop(env, [0])
            view = public.reset(0)
            fields = list(asdict(view))
            transition = public.step("search[shirt]")
            items = [
                s for s in transition.view.clickables if re.fullmatch("[a-z0-9]{10}", s)
            ]
            if not items:
                raise ValueError("No public product link after search")
            steps = [
                dict(
                    action_type="search", reward=transition.reward, done=transition.done
                )
            ]
            transition = public.step("click[" + items[0] + "]")
            steps.append(
                dict(
                    action_type="click_public_product",
                    reward=transition.reward,
                    done=transition.done,
                )
            )
            transition = public.step("click[buy now]")
            steps.append(
                dict(
                    action_type="click_buy",
                    reward=transition.reward,
                    done=transition.done,
                )
            )
            if (
                not transition.done
                or transition.view.observation != "Episode finished."
            ):
                raise ValueError("Missing terminal transition or terminal debug leaked")
            env.close()
        if before != {str(p.relative_to(source)): file_sha(p) for p in code}:
            raise ValueError("Original source bytes changed")
        state.update(
            status="public_policy_boundary_preview_passed",
            finished_unix=time.time(),
            products=1000,
            synthetic_goals=6910,
            public_view_fields=fields,
            terminal_debug_excluded=True,
            source_unchanged=True,
            source_sha256=before,
            steps=steps,
            note="Predetermined public actions on synthetic preview; not an agent-performance or full-benchmark result.",
        )
        save(receipt, state)
        print(json.dumps(state), flush=True)
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__)
        save(receipt, state)
        raise


if __name__ == "__main__":
    main()
