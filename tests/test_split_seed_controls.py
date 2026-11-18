import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace

import pytest


def load():
    path = Path(__file__).resolve().parents[1] / "scripts/train_lora_split_seeds.py"
    spec = importlib.util.spec_from_file_location("split_seed_controls", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def schedule(length, budget, seed):
    return random.Random(seed).sample(range(length), budget)


def test_sampling_is_independent_of_initialization_argument():
    module = load()
    controlled = module.schedule_for_sampling_seed(schedule, 101)
    assert controlled(732, 256, 101) == controlled(732, 256, 202)
    other = module.schedule_for_sampling_seed(schedule, 202)
    assert other(732, 256, 101) != controlled(732, 256, 101)


def test_nonfixed_sampler_is_rejected():
    with pytest.raises(ValueError, match="fixed-budget"):
        load().validate_config(dict(max_steps=64, global_batch_size=4))


def test_initialization_seed_is_forwarded_to_the_frozen_trainer():
    args = SimpleNamespace(data="data", model="model", revision="revision", config="config", out="out",
                           initialization_seed=202, sampling_seed=101)
    forwarded = load().training_arguments(args)
    assert forwarded[forwarded.index("--seed") + 1] == "202"
    assert forwarded[forwarded.index("--method") + 1] == "target_sft"
