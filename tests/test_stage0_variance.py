import importlib.util

import numpy as np
import pytest


def load():
    spec = importlib.util.spec_from_file_location("stage0", "scripts/stage0_variance_share.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_identical_replicates_give_zero_share():
    stage0 = load()
    d = np.array([[0.10, 0.10, 0.10], [0.30, 0.30, 0.30], [-0.20, -0.20, -0.20]])
    result = stage0.variance_share(d)
    assert result["share"] == 0.0
    assert result["share_interval"] == [0.0, 0.0]
    assert result["seed_variance"] == pytest.approx(np.var([0.10, 0.30, -0.20], ddof=1))


def test_no_seed_differences_give_full_share():
    stage0 = load()
    d = np.array([[0.0, 0.1, -0.1], [0.1, -0.1, 0.0], [-0.1, 0.0, 0.1]])
    result = stage0.variance_share(d)
    assert result["seed_variance"] == 0.0 and result["seed_variance_truncated_at_zero"]
    assert result["share"] == 1.0


def test_constant_matrix_is_undefined_and_takes_default_branch():
    stage0 = load()
    result = stage0.variance_share(np.full((3, 3), 0.25))
    assert result["share"] is None
    assert stage0.decide(result["share"], 3) == "share undefined: Stage 1 in full"


def test_known_components_are_recovered():
    stage0 = load()
    # Seed means 0, 0.2, 0.4 and a replicate pattern of -0.1, 0, 0.1 in every seed.
    d = np.array([[m - 0.1, m, m + 0.1] for m in (0.0, 0.2, 0.4)])
    result = stage0.variance_share(d)
    assert result["msw"] == pytest.approx(0.01)
    assert result["msb"] == pytest.approx(3 * 0.04)
    assert result["seed_variance"] == pytest.approx((0.12 - 0.01) / 3)
    assert result["share"] == pytest.approx(0.01 / (0.01 + (0.12 - 0.01) / 3))
    low, high = result["share_interval"]
    assert 0.0 <= low <= result["share"] <= high <= 1.0


def test_spending_rule_branches():
    stage0 = load()
    assert stage0.decide(0.50, 3).startswith("evaluation dominates")
    assert stage0.decide(0.49, 3) == "Stage 1 in full"
    assert stage0.decide(0.90, 1).startswith("inconclusive")


def test_paired_difference_weights_families_equally():
    stage0 = load()
    tasks = [f"t{i}" for i in range(4)]
    cells = {}
    for seed in stage0.B_SEEDS:
        cells["cool", "api", seed] = {t: 1 for t in tasks}
        cells["cool", "generic", seed] = {t: 0 for t in tasks}
        cells["heat", "api", seed] = {t: 0 for t in tasks}
        cells["heat", "generic", seed] = {t: 0 for t in tasks}
    d = stage0.paired_differences([cells])
    assert d.shape == (3, 1) and np.allclose(d, 0.5)
