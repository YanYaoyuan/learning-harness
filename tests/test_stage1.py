import importlib.util
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_stage1_crossed_seeds as runner
import stage1_variance_report as report


def test_crossed_plan_balances_hardware_and_repeats():
    cases = runner.plan()
    assert len(cases) == 34
    assert len({c["name"] for c in cases}) == 34
    for source in ("api", "generic"):
        grid = [c for c in cases if c["source"] == source and c["repeat"] == 0]
        assert len(grid) == 16
        assert sum(c["gpu"] == 0 for c in grid) == 8
        repeat = next(c for c in cases if c["source"] == source and c["repeat"] == 1)
        original = next(c for c in grid if c["initialization_seed"] == 101 and c["sampling_seed"] == 101)
        assert repeat["gpu"] == original["gpu"]


def test_budget_reservations_bound_concurrent_jobs():
    assert runner.allocation_seconds(21600, 19000, 1200) == 1200
    assert runner.allocation_seconds(21600, 20000, 1200) == 0
    assert runner.allocation_seconds(21600, 20900, 0) == 700
    assert runner.allocation_seconds(21600, 22000, 0) == 0


def test_variance_decomposition_factor_orientation_and_zero():
    x = np.arange(4.)
    initialization = report.components(np.repeat(x[:, None], 4, axis=1))
    sampling = report.components(np.repeat(x[None, :], 4, axis=0))
    assert initialization["initialization_share"] == 1
    assert initialization["sampling_share"] == 0
    assert sampling["sampling_share"] == 1
    assert report.components(np.full((4, 4), .37))["sampling_share"] is None


def test_interaction_is_not_mislabelled_as_sampling():
    x = np.array([-1., -1., 1., 1.])
    fit = report.components(np.outer(x, x))
    assert fit["sampling_variance_raw"] < 0
    assert fit["sampling_share"] == 0
    assert fit["interaction_plus_noise_variance"] > 0


def test_allocation_and_sensitivity_are_descriptive():
    assert "recommended" in report.allocation({"sampling_share": .3}, {"sampling_share": None})
    assert "not indicated" in report.allocation({"sampling_share": .299}, {"sampling_share": None})
    result = report.sensitivity(np.arange(16.).reshape(4, 4))
    assert len(result["fits"]) == 8
    assert "not a confidence interval" in result["method"]
