from collections import Counter
import json
from pathlib import Path
import signal
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_lora import fixed_schedule
from train_lora_sampling_intervention import schedule
import run_stage2_sampling as controller
from learning_harness.bounded_queue import allocation, run


@pytest.mark.parametrize("length", [721, 732])
@pytest.mark.parametrize("seed", [101, 202, 303, 404])
def test_equal_budget_shared_prefix_and_actual_membership(length, seed):
    full = schedule(fixed_schedule, length, 768, seed, "full_pass")
    replay = schedule(fixed_schedule, length, 768, seed, "subset_replay")
    old = fixed_schedule(length, 256, seed)
    assert full[:256] == replay[:256] == old
    assert len(full) == len(replay) == 768
    assert len(set(full)) == length
    assert len(set(replay)) == 256
    assert set(Counter(replay).values()) == {3}
    assert schedule(fixed_schedule, length, 4, seed, "subset_replay") == full[:4]


def test_policy_controls_preserve_hardware_and_coverage_grid():
    cases = controller.plan()
    assert len(cases) == len({c["name"] for c in cases}) == 68
    for arm in controller.ARMS:
        group = [c for c in cases if c["arm"] == arm]
        assert len(group) == 34
        assert sum(c["gpu"] == 0 for c in group) == 17
    for case in cases:
        mate = next(c for c in cases if c["arm"] != case["arm"] and
                    all(c[k] == case[k] for k in ("source", "initialization_seed", "sampling_seed", "repeat")))
        assert mate["gpu"] == case["gpu"]


def test_reservations_never_spend_concurrent_budget_twice():
    assert allocation(500, 100, 300, 300, 90) == 100
    assert allocation(500, 100, 320, 300, 90) == 0


def test_report_separates_mean_effect_from_variance_reduction():
    from stage2_sampling_report import analyze
    from run_stage1_crossed_seeds import plan, SEEDS
    rows = []
    for arm, scale in (("partial_256", 3), ("full_pass", 1), ("subset_replay", 2)):
        for case in plan():
            s = SEEDS.index(case["sampling_seed"])
            i = SEEDS.index(case["initialization_seed"])
            success = (.2 + scale*s*.01 + i*.002 if case["source"] == "api" else
                       .1 + scale*s*.005 + i*.001)
            rows.append(dict(case, arm=arm, mean_success=success, mean_nll=1-success))
    matrices, fits, effects = analyze(rows)
    paired = effects["mean_success"]["paired_api_minus_generic"]
    assert paired["mean_difference"] == pytest.approx(-.0075)
    assert paired["sampling_variance_reduction_fraction"] == pytest.approx(.75)
    assert effects["mean_success"]["api"]["mean_difference"] == pytest.approx(-.015)


def test_queue_completes_arbitrary_job_count_and_charges_elapsed(tmp_path):
    (tmp_path / "freeze.json").write_text('{}')
    cases = [dict(name=f"job{i}", gpu=i % 2) for i in range(3)]
    frozen = dict(cases=cases, cap_gpu_hours=10/3600, allocation_seconds=2,
                  minimum_allocation_seconds=.1, gpus=[0, 1])
    def command(case, gpu, seconds):
        result = tmp_path / "cases" / case["name"] / "worker-result.json"
        return [sys.executable, "-c", "import json,sys;from pathlib import Path;Path(sys.argv[1]).write_text(json.dumps({'complete':True}))", str(result)]
    old = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        ledger = run(tmp_path, frozen, command, lambda gpu: None, preparatory_seconds=.01)
    finally:
        for s, handler in old.items():
            signal.signal(s, handler)
    assert ledger["status"] == "evaluation_completed"
    assert ledger["completed_cases"] == 3
    assert not ledger["open"]
    assert ledger["gpu_seconds"] > .01


def test_queue_records_failure_without_inventing_results(tmp_path):
    (tmp_path / "freeze.json").write_text('{}')
    cases = [dict(name=f"job{i}", gpu=0) for i in range(2)]
    frozen = dict(cases=cases, cap_gpu_hours=10/3600, allocation_seconds=2,
                  minimum_allocation_seconds=.1, gpus=[0])
    old = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        ledger = run(tmp_path, frozen, lambda *args: [sys.executable, "-c", "raise SystemExit(1)"], lambda gpu: None)
    finally:
        for s, handler in old.items():
            signal.signal(s, handler)
    assert ledger["status"] == "failed"
    assert ledger["completed_cases"] == 0
    assert "job1" not in ledger["cases"]
    assert ledger["cases"]["job0"]["seconds"] > 0
