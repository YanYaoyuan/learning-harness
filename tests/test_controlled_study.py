from copy import deepcopy
from pathlib import Path
import sys

import pytest

from learning_harness.controlled_study import (
    audit_episodes,
    paired_effects,
    validate_plan,
    validate_reservation,
)


def plan():
    cases = [
        dict(
            name=f"{s}-{p}",
            source=s,
            policy=p,
            initialization_seed=101,
            sampling_seed=101,
            repeat=0,
            gpu=g,
            schedule=p + ".json",
        )
        for s, g in [("api", 0), ("generic", 1)]
        for p in ("random", "min_nll")
    ]
    return dict(
        protocol="task-action-controls-v1",
        teacher_calls=0,
        official_tests_used=False,
        primary_metric="mean_success",
        primary_contrast="min_nll_minus_random",
        gpus=[0, 1],
        cap_gpu_hours=1,
        allocation_seconds=1200,
        minimum_allocation_seconds=600,
        planned_episodes=256,
        cases=cases,
    )


def test_paired_plan_counts_actual_models_and_repeats():
    p = plan()
    for c in list(p["cases"]):
        p["cases"].append(dict(c, name=c["name"] + "-repeat", repeat=1))
    p["planned_episodes"] = 512
    result = validate_plan(p)
    assert result == dict(
        primary_seed_cells=[(101, 101)], planned_models=8, repeated_models=4
    )


@pytest.mark.parametrize(
    "change", ["missing", "gpu", "duplicate", "source", "negative_cost"]
)
def test_unpaired_or_misaccounted_plans_fail(change):
    p = plan()
    if change == "missing":
        p["cases"].pop()
    elif change == "gpu":
        p["cases"][-1]["gpu"] = 0
    elif change == "duplicate":
        p["cases"].append(dict(p["cases"][0], name="duplicate"))
    elif change == "source":
        p["cases"] = p["cases"][:2]
    else:
        p["preparatory_gpu_seconds"] = -1
    p["planned_episodes"] = 64 * len(p["cases"])
    with pytest.raises(ValueError):
        validate_plan(p)


def result_rows():
    return [
        dict(
            c,
            mean_success=0.2 if c["policy"] == "random" else 0.3,
            mean_nll=0.8 if c["policy"] == "random" else 0.7,
        )
        for c in plan()["cases"]
    ]


def test_paired_direction_and_conditional_repeat_exclusion():
    rows = result_rows()
    rows.append(dict(rows[1], repeat=1, mean_success=0.99, mean_nll=100))
    effects = paired_effects(rows)
    for source in ("api", "generic"):
        success = effects["mean_success"]["nll_minus_random"][source]
        nll = effects["mean_nll"]["nll_minus_random"][source]
        assert success["mean_difference"] == pytest.approx(0.1)
        assert nll["mean_difference"] == pytest.approx(-0.1)
        assert success["wins"] == nll["wins"] == 1


def test_incomplete_and_nonfinite_results_cannot_be_silently_averaged():
    rows = result_rows()
    with pytest.raises(ValueError):
        paired_effects(rows[:-1])
    rows[0]["mean_success"] = float("nan")
    with pytest.raises(ValueError):
        paired_effects(rows)


def episode_fixture():
    manifest = [
        dict(
            task_id=t,
            task_group=t,
            official_split="train",
            purpose="development",
            partition="probe",
        )
        for t in ("a", "b")
    ]
    episodes = [
        dict(
            task_id=t,
            task_group=t,
            partition="probe",
            arm="none",
            rollout_seed=11,
            status="completed",
            success=t == "a",
        )
        for t in ("a", "b")
    ]
    return episodes, manifest


def test_raw_success_is_recomputed_from_real_boolean_outcomes():
    episodes, manifest = episode_fixture()
    assert audit_episodes(episodes, manifest) == 0.5


@pytest.mark.parametrize("change", ["duplicate", "condition", "seed"])
def test_wrong_task_condition_or_environment_seed_cannot_enter_report(change):
    episodes, manifest = episode_fixture()
    if change == "duplicate":
        episodes[1] = deepcopy(episodes[0])
    elif change == "condition":
        episodes[0]["arm"] = "generic"
    else:
        episodes[0]["rollout_seed"] = 23
    with pytest.raises(ValueError):
        audit_episodes(episodes, manifest)


@pytest.mark.parametrize("change", [None, "pid", "allocation", "freeze", "status"])
def test_worker_must_own_its_durable_reservation(change):
    ledger = dict(
        status="running",
        freeze_sha256="sha",
        open={"case": dict(pid=10, gpu=0, allocation=1200)},
    )
    if change == "pid":
        ledger["open"]["case"]["pid"] = 11
    elif change == "allocation":
        ledger["open"]["case"]["allocation"] = 1800
    elif change == "freeze":
        ledger["freeze_sha256"] = "other"
    elif change == "status":
        ledger["status"] = "failed"
    if change:
        with pytest.raises(ValueError):
            validate_reservation(ledger, "case", 10, 0, 1200, "sha")
    else:
        validate_reservation(ledger, "case", 10, 0, 1200, "sha")


def test_gpu_preflight_refuses_existing_compute_without_stopping_it(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_action_controls as runner

    values = iter(["0, GPU-a\n1, GPU-b\n", "GPU-a, 123\n"])
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: next(values))
    with pytest.raises(ValueError, match="already has a compute process"):
        runner.gpu_preflight([0, 1])
