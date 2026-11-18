from copy import deepcopy
import json

import pytest

from learning_harness.research_aggregates import aggregate_report


def fixture():
    cases = [
        dict(
            name=f"private-condition-{source}-{policy}",
            source=source,
            policy=policy,
            initialization_seed=101,
            sampling_seed=101,
            repeat=0,
            gpu=gpu,
            schedule="local-only.json",
        )
        for source, gpu in (("api", 0), ("generic", 1))
        for policy in ("random", "min_nll")
    ]
    plan = dict(
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
    rows = [
        dict(
            c,
            mean_success=0.2 if c["policy"] == "random" else 0.3,
            mean_nll=0.8,
            cool_success=0.2 if c["policy"] == "random" else 0.3,
            heat_success=0.2 if c["policy"] == "random" else 0.3,
            task_id="PRIVATE_TASK_SENTINEL",
        )
        for c in cases
    ]
    report = dict(
        complete=True,
        cases=4,
        actual_evaluation_episodes=256,
        freeze_sha256="a" * 64,
        teacher_calls=0,
        official_tests_used=False,
        results=rows,
        exposure={"PRIVATE_TASK_SENTINEL": 9},
        note="PRIVATE_FREE_TEXT",
    )
    return report, plan


def test_export_has_no_task_trajectory_case_names_or_free_text():
    report, plan = fixture()
    result = aggregate_report(report, plan, "a" * 64)
    serialized = json.dumps(result)
    for forbidden in (
        "PRIVATE_",
        "private-condition",
        "task_id",
        "exposure",
        "local-only",
    ):
        assert forbidden not in serialized
    assert result["raw_records_uploaded"] is False
    assert result["min_nll_minus_random"]["api"][
        "success_difference_pp"
    ] == pytest.approx(10)


@pytest.mark.parametrize(
    "change", ["incomplete", "missing", "wrong_condition", "wrong_mean", "nonfinite"]
)
def test_incomplete_or_mismatched_report_is_not_exported(change):
    report, plan = fixture()
    report = deepcopy(report)
    if change == "incomplete":
        report["complete"] = False
    elif change == "missing":
        report["results"].pop()
    elif change == "wrong_condition":
        report["results"][0]["sampling_seed"] = 202
    elif change == "wrong_mean":
        report["results"][0]["mean_success"] = 0.9
    else:
        report["results"][0]["mean_nll"] = float("nan")
    with pytest.raises(ValueError):
        aggregate_report(report, plan, "a" * 64)
