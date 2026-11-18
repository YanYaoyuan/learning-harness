import pytest

from learning_harness.action_matched_sampling import action_matched_schedule
from learning_harness.nll_range_sampling import range_schedule


def test_extremes_and_middle_select_distinct_content_rank():
    tasks, actions, content = ["t"] * 5, ["go"] * 5, list("abcde")
    scores = [0.1, 0.2, 0.7, 1.0, 9.0]
    selected = {
        p: range_schedule(tasks, actions, content, [0] * 3, 101, p, scores)
        for p in ("low", "middle", "high")
    }
    assert set(selected["low"]) == {0}
    assert set(selected["middle"]) == {2}
    assert set(selected["high"]) == {4}


def test_duplicate_rows_do_not_weight_the_middle_center():
    # Distinct values are 0, 2, 10; duplicating zero must not move the median.
    assert set(
        range_schedule(
            ["t"] * 6,
            ["go"] * 6,
            list("aaaabc"),
            [0] * 4,
            101,
            "middle",
            [0, 0, 0, 0, 2, 10],
        )
    ) == {4}


def test_task_action_positions_and_content_quotas_are_fixed():
    tasks, actions = ["a"] * 4 + ["b"] * 3, ["go"] * 4 + ["take"] * 3
    content, scores, reference = list("abcdxyz"), [1, 2, 3, 4, 8, 6, 4], [0, 4, 1, 4, 0]
    for policy in ("random", "low", "middle", "high"):
        result = range_schedule(tasks, actions, content, reference, 3, policy, scores)
        assert [tasks[i] for i in result] == [tasks[i] for i in reference]
        assert len({content[i] for i in result if tasks[i] == "a"}) == 2
        assert len({content[i] for i in result if tasks[i] == "b"}) == 1
    assert range_schedule(
        tasks, actions, content, reference, 3, "low", scores
    ) == action_matched_schedule(
        tasks, actions, content, reference, 3, "min_nll", scores
    )


def test_invalid_scores_are_rejected():
    with pytest.raises(ValueError, match="Finite"):
        range_schedule(["t"], ["go"], ["x"], [0], 1, "high", [float("nan")])
