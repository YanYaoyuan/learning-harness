import json

import pytest

from learning_harness.experience_value_study import content_sha, select
from learning_harness.global_experience_selection import global_schedule, matrix


def pool():
    return [
        dict(
            task_id=f"t{i % 3}",
            partition="train",
            messages=[
                dict(role="user", content=str(i)),
                dict(
                    role="assistant", content=json.dumps(dict(action="go to table 1"))
                ),
            ],
        )
        for i in range(300)
    ]


def test_gradient_reference_is_exactly_the_completed_implementation():
    rows = pool()
    values = [i / 300 for i in range(300)]
    for seed in (101, 202):
        assert global_schedule(rows, seed, "gradient_global", values) == select(
            rows, values, values, list(range(256)) * 3, seed, "gradient_global"
        )


def test_global_high_selects_top_content_and_replays_each_exactly_three_times():
    from collections import Counter

    indices = global_schedule(pool(), 101, "global_high", list(range(300)))
    assert set(indices) == set(range(44, 300))
    assert set(Counter(indices).values()) == {3}


def test_duplicate_rows_do_not_change_random_content_probabilities_or_replay():
    rows = pool()
    duplicated = rows + rows[:40] * 4
    a = global_schedule(rows, 101, "global_random")
    b = global_schedule(duplicated, 101, "global_random")
    assert [content_sha(rows[i]) for i in a] == [content_sha(duplicated[i]) for i in b]
    assert a != global_schedule(rows, 202, "global_random")
    with pytest.raises(ValueError, match="256"):
        global_schedule(rows[:255], 101, "global_random")


def test_new_matrix_adds_only_the_missing_64_students():
    cases = matrix()
    assert len(cases) == len({c["name"] for c in cases}) == 64
    assert {c["policy"] for c in cases} == {"global_random", "global_high"}
    assert {c["initialization_seed"] for c in cases} == {
        101,
        202,
        303,
        404,
        505,
        606,
        707,
        808,
    }
    assert {c["gpu"] for c in cases} == {0}
