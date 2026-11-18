import pytest

from learning_harness.global_confirmation import aggregate, matrix, validate_queries


def test_reuses_every_initialization_and_sampling_cell_without_cherry_picking():
    cases = matrix()
    assert len(cases) == len({c["name"] for c in cases}) == 97
    assert cases[0]["name"] == "base"
    assert sum(c["policy"] == "gradient_global" for c in cases) == 32
    assert {c["gpu"] for c in cases} == {0}


def test_confirmation_rejects_development_role_and_group_overlap():
    rows = [
        dict(
            task_id=str(i),
            task_group=str(i),
            fresh_role="confirmation",
            official_split="train",
            partition="fresh_confirmation",
            task_type="pick_and_place_simple"
            if i < 64
            else "pick_clean_then_place_in_recep",
        )
        for i in range(128)
    ]
    validate_queries(rows, [], [])
    with pytest.raises(ValueError, match="isolation"):
        validate_queries(rows, [dict(task_group="1")], [])
    rows[0]["fresh_role"] = "development"
    with pytest.raises(ValueError, match="isolation"):
        validate_queries(rows, [], [])


def test_aggregate_pairs_sampling_before_initializations_and_requires_all_models():
    records = []
    for case in matrix():
        success = 0.75 if case["policy"] == "gradient_global" else 0.5
        records.append(dict(case=case, episodes=128, mean_success=success))
    result = aggregate(records)
    assert result["contrasts"]["global_random"]["mean_difference"] == 0.25
    assert len(result["contrasts"]["global_random"]["per_initialization"]) == 8
    assert result["contrasts"]["global_random"]["initialization_sd"] == 0
    with pytest.raises(ValueError, match="full 97"):
        aggregate(records[:-1])
    with pytest.raises(ValueError, match="full 97"):
        aggregate(records + records[:1])
