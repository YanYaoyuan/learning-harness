import pytest

from learning_harness.experience_value_study import (
    POLICIES,
    SOURCES,
    aggregate_cells,
    matrix,
    select,
    validate_queries,
)


def test_raw_evaluation_rejects_changed_executor_and_incomplete_usage():
    from learning_harness.experience_value_study import audit_episodes
    from learning_harness.json_demo_executor import actor_messages

    initial = "Room. Your task is to: put an apple on table."
    query = dict(task_id="t", task_group="g", task_type="pick_and_place_simple")
    row = dict(
        query,
        run_id="r",
        seed=11,
        method="json_direct",
        trials=[{}],
        calls=1,
        success=False,
        first_success=False,
        input_tokens=3,
        output_tokens=2,
    )
    event = dict(
        run_id="r",
        task_id="t",
        seed=11,
        method="json_direct",
        trial=1,
        call=1,
        phase="act",
        thought=False,
        success=False,
        input_tokens=3,
        output_tokens=2,
        invalid_format=False,
        messages=actor_messages(initial, []),
    )
    usage = dict(
        complete_usage=True,
        error_records=0,
        attempted_requests=1,
        known_input_tokens=3,
        known_output_tokens=2,
    )
    assert audit_episodes([row], [event], usage, [query])["episodes"] == 1
    with pytest.raises(ValueError, match="Incomplete"):
        audit_episodes([row], [event], dict(usage, complete_usage=False), [query])
    event["messages"][0]["content"] = "A different executor"
    with pytest.raises(ValueError, match="executor changed"):
        audit_episodes([row], [event], usage, [query])


def test_matrix_is_fully_crossed_and_baseline_never_duplicates_student():
    cases = matrix()
    assert len(cases) == 225 and len({c["name"] for c in cases}) == 225
    assert cases[1]["name"] == "base"
    cells = {
        (c["initialization_seed"], c["sampling_seed"], c["policy"], c["source"])
        for c in cases
        if c["name"] != "base"
    }
    assert len(cells) == 224
    assert {c["gpu"] for c in cases} == {0}


def test_zero_branch_excludes_foreign_adapter_export_keys():
    import torch
    from learning_harness.experience_value_study import clean_zero_branch

    state = {
        "q.lora_A.weight": torch.ones(2, 3),
        "q.lora_B.weight": torch.zeros(3, 2),
        "q.lora_B.A.weight": torch.ones(3, 2),
    }
    active, foreign = clean_zero_branch(state)
    assert len(active) == 2 and foreign == ["q.lora_B.A.weight"]
    state["q.lora_B.weight"][0, 0] = 1
    with pytest.raises(ValueError, match="nonzero"):
        clean_zero_branch(state)


def test_development_cannot_reuse_calibration_groups_or_change_roles():
    tasks = [
        dict(
            task_id=str(i),
            task_group=f"g{i}",
            task_type="pick_and_place_simple"
            if i < 64
            else "pick_clean_then_place_in_recep",
            fresh_role="development",
            partition="fresh_development",
            official_split="train",
        )
        for i in range(128)
    ]
    validate_queries(tasks, [], [])
    with pytest.raises(ValueError, match="isolation"):
        validate_queries(tasks, [dict(task_group="g1")], [])
    tasks[0]["fresh_role"] = "confirmation"
    with pytest.raises(ValueError, match="isolation"):
        validate_queries(tasks, [], [])


def selection_fixture():
    import json

    rows = [
        dict(
            task_id=f"t{i % 2}",
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
    reference = list(range(256)) * 3
    scores = [i / 300 for i in range(300)]
    return rows, reference, scores


def test_high_low_gradient_intervene_inside_same_task_action_budget():
    rows, reference, scores = selection_fixture()
    high = select(rows, scores, scores, reference, 101, "gradient_high")
    low = select(rows, scores, scores, reference, 101, "gradient_low")
    assert len(high) == len(low) == 768
    assert len(set(high)) == len(set(low)) == 256
    assert [rows[i]["task_id"] for i in high] == [rows[i]["task_id"] for i in reference]
    assert min(high) > min(low) and max(high) > max(low)


def test_global_selector_is_explicitly_not_coverage_matched():
    rows, reference, scores = selection_fixture()
    indices = select(rows, scores, scores, reference, 101, "gradient_global")
    assert len(indices) == 768 and set(indices) == set(range(44, 300))


def complete_block(offset=0):
    return [
        dict(
            case=dict(
                block=1, source=s, policy=p, initialization_seed=101, sampling_seed=101
            ),
            mean_success=0.3 + offset + (0.1 if p == "gradient_high" else 0),
        )
        for s in SOURCES
        for p in POLICIES
    ]


def test_partial_blocks_and_duplicate_cells_never_become_scientific_aggregate():
    rows = complete_block()
    with pytest.raises(ValueError, match="complete balanced"):
        aggregate_cells(rows[:-1], [1])
    with pytest.raises(ValueError):
        aggregate_cells(rows + [rows[0]], [1])
    result = aggregate_cells(rows, [1])
    assert result["contrasts"]["gradient_high_minus_random"][
        "mean_difference"
    ] == pytest.approx(0.1)
    assert not result["inferential_tests_performed"]
