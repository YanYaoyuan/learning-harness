import pytest

from learning_harness.fresh_tasks import (
    FAMILIES,
    allocate_roles,
    candidate_groups,
    reset_signature,
    task_group,
)


def rows():
    return [
        dict(
            task_id=f"alfworld:train/{f}-g{i}/v{j}",
            task_group=f"train/{f}-g{i}",
            official_split="train",
            partition="train",
            task_type=f,
        )
        for f in FAMILIES
        for i in range(5)
        for j in range(2)
    ]


def test_group_exclusion_includes_alternate_game_instances():
    data = rows()
    excluded = {data[0]["task_group"]}
    by = candidate_groups(data, excluded, set())
    assert len(by[FAMILIES[0]]) == 4
    assert all(r["task_group"] not in excluded for r in by[FAMILIES[0]])
    assert len({r["task_group"] for r in by[FAMILIES[1]]}) == 5


def test_group_can_be_recovered_from_old_action_only_record():
    assert (
        task_group(dict(task_id="alfworld:train/goal-group/game-instance"))
        == "train/goal-group"
    )
    assert task_group(dict(task_id="webshop:train/1")) is None


def test_final_split_cannot_enter_candidate_inventory():
    data = rows()
    data[0]["official_split"] = "valid_unseen"
    with pytest.raises(ValueError, match="training inventory"):
        candidate_groups(data, set(), set())


def test_roles_are_group_disjoint_balanced_and_deterministic():
    pool = candidate_groups(rows(), set(), set())
    roles = allocate_roles(pool, dict(calibration=1, development=2, confirmation=2))
    assert roles == allocate_roles(
        pool, dict(calibration=1, development=2, confirmation=2)
    )
    groups = [r["task_group"] for tasks in roles.values() for r in tasks]
    assert len(groups) == len(set(groups)) == 10
    assert all(
        r["fresh_role"] == role and r["partition"] != "train"
        for role, tasks in roles.items()
        for r in tasks
    )


def test_not_enough_groups_fails_before_silently_shrinking_design():
    with pytest.raises(ValueError, match="Insufficient"):
        allocate_roles(
            candidate_groups(rows(), set(), set()),
            dict(calibration=2, development=2, confirmation=2),
        )


def test_public_signature_ignores_receptacle_order_and_goal_whitespace():
    a = "Looking quickly around you, you see a table 1, a fridge 2. Your task is to: put an apple on table."
    b = "Looking quickly around you, you see a fridge 2, a table 1. Your task is to: PUT  AN apple on table."
    assert reset_signature(a) == reset_signature(b)
    with pytest.raises(ValueError):
        reset_signature("Unrecognized observation")
