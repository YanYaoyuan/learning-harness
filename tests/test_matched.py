import importlib.util

import pytest


def module():
    spec = importlib.util.spec_from_file_location("matched_study", "scripts/matched_study.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_matching_preserves_pairs_counts_endpoints_and_shuffled_alignment():
    m = module()

    def rows(task, seed, length):
        return [dict(task_id=task, rollout_seed=seed, step=i, partition="train") for i in range(length)]

    corpora = dict(none=rows("shared", 11, 4) + rows("shared", 23, 3) + rows("only_none", 11, 2),
                   early=rows("shared", 11, 7) + rows("shared", 23, 5) + rows("only_early", 11, 8))
    matched, pairs = m.match_rows(corpora)
    assert len(matched["none"]) == len(matched["early"]) == 7
    assert len(pairs) == 2
    assert all(row["task_id"] == "shared" for arm in matched.values() for row in arm)
    for arm in matched.values():
        for seed, last in ((11, 3 if arm is matched["none"] else 6), (23, 2 if arm is matched["none"] else 4)):
            steps = {r["step"] for r in arm if r["rollout_seed"] == seed}
            assert 0 in steps and last in steps
    key = lambda r: (r["task_id"], r["rollout_seed"], r["matched_index"])
    assert [key(r) for r in matched["none"]] == [key(r) for r in matched["early"]]
    corpora["none"][0]["partition"] = "probe"
    with pytest.raises(ValueError, match="cannot consume"):
        m.match_rows(corpora)


def test_uncertainty_resamples_training_seeds_and_keeps_pairing():
    m = module()
    # All tasks move together with the training seed. Task-only resampling would
    # incorrectly report no uncertainty in the mean across these two seeds.
    assert m.crossed_interval([[-1] * 8, [1] * 8]) == [-1, 1]
    assert m.crossed_interval([[0] * 8] * 3) == [0, 0]


def test_training_budget_has_no_short_epoch_tail_updates():
    spec = importlib.util.spec_from_file_location("train_lora", "scripts/train_lora.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    schedule = trainer.fixed_schedule(5, 256, 101)
    assert len(schedule) == 256
    assert set(schedule[:5]) == set(range(5))
    assert schedule == trainer.fixed_schedule(5, 256, 101)
    assert schedule != trainer.fixed_schedule(5, 256, 202)
    assert len(set(trainer.fixed_schedule(1000, 256, 101))) == 256
