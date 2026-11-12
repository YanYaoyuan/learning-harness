from copy import deepcopy

import pytest

from learning_harness.controlled_study import completed_policy_pairs


def fixture():
    cases = [
        dict(
            name=f"{init}-{source}-{policy}",
            initialization_seed=init,
            sampling_seed=101,
            source=source,
            policy=policy,
            repeat=0,
        )
        for init in (101, 202)
        for source in ("api", "generic")
        for policy in ("random", "min_nll")
    ]
    records = [dict(case=c, complete=True, smoke=False) for c in cases]
    return dict(cases=cases), records


def test_partial_cell_and_repeats_do_not_enter_primary_snapshot():
    frozen, records = fixture()
    repeat = dict(frozen["cases"][0], name="repeat", repeat=1)
    frozen["cases"].append(repeat)
    records = records[:7] + [dict(case=repeat, complete=True, smoke=False)]
    keys, cases = completed_policy_pairs(frozen, records)
    assert keys == [(101, 101)]
    assert len(cases) == 4 and all(c["repeat"] == 0 for c in cases)


@pytest.mark.parametrize("change", ["duplicate", "mismatch", "incomplete"])
def test_corrupt_completed_inventory_is_rejected(change):
    frozen, records = fixture()
    records = deepcopy(records)
    if change == "duplicate":
        records.append(records[0])
    elif change == "mismatch":
        records[0]["case"]["policy"] = "other"
    else:
        records[0]["complete"] = False
    with pytest.raises(ValueError):
        completed_policy_pairs(frozen, records)
