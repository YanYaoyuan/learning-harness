from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from stage3_interim_report import paired


def records(sample=101):
    result = []
    for source in ('api', 'generic'):
        for arm, score, nll in (('random', .3, .4), ('min_nll', .1, .7)):
            result.append(dict(case=dict(initialization_seed=101, sampling_seed=sample,
                                         source=source, arm=arm, repeat=0),
                               control=dict(initial_parameter_sha256='same'),
                               exposure=dict(task_exposures={'task':768}, task_order_sha256='order',
                                             processed_examples=768, unique_task_content_examples=256),
                               mean_success=score, mean_nll=nll))
    return result


def test_unpaired_cells_and_conditional_repeats_cannot_change_paired_means():
    group = records()
    partial = records(202)[:-1]
    repeat = deepcopy(group[0])
    repeat['case']['repeat'] = 1
    repeat['mean_success'] = 1.0
    keys, rows = paired(group + partial + [repeat])
    assert keys == [(101, 101)]
    assert len(rows) == 2
    assert all(r['success_difference_pp'] == pytest.approx(-20) for r in rows)
    assert all(r['nll_difference'] == pytest.approx(.3) for r in rows)


@pytest.mark.parametrize('field,value', [('task_exposures', {'other':768}),
                                       ('task_order_sha256', 'different'),
                                       ('processed_examples', 256),
                                       ('unique_task_content_examples', 128)])
def test_mismatched_task_or_content_controls_fail_before_reporting(field, value):
    group = records()
    group[1]['exposure'][field] = value
    with pytest.raises(ValueError):
        paired(group)


def test_duplicate_cells_and_changed_initialization_are_rejected():
    group = records()
    with pytest.raises(ValueError):
        paired(group + [deepcopy(group[0])])
    group[1]['control']['initial_parameter_sha256'] = 'changed'
    with pytest.raises(ValueError):
        paired(group)
