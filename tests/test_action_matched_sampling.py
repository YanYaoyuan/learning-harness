from collections import Counter

import pytest

from learning_harness.action_matched_sampling import action_matched_schedule


def fixture():
    # The reference omits an eligible action family and repeats one content.
    tasks = ['a'] * 7 + ['b'] * 5
    actions = ['go', 'go', 'go', 'move', 'move', 'move', 'heat',
               'go', 'go', 'move', 'move', 'move']
    content = ['g0', 'g0', 'g1', 'm0', 'm1', 'm2', 'h0',
               'bg0', 'bg1', 'bm0', 'bm1', 'bm2']
    scores = [.1, .3, .5, .8, .2, .6, .01, .7, .1, .9, .3, .2]
    reference = [0, 3, 7, 9, 0, 4, 8, 9, 3, 0]
    return tasks, actions, content, scores, reference


@pytest.mark.parametrize('seed', [101, 202, 303, 404])
def test_every_task_action_position_and_within_stratum_support_are_preserved(seed):
    tasks, actions, content, nll, reference = fixture()
    strata = list(zip(tasks, actions))
    for policy in ('random', 'min_nll'):
        result = action_matched_schedule(tasks, actions, content, reference, seed, policy, nll)
        assert [strata[i] for i in result] == [strata[i] for i in reference]
        assert Counter(tasks[i] for i in result) == Counter(tasks[i] for i in reference)
        for s in set(strata[i] for i in reference):
            assert len({content[i] for i in result if strata[i] == s}) == len({content[i] for i in reference if strata[i] == s})
        assert not any(actions[i] == 'heat' for i in result)
        assert result == action_matched_schedule(tasks, actions, content, reference, seed, policy, nll)


def test_nll_selection_averages_duplicate_copies_and_preserves_rare_placement_slots():
    tasks, actions, content, nll, reference = fixture()
    nll[0], nll[1], nll[2] = 0., 1., .3
    result = action_matched_schedule(tasks, actions, content, reference, 101, 'min_nll', nll)
    # Averaging duplicate copies makes g0 (.5) worse than g1 (.3).
    # Taking the first/minimum duplicate score instead would choose g0 (0.).
    assert {content[i] for i in result if tasks[i] == 'a' and actions[i] == 'go'} == {'g1'}
    assert {content[i] for i in result if tasks[i] == 'a' and actions[i] == 'move'} == {'m1', 'm2'}
    assert {content[i] for i in result if tasks[i] == 'b' and actions[i] == 'move'} == {'bm2'}


def test_invalid_reference_and_conflicting_content_labels_fail():
    tasks, actions, content, nll, reference = fixture()
    with pytest.raises(ValueError):
        action_matched_schedule(tasks, actions, content, reference + [-1], 101, 'random')
    with pytest.raises(ValueError):
        action_matched_schedule(tasks, actions, content, reference, 101, 'min_nll')
    actions[1] = 'move'
    with pytest.raises(ValueError):
        action_matched_schedule(tasks, actions, content, reference, 101, 'random')
