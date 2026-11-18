from collections import Counter
import random

import pytest

from learning_harness.task_matched_sampling import allocate_support,matched_schedule


def reference(length,budget,seed):
    rng=random.Random(seed)
    values=[]
    while len(values)<budget:
        order=list(range(length));rng.shuffle(order)
        values.extend(order[:budget-len(values)])
    return values


def test_capped_allocation_preserves_rare_tasks_and_exact_support():
    caps={str(i):n for i,n in enumerate([2,7,13,30,40,100,200,329])}
    quotas=allocate_support(caps)
    assert sum(quotas.values())==256
    assert all(1<=quotas[k]<=caps[k] for k in caps)
    assert allocate_support(dict(reversed(list(caps.items()))))==quotas


@pytest.mark.parametrize('seed',[101,202,303,404])
def test_new_policies_match_every_task_position_and_select_exact_support(seed):
    tasks=[f'task{i}' for i,n in enumerate([2,7,13,30,40,100,200,329]) for _ in range(n)]
    ref=reference(len(tasks),768,seed)
    nll=[float(i) for i in range(len(tasks))]
    random_schedule=matched_schedule(tasks,ref,seed,'random')
    selected=matched_schedule(tasks,ref,seed,'min_nll',nll)
    for values in (random_schedule,selected):
        assert len(values)==768 and len(set(values))==256
        assert [tasks[i] for i in values]==[tasks[i] for i in ref]
        assert Counter(tasks[i] for i in values)==Counter(tasks[i] for i in ref)
    for task in set(tasks):
        candidates=[i for i,t in enumerate(tasks) if t==task]
        actual=sorted(i for i in set(selected) if tasks[i]==task)
        assert actual==candidates[:len(actual)]


def test_random_support_varies_with_seed_and_missing_scores_are_rejected():
    tasks=['a']*400+['b']*332
    a=matched_schedule(tasks,reference(732,768,101),101,'random')
    b=matched_schedule(tasks,reference(732,768,202),202,'random')
    assert set(a)!=set(b)
    with pytest.raises(ValueError,match='NLL'):
        matched_schedule(tasks,reference(732,768,101),101,'min_nll')


def test_duplicate_messages_cannot_reduce_content_support_in_either_policy():
    tasks=['a']*400+['b']*332
    keys=[str(i//2) for i in range(len(tasks))]
    scores=[float(i) for i in range(len(tasks))]
    ref=reference(732,768,101)
    for policy in ('random','min_nll'):
        values=matched_schedule(tasks,ref,101,policy,scores,content_keys=keys)
        assert len(set(values))==256
        assert len({(tasks[i],keys[i]) for i in values})==256
        assert [tasks[i] for i in values]==[tasks[i] for i in ref]
