import hashlib
import json
from pathlib import Path
import sys

import pytest

from learning_harness.action_control_audit import audit_recipe, digest
from learning_harness.action_matched_sampling import action_matched_schedule


def fixture():
    actions = ['go to cabinet 1', 'go to cabinet 2', 'move apple 1 to shelf 1', 'move apple 2 to shelf 1']
    rows = [dict(partition='train', task_id='a', messages=[dict(role='user', content=f'public state {i}'),
             dict(role='assistant', content=json.dumps(dict(action=a)))]) for i,a in enumerate(actions)]
    content = [hashlib.sha256(json.dumps(r['messages'], sort_keys=True, ensure_ascii=False).encode()).hexdigest() for r in rows]
    scores = [.9, .1, .4, .2]
    reference = dict(policy='random', source='api', sampling_seed=101, indices=[0,2,0,3,0])
    indices = action_matched_schedule(['a']*4, ['go','go','move','move'], content,
                                     reference['indices'], 101, 'min_nll', scores)
    order = [('a','go'), ('a','move'), ('a','go'), ('a','move'), ('a','go')]
    recipe = dict(policy='min_nll', source='api', sampling_seed=101, indices=indices,
                  schedule_sha256=digest(indices), presentations=5, unique_task_content_examples=3,
                  task_action_order_sha256=digest(order), reference_task_action_order_sha256=digest(order),
                  strata=[dict(task_id='a', action_family='go', presentations=3, distinct_support=1),
                          dict(task_id='a', action_family='move', presentations=2, distinct_support=2)])
    return recipe, rows, scores, reference


def test_positive_audit_retains_action_and_content_budgets():
    recipe, rows, scores, reference = fixture()
    result = audit_recipe(recipe, rows, scores, reference)
    assert result['processed_examples'] == 5
    assert result['unique_task_content_examples'] == 3
    assert result['regenerated_schedule_verified']
    assert recipe['indices'][0] == 1


def test_same_action_counts_cannot_hide_wrong_nll_content():
    recipe, rows, scores, reference = fixture()
    recipe['indices'] = [0 if i == 1 else i for i in recipe['indices']]
    recipe['schedule_sha256'] = digest(recipe['indices'])
    with pytest.raises(ValueError, match='regenerated'):
        audit_recipe(recipe, rows, scores, reference)


def test_nontraining_content_is_rejected():
    recipe, rows, scores, reference = fixture()
    rows[0]['partition'] = 'dev'
    with pytest.raises(ValueError, match='training'):
        audit_recipe(recipe, rows, scores, reference)


@pytest.mark.parametrize('change', ['source', 'order', 'support'])
def test_misreported_reference_order_and_group_support_are_rejected(change):
    recipe, rows, scores, reference = fixture()
    if change == 'source':
        reference['source'] = 'generic'
    elif change == 'order':
        recipe['task_action_order_sha256'] = 'wrong'
    else:
        recipe['strata'][0]['distinct_support'] = 2
    with pytest.raises(ValueError):
        audit_recipe(recipe, rows, scores, reference)


def test_worker_refuses_training_while_stage3_is_running(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    import action_control_worker as worker
    parent = tmp_path/'parent'
    parent.mkdir()
    (parent/'execution.json').write_text(json.dumps(dict(status='running')))
    monkeypatch.setattr(worker.parent, 'ROOT', parent)
    def unexpected(*args, **kwargs):
        pytest.fail('Training command must not run')
    monkeypatch.setattr(worker.engine, 'command', unexpected)
    with pytest.raises(ValueError, match='Stage 3 must finish'):
        worker.run_case(tmp_path/'followup', 'case', 0)
