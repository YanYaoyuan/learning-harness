import importlib.util
import json
from pathlib import Path

import pytest

from learning_harness.common import digest,file_sha
from learning_harness.source_comparison import target_view,match_sources
from learning_harness.support import SUPPORT_PREFIX,SYSTEM_PROMPT


def event(call,action,observation,thought=False,invalid=False):
    return dict(call=call,turn=call-1,task_id='t',trial=1,phase='act',seed=11,
        action=action,observation=observation,thought=thought,invalid_format=invalid,
        messages=[dict(role='system',content='SECRET_EXAMPLE'),dict(role='user',content='Public room and task'),
                  dict(role='user',content=SUPPORT_PREFIX+'SECRET_HINT')])


def test_target_view_removes_source_privileged_context_and_thoughts():
    tasks={'t':dict(partition='train',official_split='train',task_group='g',task_type='place')}
    events=[event(1,'think: SECRET_THOUGHT','OK.',True),event(2,'go to table 1','Public apple 1'),
            event(3,'bad','Nothing happens.',invalid=True),event(4,'put apple 1 in/on table 1','Placed')]
    rows=target_view(events,tasks,'react')
    assert len(rows)==2 and rows[0]['messages'][0]['content']==SYSTEM_PROMPT
    serialized=json.dumps(rows)
    assert 'SECRET_' not in serialized and SUPPORT_PREFIX not in serialized and 'OK.' not in serialized
    assert 'Public apple 1' in serialized and 'Nothing happens.' in serialized
    assert json.loads(rows[-1]['messages'][-1]['content'])['action']=='move apple 1 to table 1'
    assert rows[-1]['messages'][-2]['content']=='Nothing happens.'


def test_export_rejects_query_and_multi_trial():
    t={'t':dict(partition='probe',official_split='train',task_group='g',task_type='place')}
    with pytest.raises(ValueError,match='training'):target_view([event(1,'look','Public')],t,'api')
    t['t']['partition']='train';e=event(1,'look','Public');e['trial']=2
    with pytest.raises(ValueError,match='single'):target_view([e],t,'api')


def test_pairing_matches_task_seed_exposure_and_excludes_source_only_success():
    row=lambda t,s,i:dict(task_id=t,rollout_seed=s,step=i,partition='train')
    sources={'api':[row('t',11,i) for i in range(5)]+[row('only_api',11,0)],
             'react':[row('t',11,i) for i in range(8)]+[row('t',23,0)]}
    matched,selection=match_sources(sources,{'t':dict(task_group='g')},minimum_groups=1)
    assert selection['gate_passed'] and selection['matched_examples_each']==5
    keys=lambda rs:[(r['task_id'],r['rollout_seed'],r['matched_index']) for r in rs]
    assert keys(matched['api'])==keys(matched['react'])
    assert all(r['task_id']=='t' and r['rollout_seed']==11 for r in matched['react'])
    assert not match_sources(sources,{'t':dict(task_group='g')},minimum_groups=8)[1]['gate_passed']


def test_cached_hint_visible_only_first_four_turns_and_logged_in_place(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('collector','scripts/collect_source_comparison.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    bank=tmp_path/'bank.jsonl';bank.write_text(json.dumps(dict(public_reset_sha256=digest('public'),hint='PLAN'))+'\n')
    monkeypatch.setenv('LH_TEST_URL','http://localhost/v1');monkeypatch.setenv('LH_TEST_MODEL','test')
    cfg=dict(model=dict(base_url_env='LH_TEST_URL',model_env='LH_TEST_MODEL',api_key_env='LH_TEST_KEY'),
             support_bank=str(bank),support_bank_sha256=file_sha(bank))
    monkeypatch.setattr(module.BatchClient,'complete_many',lambda self,requests,*args:requests)
    client=module.ScaffoldClient(cfg)
    for turn in range(13):
        messages=[dict(role='system',content='system'),dict(role='user',content='public')]
        for i in range(turn):messages.extend([dict(role='assistant',content='action'),dict(role='user',content='observation')])
        result=client.complete_many([dict(messages=messages)],128,.7,'test')
        assert any(m['content']==SUPPORT_PREFIX+'PLAN' for m in messages)==(turn<4)
        assert result[0]['messages'] is messages
