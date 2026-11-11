import json
from pathlib import Path
import pytest

from learning_harness.benchmark import actor_messages, demonstration, parse_line, public_family, run, validate
from learning_harness.common import digest, file_sha


def config(tmp_path,method='reflexion'):
    game=tmp_path/'game';game.write_text('fake')
    task=dict(task_id='t',task_group='g',task_type='place',game_relpath='game',game_sha256=file_sha(game),purpose='development')
    manifest=tmp_path/'manifest.jsonl';manifest.write_text(json.dumps(task)+'\n')
    cfg=dict(method=method,manifest=str(manifest),revision='test',seeds=[11],max_turns=2,max_calls=6,
        max_trials=3,history_turns=12,batch_size=1,max_new_tokens=128,temperature=.7,model={})
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    return cfg,task,path


class Env:
    actions=[]
    def __init__(self,*args):self.count=0
    def reset(self):return 'Public room. Your task is to: put an apple on table.'
    def step(self,action):
        self.actions.append(action);self.count+=1
        return 'Nothing happens.',False,self.count>=2
    def close(self):pass


class Client:
    model='fake'
    def __init__(self):self.requests=[]
    def complete_many(self,requests,max_tokens,temperature,revision):
        self.requests.extend(requests)
        return dict(results=[dict(text='Plan: open the cabinet.' if 'New plan:' in r['messages'][-1]['content']
                                 else 'think: find the apple',input_tokens=20,output_tokens=8) for r in requests])


def test_public_router_and_matched_demos():
    initial='Room has a microwave. Your task is to: put some apple on table.'
    assert public_family(initial)=='put'
    for goal,family in [('heat an apple','heat'),('put a cold apple in fridge','cool'),
        ('put a clean cup on table','clean'),('look at apple under the lamp','examine'),('put two apples on table','puttwo')]:
        assert public_family('Your task is to: '+goal)==family
    react=demonstration(initial,'react_retry');act=demonstration(initial,'act')
    assert '> think:' in react and '> think:' not in act
    assert [l for l in react.splitlines() if l.startswith('> ') and not l.startswith('> think:')]==[l for l in act.splitlines() if l.startswith('> ')]
    assert actor_messages(initial,[],[],'react_retry',12)==actor_messages(initial,[],[],'reflexion',12)


def test_invalid_and_thought_parsing():
    assert parse_line('think: find cup\n> go to table 1','react_retry')==('think: find cup',False,True)
    assert parse_line('think: find cup','act')[1]
    assert parse_line('put apple 1 in/on bowl 1','act')[0]=='move apple 1 to bowl 1'
    assert parse_line('not json','json_direct')[1]


def test_reflection_cost_memory_reset_resume(tmp_path,monkeypatch):
    cfg,task,path=config(tmp_path);monkeypatch.setenv('LH_STUDENT_REVISION','test')
    client=Client();out=tmp_path/'out';Env.actions=[]
    rows=run(path,tmp_path,out,client=client,environment_factory=Env)
    assert len(rows)==1 and rows[0]['calls']==5 # No useless final reflection without a following action.
    assert len(rows[0]['trials'])==2 and len(Env.actions)==4
    assert rows[0]['output_tokens']==40 and rows[0]['input_tokens']==100
    assert any('previous failed attempts' in m['content'] for m in client.requests[3]['messages'])
    assert all('previous failed attempts' not in m['content'] for m in client.requests[0]['messages'])
    assert not rows[0]['success']
    before=len(client.requests);run(path,tmp_path,out,client=client,environment_factory=Env)
    assert len(client.requests)==before
    cfg['temperature']=.4;path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='fingerprint'):run(path,tmp_path,out,client=client,environment_factory=Env)


def test_retry_has_equal_call_cap_without_reflection(tmp_path,monkeypatch):
    cfg,task,path=config(tmp_path,'react_retry');monkeypatch.setenv('LH_STUDENT_REVISION','test')
    rows=run(path,tmp_path,tmp_path/'out',client=Client(),environment_factory=Env)
    assert rows[0]['calls']==6 and len(rows[0]['trials'])==3


def test_final_is_closed_and_freeze_checks_sources(tmp_path):
    cfg,task,path=config(tmp_path)
    task['official_split']='valid_unseen'
    with pytest.raises(ValueError,match='requires'):validate(cfg,[task])
    lock=tmp_path/'lock.json';lock.write_text(json.dumps(dict(status='frozen',config_digests=[digest(cfg)],files={})))
    with pytest.raises(ValueError,match='Incomplete'):validate(cfg,[task],lock)


def test_success_stops_retries(tmp_path,monkeypatch):
    cfg,task,path=config(tmp_path);monkeypatch.setenv('LH_STUDENT_REVISION','test')
    class Won(Env):
        def step(self,action):return 'You win.',True,True
    rows=run(path,tmp_path,tmp_path/'out',client=Client(),environment_factory=Won)
    assert rows[0]['calls']==1 and rows[0]['first_success'] and len(rows[0]['trials'])==1


def test_memory_never_crosses_tasks_and_orphans_are_not_results(tmp_path,monkeypatch):
    cfg,task,path=config(tmp_path);monkeypatch.setenv('LH_STUDENT_REVISION','test')
    task2=dict(task,task_id='t2',task_group='g2')
    Path(cfg['manifest']).write_text(json.dumps(task)+'\n'+json.dumps(task2)+'\n')
    client=Client();out=tmp_path/'out';out.mkdir()
    (out/'events.jsonl').write_text(json.dumps(dict(run_id='orphan',task_id='t'))+'\n')
    rows=run(path,tmp_path,out,client=client,environment_factory=Env)
    assert len(rows)==2 and all(r['run_id']!='orphan' for r in rows)
    assert len(client.requests)==10
    assert all('previous failed attempts' not in m['content'] for m in client.requests[5]['messages'])


@pytest.mark.parametrize('goal',['put apple on table','put a clean cup on table','heat an apple',
    'put a cold cup on table','look at apple under lamp','put two apples on table'])
def test_all_act_examples_remove_thoughts(goal):
    assert '> think:' not in demonstration('Your task is to: '+goal,'act')
