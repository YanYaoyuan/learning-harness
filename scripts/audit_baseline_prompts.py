#!/usr/bin/env python3
"""Detect public demonstration overlap without opening final benchmark games."""
import json
from pathlib import Path
import re

from learning_harness.common import digest, read_jsonl


PATTERN=r'Looking quickly around you, you see (.*?)\.\s*Your task is to:\s*([^\n]+)'


def signature(room,goal):
    return tuple(sorted(re.findall(r'[a-z]+ \d+',room))),re.sub(r'\s+',' ',goal.lower().strip())


def audit(root):
    actor=json.loads(Path('third_party/ReAct/prompts/alfworld_3prompts.json').read_text())
    sources={k:v for k,v in actor.items() if k.startswith('react_') and k.endswith(('_0','_1'))}
    sources['reflexion_examples']=Path('third_party/Reflexion/alfworld_runs/reflexion_few_shot_examples.txt').read_text()
    examples={}
    for key,text in sources.items():
        for index,match in enumerate(re.finditer(PATTERN,text,re.S)):
            examples[key+':'+str(index)]=signature(*match.groups())
    initial={};first_requests={}
    for path in root.glob('development/*/events.jsonl'):
        method=path.parent.name
        committed={r['run_id'] for r in read_jsonl(path.parent/'tasks.jsonl')}
        with path.open() as f:
            for line in f:
                try:e=json.loads(line)
                except json.JSONDecodeError:continue
                if e['run_id'] in committed and e['call']==1 and method in ('react_retry','reflexion'):
                    first_requests[(method,e['task_id'],e['seed'])]=dict(messages=digest(e['messages']),
                        request_seed=e['request_seed'],output=digest(e['response']))
                if e['phase']=='act' and e['task_id'] not in initial:
                    match=re.search(PATTERN,e['messages'][1]['content'],re.S)
                    if not match:raise ValueError('Cannot audit public reset: '+e['task_id'])
                    initial[e['task_id']]=signature(*match.groups())
    collisions=[dict(task_id=task,example=key) for task,value in initial.items() for key,ex in examples.items() if value==ex]
    replay=Path('output/baseline-benchmark-v1/demo-replay-audit.json')
    confirmations=[]
    if replay.exists():
        r=json.loads(replay.read_text())
        def normalize(text):
            text=re.sub(r'^You arrive at [^.]+\.\s*','',text)
            text=re.sub(r'You put the (.+?) in/on the (.+?)\.',r'You move the \1 to the \2.',text)
            return re.sub(r'\s+',' ',text).strip()
        confirmations.append(dict(task_id=r['task_id'],example=r['demo_key'],
            demo_actions_solve_task=r['demo_actions_solve_task'],
            observation_path_matches_after_navigation_prefix_and_put_move_normalization=
                all(normalize(e['expected'])==normalize(e['actual']) for e in r['events'])))
    planned=len(read_jsonl('data/manifests/alfworld/dev.jsonl'))
    pairs=[]
    for (method,task,seed),r in first_requests.items():
        other=first_requests.get(('reflexion',task,seed))
        if method=='react_retry' and other is not None:
            pairs.append(dict(prompt_matches=r['messages']==other['messages'],
                request_seed_matches=r['request_seed']==other['request_seed'],output_matches=r['output']==other['output']))
    if pairs and not all(r['prompt_matches'] and r['request_seed_matches'] for r in pairs):
        raise ValueError('ReAct/Reflexion first-request pairing failed')
    return dict(complete=len(initial)==planned,observed_tasks=len(initial),planned_tasks=planned,
        first_request_pairing=dict(paired_tasks=len(pairs),same_prompt_and_seed=all(r['prompt_matches'] and r['request_seed_matches'] for r in pairs),
            identical_raw_outputs=sum(r['output_matches'] for r in pairs)),
        match_rule='Exact public goal + multiset of visible numbered receptacles; conservative exclusion, not proof of unseen hidden-state identity',
        collisions=collisions,excluded_task_ids=sorted({r['task_id'] for r in collisions}),
        public_replay_confirmations=confirmations,
        limitations='No guarantee against pretrained model contamination. No final games opened.')


if __name__=='__main__':
    print(json.dumps(audit(Path('runs/baseline-benchmark-v1')),indent=2))
