#!/usr/bin/env python3
"""Common B teaching data; query/calibration partitions cannot enter this tool."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import multiprocessing
from pathlib import Path
import uuid

from learning_harness.common import digest, file_sha, game_path, read_jsonl, write_jsonl
from learning_harness.environment import AlfWorld
from learning_harness.reference_cooling import cooling_plan
from learning_harness.support import SYSTEM_PROMPT, messages_for


def episode(job):
    task,seed,cap=job
    if task['partition']!='train' or task.get('adaptation_role')!='b_support_candidates':
        raise ValueError('Only B teaching candidates may enter the reference collector')
    env=AlfWorld(game_path('data/alfworld',task),seed,cap)
    events=[];attempt=uuid.uuid4().hex;success=False;status='completed';error=None
    try:
        initial=env.reset();policy=cooling_plan(initial);action=next(policy)
        for turn in range(cap):
            messages=messages_for(initial,events,history_turns=12)
            observation,success,done=env.step(action)
            events.append(dict(attempt_id=attempt,task_id=task['task_id'],task_type=task['task_type'],
                partition='train',arm='reference',rollout_seed=seed,step=turn,messages=messages,
                response=json.dumps({'action':action}),action=action,raw_action=action,
                observation=observation,success=success,invalid_action=False,support_active=False,
                input_tokens=0,output_tokens=0,response_model='public-text-scripted-reference'))
            if success or done:break
            try:action=policy.send(observation)
            except StopIteration:break
    except Exception as exc:
        status='error';error=type(exc).__name__
    finally:env.close()
    return dict(attempt_id=attempt,task_id=task['task_id'],task_group=task['task_group'],
        task_type=task['task_type'],partition='train',arm='reference',rollout_seed=seed,
        success=bool(success),status=status,error_type=error,steps=len(events),invalid_actions=0,
        support_injected=False,input_tokens=0,output_tokens=0),events


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',default='runs/learning-speed-v1/b_support_candidates.jsonl')
    parser.add_argument('--out',required=True)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--max-steps',type=int,default=120)
    args=parser.parse_args()
    tasks=read_jsonl(args.manifest)
    if any(t['partition']!='train' or t.get('adaptation_role')!='b_support_candidates' for t in tasks):
        raise ValueError('Reference collection is restricted to B teaching candidates')
    if args.limit:tasks=tasks[:args.limit]
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    config=dict(manifest_sha256=file_sha(args.manifest),tasks=[t['task_id'] for t in tasks],seed=11,
        max_steps=args.max_steps,reference_policy_sha256=file_sha('src/learning_harness/reference_cooling.py'),
        system_prompt_hash=digest(SYSTEM_PROMPT),interface='public-text-only')
    fingerprint=digest(config)
    meta=out/'run.json'
    if meta.exists():
        if json.loads(meta.read_text())['fingerprint']!=fingerprint:raise ValueError('Changed reference protocol')
    else:meta.write_text(json.dumps(dict(fingerprint=fingerprint,planned_episodes=len(tasks),config=config,
        paid_api_calls=0,model_generation_calls=0,teacher='scripted_public_observation_reference'),indent=2)+'\n')
    path=out/'episodes.jsonl';old=read_jsonl(path) if path.exists() else []
    assert len(old)==len({r['task_id'] for r in old})
    if any(r['status']=='error' for r in old):raise ValueError('Inspect failed collection before resume')
    done={r['task_id'] for r in old};jobs=[(t,11,args.max_steps) for t in tasks if t['task_id'] not in done]
    with (out/'episodes.jsonl').open('a') as ep,(out/'steps.jsonl').open('a') as step:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for result,events in pool.map(episode,jobs):
                for event in events:step.write(json.dumps(event,ensure_ascii=False)+'\n')
                step.flush();ep.write(json.dumps(result)+'\n');ep.flush()
                print(result['task_id'],result['success'],result['steps'],result['status'],flush=True)
                if result['status']=='error':raise RuntimeError(result['error_type'])
    rows=read_jsonl(path)
    report=dict(complete=len(rows)==len(tasks),episodes=len(rows),successes=sum(r['success'] for r in rows),
        successful_task_groups=len({r['task_group'] for r in rows if r['success']}),
        reference='scripted_public_observation_only',paid_api_calls=0,model_generation_calls=0)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
