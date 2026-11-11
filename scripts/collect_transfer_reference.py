#!/usr/bin/env python3
"""Collect public-only support or diagnostic references; never query tasks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import uuid
from learning_harness.common import digest,file_sha,game_path,read_jsonl
from learning_harness.environment import AlfWorld
from learning_harness.reference_cooling import cooling_plan
from learning_harness.reference_heating import heating_plan
from learning_harness.support import messages_for


def episode(job):
    task,cap=job
    allowed={('train','b_support_candidates'),('dev','b_diagnostic')}
    if (task['partition'],task.get('adaptation_role')) not in allowed:
        raise ValueError('Only support or diagnostic manifests may enter reference collection')
    policy_fn={'pick_cool_then_place_in_recep':cooling_plan,'pick_heat_then_place_in_recep':heating_plan}[task['task_type']]
    env=AlfWorld(game_path('data/alfworld',task),11,cap)
    attempt=uuid.uuid4().hex;events=[];success=False;status='completed';error=None
    base={k:task[k] for k in ['task_id','task_group','task_type','partition','adaptation_role']}
    base.update(attempt_id=attempt,arm='reference',rollout_seed=11)
    try:
        initial=env.reset();policy=policy_fn(initial);action=next(policy)
        for turn in range(cap):
            messages=messages_for(initial,events,history_turns=12)
            observation,success,done=env.step(action)
            events.append(dict(base,step=turn,messages=messages,response=json.dumps({'action':action}),
                action=action,observation=observation,success=success,invalid_action=False,
                support_active=False,input_tokens=0,output_tokens=0))
            if success or done:break
            try:action=policy.send(observation)
            except StopIteration:break
    except Exception as exc:status='error';error=type(exc).__name__
    finally:env.close()
    return dict(base,success=bool(success),status=status,error_type=error,steps=len(events),
        invalid_actions=0,support_injected=False,input_tokens=0,output_tokens=0),events


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--max-steps',type=int,default=120);a=p.parse_args()
    tasks=read_jsonl(a.manifest)
    assert all((r['partition'],r['adaptation_role']) in {('train','b_support_candidates'),('dev','b_diagnostic')} for r in tasks)
    a.out.mkdir(parents=True,exist_ok=True)
    identity=dict(manifest_sha256=file_sha(a.manifest),max_steps=a.max_steps,
        policy_hashes={f:file_sha(f) for f in ['src/learning_harness/reference_cooling.py',
            'src/learning_harness/reference_heating.py','scripts/collect_transfer_reference.py']})
    meta=a.out/'run.json';fingerprint=digest(identity)
    if meta.exists():assert json.loads(meta.read_text())['fingerprint']==fingerprint
    else:meta.write_text(json.dumps(dict(fingerprint=fingerprint,planned_episodes=len(tasks),
        config=identity,paid_api_calls=0,teacher='scripted public-observation reference'),indent=2)+'\n')
    path=a.out/'episodes.jsonl';old=read_jsonl(path) if path.exists() else []
    assert len({r['task_id'] for r in old})==len(old) and all(r['status']=='completed' for r in old)
    done={r['task_id'] for r in old};jobs=[(t,a.max_steps) for t in tasks if t['task_id'] not in done]
    with path.open('a') as ep,(a.out/'steps.jsonl').open('a') as steps:
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for result,events in pool.map(episode,jobs):
                for row in events:steps.write(json.dumps(row,ensure_ascii=False)+'\n')
                steps.flush();ep.write(json.dumps(result)+'\n');ep.flush()
                print(result['task_id'],result['success'],result['status'],flush=True)
                if result['status']=='error':raise RuntimeError(result['error_type'])
    rows=read_jsonl(path);summary=dict(complete=len(rows)==len(tasks),episodes=len(rows),
        successes=sum(r['success'] for r in rows),paid_api_calls=0)
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
