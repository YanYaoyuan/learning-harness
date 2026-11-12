#!/usr/bin/env python3
"""Periodic process-backed experiment checks with immediate terminal-event handling."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def process(pid, expected):
    directory=Path('/proc')/str(pid)
    try:
        command=directory.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        status=directory.joinpath('status').read_text()
    except FileNotFoundError:
        return dict(pid=pid,alive=False,identity_verified=False)
    identity=all(str(value) in command for value in expected)
    return dict(pid=pid,alive=bool(command) and '\nState:\tZ' not in status,
                identity_verified=identity)


def inspect(root):
    root=Path(root)
    ledger=read(root/'execution.json')
    jobs=read(root/'job.json')
    if not jobs and not ledger:
        return dict(root=str(root),status='not_started',processes={})
    handles={}
    for name,item in jobs.items():
        if not isinstance(item,dict) or 'pid' not in item:
            continue
        expected=[Path(x).name for x in item.get('argv',[])[1:] if x.endswith('.py')]
        handles[name]=process(item['pid'],expected)
    for name,item in ledger.get('open',{}).items():
        handles[name]=process(item['pid'],[name])
    cases=ledger.get('cases',{})
    now=time.time()
    live_seconds=sum(max(0,now-v['started_unix']) for v in ledger.get('open',{}).values())
    output=Path('output')/root.relative_to('runs')
    finalization=read(output/'finalization.json')
    archive=read(output/'archival.json')
    missing=[name for name,value in handles.items() if not value['alive'] or not value['identity_verified']]
    terminal=ledger.get('status') in ('completed','failed','interrupted','stopped_at_cap')
    return dict(root=str(root),status=ledger.get('status','unknown'),
        completed=sum(v.get('status')=='completed' for v in cases.values()),
        failed_cases=[k for k,v in cases.items() if v.get('status') not in ('completed',)],
        elapsed_gpu_hours=(ledger.get('gpu_seconds',0)+live_seconds)/3600,
        processes=handles,missing_or_mismatched=missing,
        needs_attention=bool(missing and not terminal),
        finalization=finalization,archive_uploaded=archive.get('status')=='uploaded')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots',nargs='+',required=True)
    parser.add_argument('--interval',type=float,default=1800)
    parser.add_argument('--poll',type=float,default=30)
    parser.add_argument('--out',default='runs/research-supervision-v1')
    parser.add_argument('--on-stage2-complete',action='store_true')
    args=parser.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=dict(roots=args.roots,interval_seconds=args.interval,poll_seconds=args.poll,
                on_stage2_complete=args.on_stage2_complete,automatic_gpu_restart=False)
    config_path=out/'config.json'
    if config_path.exists() and read(config_path)!=config:
        raise ValueError('Supervisor configuration changed; inspect before restart')
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    from learning_harness.bounded_queue import save
    state=read(out/'state.json') or dict(status='watching',events={},callbacks={})
    next_check=0
    while True:
        checks=[inspect(root) for root in args.roots]
        changed=[]
        for check in checks:
            key=check['root']
            event=(check['status'],check.get('archive_uploaded',False),check.get('needs_attention',False))
            if state['events'].get(key)!=list(event):
                changed.append(dict(root=key,status=event[0],archive_uploaded=event[1],needs_attention=event[2]))
                state['events'][key]=list(event)
        now=time.time()
        periodic=now>=next_check
        if periodic or changed:
            if periodic:
                next_check=now+args.interval
                try:
                    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu',
                        '--format=csv,noheader'],capture_output=True,text=True,timeout=15)
                    gpu_state=dict(returncode=gpu.returncode,output=gpu.stdout.strip())
                except (OSError,subprocess.TimeoutExpired) as error:
                    gpu_state=dict(observation_error=type(error).__name__)
            else:
                gpu_state=None
            snapshot=dict(time_unix=now,time_china=datetime.fromtimestamp(now,ZoneInfo('Asia/Shanghai')).isoformat(),
                periodic=periodic,changes=changed,studies=checks,gpus=gpu_state)
            with (out/'checks.jsonl').open('a') as handle:
                handle.write(json.dumps(snapshot,ensure_ascii=False)+'\n')
            state.update(last_check_unix=now,next_periodic_unix=next_check,last_snapshot=snapshot)
            save(out/'state.json',state)
            print(json.dumps(snapshot,ensure_ascii=False),flush=True)
        if args.on_stage2_complete:
            stage2=next((c for c in checks if Path(c['root']).name=='stage2'),None)
            if stage2 and stage2['status']=='completed' and 'stage2-mechanism-audit' not in state['callbacks']:
                script=Path('scripts/analyze_stage2_mechanisms.py')
                if script.exists():
                    state['callbacks']['stage2-mechanism-audit']=dict(status='running',started_unix=time.time())
                    save(out/'state.json',state)
                    with (out/'stage2-mechanism-audit.log').open('x') as log:
                        try:
                            result=subprocess.run([str(Path('.venv-model/bin/python').absolute()),str(script)],
                                stdout=log,stderr=subprocess.STDOUT,timeout=180)
                            status='completed' if result.returncode==0 else 'failed'
                        except subprocess.TimeoutExpired:
                            status='timed_out'
                    state['callbacks']['stage2-mechanism-audit'].update(status=status,ended_unix=time.time())
                    save(out/'state.json',state)
        time.sleep(args.poll)


if __name__=='__main__':
    main()
