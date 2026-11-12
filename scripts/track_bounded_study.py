#!/usr/bin/env python3
"""Independent progress observer and archival completion for a frozen bounded study."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def read(path):return json.loads(path.read_text()) if path.exists() else {}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();root,out=Path(args.root),Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    frozen=read(root/'freeze.json')
    if not frozen:raise ValueError('Study is not frozen')
    identity=hashlib.sha256((root/'freeze.json').read_bytes()).hexdigest()[:12]
    target=read(Path('output/wandb/connection.json'));host=target.get('base_url',target.get('host'))
    if host:os.environ['WANDB_BASE_URL']=host
    import wandb
    run=wandb.init(entity=target['entity'],project=target['project'],id=identity+'-progress',
        name=frozen['name']+' / progress only',group='variance-diagnostic-v1',job_type='progress',
        resume='allow',mode='online',dir=str(out.absolute()),save_code=False,
        settings=wandb.Settings(console='off',disable_code=True,disable_git=True,save_code=False,
            disable_job_creation=True,x_disable_stats=True,init_timeout=60))
    receipt=dict(status='watching',run_id=run.id,run_url=run.url,progress_only=True,measured_result_run_id=identity)
    path=out/'progress-wandb.json';path.write_text(json.dumps(receipt,indent=2)+'\n')
    deadline=time.monotonic()+18*3600
    try:
        while True:
            ledger=read(root/'execution.json');state=ledger.get('status','not_started')
            completed=sum(v.get('status')=='completed' for v in ledger.get('cases',{}).values())
            elapsed=ledger.get('gpu_seconds',0)+sum(max(0,time.time()-v['started_unix']) for v in ledger.get('open',{}).values())
            if state=='evaluation_completed':
                pid=read(root/'job.json').get('pipeline',{}).get('pid')
                alive=True
                if pid:
                    try:os.kill(pid,0)
                    except ProcessLookupError:alive=False
                if not alive:
                    env=dict(os.environ,PYTHONPATH=str(Path('src').absolute()),MPLCONFIGDIR='/tmp/lh-mpl')
                    with (out/'report-recovery.log').open('w') as log:
                        subprocess.run([str(Path('.venv-model/bin/python').absolute()),frozen['report_script']],
                            env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
                    state='report_recovered'
            run.log({'progress/completed_adapters':completed,'progress/planned_adapters':len(frozen['cases']),
                     'progress/completed_episodes':completed*64,'progress/elapsed_gpu_hours':elapsed/3600,
                     'progress/cap_gpu_hours':frozen['cap_gpu_hours'],'progress/state':state})
            if state in ('completed','report_recovered','failed','interrupted','stopped_at_cap'):break
            if time.monotonic()>deadline:raise TimeoutError('Study observer deadline')
            time.sleep(30)
        run.summary.update(dict(progress_only=True,termination_state=state,completed_adapters=completed))
        receipt.update(status=state,completed_adapters=completed);path.write_text(json.dumps(receipt,indent=2)+'\n')
    finally:run.finish()
    if state not in ('completed','report_recovered'):
        (out/'finalization.json').write_text(json.dumps(dict(status='stopped',execution_state=state),indent=2)+'\n')
        return
    final=dict(local_report_complete=True,cloud_status='pending')
    (out/'finalization.json').write_text(json.dumps(final,indent=2)+'\n')
    try:
        with (out/'upload.log').open('w') as log:
            subprocess.run([str(Path('.venv-wandb/bin/python').absolute()),'scripts/archive_bounded_study.py',
                '--root',str(root),'--out',str(out)],env=dict(os.environ,PYTHONPATH=str(Path('src').absolute())),
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        final['cloud_status']='uploaded'
    except Exception as error:final.update(cloud_status='failed',error_type=type(error).__name__)
    (out/'finalization.json').write_text(json.dumps(final,indent=2)+'\n')


if __name__=='__main__':main()
