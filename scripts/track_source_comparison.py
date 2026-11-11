#!/usr/bin/env python3
"""Read-only W&B observer and automatic measured-report finalizer."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import wandb

ROOT=Path('runs/source-comparison-v1');OUT=Path('output/source-comparison-v1')


def records(path):
    if not path.exists():return []
    result=[]
    for line in path.read_text().splitlines():
        try:result.append(json.loads(line))
        except json.JSONDecodeError:pass
    return result


def report():
    env=dict(os.environ,PYTHONPATH=str(Path('src').absolute()),MPLCONFIGDIR='/tmp/lh-mpl')
    with (ROOT/'report.log').open('w') as log:
        subprocess.run(['.venv-model/bin/python','scripts/source_comparison_report.py'],env=env,
            stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    target=json.loads(Path('output/wandb/connection.json').read_text());design=json.loads((ROOT/'design.json').read_text())
    OUT.mkdir(parents=True,exist_ok=True)
    identity=hashlib.sha256((ROOT/'inputs.sha256.json').read_bytes()).hexdigest()[:12]
    run=wandb.init(entity=target['entity'],project=target['project'],id=identity,
        name='API-scaffold vs ReAct / matched SFT and adaptation',group='source-comparison-v1',job_type='pipeline-comparison',
        mode='online',resume='allow',dir=str(OUT.absolute()),save_code=False,
        settings=wandb.Settings(console='off',disable_code=True,disable_git=True,save_code=False,
            disable_job_creation=True,x_disable_stats=True,init_timeout=60),config=design)
    def receipt(status,**extra):
        (OUT/'wandb.json').write_text(json.dumps(dict(run_id=run.id,run_url=run.url,status=status,
            official_final_tests_used=False,**extra),indent=2)+'\n')
    receipt('watching');started=time.monotonic();selection_logged=False
    try:
        while True:
            metrics={}
            for source in ['api','react']:
                rows=records(ROOT/'collection'/source/'tasks.jsonl')
                metrics[f'collection/{source}/completed']=len(rows)
                if rows:
                    metrics[f'collection/{source}/successes']=sum(r['success'] for r in rows)
                    metrics[f'collection/{source}/input_tokens']=sum(r['input_tokens'] for r in rows)
                    metrics[f'collection/{source}/output_tokens']=sum(r['output_tokens'] for r in rows)
            metas=list((ROOT/'a-checkpoints').glob('*/experiment.json'));metrics['training/A_completed']=len(metas)
            metrics['training/B_completed']=len(list((ROOT/'trajectories').glob('*/*/trajectory.json')))
            evaluation=sum(len(records(path)) for path in (ROOT/'evaluation').glob('*/*/*/*/step-*/tasks.jsonl'))
            metrics['evaluation/completed_task_runs']=evaluation
            status=json.loads((ROOT/'execution.json').read_text()) if (ROOT/'execution.json').exists() else {'status':'starting'}
            metrics['pipeline/status']=status['status'];run.log(metrics)
            if (ROOT/'selection.json').exists() and not selection_logged:
                selection=json.loads((ROOT/'selection.json').read_text());run.summary['selection']=selection
                report()
                if (OUT/'collection.png').exists():run.log({'figures/collection':wandb.Image(str(OUT/'collection.png'))})
                selection_logged=True
            if status['status']=='completed':
                report();result=json.loads((OUT/'summary.json').read_text())
                if not result['complete']:raise ValueError('Pipeline finished without all planned evaluations')
                for path in sorted(OUT.glob('*.png')):run.log({'figures/'+path.stem:wandb.Image(str(path))})
                for name in ['seed-scores','training-cost']:
                    with (OUT/f'{name}.csv').open() as handle:
                        reader=csv.DictReader(handle);rows=list(reader)
                        run.log({'tables/'+name:wandb.Table(columns=reader.fieldnames,data=[[r[c] for c in reader.fieldnames] for r in rows])})
                artifact=wandb.Artifact('source-comparison-v1-measured-results',type='research-results')
                paths=[ROOT/'design.json',ROOT/'inputs.sha256.json',ROOT/'training-inputs.sha256.json',
                    ROOT/'analysis-lock.json',ROOT/'selection.json',ROOT/'paired-a-schedules.json',
                    Path('docs/SOURCE_COMPARISON_V1.zh-CN.md')]
                paths += [p for p in OUT.iterdir() if p.suffix in ['.csv','.png','.pdf','.md'] or p.name=='summary.json']
                for path in paths:artifact.add_file(str(path),name=str(path))
                run.log_artifact(artifact)
                run.summary.update(dict(complete=True,evaluation_task_runs=evaluation,token_and_budget_audit_passed=True,
                    primary_api_minus_react=result['primary_api_minus_react'],official_final_tests_used=False))
                receipt('uploaded',evaluation_task_runs=evaluation);break
            if status['status'] in ['failed','matching_gate_failed']:
                run.summary.update(dict(complete=False,termination_status=status['status']));receipt(status['status']);break
            if not args.watch or time.monotonic()-started>24*3600:break
            time.sleep(30)
    except Exception as exc:
        receipt('observer_failed',error_type=type(exc).__name__)
        raise
    finally:run.finish();wandb.teardown()


if __name__=='__main__':main()
