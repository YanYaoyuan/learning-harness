#!/usr/bin/env python3
"""Read-only cloud observer; uploads only explicit metrics and research artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import wandb

ROOT=Path('runs/baseline-benchmark-v1')
OUT=Path('output/baseline-benchmark-v1')
METHODS=['json_direct','act','react_retry','reflexion']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    target=json.loads(Path('output/wandb/connection.json').read_text())
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'pre-run-lock.json').read_bytes()
    run=wandb.init(entity=target['entity'],project=target['project'],
        id=hashlib.sha256(b'baseline-benchmark-v1'+lock).hexdigest()[:12],
        name='ALFWorld / controlled baseline development v1',group='baseline-benchmark-v1',
        job_type='evaluation',mode='online',resume='allow',dir=str(OUT.absolute()),save_code=False,
        settings=wandb.Settings(console='off',disable_code=True,disable_git=True,save_code=False,
            disable_job_creation=True,x_disable_stats=True,init_timeout=60),
        config=dict(protocol='baseline-benchmark-v1',model='Qwen3-4B-Instruct-2507',
            split='reused development',tasks_per_method=40,methods=METHODS,
            max_calls=150,max_trials=3,max_turns=50,reflection_in_budget=True,
            final_test_executed=False,source_lock_sha256=hashlib.sha256(lock).hexdigest()))
    receipt=OUT/'wandb.json'
    receipt.write_text(json.dumps(dict(run_id=run.id,run_url=run.url,status='watching',final_test_executed=False),indent=2)+'\n')
    started=time.monotonic()
    try:
        while True:
            metrics={};complete=True;all_rows=[]
            for method in METHODS:
                path=ROOT/'development'/method/'tasks.jsonl'
                rows=[]
                if path.exists():
                    for line in path.read_text().splitlines():
                        try:rows.append(json.loads(line))
                        except json.JSONDecodeError:pass
                complete &= len(rows)==40
                metrics[method+'/completed']=len(rows)
                if rows:
                    metrics.update({method+'/first_success_rate':sum(r['first_success'] for r in rows)/len(rows),
                        method+'/cumulative_success_rate':sum(r['success'] for r in rows)/len(rows),
                        method+'/mean_calls':sum(r['calls'] for r in rows)/len(rows),
                        method+'/total_input_tokens':sum(r['input_tokens'] for r in rows),
                        method+'/total_output_tokens':sum(r['output_tokens'] for r in rows)})
                    all_rows.extend(rows)
            metrics['complete']=complete;run.log(metrics)
            report=OUT/'summary.json'
            if complete and report.exists() and json.loads(report.read_text()).get('complete'):
                columns=['method','task_id','seed','first_success','success','calls','input_tokens','output_tokens']
                run.log({'tasks':wandb.Table(columns=columns,data=[[r[c] for c in columns] for r in all_rows])})
                for path in sorted(OUT.glob('*.png')):run.log({'figures/'+path.stem:wandb.Image(str(path))})
                artifact=wandb.Artifact('baseline-benchmark-v1-development',type='evaluation')
                for path in [ROOT/'pre-run-lock.json',ROOT/'checkpoint-registry.json',
                    Path('docs/BASELINE_EVALUATION_PROTOCOL_V1.zh-CN.md')]+sorted((ROOT/'configs').glob('*.json'))+[
                        p for p in OUT.iterdir() if p.suffix in ['.csv','.pdf','.png'] or p.name=='summary.json']:
                    artifact.add_file(str(path),name=str(path))
                run.log_artifact(artifact);run.summary.update(dict(complete=True,completed_task_methods=len(all_rows),final_test_executed=False))
                receipt.write_text(json.dumps(dict(run_id=run.id,run_url=run.url,status='uploaded',
                    completed_task_methods=len(all_rows),final_test_executed=False),indent=2)+'\n')
                break
            if not args.watch or time.monotonic()-started>6*3600:break
            time.sleep(30)
    finally:run.finish();wandb.teardown()


if __name__=='__main__':main()
