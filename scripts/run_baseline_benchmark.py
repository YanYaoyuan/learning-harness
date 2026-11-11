#!/usr/bin/env python3
"""Prepare and execute the development baseline suite on two local GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

from learning_harness.benchmark import METHODS, SOURCE_FILES
from learning_harness.common import digest, file_sha, read_jsonl, write_jsonl
from run_api_harness import server, REV, CPU_PY

ROOT=Path('runs/baseline-benchmark-v1')


def immutable_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value: raise ValueError('Frozen preparation changed: '+str(path))
    else: path.write_text(json.dumps(value,indent=2)+'\n')


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    manifest='data/manifests/alfworld/dev.jsonl'
    tasks=read_jsonl(manifest)
    base=dict(protocol_id='alfworld-baseline-v1',revision=REV,manifest=manifest,
        max_turns=50,max_calls=150,max_trials=3,max_new_tokens=128,temperature=.7,
        history_turns=12,batch_size=8,seeds=[11],
        model=dict(base_url_env='LH_STUDENT_BASE_URL',model_env='LH_STUDENT_MODEL',api_key_env='LH_STUDENT_API_KEY'))
    for method in METHODS:
        immutable_json(ROOT/'configs'/f'{method}.json',dict(base,method=method))
        for split in ['seen','unseen']:
            immutable_json(ROOT/'final-configs'/f'{split}-{method}.json',
                dict(base,method=method,manifest=f'data/manifests/alfworld/test_{split}.jsonl',seeds=[11,23,37]))
    registry=[]
    for source in ['none','generic','api']:
        for seed in [101,202,303]:
            path=Path(f'runs/api-harness-v1/a-checkpoints/{source}-seed{seed}')
            weights=path/'adapter_model.safetensors'
            if not weights.exists(): raise ValueError('Missing existing SFT checkpoint: '+str(path))
            registry.append(dict(name=f'success-sft-{source}-seed{seed}',path=str(path),
                sha256=file_sha(weights),training_seed=seed,source=source,
                role='existing matched-success SFT comparator; not on-policy distillation or RL',
                evaluation_status='registered_not_run_in_this_suite'))
    inputs={p:file_sha(p) for p in SOURCE_FILES+[manifest,
        'data/manifests/alfworld/test_seen.jsonl','data/manifests/alfworld/test_unseen.jsonl',
        'scripts/run_baseline_benchmark.py']}
    immutable_json(ROOT/'pre-run-lock.json',dict(files=inputs,
        configs={str(p):file_sha(p) for p in sorted((ROOT/'configs').glob('*.json'))},
        development_tasks=len(tasks),final_test_executed=False))
    immutable_json(ROOT/'checkpoint-registry.json',registry)
    immutable_json(ROOT/'final-freeze.template.json',dict(status='development_only',
        config_digests=[digest(json.loads(p.read_text())) for p in sorted((ROOT/'final-configs').glob('*.json'))],
        files=inputs,notes='Candidate protocol, not preregistered final evidence. Freeze methods, checkpoints and analysis before activation.'))
    print('Prepared: 40 development tasks, 4 methods; final configurations remain sealed.',flush=True)


def worker(gpu,methods):
    for method in methods:
        out=ROOT/'development'/method;out.mkdir(parents=True,exist_ok=True)
        with server(gpu,'baseline-qwen3-4b',REV,out/'server.log') as env:
            with (out/'worker.log').open('a') as log:
                subprocess.run([CPU_PY,'-m','learning_harness.benchmark','--config',
                    str(ROOT/'configs'/f'{method}.json'),'--out',str(out)],env=env,
                    stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Completed '+method,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    prepare()
    if args.run:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,0,['json_direct','act']),pool.submit(worker,1,['react_retry','reflexion'])]
            for future in futures:future.result()


if __name__=='__main__':main()
