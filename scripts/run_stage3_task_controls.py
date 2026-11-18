#!/usr/bin/env python3
"""Freeze and run task-order/exposure-matched random and base-NLL controls."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import run_api_harness as engine
import run_stage1_crossed_seeds as stage1
import run_stage2_sampling as stage2
from learning_harness.bounded_queue import run,save
from learning_harness.common import file_sha,read_jsonl
from train_lora import fixed_schedule

ROOT=Path('runs/variance-diagnostic-v1/stage3')
OUT=Path('output/variance-diagnostic-v1/stage3')
ARMS=('random','min_nll')
FILES=['docs/VARIANCE_DIAGNOSTIC_STAGE3_SPEC.md','scripts/run_stage3_task_controls.py',
       'scripts/stage3_worker.py','scripts/stage3_task_report.py','scripts/train_lora_fixed_schedule.py',
       'scripts/track_bounded_study.py','scripts/archive_bounded_study.py',
       'src/learning_harness/task_matched_sampling.py','scripts/score_training_nll.py',
       'scripts/analyze_stage2_mechanisms.py']


def recipe_path(arm,source,seed):
    return ROOT/'inputs/schedules'/f'{arm}-{source}-s{seed}.json'


def plan():
    cases=[]
    for source,gpu in (('api',0),('generic',1)):
        cases.append(dict(name=f'full_reference-{source}-i101-s101-r2',arm='full_reference',source=source,
            initialization_seed=101,sampling_seed=101,repeat=2,gpu=gpu,
            schedule=str(recipe_path('full_reference',source,101))))
    for base in stage1.plan():
        parity=(stage1.SEEDS.index(base['initialization_seed'])+stage1.SEEDS.index(base['sampling_seed']))%2
        for arm in (ARMS if parity==0 else ARMS[::-1]):
            cases.append(dict(base,name=f'task_matched_{arm}-{base["name"]}',arm=arm,
                              schedule=str(recipe_path(arm,base['source'],base['sampling_seed']))))
    return cases


def prepare_references():
    for source in ('api','generic'):
        data=Path(f'runs/transfer-factorial-v1/corpora/full-{source}.jsonl')
        rows=read_jsonl(data);tasks=[r['task_id'] for r in rows]
        indices=fixed_schedule(len(rows),768,101)
        digest=lambda value:hashlib.sha256(json.dumps(value).encode()).hexdigest()
        order=digest([tasks[i] for i in indices])
        value=dict(policy='full_reference',source=source,sampling_seed=101,data=str(data),data_sha256=file_sha(data),
            indices=indices,schedule_sha256=digest(indices),reference_schedule_sha256=digest(indices),
            task_order_sha256=order,reference_task_order_sha256=order,unique_examples=len(rows),
            presentations=768,exposures_by_task=dict(Counter(tasks[i] for i in indices)),
            exact_task_order_matched=True,exact_task_exposure_matched=True)
        path=recipe_path('full_reference',source,101)
        if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Existing reference recipe differs')
        save(path,value)


def smoke():
    if (ROOT/'freeze.json').exists():raise ValueError('Smoke belongs before freezing')
    result_path=ROOT/'smoke/result.json'
    if result_path.exists():
        if not json.loads(result_path.read_text()).get('passed'):raise ValueError('Prior smoke failed; inspect')
        return
    stage2.load_freeze()
    audit=json.loads(Path('output/variance-diagnostic-v1/stage2/mechanism-audit/summary.json').read_text())
    if not audit['next_control_ready']:raise ValueError('Task-matched recipes are incomplete')
    cfg=json.loads((stage2.ROOT/'a-lora.json').read_text());cfg['max_steps']=1
    save(ROOT/'smoke/config.json',cfg)
    result=dict(passed=False,task_evaluations=0,teacher_calls=0,controls=[])
    started=time.monotonic()
    try:
        for arm in ARMS:
            directory=ROOT/'smoke'/arm
            recipe=recipe_path(arm,'api',101)
            engine.command([engine.MODEL_PY,'scripts/train_lora_fixed_schedule.py','--schedule',str(recipe),
                '--data','runs/transfer-factorial-v1/corpora/full-api.jsonl','--model',engine.MODEL,
                '--revision',engine.REV,'--config',str(ROOT/'smoke/config.json'),'--out',str(directory),
                '--initialization-seed','101','--sampling-seed','101'],stage1.environment(0),ROOT/'smoke'/f'{arm}.log',timeout=180)
            ctrl=json.loads((directory/'seed-controls.json').read_text())
            indices=json.loads(recipe.read_text())['indices'][:4]
            expected=hashlib.sha256(json.dumps(indices).encode()).hexdigest()
            if not ctrl['completed'] or ctrl['selection_policy']!=arm or ctrl['schedule_sha256']!=expected:
                raise ValueError('GPU schedule binding failed')
            result['controls'].append(ctrl)
        a,b=result['controls']
        old=json.loads((stage2.ROOT/'cases/full_pass-api-i101-s101-r0/worker-result.json').read_text())
        if not a['initial_parameter_sha256']==b['initial_parameter_sha256']==old['control']['initial_parameter_sha256']:
            raise ValueError('Policy changed initialization')
        result['passed']=True
    finally:
        result['gpu_seconds']=time.monotonic()-started
        save(result_path,result)


def optimized_smoke():
    """Two updates cross the warmup boundary and verify real LoRA parameter changes."""
    root=ROOT/'smoke-optimized';result_path=root/'result.json'
    if result_path.exists():
        if not json.loads(result_path.read_text()).get('passed'):raise ValueError('Prior optimized smoke failed')
        return
    if (ROOT/'freeze.json').exists():raise ValueError('Smoke belongs before freeze')
    cfg=json.loads((stage2.ROOT/'a-lora.json').read_text());cfg['max_steps']=2
    save(root/'config.json',cfg)
    result=dict(passed=False,updates=2,task_evaluations=0,teacher_calls=0,cases=[])
    started=time.monotonic()
    try:
        from safetensors.torch import load_file
        for arm in ARMS:
            directory=root/arm
            recipe=recipe_path(arm,'api',101)
            engine.command([engine.MODEL_PY,'scripts/train_lora_fixed_schedule.py','--schedule',str(recipe),
                '--data','runs/transfer-factorial-v1/corpora/full-api.jsonl','--model',engine.MODEL,
                '--revision',engine.REV,'--config',str(root/'config.json'),'--out',str(directory),
                '--initialization-seed','101','--sampling-seed','101'],stage1.environment(0),root/f'{arm}.log',timeout=180)
            control=json.loads((directory/'seed-controls.json').read_text())
            tensors=load_file(str(directory/'adapter_model.safetensors'),device='cpu')
            nonzero=sum(int(v.count_nonzero()) for k,v in tensors.items() if 'lora_B' in k)
            if not control['completed'] or control['scheduled_prefix_length']!=8 or nonzero==0:
                raise ValueError('Real optimizer updates or schedule consumption were not verified')
            result['cases'].append(dict(arm=arm,nonzero_lora_b_parameters=nonzero,control=control))
        a,b=[v['control'] for v in result['cases']]
        if a['initial_parameter_sha256']!=b['initial_parameter_sha256'] or a['trained_adapter_sha256']==b['trained_adapter_sha256']:
            raise ValueError('Policy inputs did not produce distinct optimized weights from the same initialization')
        result['passed']=True
    finally:
        result['gpu_seconds']=time.monotonic()-started
        save(result_path,result)


def freeze():
    if (ROOT/'freeze.json').exists():raise ValueError('Stage 3 freeze is never overwritten')
    parent=stage2.load_freeze()
    completed=json.loads((stage2.OUT/'summary.json').read_text())
    if not completed['complete'] or completed['actual_evaluation_episodes']!=4352:raise ValueError('Stage 2 incomplete')
    if not json.loads((ROOT/'smoke/result.json').read_text())['passed']:raise ValueError('GPU smoke not passed')
    if not json.loads((ROOT/'smoke-optimized/result.json').read_text())['passed']:raise ValueError('Optimized GPU smoke not passed')
    if not json.loads((ROOT/'base-nll/result.json').read_text())['status']=='completed':raise ValueError('Base NLL incomplete')
    prepare_references()
    config=json.loads((stage2.ROOT/'a-lora.json').read_text());save(ROOT/'a-lora.json',config)
    paths=set(FILES)|set(parent['files'])|{str(stage2.ROOT/'freeze.json'),str(stage2.OUT/'summary.json'),
        str(ROOT/'a-lora.json'),str(ROOT/'smoke/result.json'),str(ROOT/'base-nll/result.json'),
        str(ROOT/'smoke-optimized/result.json'),
        str(ROOT/'preparation-revision.json'),
        str(ROOT/'preparation-v0-row-support/smoke/result.json'),
        str(ROOT/'preparation-v0-row-support/smoke-optimized/result.json'),
        str(ROOT/'base-nll/api.json'),str(ROOT/'base-nll/generic.json'),
        'output/variance-diagnostic-v1/stage2/mechanism-audit/summary.json'}
    for case in stage2.plan():paths.add(str(stage2.ROOT/'cases'/case['name']/'worker-result.json'))
    for case in plan():paths.add(case['schedule'])
    weights=(ROOT/'smoke/random/adapter_model.safetensors').stat().st_size
    if shutil.disk_usage(ROOT).free<2*weights*70+2*1024**3:raise ValueError('Need 70 A/B checkpoints plus 2 GiB reserve')
    save(ROOT/'freeze.json',dict(protocol='variance-diagnostic-v1-stage3',name='Stage 3 / exact task-order controls',
        frozen_unix=time.time(),cases=plan(),files={p:file_sha(p) for p in sorted(paths)},gpus=[0,1],
        cap_gpu_hours=14,allocation_seconds=1800,minimum_allocation_seconds=900,planned_episodes=4480,
        report_script='scripts/stage3_task_report.py',teacher_calls=0,official_tests_used=False,
        primary_contrasts=['random minus historical full_pass, within each source',
                           'min_nll minus random, within each source'],
        reused_reference_cases=34,fresh_reference_cases=2,presentations=768,optimizer_updates=192))


def load_freeze():
    value=json.loads((ROOT/'freeze.json').read_text())
    if value['cases']!=plan() or value['cap_gpu_hours']!=14:raise ValueError('Stage 3 design changed')
    for path,sha in value['files'].items():
        if file_sha(path)!=sha:raise ValueError('Stage 3 frozen file changed: '+path)
    stage2.load_freeze()
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--smoke',action='store_true');mode.add_argument('--optimized-smoke',action='store_true');mode.add_argument('--freeze',action='store_true')
    args=parser.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.smoke:smoke();return
    if args.optimized_smoke:optimized_smoke();return
    if args.freeze:freeze();return
    frozen=load_freeze()
    for gpu in frozen['gpus']:
        if not stage1.stage0.port_is_free(gpu):raise ValueError('Existing model server occupies GPU '+str(gpu))
    preparation=sum(json.loads((ROOT/p/'result.json').read_text())['gpu_seconds'] for p in ('base-nll','smoke','smoke-optimized'))
    preparation+=sum(json.loads((ROOT/'preparation-v0-row-support'/p/'result.json').read_text())['gpu_seconds'] for p in ('smoke','smoke-optimized'))
    def command(case,gpu,seconds):
        return [engine.MODEL_PY,'scripts/stage3_worker.py','--case',case['name'],'--gpu',str(gpu),'--timeout',str(seconds)]
    ledger=run(ROOT,frozen,command,stage1.environment,preparation)
    if ledger['status']=='evaluation_completed':
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'report.log').open('w') as log:
            subprocess.run([engine.MODEL_PY,frozen['report_script']],env=stage1.environment(0),stdout=log,
                           stderr=subprocess.STDOUT,check=True,timeout=180)
        ledger['status']='completed';save(ROOT/'execution.json',ledger)


if __name__=='__main__':main()
