#!/usr/bin/env python3
"""Execute the frozen transfer factorial, with calibration and diagnostic gates."""
from collections import defaultdict,Counter
import argparse
import json
from pathlib import Path
import random
import shutil
import subprocess

import run_api_harness as engine
from learning_harness.analysis import export_sft,summarize
from learning_harness.common import file_sha,read_jsonl,write_jsonl
from learning_harness.adaptation import composition_revision

ROOT=Path('runs/transfer-factorial-v1')
FAMILIES=['cool','heat']


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.part');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def frozen():
    for name in ['inputs.sha256.json','curricula.sha256.json','execution-code.sha256.json']:
        path=ROOT/name
        if path.exists():
            for p,h in json.loads(path.read_text()).items():
                if file_sha(p)!=h:raise ValueError(f'Frozen input/code changed: {p}')


def references():
    for family in FAMILIES:
        root=ROOT/family
        for role in ['support_candidates','diagnostic']:
            out=root/('reference-'+role)
            engine.command([engine.CPU_PY,'scripts/collect_transfer_reference.py','--manifest',
                str(root/f'b_{role}.jsonl'),'--out',str(out)],engine.environment(0),root/f'reference-{role}.log')
    from transformers import AutoTokenizer
    from train_lora import encode_example
    from matched_study import evenly_spaced
    tokenizer=AutoTokenizer.from_pretrained(engine.MODEL,local_files_only=True)
    outputs=[]
    for family in FAMILIES:
        root=ROOT/family
        if (root/'curriculum.json').exists():continue
        design=json.loads((root/'design.json').read_text())
        path=root/'b_examples.jsonl';meta=export_sft(root/'reference-support_candidates','reference',path)
        feedback={(r['task_id'],r['step']):r['observation'].strip()
                  for r in read_jsonl(root/'reference-support_candidates/steps.jsonl')}
        rows=[r for r in read_jsonl(path) if feedback[r['task_id'],r['step']]!='Nothing happens.'
              and encode_example(tokenizer,r,2048) is not None]
        groups=defaultdict(list)
        for i,r in enumerate(rows):groups[r['task_id']].append(i)
        if len(groups)<24 or len(rows)<200:raise ValueError('Reference data gate failed; keep query closed')
        rng=random.Random(777);queues={g:[] for g in groups};order=[]
        while len(order)<512:
            cycle=sorted(groups);rng.shuffle(cycle)
            for g in cycle:
                if not queues[g]:queues[g]=list(groups[g]);rng.shuffle(queues[g])
                order.append(queues[g].pop())
                if len(order)==512:break
        schedule=[dict(rows[i],shared_index=k,original_example_index=i) for k,i in enumerate(order)]
        counts=Counter(r['task_id'] for r in schedule);assert max(counts.values())-min(counts.values())<=1
        write_jsonl(path,rows);write_jsonl(root/'b_schedule.jsonl',schedule)
        diagnostic_dir=root/'reference-diagnostic'
        successes={r['attempt_id'] for r in read_jsonl(diagnostic_dir/'episodes.jsonl') if r['success'] and r['status']=='completed'}
        diagnostic_groups=defaultdict(list)
        for r in read_jsonl(diagnostic_dir/'steps.jsonl'):
            if r['attempt_id'] not in successes or r['observation'].strip()=='Nothing happens.':continue
            assert r['partition']=='dev' and r['adaptation_role']=='b_diagnostic'
            row={k:r[k] for k in ['task_id','task_group','partition','adaptation_role','step']}
            row['messages']=r['messages']+[dict(role='assistant',content=r['response'])]
            if encode_example(tokenizer,dict(row,partition='train'),2048) is not None:diagnostic_groups[r['task_id']].append(row)
        if len(diagnostic_groups)<design['diagnostic']['min_successful_groups']:
            raise ValueError('Diagnostic coverage gate failed; keep query closed')
        diagnostic=[r for t in sorted(diagnostic_groups)
                    for r in evenly_spaced(diagnostic_groups[t],min(8,len(diagnostic_groups[t])))]
        assert not {r['task_id'] for r in diagnostic}&set(groups)
        write_jsonl(root/'diagnostic.jsonl',diagnostic)
        save(root/'curriculum.json',dict(complete=True,training_examples=len(rows),training_tasks=len(groups),
            processed_examples=512,diagnostic_examples=len(diagnostic),diagnostic_tasks=len(diagnostic_groups),
            diagnostic_used_for_optimization=False,paid_api_calls=0,task_balanced_training_schedule=True))
    for family in FAMILIES:
        outputs.extend(ROOT/family/n for n in ['b_examples.jsonl','b_schedule.jsonl','diagnostic.jsonl','curriculum.json'])
    save(ROOT/'curricula.sha256.json',{str(p):file_sha(p) for p in outputs})


def diagnostic_invariance():
    """Verify process isolation and checkpoint immutability, not GPU bit repeatability."""
    audit=ROOT/'diagnostic-isolation.json'
    if audit.exists():
        assert json.loads(audit.read_text())['passed'];return
    trajectory=ROOT/'diagnostic-invariance/without-diagnostic'
    if not (trajectory/'trajectory.json').exists():
        raise ValueError('Prepare the bounded 8-step validation trajectory first')
    paths=list(trajectory.rglob('adapter_model.safetensors'))
    before={str(p):file_sha(p) for p in paths}
    engine.command([engine.MODEL_PY,'scripts/score_transfer_diagnostic.py','--trajectory',str(trajectory),
        '--data',str(ROOT/'cool/diagnostic.jsonl')],engine.environment(0),ROOT/'diagnostic-isolation.log')
    after={str(p):file_sha(p) for p in paths}
    records=json.loads((trajectory/'diagnostic-nll.json').read_text())
    passed=before==after and all(r['separate_process'] and r['checkpoint_unchanged'] for r in records.values())
    save(audit,dict(passed=passed,checkpoint_hashes=after,separate_process=True,query_used=False,
        validation_checkpoints=[0,8],training_script_restored=True,
        original_bitwise_comparison='Failed, including a same-GPU no-diagnostic repeat; not attributable to diagnostic alone',
        interpretation='No training process is alive while diagnostic reads its saved checkpoints'))
    if not passed:raise ValueError('Independent diagnostic modified a checkpoint')


def train_a():
    engine.ROOT=ROOT
    jobs=[dict(j,name=f"{j['condition']}-seed{j['a_seed']}") for j in json.loads((ROOT/'design.json').read_text())['initializations'] if not j['reuse']]
    def work(job,gpu,status):
        out=Path(job['a_adapter']);data=job['data']
        if not (out/'experiment.json').exists():
            engine.command([engine.MODEL_PY,'scripts/train_lora.py','--model',engine.MODEL,'--revision',engine.REV,
                '--data',data,'--config',str(ROOT/'a-lora.json'),'--seed',str(job['a_seed']),'--out',str(out)],
                engine.environment(gpu),ROOT/'logs'/f"A-{job['name']}.log")
        meta=json.loads((out/'experiment.json').read_text())
        assert meta['processed_examples_rank0']==256 and meta['training_data_sha256']==file_sha(data)
        assert meta['training_seed']==job['a_seed'] and meta['config']==json.loads((ROOT/'a-lora.json').read_text())
    engine.gpu_jobs('a-training',jobs,work)


def evaluate(root,gpu,case,checkpoint,role,step,calibration=False):
    revision=composition_revision(checkpoint)
    directory=root/'evaluation'/('calibration' if calibration else 'main')/case/role/f'step-{step:04d}'
    directory.mkdir(parents=True,exist_ok=True)
    manifest=root/('b_calibration.jsonl' if calibration else f'{role.lower()}_query.jsonl')
    config=json.loads(Path('configs/probe.json').read_text())
    config.update(protocol_id=f'transfer-factorial-v1-{root.name}-{role}',manifest=str(manifest),
        batch_size=8,max_steps=40,max_student_output_tokens=5120)
    cfg=directory/'config.json'
    if cfg.exists():assert json.loads(cfg.read_text())==config
    else:save(cfg,config)
    if (directory/'summary.json').exists() and json.loads((directory/'summary.json').read_text())['complete']:
        assert json.loads((directory/'run.json').read_text())['revision']==revision;return
    with engine.server(gpu,f'{root.name}-{case}-{role}-{step}',revision,directory/'server.log',checkpoint) as env:
        engine.command([engine.CPU_PY,'-m','learning_harness.batch_runner','--config',str(cfg),
            '--evaluation','--out',str(directory)],env,directory/'rollout.log')
    result=summarize(directory);assert result['complete'] and result['metrics']['none']['error_count']==0


def run_family(family,calibration):
    root=ROOT/family;engine.ROOT=root;design=json.loads((root/'design.json').read_text())
    jobs=[dict(j,name=f"{j['condition']}-seed{j['b_seed']}") for j in design['initializations']
          if not calibration or j['condition']=='base']
    def work(job,gpu,status):
        case=job['name'];trajectory=root/'trajectories'/case
        if not (trajectory/'trajectory.json').exists():
            args=[engine.MODEL_PY,'scripts/train_adaptation.py','--data',str(root/'b_schedule.jsonl'),
                '--config',str(root/'b-lora.json'),
                '--seed',str(job['b_seed']),'--out',str(trajectory)]
            if job['a_adapter']:args+=['--a-adapter',job['a_adapter']]
            status('training-B');engine.command(args,engine.environment(gpu),root/'logs'/f'B-{case}.log')
        meta=json.loads((trajectory/'trajectory.json').read_text())
        assert meta['complete'] and meta['processed_examples']==512 and meta['a_frozen_verified'] and meta['b_updated_verified']
        assert meta['data_sha256']==file_sha(root/'b_schedule.jsonl') and meta['b_seed']==job['b_seed']
        assert meta['config']==json.loads((root/'b-lora.json').read_text())
        status('scoring-held-out-actions')
        engine.command([engine.MODEL_PY,'scripts/score_transfer_diagnostic.py','--trajectory',str(trajectory),
            '--data',str(root/'diagnostic.jsonl')],engine.environment(gpu),root/'logs'/f'diagnostic-{case}.log')
        for step in ([0,128] if calibration else design['b_training']['checkpoints']):
            checkpoint=trajectory/f'step-{step:04d}'
            status(f'evaluating-B-{step}');evaluate(root,gpu,case,checkpoint,'B',step,calibration)
            if not calibration and step in [0,128]:
                status(f'evaluating-A-{step}');evaluate(root,gpu,case,checkpoint,'A',step)
    engine.gpu_jobs('calibration' if calibration else 'adaptation',jobs,work)
    if calibration:
        gains=[]
        for job in jobs:
            wins=[]
            for step in [0,128]:
                rows=read_jsonl(root/'evaluation/calibration'/job['name']/'B'/f'step-{step:04d}'/'episodes.jsonl')
                assert len(rows)==12 and all(r['status']=='completed' for r in rows)
                wins.append(sum(r['success'] for r in rows))
            gains.append(wins[1]-wins[0])
        save(root/'calibration-result.json',dict(gate_passed=sum(g>=2 for g in gains)>=2,additional_wins=gains,
            query_opened=False,rule='At least 2 extra wins out of 12 in at least 2 of 3 base seeds'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['references','validate','train-a','calibration','main','all'],default='all');args=p.parse_args()
    frozen()
    if args.stage=='all' and shutil.disk_usage(ROOT).free<10*1024**3:raise RuntimeError('Need at least 10 GiB free; no automatic checkpoint deletion')
    save(ROOT/'execution.json',dict(status='running',stage=args.stage,paid_api_calls=0))
    try:
        if args.stage in ['references','all']:references()
        frozen()
        if args.stage in ['validate','all']:diagnostic_invariance()
        if args.stage in ['train-a','all']:train_a()
        if args.stage in ['calibration','all']:
            for family in FAMILIES:run_family(family,True)
        if args.stage in ['main','all']:
            gates=[json.loads((ROOT/f/'calibration-result.json').read_text())['gate_passed'] for f in FAMILIES]
            if not all(gates):
                save(ROOT/'execution.json',dict(status='calibration_gate_failed',family_gates=dict(zip(FAMILIES,gates)),
                    query_opened=False,paid_api_calls=0));print('Calibration gate failed; query stays closed.',flush=True);return
            save(ROOT/'execution.json',dict(status='running_main',query_opened=True,paid_api_calls=0))
            for family in FAMILIES:
                frozen();run_family(family,False)
                engine.command(['python3','scripts/transfer_factorial_report.py','--family',family],engine.environment(0),ROOT/family/'report.log')
            engine.command(['python3','scripts/transfer_factorial_report.py'],engine.environment(0),ROOT/'report.log')
            save(ROOT/'execution.json',dict(status='completed',query_opened=True,paid_api_calls=0))
    except Exception as exc:
        save(ROOT/'execution.json',dict(status='failed',stage=args.stage,error_type=type(exc).__name__))
        raise


if __name__=='__main__':main()
