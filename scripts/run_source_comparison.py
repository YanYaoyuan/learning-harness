#!/usr/bin/env python3
"""Frozen matched-data comparison: API-scaffold -> SFT versus ReAct -> SFT."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import re
import shutil

import run_api_harness as engine
from learning_harness.adaptation import composition_revision
from learning_harness.benchmark import SOURCE_FILES
from learning_harness.common import digest,file_sha,read_jsonl,write_jsonl
from learning_harness.source_comparison import target_view,match_sources
from audit_baseline_prompts import PATTERN,signature

ROOT=Path('runs/source-comparison-v1');OLD=Path('runs/transfer-factorial-v1')
SOURCES=['api','react'];SEEDS=[101,202,303];STEPS=[0,8,32,128];FAMILIES=['cool','heat']
engine.ROOT=ROOT


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.part');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(path)


def frozen():
    for name in ['inputs.sha256.json','training-inputs.sha256.json']:
        p=ROOT/name
        if p.exists():
            for path,sha in json.loads(p.read_text()).items():
                if file_sha(path)!=sha:raise ValueError('Frozen input changed: '+path)
    return json.loads((ROOT/'design.json').read_text())


def base_config(manifest,method,revision=engine.REV):
    return dict(protocol_id='source-comparison-v1',manifest=str(manifest),method=method,revision=revision,
        max_turns=40,max_calls=40,max_trials=1,max_new_tokens=128,history_turns=12,batch_size=8,
        temperature=0.,seeds=[11],model=dict(base_url_env='LH_STUDENT_BASE_URL',
            model_env='LH_STUDENT_MODEL',api_key_env='LH_STUDENT_API_KEY'))


def prepare():
    if (ROOT/'design.json').exists():
        if not (ROOT/'inputs.sha256.json').exists():raise ValueError('Incomplete preparation: inspect before resuming')
        frozen();return
    if shutil.disk_usage('.').free<8*1024**3:raise ValueError('Need 8 GiB free; no automatic deletion')
    ROOT.mkdir(parents=True,exist_ok=True)
    raw={'train':read_jsonl('runs/api-harness-v1/train.jsonl'),'a_query':read_jsonl(OLD/'a_query.jsonl')}
    raw.update({f'{f}_query':read_jsonl(OLD/f/'b_query.jsonl') for f in FAMILIES})
    initial={}
    logs=[Path('runs/api-harness-v1/collection/api/steps.jsonl'),
        OLD/'cool/evaluation/main/base-seed1101/A/step-0000/steps.jsonl']+[
        OLD/f/'evaluation/main/base-seed1101/B/step-0000/steps.jsonl' for f in FAMILIES]
    for p in logs:
        with p.open() as handle:
            for line in handle:
                r=json.loads(line)
                initial.setdefault(r['task_id'],r['messages'][1]['content'])
    prompts=json.loads(Path('third_party/ReAct/prompts/alfworld_3prompts.json').read_text())
    demo_signatures={}
    for key,value in prompts.items():
        if key.startswith('react_') and key.endswith(('_0','_1')):
            m=re.search(PATTERN,value,re.S)
            if not m:raise ValueError('Unparsed demonstration')
            demo_signatures[key]=signature(*m.groups())
    excluded=[];selected={}
    for name,rows in raw.items():
        kept=[]
        for row in rows:
            m=re.search(PATTERN,initial[row['task_id']],re.S)
            if not m:raise ValueError('Unparsed public reset')
            matches=[key for key,sig in demo_signatures.items() if sig==signature(*m.groups())]
            if matches:excluded.append(dict(split=name,task_id=row['task_id'],demonstrations=matches))
            else:kept.append(row)
        if not kept:raise ValueError('Empty split after demonstration audit')
        selected[name]=kept;write_jsonl(ROOT/f'{name}.jsonl',kept)
    for f in FAMILIES:
        # Reuse already frozen B curricula, but never reuse old evaluation scores.
        shutil.copyfile(OLD/f/'b_schedule.jsonl',ROOT/f'{f}_schedule.jsonl')
        config=json.loads((OLD/f/'b-lora.json').read_text());config['checkpoints']=STEPS
        save(ROOT/f'{f}-lora.json',config)
    train_groups={r['task_group'] for r in selected['train']}
    query_groups={r['task_group'] for name,rows in selected.items() if name!='train' for r in rows}
    if train_groups&query_groups:raise ValueError('A training/query overlap')
    for f in FAMILIES:
        groups={r['task_group'] for r in read_jsonl(OLD/f/'b_support_candidates.jsonl')}
        if groups&query_groups or groups&train_groups:raise ValueError('B teaching overlap')
    shutil.copyfile('runs/api-harness-v1/api-support.jsonl',ROOT/'api-support.jsonl')
    save(ROOT/'a-lora.json',json.loads(Path('runs/api-harness-v1/a-lora.json').read_text()))
    for source in SOURCES:
        config=base_config(ROOT/'train.jsonl','json_direct' if source=='api' else 'react_retry')
        config.update(source=source,seeds=[11,23],temperature=.7)
        if source=='api':config.update(support_bank=str(ROOT/'api-support.jsonl'),support_bank_sha256=file_sha(ROOT/'api-support.jsonl'))
        save(ROOT/'configs'/f'collect-{source}.json',config)
    design=dict(protocol='source-comparison-v1',date='2026-11-11',status='frozen_before_fresh_collection',
        question='Does API-assisted collection produce a better learned initialization than strong ReAct self-collection under matched training exposure?',
        sources=SOURCES,training_seeds=SEEDS,collection=dict(tasks=len(selected['train']),seeds=[11,23],
            max_calls=40,max_trials=1,temperature=.7,same_request_rng=True,
            api='Original JSON actor + cached public-reset-only strategy, first four turns',
            react='Two fixed public task-family examples + think/action, no teacher card',
            fresh_rollouts_both_sources=True,paid_api_calls=0),
        target_view='Public initial observation + last 12 non-thought action/observation pairs; canonical JSON action labels. Remove all cards, examples and thoughts.',
        matching=dict(unit='successful task/environment-seed pair',action_count='minimum eligible count per pair; evenly spaced',minimum_task_groups=8,
            max_sequence_length=2048,exclusions='invalid format, thoughts, literal Nothing happens targets, overlength; retain failed physical actions in history'),
        a_training=dict(new_adapters=6,max_steps=64,processed_examples=256,paired_sample_keys=True),
        b_training=dict(new_trajectories=12,reused_no_A_trajectories=6,max_steps=128,processed_examples=512,
            checkpoints=STEPS,families=FAMILIES,parameterization='Frozen A + fresh B LoRA. Tests additive adaptation, not unrestricted same-parameter plasticity.'),
        evaluation=dict(tasks={k:len(v) for k,v in selected.items() if k!='train'},seeds=[11],temperature=0.,
            max_calls=40,max_trials=1,json_direct_B_steps=STEPS,react_B_steps=[0,128],
            A_retention='JSON at 0 and 128; ReAct at A start only. Shared A start evaluated once on cool branch.',
            all_scores_fresh=True,official_final_tests_used=False),
        primary='API minus ReAct: equal-weight cool/heat mean normalized raw success AUC under common JSON execution',
        secondary=['Initial success, endpoint success, baseline-subtracted gain AUC','Per-family curves; heat within-family versus cool A-unseen family',
            'Common ReAct execution at B start/end','A retention','Source collection yield and real input/output costs','No-A reference'],
        inference='Paired crossed bootstrap of 3 training-seed indices and task groups within each family, 4000 draws; preserve seed pairing across families',
        demonstration_audit=dict(excluded=excluded,rule='Public goal plus visible numbered receptacle multiset'),
        limitations=['Reused exploratory query tasks, not new confirmation evidence','Mutual-success selection changes the population',
            'Equal samples/updates do not imply identical target/input token FLOPs','The two collection harnesses differ in examples and action interface; this is a pipeline comparison, not an isolated card effect',
            'ReAct evaluation supplies cool demonstrations; label this few-shot execution, not zero-shot goal transfer',
            'Frozen A plus B and three seeds cannot establish pure plasticity or RSI','BF16 batching can change trajectories despite paired request RNG'],
        planned_evaluation_task_runs=9*(sum(len(selected[f'{f}_query'])*6 for f in FAMILIES)+len(selected['a_query'])*4))
    save(ROOT/'design.json',design)
    paths=[Path(p) for p in SOURCE_FILES]+[Path('src/learning_harness/source_comparison.py'),
        Path('scripts/collect_source_comparison.py'),Path(__file__),Path('scripts/train_lora.py'),
        Path('scripts/train_adaptation.py'),Path('src/learning_harness/adaptation.py'),Path('scripts/run_api_harness.py')]
    paths+=list(ROOT.glob('*.json'))+list(ROOT.glob('*.jsonl'))+list((ROOT/'configs').glob('*.json'))
    # Pin the reused no-A weights and curricula; do not reuse their old scores.
    for f in FAMILIES:
        for seed in SEEDS:
            base=OLD/f/'trajectories'/f'base-seed{seed+1000}'
            paths += [base/'trajectory.json']
            for step in STEPS:paths += [base/f'step-{step:04d}'/'composition.json',base/f'step-{step:04d}'/'B/adapter_model.safetensors']
    save(ROOT/'inputs.sha256.json',{str(p):file_sha(p) for p in sorted(set(paths))})
    print(json.dumps(design,ensure_ascii=False),flush=True)


def collect():
    jobs=[dict(name=s,source=s) for s in SOURCES]
    def work(job,gpu,status):
        source=job['source'];out=ROOT/'collection'/source;out.mkdir(parents=True,exist_ok=True)
        with engine.server(gpu,'source-comparison-base',engine.REV,out/'server.log') as env:
            engine.command([engine.CPU_PY,'scripts/collect_source_comparison.py','--config',
                str(ROOT/'configs'/f'collect-{source}.json'),'--out',str(out)],env,out/'worker.log')
    engine.gpu_jobs('collection',jobs,work)


def build():
    frozen()
    if (ROOT/'selection.json').exists():
        if not (ROOT/'training-inputs.sha256.json').exists():raise ValueError('Incomplete corpus build: inspect before resuming')
        return json.loads((ROOT/'selection.json').read_text())['gate_passed']
    from transformers import AutoTokenizer
    from train_lora import encode_example,fixed_schedule
    tokenizer=AutoTokenizer.from_pretrained(engine.MODEL,local_files_only=True)
    index={t['task_id']:t for t in read_jsonl(ROOT/'train.jsonl')};sources={};reports={}
    for source in SOURCES:
        directory=ROOT/'collection'/source;tasks=read_jsonl(directory/'tasks.jsonl')
        expected={(t,s) for t in index for s in [11,23]}
        if {(r['task_id'],r['seed']) for r in tasks}!=expected or len(tasks)!=len(expected):raise ValueError('Incomplete collection')
        successes={r['run_id']:r for r in tasks if r['success']};events=defaultdict(list)
        all_events={r['run_id']:[0,0,0] for r in tasks};by_task=defaultdict(set)
        with (directory/'events.jsonl').open() as handle:
            for line in handle:
                e=json.loads(line);rid=e['run_id']
                if rid not in all_events:continue
                v=all_events[rid];v[0]+=1;v[1]+=e['input_tokens'];v[2]+=e['output_tokens']
                if rid in successes:events[rid].append(e)
        rows=[];overlength=0
        for rid,es in events.items():
            for r in target_view(es,index,source):
                if encode_example(tokenizer,r,2048) is None:overlength+=1
                else:rows.append(r)
        for r in tasks:
            if all_events[r['run_id']]!=[r['calls'],r['input_tokens'],r['output_tokens']]:raise ValueError('Collection token accounting mismatch')
            if r['calls']>40 or len(r['trials'])!=1:raise ValueError('Collection budget violation')
        sources[source]=rows;write_jsonl(ROOT/'corpora'/f'eligible-{source}.jsonl',rows)
        reports[source]=dict(tasks=len(tasks),successful_episodes=len(successes),success_rate=len(successes)/len(tasks),
            eligible_examples=len(rows),overlength_dropped=overlength,input_tokens=sum(r['input_tokens'] for r in tasks),
            output_tokens=sum(r['output_tokens'] for r in tasks),calls=sum(r['calls'] for r in tasks))
    matched,selection=match_sources(sources,index)
    selection.update(collection=reports,query_used_for_matching=False)
    save(ROOT/'selection.json',selection)
    if selection['gate_passed']:
        for source,rows in matched.items():
            path=ROOT/'corpora'/f'matched-{source}.jsonl';write_jsonl(path,rows)
            save(str(path)+'.meta.json',dict(source=source,examples=len(rows),common_task_groups=selection['common_task_groups'],
                source_run_fingerprint=json.loads((ROOT/'collection'/source/'run.json').read_text())['fingerprint'],
                target_view='canonical public JSON actions, no hints/examples/thoughts',query_used=False))
        schedules={}
        for seed in SEEDS:
            order=fixed_schedule(selection['matched_examples_each'],256,seed)
            keys=[(matched['api'][i]['task_id'],matched['api'][i]['rollout_seed'],matched['api'][i]['matched_index']) for i in order]
            assert keys==[(matched['react'][i]['task_id'],matched['react'][i]['rollout_seed'],matched['react'][i]['matched_index']) for i in order]
            schedules[str(seed)]=dict(processed_examples=256,unique_examples=len(set(order)),paired_training_keys_sha256=digest(keys))
        save(ROOT/'paired-a-schedules.json',schedules)
    paths=list((ROOT/'corpora').glob('*'))+[ROOT/'selection.json']
    if (ROOT/'paired-a-schedules.json').exists():paths.append(ROOT/'paired-a-schedules.json')
    save(ROOT/'training-inputs.sha256.json',{str(p):file_sha(p) for p in paths})
    print(json.dumps(selection),flush=True)
    return selection['gate_passed']


def train_a():
    frozen()
    if not json.loads((ROOT/'selection.json').read_text())['gate_passed']:raise ValueError('Matching gate not passed')
    jobs=[dict(name=f'{s}-seed{seed}',source=s,seed=seed) for seed in SEEDS for s in SOURCES]
    def work(job,gpu,status):
        out=ROOT/'a-checkpoints'/job['name'];data=ROOT/'corpora'/f'matched-{job["source"]}.jsonl'
        if not (out/'experiment.json').exists():
            engine.command([engine.MODEL_PY,'scripts/train_lora.py','--model',engine.MODEL,'--revision',engine.REV,
                '--data',str(data),'--config',str(ROOT/'a-lora.json'),'--seed',str(job['seed']),'--out',str(out)],
                engine.environment(gpu),ROOT/'logs'/f'A-{job["name"]}.log')
        meta=json.loads((out/'experiment.json').read_text())
        assert meta['processed_examples_rank0']==256 and meta['overlength_dropped']==0
        assert meta['training_data_sha256']==file_sha(data) and meta['training_seed']==job['seed']
        assert meta['config']==json.loads((ROOT/'a-lora.json').read_text())
    engine.gpu_jobs('a-training',jobs,work)


def evaluate(job,gpu,checkpoint,role,step,harness):
    revision=composition_revision(checkpoint);family=job['family'];case=f'{job["source"]}-seed{job["seed"]}'
    out=ROOT/'evaluation'/family/case/harness/role/f'step-{step:04d}';out.mkdir(parents=True,exist_ok=True)
    manifest=ROOT/(f'{family}_query.jsonl' if role=='B' else 'a_query.jsonl')
    cfg=base_config(manifest,harness,revision);path=out/'config.json'
    if path.exists():
        if json.loads(path.read_text())!=cfg:raise ValueError('Evaluation config changed')
    else:save(path,cfg)
    if (out/'tasks.jsonl').exists() and len(read_jsonl(out/'tasks.jsonl'))==len(read_jsonl(manifest)):
        if json.loads((out/'run.json').read_text())['revision']!=revision:raise ValueError('Evaluated model changed')
        return
    with engine.server(gpu,f'{family}-{case}-{harness}-{role}-{step}',revision,out/'server.log',checkpoint) as env:
        engine.command([engine.CPU_PY,'-m','learning_harness.benchmark','--config',str(path),'--out',str(out)],env,out/'worker.log')


def adapt():
    frozen()
    jobs=[dict(name=f'{f}-{s}-seed{seed}',family=f,source=s,seed=seed) for seed in SEEDS for f in FAMILIES for s in ['base']+SOURCES]
    def work(job,gpu,status):
        f=job['family'];source=job['source'];seed=job['seed'];case=f'{source}-seed{seed}'
        data=ROOT/f'{f}_schedule.jsonl';config=ROOT/f'{f}-lora.json'
        if source=='base':trajectory=OLD/f/'trajectories'/f'base-seed{seed+1000}'
        else:
            trajectory=ROOT/'trajectories'/f/case
            if not (trajectory/'trajectory.json').exists():
                status('training-B')
                engine.command([engine.MODEL_PY,'scripts/train_adaptation.py','--a-adapter',
                    str(ROOT/'a-checkpoints'/case),'--data',str(data),'--config',str(config),
                    '--seed',str(seed+1000),'--out',str(trajectory)],engine.environment(gpu),ROOT/'logs'/f'B-{f}-{case}.log')
        meta=json.loads((trajectory/'trajectory.json').read_text())
        assert meta['complete'] and meta['processed_examples']==512 and meta['a_frozen_verified'] and meta['b_updated_verified']
        assert meta['data_sha256']==file_sha(data) and meta['b_seed']==seed+1000
        expected=json.loads(config.read_text());actual=dict(meta['config'])
        actual.pop('checkpoints');expected.pop('checkpoints');assert actual==expected
        for step in STEPS:
            checkpoint=trajectory/f'step-{step:04d}'
            status(f'evaluating-B-json-{step}');evaluate(job,gpu,checkpoint,'B',step,'json_direct')
            if step in [0,128]:
                status(f'evaluating-B-react-{step}');evaluate(job,gpu,checkpoint,'B',step,'react_retry')
                if step==128 or f=='cool':
                    status(f'evaluating-A-json-{step}');evaluate(job,gpu,checkpoint,'A',step,'json_direct')
                if step==0 and f=='cool':
                    status('evaluating-A-react-0');evaluate(job,gpu,checkpoint,'A',step,'react_retry')
    engine.gpu_jobs('adaptation',jobs,work)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','collect','build','train-a','adapt','all'],default='all');args=p.parse_args()
    prepare();frozen()
    if args.stage=='prepare':return
    save(ROOT/'execution.json',dict(status='running',stage=args.stage,paid_api_calls=0))
    try:
        if args.stage in ['all','collect']:collect()
        if args.stage in ['all','build']:
            if not build():
                save(ROOT/'execution.json',dict(status='matching_gate_failed',paid_api_calls=0));return
        if args.stage in ['all','train-a']:train_a()
        if args.stage in ['all','adapt']:adapt()
        save(ROOT/'execution.json',dict(status='completed' if args.stage=='all' else 'stage_completed',stage=args.stage,paid_api_calls=0))
    except Exception as exc:
        save(ROOT/'execution.json',dict(status='failed',stage=args.stage,error_type=type(exc).__name__,paid_api_calls=0));raise


if __name__=='__main__':main()
