#!/usr/bin/env python3
"""Freeze a new two-family, source-by-selection experiment before new outcomes."""
from collections import defaultdict,Counter
import json
from pathlib import Path
from learning_harness.common import digest,file_sha,read_jsonl,write_jsonl

ROOT=Path('runs/transfer-factorial-v1')
OLD=Path('runs/api-harness-v1')
FAMILIES={'cool':'pick_cool_then_place_in_recep','heat':'pick_heat_then_place_in_recep'}
SEEDS=[101,202,303]
STEPS=[0,8,16,32,64,128]


def save(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def main():
    if (ROOT/'design.json').exists():raise ValueError('Design already frozen')
    from transformers import AutoTokenizer
    from train_lora import encode_example
    tokenizer=AutoTokenizer.from_pretrained('data/models/Qwen3-4B-Instruct-2507',local_files_only=True)
    pool=read_jsonl('data/manifests/alfworld/train.jsonl');index={r['task_id']:r for r in pool}
    blocked=set()
    manifests=[Path('data/manifests/alfworld')/(n+'.jsonl') for n in ['pilot','dev','probe','train_pilot']]
    manifests += [p for root in ['runs/dual5090-v2','runs/learning-speed-v1','runs/api-harness-v1']
                  for p in Path(root).glob('*.jsonl')]
    for p in manifests:
        for r in read_jsonl(p):
            if 'task_group' in r:blocked.add(r['task_group'])
            elif r.get('task_id') in index:blocked.add(index[r['task_id']]['task_group'])
    owned=set(blocked)
    def choose(family,n,role,partition):
        groups=defaultdict(list)
        for r in pool:
            if r['task_type']==family and r['task_group'] not in owned:groups[r['task_group']].append(r)
        chosen=sorted(groups,key=lambda g:digest([20261110,'transfer-factorial-v1',role,g]))[:n]
        if len(chosen)!=n:raise ValueError(f'Insufficient unused groups for {family}/{role}: {len(chosen)}')
        owned.update(chosen)
        return [dict(min(groups[g],key=lambda r:digest(['representative',r['task_id']])),
            partition=partition,purpose='development',original_partition='train',adaptation_role=role) for g in chosen]
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'corpora').mkdir(exist_ok=True)
    a_query=[r for family in ['pick_and_place_simple','pick_clean_then_place_in_recep','pick_heat_then_place_in_recep']
             for r in choose(family,8,'a_retention','probe')]
    write_jsonl(ROOT/'a_query.jsonl',a_query)
    initializations=[];coverage={};dependencies=[]
    for source in ['generic','api']:
        rows=read_jsonl(OLD/'corpora'/f'raw-{source}.jsonl')
        feedback={(r['task_id'],r['rollout_seed'],r['step']):r['observation'].strip()
                  for r in read_jsonl(OLD/'collection'/source/'steps.jsonl')}
        full=[r for r in rows if feedback[r['task_id'],r['rollout_seed'],r['step']]!='Nothing happens.'
              and encode_example(tokenizer,r,2048) is not None]
        # Match the original train_lora eligibility, including teacher views if present.
        assert not any('teacher_messages' in r for r in full)
        expected=json.loads((OLD/'data-selection.json').read_text())['eligible'][source]['eligible_examples']
        assert len(full)==expected
        write_jsonl(ROOT/'corpora'/f'full-{source}.jsonl',full)
        meta=json.loads((OLD/'corpora'/f'raw-{source}.jsonl.meta.json').read_text())
        save(ROOT/'corpora'/f'full-{source}.jsonl.meta.json',dict(meta,examples=len(full),
            regime='all eligible successful examples; fixed 256-example training budget',query_used=False))
        coverage[source]=dict(eligible_examples=len(full),task_groups=len({index[r['task_id']]['task_group'] for r in full}),
            tasks_by_type=dict(Counter(index[t]['task_type'] for t in {r['task_id'] for r in full})))
        for seed in SEEDS:
            for regime in ['matched','full']:
                adapter=(OLD/'a-checkpoints'/f'{source}-seed{seed}' if regime=='matched'
                         else ROOT/'a-checkpoints'/f'full-{source}-seed{seed}')
                data=(OLD/'corpora'/f'matched-{source}.jsonl' if regime=='matched'
                      else ROOT/'corpora'/f'full-{source}.jsonl')
                initializations.append(dict(condition=f'{regime}-{source}',a_seed=seed,b_seed=seed+1000,
                    a_adapter=str(adapter),data=str(data),reuse=regime=='matched'))
                if regime=='matched':dependencies += [adapter/'adapter_model.safetensors',adapter/'experiment.json',data]
    for seed in SEEDS:initializations.append(dict(condition='base',a_seed=None,b_seed=seed+1000,a_adapter=None,data=None,reuse=True))
    a_config=json.loads((OLD/'a-lora.json').read_text());save(ROOT/'a-lora.json',a_config)
    for short,family in FAMILIES.items():
        root=ROOT/short;root.mkdir(exist_ok=True)
        splits={'b_support_candidates':(48,'train'),'b_calibration':(12,'dev'),
                'b_diagnostic':(8,'dev'),'b_query':(32,'probe')}
        for role,(n,partition) in splits.items():
            # Only 52 unused cooling groups remain. Reuse its frozen teaching
            # tasks, reserving every unused group for calibration/diagnostic/query.
            rows=(read_jsonl('runs/learning-speed-v1/b_support_candidates.jsonl')
                  if short=='cool' and role=='b_support_candidates' else choose(family,n,role,partition))
            write_jsonl(root/f'{role}.jsonl',rows)
        write_jsonl(root/'a_query.jsonl',a_query)
        config=json.loads((OLD/'b-lora.json').read_text());config['checkpoints']=STEPS
        save(root/'b-lora.json',config)
        design=dict(protocol='transfer-factorial-v1',family=family,study_name=f'transfer-factorial-v1-{short}',
            status='frozen_before_new_reference_collection_and_query',initializations=initializations,
            conditions=['base','matched-generic','matched-api','full-generic','full-api'],
            b_training=dict(max_steps=128,processed_examples=512,checkpoints=STEPS,
                shared_schedule=str(root/'b_schedule.jsonl'),diagnostic=str(root/'diagnostic.jsonl')),
            evaluation=dict(b_query_tasks=32,a_query_tasks=24,a_retention_at=[0,128],batch_size=8,max_steps=40),
            diagnostic=dict(tasks=8,max_examples_per_task=8,min_successful_groups=6,partition='dev',
                optimizer_use=False,selection_use=False,checkpoints=STEPS,metric='per-task mean target-token NLL'),
            minimum_b_data=dict(successful_task_groups=24,eligible_action_examples=200),
            calibration_gate=dict(tasks=12,base_seeds=3,steps=[0,128],minimum_additional_wins=2,
                minimum_passing_seeds=2,query_closed_until_both_family_gates_pass=True),
            a_training=dict(processed_examples=256,max_steps=64,config=str(ROOT/'a-lora.json'),
                matched='Reuse the exact three-way common-success 13-group adapters; no A retraining',
                full='All eligible success-only A examples, same uniform-example sampling rule and optimizer budget'),
            primary_contrast='(full-api minus full-generic) minus (matched-api minus matched-generic): raw normalized AUC',
            secondary=['Source effects within each selection regime','gain AUC and initial success',
                'Held-out action NLL at each checkpoint; initial NLL and within-trajectory reduction',
                'A retention endpoints','B raw and gain AUC versus no-A base'],
            interpretation='Heat is within-A-family instance transfer; cool is an A-unseen goal family. Neither establishes pure plasticity.',
            limitations=['A data and seeds are reused; this is a fresh-query follow-up, not an independent A-data replication',
                'Cooling teaching tasks are reused because only 52 unused train-derived cooling groups remain',
                'Full versus matched changes coverage, action mix, corpus size and sample repetition together',
                'Equal processed examples do not imply equal token compute; report token budgets',
                'Three seeds; fixed-B sequence; two families; diagnostic reference is success-conditioned',
                'Gain AUC and NLL adjustment do not causally equalize prior knowledge',
                'Task-group disjointness does not imply room or pretraining disjointness'],
            official_final_tests_used=False,paid_api_calls=0)
        save(root/'design.json',design)
    master=dict(protocol='transfer-factorial-v1',status='frozen_before_execution',families=FAMILIES,
        new_A_trainings=6,reused_A_adapters=6,B_trajectories=30,
        planned_main_evaluation_episodes=30*(32*len(STEPS)+24*2),calibration_episodes=2*3*12*2,
        primary='Equal-weight mean of family-specific source-by-selection interactions in normalized raw AUC',
        inference='Paired crossed bootstrap over A/B seed indices shared across families and task groups within each family; 4000 draws',
        diagnostic='Descriptive prior-knowledge versus update-gain evidence; no pure-plasticity claim',
        reserved_families_preserved=['pick_two_obj_and_place','look_at_obj_in_light'],
        full_source_coverage=coverage,initializations=initializations,paid_api_calls=0,
        design_changes_based_on_previous_results=True,new_query_used_for_design=False)
    save(ROOT/'design.json',master)
    dependencies+=list(ROOT.rglob('*.json'))+list(ROOT.rglob('*.jsonl'))
    # Freeze upstream collection inputs and all newly selected game bytes too.
    dependencies += [OLD/'collection'/s/f for s in ['generic','api'] for f in ['episodes.jsonl','steps.jsonl']]
    for p in ROOT.rglob('*.jsonl'):
        for r in read_jsonl(p):
            if 'game_relpath' in r:
                game=Path('data/alfworld')/r['game_relpath'];assert file_sha(game)==r['game_sha256'];dependencies.append(game)
    save(ROOT/'inputs.sha256.json',{str(p):file_sha(p) for p in sorted(set(dependencies))})
    print(json.dumps(master,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
