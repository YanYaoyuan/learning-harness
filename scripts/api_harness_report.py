#!/usr/bin/env python3
"""Audit measured API-scaffold experiments and export portable scientific figures."""
from collections import defaultdict
import argparse
import csv
import json
import os
from pathlib import Path
import random
import statistics

os.environ.setdefault('MPLCONFIGDIR',str(Path('.cache/matplotlib').absolute()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from learning_harness.common import file_sha,read_jsonl
from matched_study import crossed_interval

ROOT=Path('runs/api-harness-v1')
CONDITIONS=['none','generic','api']
SEEDS=[1101,1202,1303]
COLORS={'none':'#777777','generic':'#4477AA','api':'#CC6677'}
LABELS={'none':'No scaffold','generic':'Generic scaffold','api':'API task strategy'}


def save(fig,path):
    for ext in ['png','pdf','svg']:fig.savefig(path.with_suffix('.'+ext),dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def csv_out(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def group_interval(values,draws=4000):
    rng=random.Random(20261109)
    samples=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(draws))
    return [samples[int(.025*draws)],samples[int(.975*draws)]]


def collection_result():
    records=[];cells={};episodes={}
    for c in CONDITIONS:
        p=ROOT/'collection'/c
        summary=json.loads((p/'summary.json').read_text());assert summary['complete']
        rows=read_jsonl(p/'episodes.jsonl');assert len(rows)==144
        assert all(r['status']=='completed' for r in rows)
        index={(r['task_id'],r['rollout_seed']):r for r in rows}
        assert len(index)==144;episodes[c]=index
        cells[c]=dict(wins=sum(r['success'] for r in rows),episodes=144,
            task_groups=len({r['task_group'] for r in rows}),
            input_tokens=sum(r['input_tokens'] for r in rows),output_tokens=sum(r['output_tokens'] for r in rows),
            success_by_type={family:statistics.mean(int(r['success']) for r in rows if r['task_type']==family)
                for family in sorted({r['task_type'] for r in rows})})
        records.extend(dict(condition=c,task_id=r['task_id'],task_group=r['task_group'],
            rollout_seed=r['rollout_seed'],task_type=r['task_type'],success=int(r['success']),
            steps=r['steps'],input_tokens=r['input_tokens'],output_tokens=r['output_tokens']) for r in rows)
    assert episodes['none'].keys()==episodes['generic'].keys()==episodes['api'].keys()
    contrasts={}
    for other in ['generic','none']:
        groups=defaultdict(list)
        for key,r in episodes['api'].items():
            groups[r['task_group']].append(int(r['success'])-int(episodes[other][key]['success']))
        values=[statistics.mean(v) for v in groups.values()]
        contrasts['api minus '+other]=dict(task_group_mean_delta=statistics.mean(values),
            cluster_bootstrap_95=group_interval(values),paired_task_groups=len(values))
    teacher=json.loads((ROOT/'teacher-generation.json').read_text())
    return dict(cells=cells,contrasts=contrasts,teacher=teacher,episode_scores=records,
        cost_basis='Undiscounted API list-price estimate; local GPU costs excluded')


def load_evaluation(case,role,step,expected):
    p=ROOT/'evaluation'/case/role/f'step-{step:04d}'
    s=json.loads((p/'summary.json').read_text());assert s['complete'] and s['metrics']['none']['error_count']==0
    rows=read_jsonl(p/'episodes.jsonl')
    assert len(rows)==24 and {r['task_id'] for r in rows}==expected
    assert all(r['status']=='completed' and not r['support_injected'] and r['arm']=='none' for r in rows)
    return {r['task_id']:r for r in rows}


def main_result():
    design=json.loads((ROOT/'design.json').read_text());steps=design['b_training']['checkpoints']
    b_ids=sorted(r['task_id'] for r in read_jsonl(ROOT/'b_query.jsonl'))
    a_ids=sorted(r['task_id'] for r in read_jsonl(ROOT/'a_query.jsonl'))
    scores={};records=[];audits={};b_hashes={s:set() for s in SEEDS};orders=set();tokens=set();counts=set()
    a_orders={s:set() for s in SEEDS};a_config=json.loads((ROOT/'a-lora.json').read_text())
    for c in CONDITIONS:
        for seed in SEEDS:
            case=f'{c}-seed{seed}';p=ROOT/'trajectories'/case
            meta=json.loads((p/'trajectory.json').read_text())
            assert meta['complete'] and meta['processed_examples']==512 and meta['a_frozen_verified'] and meta['b_updated_verified']
            assert meta['data_sha256']==design['b_training']['shared_schedule_sha256'] and meta['b_seed']==seed
            assert meta['initialization']['exact_zero_b_logit_check']
            b_hashes[seed].add(meta['initialization']['b_initial_parameter_sha256'])
            orders.add(meta['actual_order_sha256']);tokens.add(json.dumps(meta['token_accounting'],sort_keys=True))
            counts.add(meta['initialization']['trainable_parameters'])
            ameta=json.loads((ROOT/'a-checkpoints'/f'{c}-seed{seed-1000}'/'experiment.json').read_text())
            assert ameta['processed_examples_rank0']==256 and ameta['config']==a_config
            assert ameta['training_data_sha256']==file_sha(ROOT/'corpora'/f'matched-{c}.jsonl')
            a_orders[seed].add(ameta['schedule_sha256'])
            audits[case]=dict(initial_b_material_nll=meta['initial_b_material_nll'],
                frozen_a_verified=True,identical_fresh_b_logits_verified=True,
                a_input_tokens=ameta['input_tokens_rank0'],a_target_tokens=ameta['target_tokens_rank0'],
                b_token_accounting=meta['token_accounting'])
            for role,task_ids,observed in [('B',b_ids,steps),('A',a_ids,[0,128])]:
                for k in observed:
                    rows=load_evaluation(case,role,k,set(task_ids));scores[c,seed,role,k]=rows
                    records.extend(dict(condition=c,b_seed=seed,role=role,updates=k,task_id=t,
                        task_type=r['task_type'],success=int(r['success'])) for t,r in sorted(rows.items()))
    assert len(orders)==len(tokens)==len(counts)==1
    assert all(len(v)==1 for v in b_hashes.values()) and all(len(v)==1 for v in a_orders.values())
    curves={};auc={};gain={};retention={};seed_records=[];bands={}
    for c in CONDITIONS:
        curves[c]=[];auc[c]=[];gain[c]=[];retention[c]=[];bands[c]=[]
        for k in steps:
            bands[c].append(crossed_interval([[int(scores[c,s,'B',k][t]['success']) for t in b_ids] for s in SEEDS]))
        for seed in SEEDS:
            matrix=np.array([[int(scores[c,seed,'B',k][t]['success']) for t in b_ids] for k in steps],dtype=float)
            integrate=np.trapezoid if hasattr(np,'trapezoid') else np.trapz
            area=integrate(matrix,x=steps,axis=0)/128
            auc[c].append(area.tolist());gain[c].append((area-matrix[0]).tolist());curves[c].append(matrix.mean(axis=1).tolist())
            old=[statistics.mean(int(r['success']) for r in scores[c,seed,'A',k].values()) for k in [0,128]]
            retention[c].append(old)
            seed_records.append(dict(condition=c,b_seed=seed,initial_b_success=float(matrix[0].mean()),
                final_b_success=float(matrix[-1].mean()),normalized_auc=float(area.mean()),
                gain_auc=float((area-matrix[0]).mean()),initial_a_success=old[0],final_a_success=old[-1],
                initial_b_material_nll=audits[f'{c}-seed{seed}']['initial_b_material_nll']['token_mean']))
    contrasts={}
    for other in ['generic','none']:
        raw=np.array(auc['api'])-np.array(auc[other]);improvement=np.array(gain['api'])-np.array(gain[other])
        start=np.array(curves['api'])[:,0]-np.array(curves[other])[:,0]
        assert abs(raw.mean()-start.mean()-improvement.mean())<1e-10
        contrasts['api minus '+other]=dict(raw_auc_delta=float(raw.mean()),raw_auc_per_seed=raw.mean(axis=1).tolist(),
            raw_auc_95=crossed_interval(raw.tolist()),gain_auc_delta=float(improvement.mean()),
            gain_auc_per_seed=improvement.mean(axis=1).tolist(),gain_auc_95=crossed_interval(improvement.tolist()),
            initial_success_per_seed=start.tolist())
    by_type={}
    families=sorted({r['task_type'] for r in read_jsonl(ROOT/'a_query.jsonl')})
    for family in families:
        by_type[family]={c:[[statistics.mean(int(r['success']) for r in scores[c,s,'A',k].values()
            if r['task_type']==family) for k in [0,128]] for s in SEEDS] for c in CONDITIONS}
    return dict(complete=True,design=design,steps=steps,b_seeds=SEEDS,curves=curves,
        pointwise_95=bands,normalized_auc=auc,gain_auc=gain,contrasts=contrasts,
        a_retention=retention,a_retention_by_type=by_type,initialization_audit=audits,
        shared_b_order_initialization_tokens_verified=True,a_matched_index_schedules_verified=True,
        episode_scores=records,seed_metrics=seed_records,
        training_data=json.loads((ROOT/'data-selection.json').read_text()),
        collection=collection_result(),official_final_tests_used=False,
        interpretation='Conditional fixed-material adaptation pilot; not a demonstration of general plasticity, RSI, or agentic RL')


def render(data,out):
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    collection=data['collection'];cells=collection['cells']
    csv_out(out/'collection-episode-scores.csv',collection['episode_scores'])
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for i,c in enumerate(CONDITIONS):
        value=cells[c]['wins']/144*100
        axes[0].bar(i,value,color=COLORS[c],width=.65)
        axes[0].text(i,value+2,f"{cells[c]['wins']}/144",ha='center',fontsize=9)
    axes[0].set_xticks(range(3),[LABELS[c] for c in CONDITIONS],rotation=12)
    axes[0].set_ylim(0,105);axes[0].set_ylabel('Collection success (%)');axes[0].set_title('Actual supported collection',loc='left',fontweight='bold')
    axes[1].axvline(0,color='#888888',linestyle='--')
    for i,(label,c) in enumerate(collection['contrasts'].items()):
        lo,hi=np.array(c['cluster_bootstrap_95'])*100;mean=c['task_group_mean_delta']*100
        axes[1].plot([lo,hi],[i,i],color='#26374A');axes[1].scatter(mean,i,color='#26374A',marker='D')
    axes[1].set_yticks(range(2),list(collection['contrasts']));axes[1].set_xlabel('Task-group mean difference (pp)')
    axes[1].set_title('Paired collection contrasts',loc='left',fontweight='bold');axes[1].grid(axis='x',color='#E7E7E7')
    fig.subplots_adjust(bottom=.25,wspace=.6)
    fig.text(.5,.015,'72 tasks, 71 groups, 2 rollout seeds/source; same local student. Intervals: paired task-group bootstrap, 95%.',ha='center',fontsize=8)
    save(fig,out/'collection_effects')
    families=sorted(cells['none']['success_by_type']);x=np.arange(len(families));fig,ax=plt.subplots(figsize=(8.4,4.6))
    for i,c in enumerate(CONDITIONS):ax.bar(x+(i-1)*.24,[cells[c]['success_by_type'][f]*100 for f in families],width=.23,color=COLORS[c],label=LABELS[c])
    ax.set_xticks(x,['Place','Clean + place','Heat + place']);ax.set_ylim(0,105);ax.set_ylabel('Collection success (%)')
    ax.set_title('Collection by goal family',loc='left',fontweight='bold');ax.legend(frameon=False);fig.subplots_adjust(bottom=.18)
    fig.text(.5,.015,'48 actual episodes per family/source. Success does not directly measure subsequent learning.',ha='center',fontsize=8)
    save(fig,out/'collection_by_type')
    if data['stage']=='collection':return
    result=data['result'];steps=result['steps'];curves=result['curves']
    csv_out(out/'episode-scores.csv',result['episode_scores']);csv_out(out/'seed-metrics.csv',result['seed_metrics'])
    curve_rows=[dict(condition=c,b_seed=s,updates=k,success=curves[c][i][j])
        for c in CONDITIONS for i,s in enumerate(SEEDS) for j,k in enumerate(steps)]
    csv_out(out/'curve-values.csv',curve_rows)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for c in CONDITIONS:
        arr=np.array(curves[c])*100
        for row in arr:
            axes[0].plot(steps,row,color=COLORS[c],alpha=.28,linewidth=.8,marker='o',markersize=2.5)
            axes[1].plot(steps,row-row[0],color=COLORS[c],alpha=.28,linewidth=.8,marker='o',markersize=2.5)
        axes[0].plot(steps,arr.mean(axis=0),color=COLORS[c],label=LABELS[c],linewidth=2.1,marker='o')
        lo,hi=np.array(result['pointwise_95'][c]).T*100;axes[0].fill_between(steps,lo,hi,color=COLORS[c],alpha=.1)
        axes[1].plot(steps,(arr-arr[:,[0]]).mean(axis=0),color=COLORS[c],label=LABELS[c],linewidth=2.1,marker='o')
    for ax in axes:ax.set_xticks(steps);ax.set_xlabel('B-stage updates');ax.grid(color='#E7E7E7',linewidth=.5)
    axes[0].set_ylim(-2,102);axes[0].set_ylabel('Unsupported B success (%)');axes[0].set_title('Same B material and sample order',loc='left',fontweight='bold')
    axes[1].set_ylabel('Change from own start (pp)');axes[1].set_title('Adaptation gain',loc='left',fontweight='bold');axes[1].legend(frameon=False,fontsize=8)
    fig.subplots_adjust(bottom=.22,wspace=.3);fig.text(.5,.015,'24 new B tasks; 3 paired seeds. Raw-success bands are pointwise 95% crossed bootstrap. Dots are actual checkpoints; lines connect observations.',ha='center',fontsize=8)
    save(fig,out/'adaptation_curves')
    fig,axes=plt.subplots(1,2,figsize=(11.8,3.9),sharey=True)
    for ax,prefix,title in [(axes[0],'raw','Raw learning-curve AUC'),(axes[1],'gain','Gain relative to own start')]:
        ax.axvline(0,color='#888888',linestyle='--')
        for i,(label,c) in enumerate(result['contrasts'].items()):
            lo,hi=np.array(c[prefix+'_auc_95'])*100;mean=c[prefix+'_auc_delta']*100
            ax.plot([lo,hi],[i,i],color='#26374A',linewidth=1.6);ax.scatter(mean,i,color='#26374A',marker='D',zorder=3)
            for shift,color,value in zip([-.12,0,.12],['#4477AA','#EE7733','#228833'],c[prefix+'_auc_per_seed']):
                ax.scatter(value*100,i+shift,color=color,s=24)
        ax.set_yticks(range(2),list(result['contrasts']));ax.set_xlabel('API minus comparator (pp)');ax.set_title(title,loc='left',fontweight='bold');ax.grid(axis='x',color='#E7E7E7')
    fig.subplots_adjust(left=.19,bottom=.24,wspace=.25)
    fig.text(.5,.015,'Mean diamonds; seed dots; paired crossed-bootstrap 95% intervals. Raw AUC includes initial ability; gain AUC depends on headroom.',ha='center',fontsize=8)
    save(fig,out/'auc_effects')
    fig,ax=plt.subplots(figsize=(8.4,4.6))
    for c in CONDITIONS:
        arr=np.array(result['a_retention'][c])*100
        for row in arr:ax.plot([0,128],row,color=COLORS[c],alpha=.3,linewidth=1,marker='o',markersize=3)
        ax.plot([0,128],arr.mean(axis=0),color=COLORS[c],marker='o',linewidth=2.1,label=LABELS[c])
    ax.set_xticks([0,128]);ax.set_ylim(-2,102);ax.set_ylabel('Unsupported A success (%)');ax.set_xlabel('B-stage updates')
    ax.set_title('Old-family retention: two observed endpoints',loc='left',fontweight='bold');ax.legend(frameon=False);ax.grid(color='#E7E7E7')
    fig.subplots_adjust(bottom=.2);fig.text(.5,.015,'24 held-out A tasks (8/family), 3 paired seeds. No intermediate A evaluations; the segment does not estimate a retention trajectory.',ha='center',fontsize=8)
    save(fig,out/'a_retention_endpoints')
    fig,axes=plt.subplots(1,3,figsize=(11.8,4.4),sharey=True)
    names={'pick_and_place_simple':'Place','pick_clean_then_place_in_recep':'Clean + place','pick_heat_then_place_in_recep':'Heat + place'}
    for ax,family in zip(axes,sorted(result['a_retention_by_type'])):
        for c in CONDITIONS:
            arr=np.array(result['a_retention_by_type'][family][c])*100
            for row in arr:ax.plot([0,128],row,color=COLORS[c],alpha=.25,linewidth=.8)
            ax.plot([0,128],arr.mean(axis=0),color=COLORS[c],marker='o',label=LABELS[c])
        ax.set_xticks([0,128]);ax.set_ylim(-2,102);ax.set_xlabel('B updates');ax.set_title(names[family],loc='left',fontweight='bold');ax.grid(color='#E7E7E7')
    axes[0].set_ylabel('Unsupported A success (%)');axes[-1].legend(frameon=False,fontsize=8)
    fig.subplots_adjust(bottom=.2,wspace=.2);fig.text(.5,.015,'8 tasks/family; 3 paired seeds. Examine initial floors before interpreting changes. Only two A endpoints were measured.',ha='center',fontsize=8)
    save(fig,out/'a_retention_by_type')


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['collection','main']);p.add_argument('--data',type=Path);p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.data:
        data=json.loads(args.data.read_text());assert data['measured_only']
    else:
        if not args.stage:p.error('--stage or --data is required')
        collection=collection_result();data=dict(stage=args.stage,measured_only=True,collection=collection)
        if args.stage=='main':
            result=main_result();data['result']=result
            (ROOT/'main-result.json').write_text(json.dumps(result,indent=2)+'\n')
        else:(ROOT/'collection-result.json').write_text(json.dumps(collection,indent=2)+'\n')
    out=args.out or ROOT/'figures'/data['stage'];render(data,out)
    (out/'measured-data.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(stage=data['stage'],figures=str(out),complete=True)))


if __name__=='__main__':main()
