#!/usr/bin/env python3
"""Measured-only reports for source-by-selection transfer; no outcome imputation."""
import argparse
import json
from pathlib import Path
import numpy as np
from api_harness_report import save,csv_out,plt
from matched_study import crossed_interval
from learning_harness.common import file_sha,read_jsonl

ROOT=Path('runs/transfer-factorial-v1')
CONDITIONS=['base','matched-generic','matched-api','full-generic','full-api']
SEEDS=[1101,1202,1303]
COLORS={'base':'#777777','matched-generic':'#4477AA','matched-api':'#CC6677',
        'full-generic':'#228833','full-api':'#AA3377'}


def contrasts(values):
    return {'matched: API - generic':values['matched-api']-values['matched-generic'],
            'full: API - generic':values['full-api']-values['full-generic'],
            'interaction: full - matched':values['full-api']-values['full-generic']-values['matched-api']+values['matched-generic']}


def measured_family(family):
    root=ROOT/family;design=json.loads((root/'design.json').read_text());steps=design['b_training']['checkpoints']
    manifest={role:read_jsonl(root/f'{role.lower()}_query.jsonl') for role in ['A','B']}
    ids={role:sorted(r['task_id'] for r in rows) for role,rows in manifest.items()}
    records=[];seed_records=[];curves={};areas={};gains={};retention={};nll={};audits={}
    hashes={s:set() for s in SEEDS};orders=set();tokens=set();parameter_counts=set();nll_hashes=set()
    for condition in CONDITIONS:
        curves[condition]=[];areas[condition]=[];gains[condition]=[];retention[condition]=[];nll[condition]=[]
        for seed in SEEDS:
            case=f'{condition}-seed{seed}';trajectory=root/'trajectories'/case
            meta=json.loads((trajectory/'trajectory.json').read_text())
            assert meta['complete'] and meta['b_seed']==seed and meta['processed_examples']==512
            assert meta['a_frozen_verified'] and meta['b_updated_verified'] and meta['initialization']['exact_zero_b_logit_check']
            assert meta['data_sha256']==file_sha(root/'b_schedule.jsonl')
            assert meta['config']==json.loads((root/'b-lora.json').read_text())
            hashes[seed].add(meta['initialization']['b_initial_parameter_sha256']);orders.add(meta['actual_order_sha256'])
            tokens.add(json.dumps(meta['token_accounting'],sort_keys=True));parameter_counts.add(meta['initialization']['trainable_parameters'])
            ameta=None
            if condition!='base':
                job=next(j for j in design['initializations'] if j['condition']==condition and j['b_seed']==seed)
                ameta=json.loads((Path(job['a_adapter'])/'experiment.json').read_text())
                assert ameta['processed_examples_rank0']==256 and ameta['training_seed']==seed-1000
                assert ameta['training_data_sha256']==file_sha(job['data'])
            matrices={}
            for role,observed in [('B',steps),('A',[0,128])]:
                matrix=[]
                for step in observed:
                    directory=root/'evaluation/main'/case/role/f'step-{step:04d}'
                    summary=json.loads((directory/'summary.json').read_text())
                    assert summary['complete'] and summary['metrics']['none']['error_count']==0
                    rows=read_jsonl(directory/'episodes.jsonl');byid={r['task_id']:r for r in rows}
                    assert len(rows)==len(ids[role]) and set(byid)==set(ids[role])
                    assert all(r['status']=='completed' and not r['support_injected'] and r['arm']=='none' for r in rows)
                    matrix.append([int(byid[t]['success']) for t in ids[role]])
                    records.extend(dict(family=family,condition=condition,b_seed=seed,role=role,updates=step,
                        task_id=t,success=int(byid[t]['success'])) for t in ids[role])
                matrices[role]=np.array(matrix,dtype=float)
            integrate=np.trapezoid if hasattr(np,'trapezoid') else np.trapz
            area=integrate(matrices['B'],x=steps,axis=0)/128
            curves[condition].append(matrices['B'].mean(axis=1).tolist())
            areas[condition].append(area);gains[condition].append(area-matrices['B'][0])
            retention[condition].append(matrices['A'].mean(axis=1).tolist())
            diagnostic=json.loads((trajectory/'diagnostic-nll.json').read_text())
            assert set(map(int,diagnostic))==set(steps)
            nll_ids=sorted(diagnostic['0']['per_task_nll'])
            for step in steps:
                d=diagnostic[str(step)]
                assert d['data_sha256']==file_sha(root/'diagnostic.jsonl') and not d['used_for_updates'] and not d['used_for_selection']
                assert sorted(d['per_task_nll'])==nll_ids;nll_hashes.add(d['data_sha256'])
            nll[condition].append([[diagnostic[str(k)]['per_task_nll'][t] for t in nll_ids] for k in steps])
            seed_records.append(dict(family=family,condition=condition,b_seed=seed,
                initial_success=float(matrices['B'][0].mean()),final_success=float(matrices['B'][-1].mean()),
                raw_auc=float(area.mean()),gain_auc=float((area-matrices['B'][0]).mean()),
                initial_diagnostic_nll=diagnostic['0']['task_mean_nll'],final_diagnostic_nll=diagnostic['128']['task_mean_nll']))
            audits[case]=dict(B_processed_examples=512,B_tokens=meta['token_accounting'],
                A_processed_examples=ameta['processed_examples_rank0'] if ameta else 0,
                A_input_tokens=ameta['input_tokens_rank0'] if ameta else 0,
                A_target_tokens=ameta['target_tokens_rank0'] if ameta else 0,
                reused_A=condition.startswith('matched-'),frozen_A_verified=True)
    assert len(orders)==len(tokens)==len(parameter_counts)==len(nll_hashes)==1 and all(len(v)==1 for v in hashes.values())
    effects={}
    for metric,source in [('raw',areas),('gain',gains)]:
        for label,arr in contrasts({c:np.array(v) for c,v in source.items()}).items():
            effects.setdefault(label,{})[metric]=dict(mean=float(arr.mean()),per_seed=arr.mean(axis=1).tolist(),
                crossed_bootstrap_95=crossed_interval(arr.tolist()),per_seed_task=arr.tolist())
    return dict(complete=True,family=family,steps=steps,conditions=CONDITIONS,seeds=SEEDS,
        curves=curves,retention=retention,diagnostic_nll=nll,effects=effects,
        episode_scores=records,seed_metrics=seed_records,parameter_and_budget_audits=audits,
        reused_A_data_and_seed_cohort=True,diagnostic_inference='Descriptive; no causal adjustment of prior knowledge',
        source_design_sha256=file_sha(root/'design.json'))


def render(data,out):
    out.mkdir(parents=True,exist_ok=True);steps=data['steps']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    csv_out(out/'episode-scores.csv',data['episode_scores']);csv_out(out/'seed-metrics.csv',data['seed_metrics'])
    for key,label,filename in [('curves','Unsupported B success (%)','learning_curves'),
                                ('diagnostic_nll','Held-out action NLL (lower is better)','diagnostic_nll')]:
        fig,axes=plt.subplots(1,2,figsize=(11,4.4))
        for ax,regime in zip(axes,['matched','full']):
            for c in ['base',regime+'-generic',regime+'-api']:
                values=np.array(data[key][c]);values=values.mean(axis=2) if key=='diagnostic_nll' else values*100
                for row in values:ax.plot(steps,row,color=COLORS[c],alpha=.25,linewidth=.8)
                ax.plot(steps,values.mean(axis=0),color=COLORS[c],marker='o',label=c,linewidth=2)
            ax.set_title(f"{data['family']}: {regime} A data",loc='left');ax.set_xlabel('B optimizer updates');ax.set_xticks(steps)
            ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
            if key=='curves':ax.set_ylim(-2,102)
        axes[0].set_ylabel(label);fig.subplots_adjust(bottom=.22,wspace=.26)
        fig.text(.5,.02,'Three paired seeds. Lines connect measured checkpoints; diagnostic data never enter the optimizer.',ha='center',fontsize=8)
        save(fig,out/filename)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
    labels=list(data['effects'])
    for ax,metric in zip(axes,['raw','gain']):
        ax.axvline(0,color='gray',linestyle='--')
        for i,label in enumerate(labels):
            d=data['effects'][label][metric];lo,hi=np.array(d['crossed_bootstrap_95'])*100
            ax.plot([lo,hi],[i,i],color='#26374A');ax.scatter(d['mean']*100,i,marker='D',color='#26374A')
            for shift,value in zip([-.12,0,.12],d['per_seed']):ax.scatter(value*100,i+shift,s=18,color='#CC6677')
        ax.set_yticks(range(len(labels)),labels);ax.set_xlabel('Difference in normalized AUC (pp)');ax.set_title(metric+' AUC',loc='left');ax.grid(axis='x',alpha=.2)
    fig.subplots_adjust(left=.27,bottom=.22,wspace=.2)
    fig.text(.5,.02,'Paired crossed-bootstrap 95% intervals. Interaction changes selection, coverage and repetition together.',ha='center',fontsize=8)
    save(fig,out/'source_selection_effects')
    fig,ax=plt.subplots(figsize=(8,4.6))
    for c in CONDITIONS:
        values=np.array(data['retention'][c])*100
        for row in values:ax.plot([0,128],row,color=COLORS[c],alpha=.2,linewidth=.8)
        ax.plot([0,128],values.mean(axis=0),color=COLORS[c],marker='o',label=c)
    ax.set_xticks([0,128]);ax.set_ylim(-2,102);ax.set_ylabel('Unsupported A success (%)');ax.set_xlabel('B updates');ax.legend(frameon=False)
    fig.subplots_adjust(bottom=.2);fig.text(.5,.02,'Two endpoints only, 24 A tasks. Segments do not estimate intermediate retention.',ha='center',fontsize=8)
    save(fig,out/'retention_endpoints')
    (out/'measured-data.json').write_text(json.dumps(data,indent=2)+'\n')


def combined_result(families):
    effects={};rng=np.random.default_rng(20261110)
    for metric in ['raw','gain']:
        arrays=[np.array(d['effects']['interaction: full - matched'][metric]['per_seed_task']) for d in families]
        point=float(np.mean([a.mean() for a in arrays]));samples=[]
        for _ in range(4000):
            seed_indices=rng.integers(0,3,3)
            samples.append(np.mean([a[np.ix_(seed_indices,rng.integers(0,a.shape[1],a.shape[1]))].mean() for a in arrays]))
        effects[metric]=dict(mean=point,bootstrap_95=np.quantile(samples,[.025,.975]).tolist(),
            per_family=[float(a.mean()) for a in arrays])
    result=dict(complete=True,primary_metric='raw source-by-selection interaction; equal weight across two families',
        interaction=effects,families=[d['family'] for d in families],paired_seeds=SEEDS,
        final_tests_used=False,new_api_calls=0,interpretation='Coverage/selection follow-up, not a causal demonstration of plasticity')
    (ROOT/'main-result.json').write_text(json.dumps(result,indent=2)+'\n')
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),sharey=True)
    for ax,metric in zip(axes,['raw','gain']):
        ax.axvline(0,color='gray',linestyle='--')
        for i,d in enumerate(families):
            e=d['effects']['interaction: full - matched'][metric]
            ax.plot(np.array(e['crossed_bootstrap_95'])*100,[i,i],color='#4477AA');ax.scatter(e['mean']*100,i,color='#4477AA')
        e=effects[metric];ax.plot(np.array(e['bootstrap_95'])*100,[2,2],color='#CC6677');ax.scatter(e['mean']*100,2,color='#CC6677',marker='D')
        ax.set_yticks([0,1,2],['Cool','Heat','Equal-weight mean']);ax.set_title(metric+' AUC interaction',loc='left');ax.set_xlabel('Interaction (pp)')
    fig.subplots_adjust(bottom=.22,wspace=.45)
    fig.text(.5,.02,'95% crossed bootstrap; shared seed resampling across families, independent task-group resampling within each family.',ha='center',fontsize=7)
    save(fig,out/'cross_family_interaction')
    print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['cool','heat']);p.add_argument('--data',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.data:
        if a.out is None:p.error('--data requires --out')
        render(json.loads(a.data.read_text()),a.out);return
    if a.family:
        result=measured_family(a.family)
        (ROOT/a.family/'main-result.json').write_text(json.dumps(result,indent=2)+'\n')
        render(result,ROOT/a.family/'figures/main');print(json.dumps(dict(family=a.family,complete=True)));return
    combined_result([json.loads((ROOT/f/'main-result.json').read_text()) for f in ['cool','heat']])


if __name__=='__main__':main()
