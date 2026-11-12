#!/usr/bin/env python3
"""Within-source contrasts under identical per-task exposure and task ordering."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import run_stage3_task_controls as study
import run_stage2_sampling as parent
from learning_harness.adaptation import composition_revision
from learning_harness.bounded_queue import save
from learning_harness.common import file_sha,read_jsonl
from stage1_variance_report import components,sensitivity,SEEDS

ARMS=('full_reference','random','min_nll')
SOURCES=('api','generic')
ROOT,OUT=study.ROOT,study.OUT


def collect(frozen):
    rows,exposures=[],[];initializations={};actual=0
    content_counts={}
    for source in SOURCES:
        pool=read_jsonl(f'runs/transfer-factorial-v1/corpora/full-{source}.jsonl')
        keys=[hashlib.sha256(json.dumps(r['messages'],sort_keys=True,ensure_ascii=False).encode()).hexdigest() for r in pool]
        content_counts[source]=(len({(r['task_id'],key) for r,key in zip(pool,keys)}),len(set(keys)))
    old=[c for c in parent.plan() if c['arm']=='full_pass']
    for root,cases,historical in ((parent.ROOT,old,True),(ROOT,frozen['cases'],False)):
        for case in cases:
            directory=root/'cases'/case['name'];r=json.loads((directory/'worker-result.json').read_text())
            if not r['complete'] or r['smoke'] or r['case']!=case:raise ValueError('Incomplete or wrong case')
            control=r['control'];arm='full_reference' if historical else case['arm']
            if (not control['completed'] or control['initialization_seed']!=case['initialization_seed']
                    or control['sampling_seed']!=case['sampling_seed']
                    or control['trained_adapter_sha256']!=file_sha(directory/'A/adapter_model.safetensors')
                    or composition_revision(directory/'checkpoint')!=r['composition_revision']):
                raise ValueError('Weight or seed audit failed')
            init=case['initialization_seed']
            if init in initializations and initializations[init]!=control['initial_parameter_sha256']:
                raise ValueError('Initialization changed with policy/source')
            initializations[init]=control['initial_parameter_sha256']
            for family in ('cool','heat'):
                s=json.loads((directory/'evaluation'/family/'summary.json').read_text())
                run=json.loads((directory/'evaluation'/family/'run.json').read_text())
                if (not s['complete'] or s['recorded']!=32 or s['metrics']['none']['error_count']
                        or run['revision']!=r['composition_revision'] or s['metrics']['none']['success_rate']!=r['success'][family]):
                    raise ValueError('Evaluation audit failed')
                if not historical:actual+=s['recorded']
            nll=json.loads((directory/'diagnostic-nll.json').read_text())
            if not nll['checkpoint_unchanged'] or nll['composition_revision']!=r['composition_revision']:
                raise ValueError('NLL audit failed')
            if r['mean_success']!=sum(r['success'].values())/2 or r['mean_nll']!=sum(nll[f]['task_mean_nll'] for f in ('cool','heat'))/2:
                raise ValueError('Stored mean differs from raw scores')
            e=r['exposure']
            if e['processed_examples']!=768:raise ValueError('Presentation budget changed')
            if not historical:
                recipe=json.loads(Path(case['schedule']).read_text())
                if (control['external_schedule_sha256']!=file_sha(case['schedule']) or control['schedule_sha256']!=recipe['schedule_sha256']
                        or e['task_order_sha256']!=recipe['reference_task_order_sha256'] or e['task_exposures']!=recipe['exposures_by_task']
                        or e['unique_examples']!=recipe['unique_examples']):
                    raise ValueError('Exact-task control audit failed')
            rows.append(dict(name=case['name'],source=case['source'],arm=arm,initialization_seed=init,
                sampling_seed=case['sampling_seed'],repeat=case['repeat'],gpu=case['gpu'],historical=historical,
                mean_success=r['mean_success'],mean_nll=r['mean_nll'],cool_success=r['success']['cool'],heat_success=r['success']['heat']))
            exposures.append(dict(name=case['name'],arm=arm,source=case['source'],
                **{k:e[k] for k in ('processed_examples','unique_examples','unique_tasks','input_tokens','target_tokens')},
                unique_task_content_examples=e.get('unique_task_content_examples',content_counts[case['source']][0]),
                distinct_content_examples=e.get('distinct_content_examples',content_counts[case['source']][1]),
                task_type_presentations=json.dumps(e['task_type_presentations'],sort_keys=True)))
    if actual!=4480 or len(set(initializations.values()))!=4:raise ValueError('Episode or initialization totals differ')
    return rows,exposures,actual


def analyze(rows):
    matrices={};fits={};effects={}
    for metric in ('mean_success','mean_nll'):
        matrices[metric]={};fits[metric]={};effects[metric]={}
        for arm in ARMS:
            group={source:np.array([[next(r[metric] for r in rows if r['arm']==arm and r['source']==source and
                r['initialization_seed']==i and r['sampling_seed']==s and r['repeat']==0) for s in SEEDS] for i in SEEDS]) for source in SOURCES}
            group['paired_api_minus_generic']=group['api']-group['generic']
            matrices[metric][arm]=group
            fits[metric][arm]={s:dict(components=components(v),sensitivity=sensitivity(v)) for s,v in group.items()}
        for label,a,b in (('random_minus_full','random','full_reference'),('nll_minus_random','min_nll','random'),('nll_minus_full','min_nll','full_reference')):
            effects[metric][label]={}
            for source in SOURCES:
                delta=matrices[metric][a][source]-matrices[metric][b][source]
                better=delta if metric=='mean_success' else -delta
                effects[metric][label][source]=dict(mean_difference=float(delta.mean()),cells=delta.tolist(),
                    wins=int((better>0).sum()),ties=int((delta==0).sum()),losses=int((better<0).sum()),
                    per_initialization_mean=[float(v.mean()) for v in delta])
    return matrices,fits,effects


def csv_file(name,rows):
    with (OUT/name).open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def figure(fig,name):
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUT/(name+'.'+ext),dpi=180)
    plt.close(fig)


def main():
    frozen=study.load_freeze();rows,exposure,episodes=collect(frozen);matrices,fits,effects=analyze(rows)
    reference_checks=[]
    for source in SOURCES:
        values=sorted([r for r in rows if r['arm']=='full_reference' and r['source']==source and
                      r['initialization_seed']==101 and r['sampling_seed']==101],key=lambda r:r['repeat'])
        old=next(r for r in values if r['repeat']==0);fresh=next(r for r in values if r['repeat']==2)
        reference_checks.append(dict(source=source,success_difference_pp=100*(fresh['mean_success']-old['mean_success']),
            nll_difference=fresh['mean_nll']-old['mean_nll'],conditional_single_pair=True))
    OUT.mkdir(parents=True,exist_ok=True);csv_file('cell-scores.csv',rows);csv_file('training-exposure.csv',exposure)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),sharey=True)
    labels=['Full pool reference','Task-matched random','Task-matched low NLL']
    for ax,source in zip(axes,SOURCES):
        for j,arm in enumerate(ARMS):
            values=matrices['mean_success'][arm][source]*100
            ax.scatter(np.full(16,j)+np.tile(np.linspace(-.15,.15,4),4),values.ravel(),s=24,alpha=.6)
            ax.plot([j-.2,j+.2],[values.mean()]*2,color='black',linewidth=2)
        ax.set_xticks(range(3),labels,rotation=12);ax.set_ylabel('Post-A success (%)');ax.set_ylim(-2,102)
        ax.set_title(source.upper());ax.grid(axis='y',alpha=.2)
    fig.suptitle('Identical task counts and task order — 16 crossed cells per policy/source')
    figure(fig,'task-matched-policy-scores')
    fig,axes=plt.subplots(2,2,figsize=(10.5,8))
    for i,label in enumerate(('random_minus_full','nll_minus_random')):
        for j,source in enumerate(SOURCES):
            ax=axes[i,j];delta=np.array(effects['mean_success'][label][source]['cells'])*100
            bound=max(float(abs(delta).max()),.01);im=ax.imshow(delta,cmap='RdBu_r',vmin=-bound,vmax=bound)
            for a in range(4):
                for b in range(4):ax.text(b,a,f'{delta[a,b]:+.1f}',ha='center',va='center',fontsize=9,
                    color='white' if abs(delta[a,b])>.6*bound else 'black')
            ax.set_xticks(range(4),SEEDS);ax.set_yticks(range(4),SEEDS)
            ax.set_xlabel('Sampling seed');ax.set_ylabel('Initialization seed');ax.set_title(source+' / '+label)
            fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('Within-source matched-policy success differences (pp)');figure(fig,'task-matched-policy-effects')
    result=dict(complete=True,cases=70,historical_reference_cases=34,actual_evaluation_episodes=episodes,
        freeze_sha256=file_sha(ROOT/'freeze.json'),results=rows,exposure=exposure,policy_effects=effects,fits=fits,
        fresh_reference_checks=reference_checks,teacher_calls=0,official_tests_used=False,
        uncertainty='All crossed cells and factor-deletion sensitivity ranges; not calibrated confidence intervals',
        selection_note='NLL membership is deterministic; smaller seed variance alone does not establish better demonstration quality')
    lines=['# Stage 3：逐任务匹配后的经验选择','','70个新模型、4,480个新评测回合完成。参考全池使用Stage 2的34个已完成模型，另有2次本轮全池检查。','',
        '| 对比 | 来源 | 成功率平均差 |','|---|---|---:|']
    for label in ('random_minus_full','nll_minus_random','nll_minus_full'):
        for source in SOURCES:lines.append(f"| {label} | {source} | {100*effects['mean_success'][label][source]['mean_difference']:+.3f}pp |")
    lines+=['','各单元、NLL、实际token与任务暴露、原始分量和留一敏感性范围见JSON/CSV；范围不是置信区间。',
            '', '![逐任务匹配分数](task-matched-policy-scores.png)','','![来源内干预效应](task-matched-policy-effects.png)','']
    (OUT/'RESULTS.zh-CN.md').write_text('\n'.join(lines));save(OUT/'summary.json',result)
    print(json.dumps(dict(complete=True,cases=70,actual_evaluation_episodes=episodes)),flush=True)


if __name__=='__main__':main()
