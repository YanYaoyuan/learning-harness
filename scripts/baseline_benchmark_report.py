#!/usr/bin/env python3
"""Audit measured baseline traces and export development-only figures."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from learning_harness.benchmark import METHODS
from learning_harness.common import file_sha, read_jsonl
from audit_baseline_prompts import audit as audit_prompts

ROOT=Path('runs/baseline-benchmark-v1')
OUT=Path('output/baseline-benchmark-v1')
LABELS={'json_direct':'Direct JSON','act':'Act (matched demos)', 'react_retry':'ReAct + retries','reflexion':'Reflexion'}


def wilson(k,n):
    if not n:return [None,None]
    z=1.96;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]


def paired_effect(a,b,key):
    # Paired cluster bootstrap over task groups, never count retries as samples.
    left={(r['task_id'],r['seed']):r for r in a};right={(r['task_id'],r['seed']):r for r in b}
    if left.keys()!=right.keys() or not left:return None
    groups=defaultdict(list)
    for k,r in left.items():groups[r['task_group']].append(float(r[key])-float(right[k][key]))
    sums=np.array([sum(v) for v in groups.values()]);counts=np.array([len(v) for v in groups.values()])
    rng=np.random.default_rng(20261110);indices=rng.integers(0,len(sums),size=(5000,len(sums)))
    boot=sums[indices].sum(axis=1)/counts[indices].sum(axis=1)
    return dict(difference_pp=100*sums.sum()/counts.sum(),ci95_pp=(100*np.quantile(boot,[.025,.975])).tolist(),
        task_groups=len(groups),paired_task_seeds=len(left),status='exploratory_development_only')


def collect():
    datasets={};summaries=[];audit={}
    for method in METHODS:
        root=ROOT/'development'/method
        if not (root/'tasks.jsonl').exists():continue
        rows=read_jsonl(root/'tasks.jsonl');datasets[method]=rows
        meta=json.loads((root/'run.json').read_text());cfg=meta['config']
        expected={(t['task_id'],seed) for t in read_jsonl(cfg['manifest']) for seed in cfg['seeds']}
        observed={(r['task_id'],r['seed']) for r in rows}
        assert len(observed)==len(rows) and observed <= expected
        assert meta['planned_tasks']==len(expected)
        valid={r['run_id']:r for r in rows};counts=Counter();tokens=defaultdict(lambda:[0,0])
        events_by_id=defaultdict(list)
        if (root/'events.jsonl').exists():
            # Stream potentially large transcripts. Only committed task IDs contribute.
            with (root/'events.jsonl').open() as handle:
                for line in handle:
                    try:e=json.loads(line)
                    except json.JSONDecodeError:continue # Writer may be midway through the last line.
                    if e['run_id'] not in valid:continue
                    rid=e['run_id'];counts[rid]+=1
                    tokens[rid][0]+=e['input_tokens'];tokens[rid][1]+=e['output_tokens']
                    events_by_id[rid].append((e['call'],e['phase'],e.get('success',False)))
                    assert e['output_tokens']<=cfg['max_new_tokens']
                    if e['phase']=='reflect':assert method=='reflexion'
        for r in rows:
            rid=r['run_id']
            assert counts[rid]==r['calls']<=cfg['max_calls']
            assert tokens[rid]==[r['input_tokens'],r['output_tokens']]
            assert r['output_tokens']<=cfg['max_calls']*cfg['max_new_tokens']
            assert [v[0] for v in events_by_id[rid]]==list(range(1,r['calls']+1))
            assert len(r['trials'])<=cfg['max_trials']
            assert all(t['turns']<=cfg['max_turns'] for t in r['trials'])
            assert r['success']==any(t['success'] for t in r['trials'])
            assert r['first_success']==r['trials'][0]['success']
        n=len(rows);first=sum(r['first_success'] for r in rows);success=sum(r['success'] for r in rows)
        summaries.append(dict(method=method,completed=n,planned=meta['planned_tasks'],first_successes=first,
            cumulative_successes=success,first_rate=first/n if n else None,cumulative_rate=success/n if n else None,
            first_ci95=wilson(first,n),cumulative_ci95=wilson(success,n),
            mean_calls=float(np.mean([r['calls'] for r in rows])) if n else None,
            mean_input_tokens=float(np.mean([r['input_tokens'] for r in rows])) if n else None,
            mean_output_tokens=float(np.mean([r['output_tokens'] for r in rows])) if n else None))
        audit[method]=dict(committed_tasks=n,token_accounting_matches=True,call_budget_matches=True,
            task_file_sha256=file_sha(root/'tasks.jsonl'))
    complete=len(summaries)==len(METHODS) and all(r['completed']==r['planned'] for r in summaries)
    effects={}
    if complete:
        for name,a,b in [('reflection_increment','reflexion','react_retry'),('reasoning_increment','react_retry','act')]:
            effects[name]={k:paired_effect(datasets[a],datasets[b],k) for k in ['first_success','success']}
    result=dict(complete=complete,scope='reused development split; not final benchmark evidence',
        final_test_executed=False,summaries=summaries,exploratory_paired_effects=effects,audit=audit)
    if complete:
        prompt_audit=audit_prompts(ROOT)
        assert prompt_audit['complete']
        assert prompt_audit['first_request_pairing']['paired_tasks']==len(datasets['react_retry'])
        excluded=set(prompt_audit['excluded_task_ids'])
        clean={m:[r for r in rs if r['task_id'] not in excluded] for m,rs in datasets.items()}
        result['prompt_audit']=prompt_audit
        result['excluding_demo_overlap']={m:dict(tasks=len(rs),first_successes=sum(r['first_success'] for r in rs),
            cumulative_successes=sum(r['success'] for r in rs)) for m,rs in clean.items()}
        result['excluding_demo_overlap_effects']={k:paired_effect(clean['reflexion'],clean['react_retry'],k)
            for k in ['first_success','success']}
        result['analysis_source_sha256']=file_sha(__file__)
    return result,datasets


def render(result,datasets):
    OUT.mkdir(parents=True,exist_ok=True)
    rows=result['summaries']
    if rows:
        with (OUT/'metrics.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if not result['complete']:
        (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        return
    colors=plt.get_cmap('tab10').colors
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    x=np.arange(len(rows));width=.35
    for offset,key,ci,label,color in [(-width/2,'first_rate','first_ci95','First attempt','#5279b8'),
        (width/2,'cumulative_rate','cumulative_ci95','Up to 3 attempts','#df9747')]:
        y=np.array([r[key]*100 for r in rows]);bounds=np.array([r[ci] for r in rows])*100
        axes[0].bar(x+offset,y,width,label=label,color=color)
        axes[0].errorbar(x+offset,y,yerr=np.array([y-bounds[:,0],bounds[:,1]-y]),fmt='none',color='black',capsize=3,lw=1)
    axes[0].set_xticks(x);axes[0].set_xticklabels([LABELS[r['method']] for r in rows],rotation=20,ha='right')
    axes[0].set_ylim(0,105);axes[0].set_ylabel('Task success (%)');axes[0].legend(fontsize=8)
    axes[0].set_title('40 development tasks; 95% Wilson intervals')
    max_observed_tokens=max(r['output_tokens'] for rs in datasets.values() for r in rs)
    token_budgets=sorted({0,max_observed_tokens}|{t['cumulative_output_tokens'] for rs in datasets.values()
        for r in rs for t in r['trials'] if t['success']})
    for i,method in enumerate(METHODS):
        rs=datasets[method]
        for ax,key,limits,label in [(axes[1],'cumulative_calls',np.arange(0,151),'Model calls (reflection included)'),
            (axes[2],'cumulative_output_tokens',token_budgets,'Generated tokens (reflection included)')]:
            points=[]
            for budget in limits:
                points.append(100*sum(any(t['success'] and t[key]<=budget for t in r['trials']) for r in rs)/len(rs))
            ax.step(limits,points,where='post',label=LABELS[method],color=colors[i]);ax.set_xlabel(label);ax.set_ylim(0,105)
    axes[1].set_title('Success under a call budget');axes[2].set_title('Output-token budget (observed range)')
    axes[2].legend(fontsize=8)
    fig.suptitle('Controlled local ports on Qwen3-4B — exploratory development evaluation',fontsize=13)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/f'success-and-cost.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    types=sorted({r['task_type'] for rs in datasets.values() for r in rs})
    matrix=np.array([[100*np.mean([r['success'] for r in datasets[m] if r['task_type']==t]) for t in types] for m in METHODS])
    fig,ax=plt.subplots(figsize=(8,4));im=ax.imshow(matrix,vmin=0,vmax=100,cmap='Blues')
    family_labels={'pick_and_place_simple':'place','pick_clean_then_place_in_recep':'clean',
        'pick_cool_then_place_in_recep':'cool','pick_heat_then_place_in_recep':'heat'}
    ax.set_xticks(range(len(types)));ax.set_xticklabels([family_labels.get(t,t) for t in types],rotation=15,ha='right')
    ax.set_yticks(range(len(METHODS)));ax.set_yticklabels([LABELS[m] for m in METHODS])
    for i in range(len(METHODS)):
        for j in range(len(types)):ax.text(j,i,f'{matrix[i,j]:.0f}%',ha='center',va='center',color='white' if matrix[i,j]>55 else 'black')
    ax.set_title('Cumulative success by task family (10 tasks/family)');fig.colorbar(im,ax=ax,label='Success (%)')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/f'task-family.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    clean=result['excluding_demo_overlap']
    with (OUT/'excluding-demo-overlap.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=['method','tasks','first_successes','cumulative_successes'])
        writer.writeheader();writer.writerows(dict(method=m,**r) for m,r in clean.items())
    fig,ax=plt.subplots(figsize=(8,4))
    x=np.arange(len(METHODS));width=.35
    ax.bar(x-width/2,[100*clean[m]['first_successes']/clean[m]['tasks'] for m in METHODS],width,label='First attempt')
    ax.bar(x+width/2,[100*clean[m]['cumulative_successes']/clean[m]['tasks'] for m in METHODS],width,label='Cumulative')
    ax.set_xticks(x);ax.set_xticklabels([LABELS[m] for m in METHODS],rotation=15,ha='right')
    ax.set_ylim(0,105);ax.set_ylabel('Task success (%)');ax.legend()
    ax.set_title(f'Sensitivity: {next(iter(clean.values()))["tasks"]} tasks after excluding demonstration overlap')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/f'excluding-demo-overlap.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    # Publish the completion marker only after all figures and audits are ready.
    temporary=OUT/'summary.json.part';temporary.write_text(json.dumps(result,indent=2)+'\n')
    temporary.replace(OUT/'summary.json')


def main():
    result,datasets=collect();render(result,datasets)
    print(json.dumps({k:v for k,v in result.items() if k!='audit'},indent=2))


if __name__=='__main__':main()
