#!/usr/bin/env python3
"""Measured pipeline comparison; paired seeds/tasks, initial/gain decomposition."""
from collections import defaultdict
import csv
import fcntl
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from learning_harness.common import read_jsonl,file_sha

ROOT=Path('runs/source-comparison-v1');OUT=Path('output/source-comparison-v1')
SEEDS=[101,202,303];STEPS=[0,8,32,128];SOURCES=['base','api','react'];FAMILIES=['cool','heat']
LABELS={'base':'No A training','api':'Candidate: API-scaffold -> SFT','react':'Baseline: ReAct -> SFT'}


def csv_write(path,rows):
    if not rows:return
    with Path(path).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def paired_bootstrap(a,b,steps=STEPS):
    """Arrays per family: paired training seed x checkpoint x task group."""
    rng=np.random.default_rng(20261111);seed_idx=rng.integers(0,3,size=(4000,3))
    metrics={};w=np.zeros(len(steps))
    for i in range(len(steps)-1):w[i:i+2]+=(steps[i+1]-steps[i])/2/(steps[-1]-steps[0])
    for metric in ['raw_auc','gain_auc','initial','final']:
        observed=[];boot=[]
        for f in FAMILIES:
            delta=a[f]-b[f]
            if metric=='raw_auc':values=(delta*w[None,:,None]).sum(axis=1)
            elif metric=='gain_auc':values=(delta*w[None,:,None]).sum(axis=1)-delta[:,0,:]
            elif metric=='initial':values=delta[:,0,:]
            else:values=delta[:,-1,:]
            n=values.shape[1];task_idx=rng.integers(0,n,size=(4000,n))
            draws=values[seed_idx[:,:,None],task_idx[:,None,:]].mean(axis=(1,2))
            observed.append(float(values.mean()));boot.append(draws)
        combined=np.mean(boot,axis=0)
        metrics[metric]=dict(difference_pp=100*float(np.mean(observed)),ci95_pp=(100*np.quantile(combined,[.025,.975])).tolist(),
            family_differences_pp={f:100*v for f,v in zip(FAMILIES,observed)})
    return metrics


def gather():
    design=json.loads((ROOT/'design.json').read_text());rows=[];traces=[]
    for path in sorted((ROOT/'evaluation').glob('*/*/*/*/step-*/tasks.jsonl')):
        rel=path.relative_to(ROOT/'evaluation').parts;family,case,harness,role,step_dir,_=rel
        source,seed_text=case.split('-seed');seed=int(seed_text);step=int(step_dir.split('-')[1])
        run=json.loads((path.parent/'run.json').read_text());cfg=run['config']
        task_manifest=read_jsonl(cfg['manifest']);task_index={t['task_id']:t for t in task_manifest}
        records=read_jsonl(path)
        if len({(r['task_id'],r['seed']) for r in records})!=len(records):raise ValueError('Duplicate evaluation task')
        for r in records:
            if r['task_id'] not in task_index or r['seed']!=11 or len(r['trials'])!=1 or r['calls']>40:raise ValueError('Evaluation protocol mismatch')
            rows.append(dict(family=family,source=source,training_seed=seed,harness=harness,role=role,step=step,
                task_id=r['task_id'],task_group=r['task_group'],success=int(r['success']),calls=r['calls'],
                input_tokens=r['input_tokens'],output_tokens=r['output_tokens']))
        traces.append(dict(path=path,tasks=records))
    complete=len(rows)==design['planned_evaluation_task_runs']
    matrices={};means=[];seed_scores=[]
    for f in FAMILIES:
        tasks=read_jsonl(ROOT/f'{f}_query.jsonl');ids=[t['task_id'] for t in tasks]
        # Each query was selected as one representative per task group.
        assert len({t['task_group'] for t in tasks})==len(tasks)
        for source in SOURCES:
            mat=np.full((3,4,len(ids)),np.nan)
            for r in rows:
                if r['family']==f and r['source']==source and r['role']=='B' and r['harness']=='json_direct':
                    mat[SEEDS.index(r['training_seed']),STEPS.index(r['step']),ids.index(r['task_id'])]=r['success']
            matrices[f,source]=mat
            for i,seed in enumerate(SEEDS):
                if not np.isnan(mat[i]).any():
                    curve=mat[i].mean(axis=1);auc=float(np.trapz(curve,STEPS)/128)
                    seed_scores.append(dict(family=f,source=source,seed=seed,initial=float(curve[0]),final=float(curve[-1]),raw_auc=auc,gain_auc=auc-float(curve[0])))
            if not np.isnan(mat).any():
                for j,step in enumerate(STEPS):means.append(dict(family=f,source=source,step=step,success_rate=float(mat[:,j,:].mean())))
    result=dict(complete=complete,completed_evaluation_task_runs=len(rows),planned_evaluation_task_runs=design['planned_evaluation_task_runs'],
        exploratory=True,official_final_tests_used=False,new_paid_api_calls=0,seed_scores=seed_scores,curves=means)
    selection=ROOT/'selection.json'
    if selection.exists():result['selection']=json.loads(selection.read_text())
    training=[]
    for path in sorted((ROOT/'a-checkpoints').glob('*/experiment.json')):
        meta=json.loads(path.read_text());source=path.parent.name.split('-seed')[0]
        training.append(dict(source=source,seed=meta['training_seed'],processed_examples=meta['processed_examples_rank0'],
            eligible_examples=meta['used_examples'],unique_scheduled_examples=meta['unique_scheduled_examples'],
            input_tokens=meta['input_tokens_rank0'],target_tokens=meta['target_tokens_rank0'],
            schedule_sha256=meta['schedule_sha256']))
    result['A_training']=training
    if complete:
        if any(np.isnan(m).any() for m in matrices.values()):raise ValueError('Missing planned learning curve')
        paired_B=[]
        for family in FAMILIES:
            for seed in SEEDS:
                metas=[]
                for source in SOURCES:
                    trajectory=(Path('runs/transfer-factorial-v1')/family/'trajectories'/f'base-seed{seed+1000}'
                        if source=='base' else ROOT/'trajectories'/family/f'{source}-seed{seed}')
                    metas.append(json.loads((trajectory/'trajectory.json').read_text()))
                assert len({m['initialization']['b_initial_parameter_sha256'] for m in metas})==1
                assert len({m['actual_order_sha256'] for m in metas})==1
                assert len({m['data_sha256'] for m in metas})==1
                assert all(m['a_frozen_verified'] and m['b_updated_verified'] and m['processed_examples']==512 for m in metas)
                paired_B.append(dict(family=family,seed=seed,identical_B_initialization=True,identical_B_sample_order=True))
        result['paired_B_validation']=paired_B
        for seed in SEEDS:
            pair=[r for r in training if r['seed']==seed]
            assert len(pair)==2 and pair[0]['schedule_sha256']==pair[1]['schedule_sha256']
            assert all(r['processed_examples']==256 for r in pair)
        for trace in traces:
            committed={r['run_id']:r for r in trace['tasks']};account=defaultdict(lambda:[0,0,0])
            with (trace['path'].parent/'events.jsonl').open() as f:
                for line in f:
                    e=json.loads(line)
                    if e['run_id'] not in committed:continue
                    v=account[e['run_id']];v[0]+=1;v[1]+=e['input_tokens'];v[2]+=e['output_tokens']
                    assert e['phase']=='act' and e['trial']==1 and e['output_tokens']<=128
                    assert all(not m['content'].startswith('[TRAINING SUPPORT]') for m in e['messages'])
            for rid,r in committed.items():assert account[rid]==[r['calls'],r['input_tokens'],r['output_tokens']]
        result['primary_api_minus_react']=paired_bootstrap({f:matrices[f,'api'] for f in FAMILIES},{f:matrices[f,'react'] for f in FAMILIES})
        result['token_and_budget_audit_passed']=True
    return result,rows,matrices


def render(result,rows,matrices):
    OUT.mkdir(parents=True,exist_ok=True);csv_write(OUT/'evaluation.csv',rows)
    csv_write(OUT/'seed-scores.csv',result['seed_scores']);csv_write(OUT/'training-cost.csv',result['A_training'])
    colors={'base':'#888888','api':'#dc7c25','react':'#357bb7'}
    if result.get('selection'):
        collection=result['selection']['collection'];fig,ax=plt.subplots(figsize=(6,4))
        ax.bar(['API scaffold','ReAct'],[collection[s]['success_rate']*100 for s in ['api','react']],color=[colors[s] for s in ['api','react']])
        for i,s in enumerate(['api','react']):
            r=collection[s];ax.text(i,r['success_rate']*100+1,f'{r["successful_episodes"]}/{r["tasks"]}',ha='center')
        ax.set_ylim(0,105);ax.set_ylabel('Collection success (%)');ax.set_title('Fresh collection, equal call caps (before SFT)')
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(OUT/f'collection.{ext}',dpi=160,bbox_inches='tight')
        plt.close(fig)
    if result['complete']:
        fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
        for column,f in enumerate(FAMILIES):
            for source in SOURCES:
                matrix=matrices[f,source];curves=matrix.mean(axis=2)*100
                for row,gain in enumerate([False,True]):
                    y=curves-curves[:,[0]] if gain else curves
                    ax=axes[row,column]
                    for line in y:ax.plot(STEPS,line,color=colors[source],alpha=.25,lw=.8)
                    ax.plot(STEPS,y.mean(axis=0),'-o',color=colors[source],label=LABELS[source],lw=2)
                    ax.set_title(f'{f.capitalize()}: '+('gain from start' if gain else 'raw success'))
                    ax.set_ylabel('Gain (percentage points)' if gain else 'Success (%)');ax.set_xticks(STEPS)
                    if not gain:ax.set_ylim(0,105)
                    if row==1:ax.set_xlabel('B optimizer updates (4 shared examples / update)')
            axes[0,column].legend(fontsize=8)
        fig.suptitle('Matched source comparison — common JSON execution; 3 paired training seeds')
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(OUT/f'adaptation-curves.{ext}',dpi=170,bbox_inches='tight')
        plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,4));x=np.arange(4);labels=['Cool, start','Cool, final','Heat, start','Heat, final']
        for ax,harness in zip(axes,['json_direct','react_retry']):
            for i,source in enumerate(SOURCES):
                values=[100*np.mean([r['success'] for r in rows if r['family']==f and r['source']==source and
                    r['harness']==harness and r['role']=='B' and r['step']==step]) for f in FAMILIES for step in [0,128]]
                ax.bar(x+(i-1)*.24,values,.24,label=LABELS[source],color=colors[source])
            ax.set_xticks(x);ax.set_xticklabels(labels,rotation=20,ha='right');ax.set_ylim(0,105)
            ax.set_title('Common '+('JSON' if harness=='json_direct' else 'ReAct')+' execution');ax.set_ylabel('Success (%)')
        axes[1].legend(fontsize=8);fig.suptitle('Execution-harness robustness — same learned checkpoints, no teacher cards')
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(OUT/f'execution-robustness.{ext}',dpi=170,bbox_inches='tight')
        plt.close(fig)
        contrasts=result['primary_api_minus_react']
        lines=['# API 辅助采集→SFT 与 ReAct 自采→SFT：实测对照','',
            '同一 Qwen3-4B 起点、共同成功任务/环境种子、逐对等量动作标签、3 个训练种子。所有评测均重新运行；正式测试未使用。','',
            '| 比较：API − ReAct | 差异（pp） | 配对 bootstrap 95% CI |','|---|---:|---:|']
        for key,label in [('raw_auc','两任务族平均原始 AUC（主比较）'),('gain_auc','相对起点的增益 AUC'),('initial','初始成功率'),('final','最终成功率')]:
            r=contrasts[key];lines.append(f'| {label} | {r["difference_pp"]:+.2f} | [{r["ci95_pp"][0]:+.2f}, {r["ci95_pp"][1]:+.2f}] |')
        lines+=['','原始 AUC 同时包含已有能力和后续收益；gain AUC 也不等于因果意义上的可塑性。'
            '本轮是冻结 A、训练新增 B LoRA 的适应实验，不是无限制继续更新同一组参数。',
            '开发任务曾用于旧实验，本轮属于探索性强基线对照。共同成功筛选改变了适用任务范围；等样本/更新不等于等 token 或 FLOPs。',
            'ReAct 评测端点使用任务类型示例，包括冷却示例；这些结果属于 few-shot 执行，不能称为零示例目标迁移。',
            '教师策略来自既有缓存，本轮新增 API 调用为 0；策略原始生成成本不能视为免费。','',
            '完整分种子结果见 seed-scores.csv；训练真实 token 见 training-cost.csv；逐任务结果见 evaluation.csv。',
            '图表：collection、adaptation-curves、execution-robustness 均有 PNG/PDF。','']
        (OUT/'RESULTS.zh-CN.md').write_text('\n'.join(lines))
    temp=OUT/'summary.json.part';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(OUT/'summary.json')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'report.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        saved=OUT/'summary.json'
        if saved.exists() and json.loads(saved.read_text()).get('complete') and '--force' not in sys.argv:
            print('Complete measured report already exists; use --force for an intentional rebuild.');return
        result,rows,matrices=gather();render(result,rows,matrices)
        print(json.dumps(dict(complete=result['complete'],completed=result['completed_evaluation_task_runs'],
            planned=result['planned_evaluation_task_runs'],primary=result.get('primary_api_minus_react')),indent=2))


if __name__=='__main__':main()
