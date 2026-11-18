#!/usr/bin/env python3
"""Audit complete Stage 2 and materialize exact-task exposure controls for the next study."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from learning_harness.bounded_queue import save
from learning_harness.common import file_sha,read_jsonl
from learning_harness.task_matched_sampling import matched_schedule
from train_lora import fixed_schedule
from train_lora_sampling_intervention import schedule
import run_stage2_sampling as study

OUT=Path('output/variance-diagnostic-v1/stage2/mechanism-audit')
NEXT=Path('runs/variance-diagnostic-v1/stage3/inputs/schedules')
SCORES=Path('runs/variance-diagnostic-v1/stage3/base-nll')


def digest(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def main():
    study.load_freeze()
    summary_path=study.OUT/'summary.json'
    result=json.loads(summary_path.read_text())
    if not result['complete'] or result['actual_evaluation_episodes']!=4352:
        raise ValueError('Stage 2 is not complete')
    if (OUT/'summary.json').exists():
        old=json.loads((OUT/'summary.json').read_text())
        if old.get('complete') and old['stage2_summary_sha256']==file_sha(summary_path):
            print('Completed audit already exists; reusing unchanged result')
            return
        raise ValueError('Existing partial/different mechanism audit retained for inspection')
    OUT.mkdir(parents=True,exist_ok=True)
    recipes,exposure_audits=[],[]
    for source in ('api','generic'):
        data=Path(f'runs/transfer-factorial-v1/corpora/full-{source}.jsonl')
        rows=read_jsonl(data)
        task_ids=[r['task_id'] for r in rows]
        content_keys=[hashlib.sha256(json.dumps(r['messages'],sort_keys=True,ensure_ascii=False).encode()).hexdigest() for r in rows]
        score_path=SCORES/(source+'.json')
        score_record=json.loads(score_path.read_text()) if score_path.exists() else None
        nll=None
        if score_record:
            if score_record['data_sha256']!=file_sha(data) or [r['index'] for r in score_record['scores']]!=list(range(len(rows))):
                raise ValueError('Base-NLL data alignment mismatch')
            nll=[r['mean_target_nll'] for r in score_record['scores']]
        for seed in (101,202,303,404):
            full=fixed_schedule(len(rows),768,seed)
            replay=schedule(fixed_schedule,len(rows),768,seed,'subset_replay')
            full_counts=Counter(task_ids[i] for i in full)
            replay_counts=Counter(task_ids[i] for i in replay)
            tvd=sum(abs(full_counts[t]-replay_counts[t]) for t in set(full_counts)|set(replay_counts))/(2*768)
            exposure_audits.append(dict(source=source,sampling_seed=seed,task_exposure_tvd=tvd,
                full_tasks=len(full_counts),replay_tasks=len(replay_counts),full_unique=len(set(full)),replay_unique=len(set(replay))))
            for policy in ('random','min_nll'):
                if policy=='min_nll' and nll is None:
                    continue
                indices=matched_schedule(task_ids,full,seed,policy,nll,content_keys=content_keys)
                exposures=Counter(task_ids[i] for i in indices)
                record=dict(policy=policy,source=source,sampling_seed=seed,data=str(data),data_sha256=file_sha(data),
                    score_sha256=file_sha(score_path) if policy=='min_nll' else None,indices=indices,
                    schedule_sha256=digest(indices),reference_schedule_sha256=digest(full),
                    task_order_sha256=digest([task_ids[i] for i in indices]),
                    reference_task_order_sha256=digest([task_ids[i] for i in full]),
                    unique_examples=len(set(indices)),eligible_examples=len(rows),
                    unique_task_content_examples=len({(task_ids[i],content_keys[i]) for i in indices}),
                    distinct_content_examples=len({content_keys[i] for i in indices}),
                    full_pool_distinct_content_examples=len(set(content_keys)),
                    support_definition='256 distinct (task ID, canonical student messages) examples; duplicate-copy NLL averaged',
                    presentations=len(indices),exposures_by_task=dict(exposures),
                    exact_task_order_matched=[task_ids[i] for i in indices]==[task_ids[i] for i in full],
                    exact_task_exposure_matched=exposures==full_counts)
                path=NEXT/f'{policy}-{source}-s{seed}.json'
                if path.exists() and json.loads(path.read_text())!=record:
                    raise ValueError('Existing schedule differs; do not overwrite')
                save(path,record)
                recipes.append(dict(path=str(path),sha256=file_sha(path),policy=policy,source=source,seed=seed,
                    exact_task_order_matched=record['exact_task_order_matched'],
                    exact_task_exposure_matched=record['exact_task_exposure_matched']))
    fits=result['fits']['mean_success']
    conditional=[]
    for init in (101,202,303,404):
        row=dict(initialization_seed=init)
        for arm in ('full_pass','subset_replay'):
            matrix=np.asarray(result['matrices']['mean_success'][arm]['paired_api_minus_generic'])
            values=matrix[(101,202,303,404).index(init)]*100
            row[arm]=dict(mean_gap_pp=float(values.mean()),sampling_sd_pp=float(values.std(ddof=1)))
        conditional.append(row)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    x=np.arange(4)
    for offset,arm,label,color in ((-.18,'full_pass','Full pool','#2878b5'),(.18,'subset_replay','Subset replay','#e58b2a')):
        values=[r[arm]['sampling_sd_pp'] for r in conditional]
        bars=axes[0].bar(x+offset,values,width=.36,label=label,color=color)
        axes[0].bar_label(bars,fmt='%.2f',fontsize=8,padding=3)
    axes[0].set_xticks(x,[101,202,303,404]);axes[0].set_ylabel('Within-initialization source-gap SD (pp)')
    axes[0].set_xlabel('Initialization / dropout seed');axes[0].legend(fontsize=8,loc='upper left')
    axes[0].set_title('All four initialization rows');axes[0].margins(y=.15)
    for source,color in (('api','#2878b5'),('generic','#e58b2a')):
        values=[r['task_exposure_tvd']*100 for r in exposure_audits if r['source']==source]
        axes[1].plot(range(4),values,'o-',label=source.upper(),color=color)
    axes[1].set_xticks(range(4),[101,202,303,404]);axes[1].set_ylabel('Per-task exposure total variation (%)')
    axes[1].set_xlabel('Sampling seed');axes[1].set_title('Task exposure still changes with sampling policy')
    axes[1].legend();axes[1].grid(alpha=.2)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUT/('stability-and-exposure.'+ext),dpi=180)
    plt.close(fig)
    value=dict(complete=True,stage2_summary_sha256=file_sha(summary_path),
        new_cases=68,new_episodes=4352,matched_compute_policy_effects=result['matched_compute_policy_effects'],
        success_variance_components={a:fits[a]['paired_api_minus_generic'] for a in fits},
        conditional_sampling=conditional,task_exposure_audits=exposure_audits,next_schedule_recipes=recipes,
        next_control_ready=len(recipes)==16,base_nll_ready=all((SCORES/(s+'.json')).exists() for s in ('api','generic')),
        next_question='At identical per-task counts and task order, how do random and base-NLL-selected 256-example supports compare with full traversal?',
        next_step_not_started_by_this_audit=True)
    lines=['# Stage 2完整结果与机制对照准备','','主比较68个模型、4,352个任务回合完成。','',
           '| 来源 | 全池相对子集复用成功率变化 | 胜/平/负（16格） |','|---|---:|---:|']
    for source in ('api','generic'):
        eff=result['matched_compute_policy_effects']['mean_success'][source]
        differences=np.asarray(eff['cell_differences'])
        lines.append(f"| {source} | {100*eff['mean_difference']:+.3f}pp | {(differences>0).sum()}/{(differences==0).sum()}/{(differences<0).sum()} |")
    lines+=['','下一步控制已固定：逐位置匹配原全池调度的任务ID，逐任务次数完全一致；比较随机选取256条与基础模型低NLL选取256条。新模型仍768次呈现、192次更新。',
            '低NLL选择在同一来源中固定示范成员；其采样种子主要改变顺序，不能将较小波动单独解释为样本质量优势。两组平均任务表现和NLL需同时报告。',
            '',f"准备好的调度配方：{len(recipes)}/16。主实验和旧冻结文件未修改。",'',
            '![稳定性与任务暴露](stability-and-exposure.png)','']
    (OUT/'RESULTS.zh-CN.md').write_text('\n'.join(lines))
    save(OUT/'summary.json',value)
    print(json.dumps(dict(complete=True,next_control_ready=value['next_control_ready'],recipes=len(recipes))),flush=True)


if __name__=='__main__':
    main()
