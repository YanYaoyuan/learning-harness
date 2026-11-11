#!/usr/bin/env python3
"""Measured learning curves and the prespecified calibration gate."""
from pathlib import Path
import csv
import json
import os
import statistics

os.environ.setdefault('MPLCONFIGDIR',str(Path('.cache/matplotlib').absolute()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from learning_harness.common import read_jsonl,file_sha

ROOT=Path('runs/learning-speed-v1')
STEPS=[0,8,32,128]
SEEDS=[1101,1202,1303]
CONDITIONS=['base','filtered-none','filtered-early']
COLORS={'base':'#777777','filtered-none':'#4477AA','filtered-early':'#CC6677'}
SEED_COLORS=['#4477AA','#EE7733','#228833']


def read_evaluation(stage,condition,seed,step):
    directory=ROOT/'evaluation'/stage/f'{condition}-seed{seed}'/f'step-{step:04d}'
    summary=json.loads((directory/'summary.json').read_text())
    assert summary['complete'] and summary['metrics']['none']['error_count']==0
    rows=read_jsonl(directory/'episodes.jsonl')
    assert len({r['task_id'] for r in rows})==len(rows)
    assert all(not r['support_injected'] and r['status']=='completed' for r in rows)
    return rows


def export_figure(fig,path):
    for ext in ['png','pdf','svg']:fig.savefig(path.with_suffix('.'+ext),dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def export_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def report(stage):
    out=ROOT/'figures'/stage;out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False})
    design=json.loads((ROOT/'design.json').read_text())
    conditions=['base'] if stage=='calibration' else CONDITIONS
    curve_records=[];curves={};episodes={}
    b_ids={r['task_id'] for r in read_jsonl(ROOT/('b_calibration.jsonl' if stage=='calibration' else 'b_query.jsonl'))}
    for condition in conditions:
        curves[condition]=[]
        for seed in SEEDS:
            ys=[]
            for step in STEPS:
                all_rows=read_evaluation(stage,condition,seed,step)
                rows=[r for r in all_rows if r['task_id'] in b_ids]
                assert len(rows)==len(b_ids)
                episodes[condition,seed,step]={r['task_id']:r for r in all_rows}
                won=sum(r['success'] for r in rows);ys.append(won/len(rows))
                curve_records.append(dict(condition=condition,b_seed=seed,updates=step,wins=won,n_tasks=len(rows),success=won/len(rows)))
            curves[condition].append(ys)
    export_csv(out/'curve-values.csv',curve_records)
    fig,ax=plt.subplots(figsize=(8.3,4.9))
    for condition in conditions:
        arr=np.array(curves[condition])*100
        if stage=='calibration':
            for seed,row,color,marker in zip(SEEDS,arr,SEED_COLORS,['o','s','^']):
                ax.plot(STEPS,row,color=color,linewidth=1.7,marker=marker,markersize=5,label=f'B seed {seed}')
            ax.plot(STEPS,arr.mean(axis=0),color='#222222',linewidth=2.3,linestyle='--',label='Mean')
        else:
            for seed,row in zip(SEEDS,arr):ax.plot(STEPS,row,color=COLORS[condition],alpha=.3,linewidth=1,marker='o',markersize=3)
            ax.plot(STEPS,arr.mean(axis=0),color=COLORS[condition],linewidth=2.4,marker='o',label=condition)
            from matched_study import crossed_interval
            bands=[]
            for k in STEPS:
                matrix=[[int(episodes[condition,s,k][t]['success']) for t in sorted(b_ids)] for s in SEEDS]
                bands.append(crossed_interval(matrix))
            lo,hi=np.array(bands).T*100
            ax.fill_between(STEPS,lo,hi,color=COLORS[condition],alpha=.1)
    ax.set_xticks(STEPS);ax.set_ylim(-2,102);ax.set_xlabel('B-stage optimizer updates')
    ax.set_ylabel('Unsupported cooling-task success (%)')
    ax.set_title('Calibration learnability control' if stage=='calibration' else 'Adaptation on identical new B data',loc='left',fontweight='bold')
    ax.grid(color='#E7E7E7',linewidth=.6);ax.legend(frameon=False)
    fig.subplots_adjust(bottom=.22)
    note=(f'{len(b_ids)} calibration tasks; 3 B seeds; actual checkpoint observations. One continuous trajectory per seed.' if stage=='calibration'
        else f'{len(b_ids)} query tasks; 3 paired seeds. Pointwise 95% crossed-bootstrap bands; only 4 observed checkpoints.')
    fig.text(.5,.015,note,ha='center',fontsize=8)
    export_figure(fig,out/'learning_curves')
    result=dict(stage=stage,complete=True,steps=STEPS,b_seeds=SEEDS,b_tasks=len(b_ids),curves=curves,
                teaching_data=json.loads((ROOT/'curriculum.json').read_text()),paid_api_calls=0)
    if stage=='calibration':
        gains=[round((r[-1]-r[0])*len(b_ids)) for r in curves['base']]
        threshold=design['positive_control']['required_additional_calibration_wins']
        passed=sum(g>=threshold for g in gains)>=design['positive_control']['required_seeds_meeting_gate']
        result.update(additional_wins=gains,required_additional_wins=threshold,gate_passed=passed,
            query_opened=False,interpretation='Learnability gate only; no harness effect tested')
        (ROOT/'calibration-result.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    # Paired per-task learning-curve area retains the time dimension.
    from matched_study import crossed_interval
    task_ids=sorted(b_ids);auc={};gain_auc={}
    for condition in CONDITIONS:
        auc[condition]=[];gain_auc[condition]=[]
        for seed in SEEDS:
            scores=np.array([[int(episodes[condition,seed,k][t]['success']) for t in task_ids] for k in STEPS],dtype=float)
            a=np.trapezoid(scores,x=STEPS,axis=0)/128 if hasattr(np,'trapezoid') else np.trapz(scores,x=STEPS,axis=0)/128
            auc[condition].append(a.tolist());gain_auc[condition].append((a-scores[0]).tolist())
    contrasts={}
    for other in ['filtered-none','base']:
        delta=[[e-n for e,n in zip(er,nr)] for er,nr in zip(gain_auc['filtered-early'],gain_auc[other])]
        raw_delta=[[e-n for e,n in zip(er,nr)] for er,nr in zip(auc['filtered-early'],auc[other])]
        initial_delta=[e[0]-n[0] for e,n in zip(curves['filtered-early'],curves[other])]
        contrasts['filtered-early minus '+other]=dict(mean_gain_auc_delta=statistics.mean(v for r in delta for v in r),
            per_seed_delta=[statistics.mean(r) for r in delta],crossed_bootstrap_95=crossed_interval(delta),
            mean_raw_auc_delta=statistics.mean(v for r in raw_delta for v in r),
            per_seed_raw_auc_delta=[statistics.mean(r) for r in raw_delta],raw_auc_crossed_bootstrap_95=crossed_interval(raw_delta),
            initial_success_deltas=initial_delta)
        c=contrasts['filtered-early minus '+other]
        assert abs(c['mean_raw_auc_delta']-statistics.mean(initial_delta)-c['mean_gain_auc_delta'])<1e-10
    retention={}
    for condition in CONDITIONS:
        retention[condition]=[]
        for seed in SEEDS:
            scores=[]
            for k in STEPS:
                rows=[r for t,r in episodes[condition,seed,k].items() if t not in b_ids]
                assert len(rows)==21;scores.append(sum(r['success'] for r in rows)/21)
            retention[condition].append(scores)
    initialization_audit={};b_initial_hashes={s:set() for s in SEEDS};trainable=set();orders=set();token_counts=set()
    for condition in CONDITIONS:
        for seed in SEEDS:
            case=f'{condition}-seed{seed}'
            meta=json.loads((ROOT/'trajectories'/case/'trajectory.json').read_text())
            assert meta['processed_examples']==512 and meta['b_seed']==seed
            assert meta['data_sha256']==file_sha(ROOT/'b_schedule.jsonl')
            assert meta['a_frozen_verified'] and meta['b_updated_verified']
            assert meta['initialization']['exact_zero_b_logit_check']
            b_initial_hashes[seed].add(meta['initialization']['b_initial_parameter_sha256'])
            trainable.add(meta['initialization']['trainable_parameters']);orders.add(meta['actual_order_sha256'])
            token_counts.add(json.dumps(meta['token_accounting'],sort_keys=True))
            initialization_audit[case]=dict(initial_nll=meta['initial_b_material_nll'],
                trainable_parameters=meta['initialization']['trainable_parameters'],
                initial_query_success=curves[condition][SEEDS.index(seed)][0])
    assert all(len(v)==1 for v in b_initial_hashes.values()) and len(trainable)==len(orders)==len(token_counts)==1
    retention_delta={c:[[x-row[0] for x in row] for row in rows] for c,rows in retention.items()}
    family_names=sorted({r['task_type'] for r in read_jsonl(ROOT/'a_retention.jsonl')})
    retention_by_type={}
    for family in family_names:
        retention_by_type[family]={}
        for condition in CONDITIONS:
            retention_by_type[family][condition]=[]
            for seed in SEEDS:
                ys=[]
                for k in STEPS:
                    rows=[r for t,r in episodes[condition,seed,k].items() if t not in b_ids and r['task_type']==family]
                    assert len(rows)==7;ys.append(sum(r['success'] for r in rows)/7)
                retention_by_type[family][condition].append(ys)
    result.update(normalized_auc=auc,gain_auc=gain_auc,contrasts=contrasts,a_retention=retention,
                  a_retention_delta=retention_delta,a_retention_by_type=retention_by_type,
                  initialization_audit=initialization_audit,
                  shared_b_initialization_and_order_verified=True,
                  limitations=['One B family','3 paired seed trajectories','Initial knowledge and plasticity are not fully separated',
                    'Trapezoidal AUC uses only four checkpoints; rapid transitions can be undersampled',
                    'Fixed batched evaluation can be numerically sensitive to batch composition'])
    (ROOT/'main-result.json').write_text(json.dumps(result,indent=2)+'\n')
    seed_records=[]
    for condition in CONDITIONS:
        for index,seed in enumerate(SEEDS):
            seed_records.append(dict(condition=condition,b_seed=seed,
                initial_success=curves[condition][index][0],final_success=curves[condition][index][-1],
                normalized_auc=statistics.mean(auc[condition][index]),
                gain_auc=statistics.mean(gain_auc[condition][index]),
                initial_b_material_token_nll=initialization_audit[f'{condition}-seed{seed}']['initial_nll']['token_mean']))
    export_csv(out/'seed-metrics.csv',seed_records)
    export_csv(out/'retention-values.csv',[
        dict(condition=c,b_seed=s,updates=k,success=retention[c][i][j],change=retention_delta[c][i][j])
        for c in CONDITIONS for i,s in enumerate(SEEDS) for j,k in enumerate(STEPS)])
    export_csv(out/'retention-by-type.csv',[
        dict(task_type=f,condition=c,b_seed=s,updates=k,success=retention_by_type[f][c][i][j])
        for f in family_names for c in CONDITIONS for i,s in enumerate(SEEDS) for j,k in enumerate(STEPS)])
    measured_scores=[dict(condition=c,b_seed=s,updates=k,task_id=t,task_type=r['task_type'],success=int(r['success']))
        for (c,s,k),rows in episodes.items() for t,r in sorted(rows.items())]
    export_csv(out/'episode-scores.csv',measured_scores)
    (out/'measured-data.json').write_text(json.dumps(dict(measured_only=True,stage=stage,
        steps=STEPS,seeds=SEEDS,b_task_ids=task_ids,scores=measured_scores,results=result),indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(8.3,4.9))
    for condition in CONDITIONS:
        arr=np.array(retention[condition])*100
        for row in arr:ax.plot(STEPS,row,color=COLORS[condition],alpha=.3,linewidth=1)
        ax.plot(STEPS,arr.mean(axis=0),color=COLORS[condition],marker='o',linewidth=2.2,label=condition)
    ax.set_xticks(STEPS);ax.set_ylim(-2,102);ax.set_xlabel('B-stage optimizer updates');ax.set_ylabel('Old A-task success (%)')
    ax.set_title('Retention during new-task adaptation',loc='left',fontweight='bold');ax.legend(frameon=False);ax.grid(color='#E7E7E7')
    fig.subplots_adjust(bottom=.2);fig.text(.5,.015,'21 previously inspected A development tasks; secondary retention diagnostic, 3 seeds.',ha='center',fontsize=9)
    export_figure(fig,out/'retention_curves')
    fig,ax=plt.subplots(figsize=(8.3,4.9))
    maximum=max(abs(v) for rows in retention_delta.values() for row in rows for v in row)*100
    for condition in CONDITIONS:
        arr=np.array(retention_delta[condition])*100
        for row in arr:ax.plot(STEPS,row,color=COLORS[condition],alpha=.3,linewidth=1)
        ax.plot(STEPS,arr.mean(axis=0),color=COLORS[condition],marker='o',linewidth=2.2,label=condition)
    ax.axhline(0,color='#444444',linestyle='--',linewidth=1);ax.set_xticks(STEPS)
    ax.set_ylim(-max(5,maximum*1.2),max(5,maximum*1.2))
    ax.set_xlabel('B-stage optimizer updates');ax.set_ylabel('Old-task change from step 0 (pp)')
    ax.set_title('Change in old-task ability',loc='left',fontweight='bold');ax.legend(frameon=False);ax.grid(color='#E7E7E7')
    fig.subplots_adjust(bottom=.2);fig.text(.5,.015,'Within-initialization changes; positive means improvement. 21 old development tasks, 3 paired seeds.',ha='center',fontsize=9)
    export_figure(fig,out/'retention_changes')
    fig,axes=plt.subplots(1,3,figsize=(12.3,4.6),sharey=True)
    for ax,family in zip(axes,family_names):
        for condition in CONDITIONS:
            arr=np.array(retention_by_type[family][condition])*100
            for row in arr:ax.plot(STEPS,row,color=COLORS[condition],alpha=.25,linewidth=.8)
            ax.plot(STEPS,arr.mean(axis=0),color=COLORS[condition],marker='o',linewidth=2,label=condition)
        ax.set_xticks(STEPS);ax.set_ylim(-2,102);ax.set_xlabel('B updates');ax.grid(color='#E7E7E7',linewidth=.5)
        title={'pick_and_place_simple':'Place','pick_clean_then_place_in_recep':'Clean + place','pick_heat_then_place_in_recep':'Heat + place'}[family]
        ax.set_title(title,loc='left',fontweight='bold')
    axes[0].set_ylabel('Old-task success (%)');axes[-1].legend(frameon=False,fontsize=8)
    fig.subplots_adjust(bottom=.22,wspace=.15);fig.text(.5,.015,'7 old tasks/type; type means can hide opposite changes. Heat starts near the floor. Secondary, previously inspected development tasks.',ha='center',fontsize=8)
    export_figure(fig,out/'retention_by_type')
    fig,ax=plt.subplots(figsize=(8,3.8));ax.axvline(0,color='#888888',linestyle='--')
    for i,(label,c) in enumerate(contrasts.items()):
        mean=c['mean_gain_auc_delta']*100;lo,hi=np.array(c['crossed_bootstrap_95'])*100
        ax.plot([lo,hi],[i,i],color='#26374A',linewidth=1.6)
        ax.scatter(mean,i,marker='D',color='#26374A',s=35,zorder=3)
        for shift,color,value in zip([-.12,0,.12],SEED_COLORS,c['per_seed_delta']):ax.scatter(value*100,i+shift,color=color,s=30)
    ax.set_yticks(range(len(contrasts)),list(contrasts));ax.set_xlabel('Time-normalized gain-AUC difference (pp)')
    ax.set_title('Paired adaptation-gain effects',loc='left',fontweight='bold');ax.grid(axis='x',color='#E7E7E7')
    fig.subplots_adjust(left=.33,bottom=.24);fig.text(.5,.015,'95% paired crossed bootstrap over 3 seeds and 24 query tasks. Gain depends on starting ability and ceiling.',ha='center',fontsize=8)
    export_figure(fig,out/'gain_auc_effects')
    fig,axes=plt.subplots(1,2,figsize=(11.7,3.9),sharey=True)
    for ax,mean_key,interval_key,seed_key,title in [
        (axes[0],'mean_raw_auc_delta','raw_auc_crossed_bootstrap_95','per_seed_raw_auc_delta','Raw learning-curve AUC'),
        (axes[1],'mean_gain_auc_delta','crossed_bootstrap_95','per_seed_delta','Gain relative to own start')]:
        ax.axvline(0,color='#888888',linestyle='--')
        for i,(label,c) in enumerate(contrasts.items()):
            mean=c[mean_key]*100;lo,hi=np.array(c[interval_key])*100
            # Draw intervals directly: percentile bootstrap intervals need not contain their point estimate.
            ax.plot([lo,hi],[i,i],color='#26374A',linewidth=1.6)
            ax.scatter(mean,i,marker='D',color='#26374A',s=35,zorder=3)
            for shift,color,value in zip([-.12,0,.12],SEED_COLORS,c[seed_key]):ax.scatter(value*100,i+shift,color=color,s=23)
        ax.set_yticks(range(len(contrasts)),list(contrasts));ax.set_xlabel('Early minus comparator (pp)')
        ax.set_title(title,loc='left',fontweight='bold');ax.grid(axis='x',color='#E7E7E7')
    fig.subplots_adjust(left=.24,bottom=.24,wspace=.25)
    fig.text(.5,.015,'Diamonds: paired mean; colored dots: seeds; lines: 95% crossed bootstrap. Raw AUC includes initial ability; gain AUC depends on headroom.',ha='center',fontsize=8)
    export_figure(fig,out/'raw_and_gain_auc_effects')
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['calibration','main'],required=True)
    args=parser.parse_args();print(json.dumps(report(args.stage),indent=2))
