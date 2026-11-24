#!/usr/bin/env python3
"""Audit archived experiments and export trace summaries.

Outputs are separate from frozen reports. Trace categories describe observed
milestones. Exploratory bootstrap intervals are conditional on the collected
data and three paired training seeds."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/transfer-factorial-v1'
CONDITIONS = ['base', 'matched-generic', 'matched-api', 'full-generic', 'full-api']
SEEDS = [1101, 1202, 1303]
STEPS = [0, 8, 16, 32, 64, 128]
SOURCES = {}


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def rows(path):
    path = Path(path)
    digest = hashlib.sha256()
    result = []
    with path.open('rb') as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                result.append(json.loads(line))
    SOURCES[str(path.relative_to(ROOT))] = digest.hexdigest()
    return result


def interval(array):
    """Paired crossed bootstrap over seed and task axes, not episodes."""
    a = np.asarray(array, dtype=float)
    rng = np.random.default_rng(20261110)
    si = rng.integers(a.shape[0], size=(4000, a.shape[0]))
    ti = rng.integers(a.shape[1], size=(4000, a.shape[1]))
    samples = a[si[:, :, None], ti[:, None, :]].mean(axis=(1, 2))
    return dict(mean=float(a.mean()), per_seed=a.mean(axis=1).tolist(),
                exploratory_95=np.quantile(samples, [.025, .975]).tolist())


def trace_record(ep, trace, family, condition, seed):
    trace = sorted(trace, key=lambda r: r['step'])
    assert len(trace) == ep['steps'] and len({r['step'] for r in trace}) == len(trace)
    public_goal = trace[0]['messages'][1]['content'].lower()
    goal = (re.search(r'(?:cool|heat) some (\w+) and put', public_goal)
            or re.search(r'put a (?:cool|hot) (\w+) in', public_goal))
    assert goal, public_goal
    obj = re.escape(goal.group(1))
    pickup = re.compile(r'You pick up the ' + obj + r' \d+\b', re.I)
    thermal = re.compile(r'You ' + family + r' the ' + obj + r' \d+\b', re.I)
    picked = any(pickup.search(r['observation']) for r in trace)
    treated = any(thermal.search(r['observation']) for r in trace)
    if ep['success']:
        assert picked and treated, 'Successful episode missing expected public milestones'
    wrong_pickup = any(re.search(r'You pick up the ', r['observation'], re.I)
                       and not pickup.search(r['observation']) for r in trace)
    visible = any(re.search(r'\b' + obj + r' \d+\b', r['observation'], re.I) for r in trace)
    bucket = ('success' if ep['success'] else
              'no_goal_pickup' if not picked else
              'pickup_without_thermal' if not treated else 'thermal_without_success')
    seen, repeated, stale = {}, 0, 0
    for r in trace:
        if r['action'].startswith('go to '):
            key = (r['action'], r['observation'])
            if key in seen:
                repeated += 1
                stale += int(r['step'] - seen[key] > 12)
            seen[key] = r['step']
    return dict(family=family, condition=condition, seed=seed, task_id=ep['task_id'],
                success=int(ep['success']), steps=ep['steps'], milestone=bucket,
                goal_pickup_observed=picked, thermal_observed=treated, wrong_object_pickup_observed=wrong_pickup,
                goal_visible_in_feedback=visible, repeated_navigation_feedback=repeated,
                repeated_navigation_gap_gt12=stale,
                nothing_happens=sum(r['observation'].strip() == 'Nothing happens.' for r in trace),
                invalid_json=ep['invalid_actions'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'output/research-audit/2026-11-10')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    collection = read(ROOT / 'runs/api-harness-v1/collection-result.json')
    collection_check = {}
    for condition in ['none', 'generic', 'api']:
        es = rows(ROOT / f'runs/api-harness-v1/collection/{condition}/episodes.jsonl')
        assert len(es) == 144 and all(e['status'] == 'completed' for e in es)
        successes = [e for e in es if e['success']]
        collection_check[condition] = dict(episodes=len(es), wins=len(successes),
            successful_groups=len({e['task_group'] for e in successes}),
            mean_success_steps=float(np.mean([e['steps'] for e in successes])),
            input_tokens=sum(e['input_tokens'] for e in es), output_tokens=sum(e['output_tokens'] for e in es))
        assert len(successes) == collection['cells'][condition]['wins']

    families, trace_rows, episode_count = {}, [], 0
    for family in ['cool', 'heat']:
        report = read(RUN / family / 'main-result.json')
        partitions = {name: rows(RUN / family / f'{name}.jsonl') for name in
                      ['b_support_candidates', 'b_calibration', 'b_diagnostic', 'b_query', 'a_query']}
        groups = {name: {r['task_group'] for r in rs} for name, rs in partitions.items()}
        assert all(len(groups[name]) == len(rs) for name, rs in partitions.items())
        assert all(not (groups[a] & groups[b]) for a in groups for b in groups if a != b)
        ids = {role: sorted(r['task_id'] for r in rows(RUN / family / f'{role.lower()}_query.jsonl'))
               for role in ['A', 'B']}
        matrices = {role: {} for role in ['A', 'B']}
        for condition in CONDITIONS:
            for role in ['A', 'B']:
                seed_values = []
                for seed in SEEDS:
                    values = []
                    for step in (STEPS if role == 'B' else [0, 128]):
                        directory = RUN / family / 'evaluation/main' / f'{condition}-seed{seed}' / role / f'step-{step:04d}'
                        es = rows(directory / 'episodes.jsonl')
                        assert len(es) == len(ids[role])
                        assert all(e['status'] == 'completed' and not e['support_injected'] for e in es)
                        by_id = {e['task_id']: e for e in es}
                        assert set(by_id) == set(ids[role])
                        values.append([int(by_id[t]['success']) for t in ids[role]])
                        episode_count += len(es)
                        if role == 'B' and step == 128:
                            config = read(directory / 'config.json')
                            assert config['history_turns'] == 12 and config['max_steps'] == 40
                            ts = defaultdict(list)
                            for r in rows(directory / 'steps.jsonl'):
                                ts[r['attempt_id']].append(r)
                            for ep in es:
                                trace_rows.append(trace_record(ep, ts[ep['attempt_id']], family, condition, seed))
                    seed_values.append(values)
                matrices[role][condition] = np.asarray(seed_values, dtype=float)
            assert np.allclose(matrices['B'][condition].mean(axis=2), report['curves'][condition])
            assert np.allclose(matrices['A'][condition].mean(axis=2), report['retention'][condition])
        integrate = getattr(np, 'trapezoid', np.trapz)
        metrics, contrasts = {}, {}
        for c in CONDITIONS:
            b = matrices['B'][c]
            area = integrate(b, x=STEPS, axis=1) / 128
            metrics[c] = dict(initial=b[:, 0], raw_auc=area, gain_auc=area-b[:, 0], final=b[:, -1])
        for regime in ['matched', 'full']:
            contrasts[regime] = {k: interval(metrics[regime+'-api'][k]-metrics[regime+'-generic'][k])
                                 for k in ['initial', 'raw_auc', 'gain_auc', 'final']}
            assert np.isclose(contrasts[regime]['raw_auc']['mean'], report['effects'][regime+': API - generic']['raw']['mean'])
        summary = {}
        for c in CONDITIONS:
            diag = np.asarray(report['diagnostic_nll'][c])
            a = matrices['A'][c]
            summary[c] = dict(curve=matrices['B'][c].mean(axis=(0, 2)).tolist(),
                curve_per_seed=matrices['B'][c].mean(axis=2).tolist(),
                **{k:float(v.mean()) for k,v in metrics[c].items()},
                initial_nll=float(diag[:, 0].mean()), final_nll=float(diag[:, -1].mean()),
                retention_initial=float(a[:, 0].mean()), retention_final=float(a[:, -1].mean()),
                retention_by_type={t: a[:, :, [i for i,x in enumerate(ids['A']) if '/'+t+'-' in x]].mean(axis=(0,2)).tolist()
                    for t in ['pick_and_place_simple','pick_clean_then_place_in_recep','pick_heat_then_place_in_recep']})
        families[family] = dict(summary=summary, contrasts=contrasts,
            within_family_manifest_groups_disjoint=True,
            manifest_group_counts={name:len(value) for name,value in groups.items()})
    assert episode_count == 7200 and len(trace_rows) == 960

    dose = []
    for job in read(RUN / 'cool/design.json')['initializations']:
        if job['condition'] == 'base':
            continue
        meta = read(ROOT / job['a_adapter'] / 'experiment.json')
        corpus = rows(ROOT / job['data'])
        assert len(corpus) == meta['used_examples']
        rng, schedule = random.Random(job['a_seed']), []
        while len(schedule) < 256:
            order = list(range(len(corpus)))
            rng.shuffle(order)
            schedule.extend(order[:256-len(schedule)])
        assert len(set(schedule)) == meta['unique_scheduled_examples']
        sampled = [corpus[i] for i in schedule]
        def types(rs):
            return dict(Counter(r['task_id'].split('/')[1].split('-')[0] for r in rs))
        dose.append(dict(condition=job['condition'], seed=job['a_seed'], pool=len(corpus),
            unique_examples=len(set(schedule)), unique_tasks=len({r['task_id'] for r in sampled}),
            processed_action_types=types(sampled), pool_action_types=types(corpus),
            input_tokens=meta['input_tokens_rank0'], target_tokens=meta['target_tokens_rank0']))

    failure_summary = {}
    for family in ['cool', 'heat']:
        selected = [r for r in trace_rows if r['family'] == family]
        failed = [r for r in selected if not r['success']]
        task_scores = defaultdict(list)
        for r in selected:
            task_scores[r['task_id']].append(r['success'])
        assert len(task_scores)==32 and all(len(v)==15 for v in task_scores.values())
        failure_summary[family] = dict(episodes=len(selected), failures=len(failed),
            all_15_failed_tasks=sum(sum(v)==0 for v in task_scores.values()),
            all_15_succeeded_tasks=sum(sum(v)==15 for v in task_scores.values()),
            milestones=dict(Counter(r['milestone'] for r in selected)),
            failed_at_limit=sum(r['steps']==40 for r in failed),
            failed_with_repeated_navigation=sum(r['repeated_navigation_feedback']>0 for r in failed),
            failed_with_navigation_gap_gt12=sum(r['repeated_navigation_gap_gt12']>0 for r in failed),
            failed_with_nothing_happens=sum(r['nothing_happens']>0 for r in failed),
            failed_with_wrong_object_pickup=sum(r['wrong_object_pickup_observed'] for r in failed),
            failed_no_pickup_but_goal_visible=sum(r['milestone']=='no_goal_pickup' and r['goal_visible_in_feedback'] for r in failed),
            by_condition={c:dict(Counter(r['milestone'] for r in selected if r['condition']==c)) for c in CONDITIONS})
    history = {}
    for name in ['dual5090-v1','dual5090-v2','dual5090-v2-filtered']:
        d = read(ROOT / 'runs' / name / 'results.json')
        history[name] = {k:d[k] for k in ['scientific_status','fresh_primary','learning_comparisons','claims_not_established'] if k in d}
    earlier_speed = read(ROOT / 'runs/learning-speed-v1/main-result.json')
    earlier_api = read(ROOT / 'runs/api-harness-v1/main-result.json')
    result = dict(status='completed_posthoc_audit_not_new_experiment', date='2026-11-10',
        inference='Exploratory pointwise intervals; no multiplicity correction; shared collection bank and only three seeds.',
        verified_main_episodes=episode_count, final_B_trace_episodes=len(trace_rows),
        collection=collection_check, collection_contrasts=collection['contrasts'],
        history=history, earlier_speed_contrasts=earlier_speed['contrasts'],
        earlier_api_contrasts=earlier_api['contrasts'], families=families,
        primary_interaction=read(RUN/'main-result.json'), a_training_dose=dose,
        trace_definitions=dict(milestones='Ever-observed goal-type pickup/thermal feedback; not a causal failure label.',
            repeated_navigation='Same go-to command and identical returned observation at a later step; revisits may be necessary.',
            gap_gt12='Most recent identical navigation/feedback >12 steps earlier; does not establish memory truncation as cause.',
            nothing_happens='Literal environment feedback, distinct from invalid JSON.'),
        failure_summary=failure_summary)
    (args.out/'audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    with (args.out/'final-trace-audit.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(trace_rows[0]));writer.writeheader();writer.writerows(trace_rows)
    SOURCES[str(Path(__file__).resolve().relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.out/'input-sha256.json').write_text(json.dumps(SOURCES,indent=2)+'\n')
    plot(result,args.out)
    print(json.dumps(dict(verified_main_episodes=episode_count, final_trace_episodes=len(trace_rows),
        collection=collection_check, failure_summary=failure_summary, output=str(args.out)),indent=2))


def plot(result,out):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),sharex=True,sharey=True)
    for i,f in enumerate(['cool','heat']):
        for j,regime in enumerate(['matched','full']):
            ax=axes[i,j]
            for c,color,label in [('base','#999999','Base'),(regime+'-generic','#4477AA','Generic'),(regime+'-api','#CC6677','API')]:
                d=result['families'][f]['summary'][c]
                for row in d['curve_per_seed']:
                    ax.plot(STEPS,np.array(row)*100,color=color,alpha=.22,lw=.8)
                ax.plot(STEPS,np.array(d['curve'])*100,color=color,marker='o',lw=2,label=label)
            ax.set_title(f'{f.capitalize()} | {regime}');ax.set_xticks(STEPS);ax.set_ylim(0,100);ax.grid(alpha=.2)
            if i==1:ax.set_xlabel('B optimizer updates')
            if j==0:ax.set_ylabel('Unassisted success (%)')
    axes[0,0].legend(frameon=False)
    fig.suptitle('Observed adaptation: means and all three paired seeds')
    fig.tight_layout(rect=[0,.035,1,.96]);fig.text(.5,.015,'Same B examples and updates within family. Heat was A-seen; cool was A-unseen. No new rollouts.',ha='center',fontsize=9)
    save(fig,out/'adaptation-audit')

    fig,axes=plt.subplots(1,4,figsize=(13,4.3),sharey=True)
    labels=[(f,r) for f in ['cool','heat'] for r in ['matched','full']]
    for ax,key,title in zip(axes,['initial','raw_auc','gain_auc','final'],['Initial score','Raw AUC','Gain AUC','Final score']):
        ax.axvline(0,color='#999999',ls='--')
        for i,(f,r) in enumerate(labels):
            d=result['families'][f]['contrasts'][r][key];lo,hi=np.array(d['exploratory_95'])*100
            ax.plot([lo,hi],[i,i],color='#4477AA');ax.scatter(d['mean']*100,i,color='#4477AA')
            ax.scatter(np.array(d['per_seed'])*100,np.array([i-.1,i,i+.1]),color='#CC6677',s=12)
        ax.set_title(title);ax.set_xlabel('API - generic (pp)');ax.grid(axis='x',alpha=.2)
        ax.set_yticks(range(4),[f'{f} / {r}' for f,r in labels])
    fig.tight_layout(rect=[0,.09,1,.93]);fig.suptitle('Separate head start, integrated performance and endpoint')
    fig.text(.5,.025,'Exploratory 95% crossed-bootstrap intervals; red = paired seeds. Gain AUC = raw AUC - initial score; not a plasticity control.',ha='center',fontsize=8)
    save(fig,out/'effect-decomposition')

    fig,axes=plt.subplots(1,2,figsize=(10,4.7),sharey=True)
    categories=['no_goal_pickup','pickup_without_thermal','thermal_without_success','success']
    colors=['#D55E00','#E69F00','#CC79A7','#009E73']
    names=['No goal pickup','Pickup, no thermal milestone','Thermal, no success','Success']
    for ax,f in zip(axes,['cool','heat']):
        left=np.zeros(5)
        for cat,col,label in zip(categories,colors,names):
            vals=np.array([result['failure_summary'][f]['by_condition'][c].get(cat,0)/96*100 for c in CONDITIONS])
            ax.barh(range(5),vals,left=left,color=col,label=label);left+=vals
        ax.set_yticks(range(5),CONDITIONS);ax.set_xlim(0,100);ax.set_title(f.capitalize());ax.set_xlabel('Final-checkpoint episodes (%)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
    fig.suptitle('Observed milestones in all 960 final B rollouts')
    fig.tight_layout(rect=[0,.15,1,.93])
    save(fig,out/'failure-milestones')


def save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=160)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    main()
