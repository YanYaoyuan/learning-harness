#!/usr/bin/env python3
"""Read-only paired Stage 3 snapshots, separate from the frozen final analysis."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('runs/variance-diagnostic-v1/stage3')
OUT = Path('output/variance-diagnostic-v1/stage3/interim')
SOURCES = ('api', 'generic')
ARMS = ('random', 'min_nll')


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def paired(records):
    eligible = {}
    for record in records:
        case = record['case']
        if case['repeat'] != 0 or case['arm'] not in ARMS:
            continue
        key = (case['initialization_seed'], case['sampling_seed'], case['source'], case['arm'])
        if key in eligible:
            raise ValueError('Duplicate seed/source/policy cell')
        eligible[key] = record
    keys = sorted({key[:2] for key in eligible
                   if all((*key[:2], source, arm) in eligible for source in SOURCES for arm in ARMS)})
    rows = []
    for init, sample in keys:
        for source in SOURCES:
            a, b = [eligible[(init, sample, source, arm)] for arm in ARMS]
            if a['control']['initial_parameter_sha256'] != b['control']['initial_parameter_sha256']:
                raise ValueError('Paired policies did not share initialization')
            for field in ('task_exposures', 'task_order_sha256', 'processed_examples'):
                if a['exposure'][field] != b['exposure'][field]:
                    raise ValueError('Paired task exposure/order/budget mismatch')
            if any(r['exposure']['unique_task_content_examples'] != 256 for r in (a, b)):
                raise ValueError('Distinct-content budget mismatch')
            row = dict(initialization_seed=init, sampling_seed=sample, source=source)
            for arm, record in zip(ARMS, (a, b)):
                row[arm + '_success'] = record['mean_success']
                row[arm + '_nll'] = record['mean_nll']
            row['success_difference_pp'] = 100 * (b['mean_success'] - a['mean_success'])
            row['nll_difference'] = b['mean_nll'] - a['mean_nll']
            rows.append(row)
    return keys, rows


def snapshot(cutoff):
    frozen = json.loads((ROOT / 'freeze.json').read_text())
    plan = {c['name']: c for c in frozen['cases']}
    records, hashes = [], {}
    for path in sorted((ROOT / 'cases').glob('*/worker-result.json')):
        if path.stat().st_mtime > cutoff:
            continue
        raw = path.read_bytes()
        r = json.loads(raw)
        if not r['complete'] or r['smoke'] or r['case'] != plan[path.parent.name]:
            raise ValueError('Incomplete or mismatched result')
        for family in ('cool', 'heat'):
            folder = path.parent / 'evaluation' / family
            s = json.loads((folder / 'summary.json').read_text())
            run = json.loads((folder / 'run.json').read_text())
            if (not s['complete'] or s['recorded'] != 32 or s['metrics']['none']['error_count']
                    or s['metrics']['none']['success_rate'] != r['success'][family]
                    or run['revision'] != r['composition_revision']):
                raise ValueError('Raw evaluation audit failed')
        nll = json.loads((path.parent / 'diagnostic-nll.json').read_text())
        if (not nll['checkpoint_unchanged'] or nll['composition_revision'] != r['composition_revision']
                or r['mean_success'] != sum(r['success'].values()) / 2
                or r['mean_nll'] != sum(nll[f]['task_mean_nll'] for f in ('cool', 'heat')) / 2):
            raise ValueError('NLL or mean-score audit failed')
        records.append(r)
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    keys, rows = paired(records)
    return dict(interim=True, final_study_complete=False, cutoff_unix=cutoff,
                completed_models=len(records), planned_models=70, matched_seed_cells=keys,
                paired_rows=rows, records=records, provenance=hashes,
                freeze_sha256=hashlib.sha256((ROOT / 'freeze.json').read_bytes()).hexdigest(),
                note='Descriptive peek requested by user; no inference or outcome-based scheduling')


def create(value):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from run_stage3_task_controls import load_freeze
    load_freeze()
    if not value['matched_seed_cells']:
        raise ValueError('No fully paired seed cells')
    out = OUT / datetime.fromtimestamp(value['cutoff_unix'], timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'inputs.json', value)
    rows = value['paired_rows']
    effects = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    for ax, source in zip(axes, SOURCES):
        selected = [r for r in rows if r['source'] == source]
        for r in selected:
            ys = [100 * r[a + '_success'] for a in ARMS]
            ax.plot([0, 1], ys, 'o-', alpha=.65,
                    label=f"init {r['initialization_seed']} / sample {r['sampling_seed']}")
        means = [100 * sum(r[a + '_success'] for r in selected) / len(selected) for a in ARMS]
        for x, mean in enumerate(means):
            ax.plot([x-.12, x+.12], [mean, mean], color='black', linewidth=3)
            ax.annotate(f'{mean:.2f}%', (x, mean), (0, 13), textcoords='offset points', ha='center',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.8))
        delta = [r['success_difference_pp'] for r in selected]
        effects[source] = dict(random_success_pct=means[0], min_nll_success_pct=means[1],
                               nll_minus_random_pp=sum(delta)/len(delta),
                               wins=sum(v > 0 for v in delta), ties=sum(v == 0 for v in delta),
                               losses=sum(v < 0 for v in delta),
                               diagnostic_nll_difference=sum(r['nll_difference'] for r in selected)/len(selected))
        ax.set(title=source.upper(), xticks=[0, 1], xticklabels=['Task-matched random', 'Low base-model NLL'],
               xlim=(-.5, 1.5), ylim=(0, 100), ylabel='Post-A success (%)')
        ax.grid(axis='y', alpha=.2)
        if len(selected) <= 4:
            ax.legend(fontsize=7, loc='upper right')
    fig.suptitle(f"INTERIM: {len(value['matched_seed_cells'])}/16 fully paired seed cells; "
                 f"{value['completed_models']}/70 models complete", fontsize=11)
    fig.text(.5, .015, '256 distinct task-content examples; identical task counts/order; 768 presentations; 192 updates',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .95))
    for extension in ('png', 'pdf'):
        fig.savefig(out / ('paired-policies.' + extension), dpi=180)
    plt.close(fig)
    summary = {k:v for k,v in value.items() if k not in ('records', 'provenance')}
    summary['effects'] = effects
    save(out / 'summary.json', summary)
    lines = ['# Stage 3中途成对观察', '',
             f"完成{value['completed_models']}/70个模型；均值只使用{len(value['matched_seed_cells'])}组两来源、两策略均完成的种子单元。", '',
             '| 来源 | 随机选择成功率 | 低NLL成功率 | 低NLL−随机 |', '|---|---:|---:|---:|']
    for source, e in effects.items():
        lines.append(f"| {source} | {e['random_success_pct']:.2f}% | {e['min_nll_success_pct']:.2f}% | {e['nll_minus_random_pp']:+.2f}pp |")
    lines += ['', '这是描述性中途观察，无显著性检验或提前停止规则。未配齐单元、参考检查和条件重复保留于inputs.json，均不混入成对均值。原正式方案、预算和评测任务不变。',
              '', '![逐种子成对结果](paired-policies.png)', '']
    (out / 'RESULTS.zh-CN.md').write_text('\n'.join(lines))
    return out


def upload(out):
    import wandb
    if out.resolve().parent != OUT.resolve():
        raise ValueError('Unexpected snapshot path')
    result = json.loads((out / 'summary.json').read_text())
    target = json.loads(Path('output/wandb/connection.json').read_text())
    host = target.get('base_url', target.get('host'))
    if host:
        os.environ['WANDB_BASE_URL'] = host
    run = wandb.init(entity=target['entity'], project=target['project'],
                     id=result['freeze_sha256'][:12] + '-interim-' + out.name,
                     name=f"Stage 3 / INTERIM / {len(result['matched_seed_cells'])} paired cells",
                     group='variance-diagnostic-v1', job_type='descriptive-interim-look',
                     mode='online', resume='allow', dir=str(out.absolute()), save_code=False,
                     settings=wandb.Settings(console='off', disable_code=True, disable_git=True,
                                             save_code=False, disable_job_creation=True, x_disable_stats=True,
                                             init_timeout=60))
    try:
        run.log({'interim/paired-policies': wandb.Image(str(out / 'paired-policies.png'))})
        run.summary.update(dict(interim=True, final_study_complete=False, effects=result['effects'],
                                matched_cells=len(result['matched_seed_cells']), completed_models=result['completed_models']))
        artifact = wandb.Artifact('stage3-interim-' + out.name, type='descriptive-interim-results')
        for p in out.iterdir():
            if p.is_file() and p.suffix in ('.json', '.png', '.pdf', '.md'):
                artifact.add_file(str(p))
        run.log_artifact(artifact).wait()
        save(out / 'archival.json', dict(status='uploaded', run_id=run.id, run_url=run.url, interim=True))
    finally:
        run.finish()


def watch():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'observer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = dict(milestones=[1, 4, 8, 12, 16], read_only=True,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                note='Set after first eight models; descriptive monitoring, not preregistered inference')
    plan_path = OUT / 'observation-plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Observation plan changed')
    save(plan_path, plan)
    state_path = OUT / 'observer-state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(status='watching', looks={})
    while True:
        value = snapshot(time.time())
        n = len(value['matched_seed_cells'])
        pending = [m for m in plan['milestones'] if m <= n and str(m) not in state['looks']]
        if pending:
            out = create(value)
            record = dict(directory=str(out), actual_paired_cells=n, cloud_status='pending')
            for m in pending:
                state['looks'][str(m)] = dict(record)
            save(state_path, state)
            with (out / 'upload.log').open('x') as log:
                try:
                    result = subprocess.run([str(Path('.venv-wandb/bin/python').absolute()), __file__, '--upload', str(out)],
                                            stdout=log, stderr=subprocess.STDOUT, timeout=300)
                    status = 'uploaded' if result.returncode == 0 else 'failed'
                except subprocess.TimeoutExpired:
                    status = 'timed_out'
            for m in pending:
                state['looks'][str(m)]['cloud_status'] = status
            save(state_path, state)
        status = json.loads((ROOT / 'execution.json').read_text())['status']
        if status in ('completed', 'failed', 'interrupted', 'stopped_at_cap'):
            state.update(status='finished', main_execution_status=status)
            save(state_path, state)
            return
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--upload', type=Path)
    args = parser.parse_args()
    if args.upload:
        upload(args.upload)
    elif args.watch:
        watch()
    else:
        print(create(snapshot(time.time())))
