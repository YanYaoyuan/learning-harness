#!/usr/bin/env python3
"""Exploratory content audit of frozen schedules; no model or outcome selection."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path('runs/variance-diagnostic-v1/stage3')
OUT = Path('output/variance-diagnostic-v1/stage3/action-mix-audit')
FAMILIES = ('go', 'open', 'look', 'take', 'move', 'clean', 'heat', 'cool', 'examine', 'inventory', 'close', 'use')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    from run_stage3_task_controls import load_freeze
    load_freeze()
    if OUT.exists():
        raise ValueError('Preserve existing exploratory audit before another run')
    rows, provenance = [], {}
    for source in ('api', 'generic'):
        data_path = Path(f'runs/transfer-factorial-v1/corpora/full-{source}.jsonl')
        data = [json.loads(line) for line in data_path.read_text().splitlines()]
        score_path = ROOT / 'base-nll' / (source + '.json')
        score_record = json.loads(score_path.read_text())
        scores = score_record['scores']
        if score_record['data_sha256'] != digest(data_path) or len(scores) != len(data):
            raise ValueError('Score/corpus mismatch')
        for index, score in enumerate(scores):
            if score['index'] != index or score['task_id'] != data[index]['task_id']:
                raise ValueError('NLL row alignment changed')
        for arm in ('random', 'min_nll'):
            for seed in (101, 202, 303, 404):
                path = ROOT / 'inputs/schedules' / f'{arm}-{source}-s{seed}.json'
                recipe = json.loads(path.read_text())
                expected_score = digest(score_path) if arm == 'min_nll' else None
                if recipe['data_sha256'] != digest(data_path) or recipe['score_sha256'] != expected_score:
                    raise ValueError('Frozen recipe provenance mismatch')
                indices = recipe['indices']
                if len(indices) != 768:
                    raise ValueError('Presentation budget changed')
                counts = Counter()
                for index in indices:
                    message = data[index]['messages'][-1]
                    if message['role'] != 'assistant':
                        raise ValueError('Missing target action')
                    action = json.loads(message['content'])['action']
                    family = 'go' if action.startswith('go to ') else action.split()[0]
                    if family not in FAMILIES:
                        raise ValueError('Unrecognized target action family')
                    counts[family] += 1
                row = dict(source=source, policy=arm, sampling_seed=seed, presentations=len(indices),
                           input_tokens=sum(scores[i]['input_tokens'] for i in indices),
                           target_tokens=sum(scores[i]['target_tokens'] for i in indices),
                           presented_mean_base_nll=sum(scores[i]['mean_target_nll'] for i in indices)/len(indices),
                           **{family:counts[family] for family in FAMILIES})
                rows.append(row)
                provenance[str(path)] = digest(path)
    OUT.mkdir(parents=True)
    with (OUT / 'action-presentations.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    active = [family for family in FAMILIES if any(r[family] for r in rows)]
    summary = {}
    for ax, source in zip(axes, ('api', 'generic')):
        summary[source] = {}
        for offset, arm, label in ((-.18, 'random', 'Task-matched random'), (.18, 'min_nll', 'Low base-model NLL')):
            group = [r for r in rows if r['source'] == source and r['policy'] == arm]
            values = np.array([[100*r[f]/r['presentations'] for f in active] for r in group])
            means = values.mean(axis=0)
            ax.bar(np.arange(len(active))+offset, means, width=.36, label=label)
            ax.errorbar(np.arange(len(active))+offset, means,
                        yerr=np.array([means-values.min(axis=0), values.max(axis=0)-means]),
                        fmt='none', ecolor='black', capsize=2, linewidth=.8)
            summary[source][arm] = dict(mean_action_share_pct=dict(zip(active, means.tolist())),
                                        placement_share_pct=float(means[active.index('move')]),
                                        mean_input_tokens=float(np.mean([r['input_tokens'] for r in group])),
                                        mean_target_tokens=float(np.mean([r['target_tokens'] for r in group])))
        ax.set(title=source.upper(), ylabel='Share of 768 target-action presentations (%)',
               xticks=np.arange(len(active)), xticklabels=active, ylim=(0, 60))
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Exploratory action-mix audit of frozen Stage 3 schedules', fontsize=12)
    fig.text(.5, .015, 'Bars: four schedule-seed means; whiskers: min–max (not confidence intervals). No model outcomes used.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .95))
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / ('action-mix.'+ext), dpi=180)
    plt.close(fig)
    result = dict(exploratory=True, defined_after_first_paired_result=True,
                  freeze_sha256=digest(ROOT/'freeze.json'), schedules=16, rows=rows,
                  provenance=provenance, summary=summary, model_calls=0, gpu_used=False,
                  interpretation='Task matching does not imply action-family matching; this audit is not a causal mediation test')
    (OUT/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Stage 3动作构成探索性审计', '',
             '在看到首组中途结果之后定义，只分析既定16份调度，不修改训练、不选择评测结果。', '',
             '| 来源 | 随机选择move呈现占比 | 低NLL选择move呈现占比 |', '|---|---:|---:|']
    for source, s in summary.items():
        lines.append(f"| {source} | {s['random']['placement_share_pct']:.2f}% | {s['min_nll']['placement_share_pct']:.2f}% |")
    lines += ['', 'move是规范化放置动作。两策略匹配任务ID、次数和顺序，但动作类型构成仍会变化。较少放置示范是值得后续干预验证的解释，当前不能据此断言它导致表现差异；动作多样性、上下文与选择难度仍会同时变化。',
              '', '每种来源每种策略汇总4个调度种子。图中误差线为最小值至最大值，非置信区间。目标/输入token量单列于CSV与JSON，不宣称精确FLOP匹配。',
              '', '![动作构成](action-mix.png)', '']
    (OUT/'RESULTS.zh-CN.md').write_text('\n'.join(lines))
    print(json.dumps(dict(directory=str(OUT), summary=summary)), flush=True)


if __name__ == '__main__':
    main()
