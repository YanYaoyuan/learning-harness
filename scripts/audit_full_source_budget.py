#!/usr/bin/env python3
"""Audit frozen API/ReAct pools and simulated exposure, with no learner outcomes."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from train_lora import encode_example, fixed_schedule

ROOT = Path('runs/source-comparison-v1')
OUT = Path('output/source-comparison-v1/supplementary/full-budget-preparation')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise ValueError('Preserve the existing audit before another run')
    import run_api_harness as engine
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(engine.MODEL, local_files_only=True)
    manifest = json.loads((ROOT/'training-inputs.sha256.json').read_text())
    pools, exposure, provenance = {}, [], {}
    historical_checks = 0
    for source in ('api', 'react'):
        pools[source] = {}
        for corpus in ('eligible', 'matched'):
            path = ROOT/'corpora'/f'{corpus}-{source}.jsonl'
            if manifest[str(path)] != sha(path):
                raise ValueError('Frozen training corpus changed')
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            content, encoded, families = [], [], []
            for row in rows:
                if row['source'] != source or row['partition'] != 'train':
                    raise ValueError('Wrong source or nontraining row')
                value = encode_example(tokenizer, row, 2048)
                if value is None:
                    raise ValueError('Previously eligible example became overlength')
                encoded.append(value)
                content.append(hashlib.sha256(json.dumps(row['messages'], sort_keys=True, ensure_ascii=False).encode()).hexdigest())
                action = json.loads(row['messages'][-1]['content'])['action']
                families.append('go' if action.startswith('go to ') else action.split()[0])
            pools[source][corpus] = dict(rows=len(rows), distinct_student_content=len(set(content)),
                                         task_ids=len({r['task_id'] for r in rows}),
                                         task_groups=len({r['task_group'] for r in rows}),
                                         task_seed_pairs=len({(r['task_id'], r['rollout_seed']) for r in rows}),
                                         action_counts=dict(Counter(families)), data_sha256=sha(path))
            for budget in (256, 1024):
                for seed in (101, 202, 303, 404):
                    indices = fixed_schedule(len(rows), budget, seed)
                    sequence_tokens = sum(len(encoded[i]['input_ids']) for i in indices)
                    target_tokens = sum(sum(v != -100 for v in encoded[i]['labels']) for i in indices)
                    if corpus == 'matched' and budget == 256 and seed in (101, 202, 303):
                        meta_path = ROOT/'a-checkpoints'/f'{source}-seed{seed}'/'experiment.json'
                        metadata = json.loads(meta_path.read_text())
                        if (metadata['schedule_sha256'] != hashlib.sha256(json.dumps(indices).encode()).hexdigest()
                                or metadata['input_tokens_rank0'] != sequence_tokens
                                or metadata['target_tokens_rank0'] != target_tokens):
                            raise ValueError('Historical tokenizer/schedule accounting mismatch')
                        historical_checks += 1
                        provenance[str(meta_path)] = sha(meta_path)
                    exposure.append(dict(source=source, corpus=corpus, presentations=budget, seed=seed,
                                         unique_rows=len(set(indices)), distinct_content=len({content[i] for i in indices}),
                                         task_ids=len({rows[i]['task_id'] for i in indices}),
                                         task_seed_pairs=len({(rows[i]['task_id'], rows[i]['rollout_seed']) for i in indices}),
                                         full_pool_seen=len(set(indices)) == len(rows),
                                         total_sequence_tokens=sequence_tokens, target_tokens=target_tokens,
                                         nominal_updates_at_batch4=budget//4))
            provenance[str(path)] = sha(path)
    if historical_checks != 6 or not all(r['full_pool_seen'] for r in exposure if r['presentations'] == 1024):
        raise ValueError('Historical or complete-pass checks incomplete')
    OUT.mkdir(parents=True)
    with (OUT/'simulated-exposure.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exposure[0]))
        writer.writeheader()
        writer.writerows(exposure)
    result = dict(descriptive=True, learner_outcomes_read=False, simulated_exposure_only=True,
                  historical_token_checks=historical_checks, pools=pools, exposure=exposure,
                  provenance=provenance, model_calls=0, gpu_used=False, official_tests_used=False,
                  tokenizer_model=engine.MODEL, tokenizer_revision=engine.REV,
                  eligibility='Valid canonical action targets from successful training episodes only; unsuccessful episodes excluded',
                  future_protocol_frozen=False)
    (OUT/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    for source, color in (('api', '#2878b5'), ('react', '#e58b2a')):
        for corpus, style in (('eligible', 'o-'), ('matched', 's--')):
            for ax, field, label in zip(axes, ('distinct_content', 'task_ids'), ('Distinct student messages seen', 'Training task IDs seen')):
                values = np.array([[r[field] for r in exposure if r['source'] == source and r['corpus'] == corpus and r['presentations'] == b]
                                   for b in (256, 1024)])
                means = values.mean(axis=1)
                ax.errorbar([256, 1024], means, yerr=[means-values.min(axis=1), values.max(axis=1)-means],
                            fmt=style, color=color, capsize=3, label=source.upper()+' / '+corpus)
                ax.set(xlabel='Simulated sample presentations', ylabel=label, xticks=[256, 1024])
                ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('API vs ReAct: training-budget preparation, not learner performance', fontsize=12)
    fig.text(.5, .015, 'Four shuffle seeds per point; whiskers: min–max, not confidence intervals. All 1024 schedules include every eligible row.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .95))
    for ext in ('png', 'pdf'):
        fig.savefig(OUT/('budget-and-coverage.'+ext), dpi=180)
    plt.close(fig)
    print(json.dumps(dict(directory=str(OUT), pools=pools, historical_token_checks=historical_checks)), flush=True)


if __name__ == '__main__':
    main()
