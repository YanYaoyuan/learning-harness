#!/usr/bin/env python3
"""CPU-only feasibility audit and candidate recipes; never start training."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from learning_harness.action_matched_sampling import action_matched_schedule
from learning_harness.task_matched_sampling import allocate_support
from train_lora import fixed_schedule

ROOT = Path('runs/variance-diagnostic-v1/stage3')
OUT = Path('output/variance-diagnostic-v1/task-action-feasibility')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def main():
    from run_stage3_task_controls import load_freeze
    load_freeze()
    if OUT.exists():
        raise ValueError('Existing candidate preparation must be preserved')
    plans, results = [], []
    provenance = {str(ROOT/'freeze.json'):sha(ROOT/'freeze.json'),
                  __file__:sha(__file__),
                  'src/learning_harness/action_matched_sampling.py':sha('src/learning_harness/action_matched_sampling.py')}
    for source in ('api', 'generic'):
        data_path = Path(f'runs/transfer-factorial-v1/corpora/full-{source}.jsonl')
        data = [json.loads(line) for line in data_path.read_text().splitlines()]
        score_path = ROOT/'base-nll'/f'{source}.json'
        scores_record = json.loads(score_path.read_text())
        if scores_record['data_sha256'] != sha(data_path) or len(scores_record['scores']) != len(data):
            raise ValueError('NLL/corpus provenance mismatch')
        scores, tasks, actions, content = [], [], [], []
        for i, (row, score) in enumerate(zip(data, scores_record['scores'])):
            if score['index'] != i or score['task_id'] != row['task_id']:
                raise ValueError('Score alignment changed')
            tasks.append(row['task_id'])
            action = json.loads(row['messages'][-1]['content'])['action']
            actions.append('go' if action.startswith('go to ') else action.split()[0])
            content.append(hashlib.sha256(json.dumps(row['messages'], sort_keys=True, ensure_ascii=False).encode()).hexdigest())
            scores.append(score['mean_target_nll'])
        strata = list(zip(tasks, actions))
        full_families = defaultdict(set)
        for task, action in strata:
            full_families[task].add(action)
        provenance[str(data_path)] = sha(data_path)
        provenance[str(score_path)] = sha(score_path)
        for seed in (101, 202, 303, 404):
            reference_path = ROOT/'inputs/schedules'/f'random-{source}-s{seed}.json'
            r = json.loads(reference_path.read_text())
            if r['data_sha256'] != sha(data_path) or r['schedule_sha256'] != digest(r['indices']):
                raise ValueError('Reference recipe provenance changed')
            reference = r['indices']
            full = fixed_schedule(len(data), 768, seed)
            if r['reference_schedule_sha256'] != digest(full):
                raise ValueError('Full reference schedule mismatch')
            quota = Counter(tasks[i] for i in set(reference))
            infeasible = [dict(task_id=t, distinct_quota=quota[t], full_pool_action_families=len(f))
                          for t, f in sorted(full_families.items()) if quota[t] < len(f)]
            capacity = Counter(tasks[i] for i in range(len(data)))
            minimum_support_same_proportions = next(
                n for n in range(256, len(data)+1)
                if all(allocate_support({t:len({content[i] for i in range(len(data)) if tasks[i] == t}) for t in capacity}, n)[t]
                       >= len(full_families[t]) for t in full_families))
            groups = sorted(set(strata[i] for i in reference))
            group_quota = {g:len({content[i] for i in reference if strata[i] == g}) for g in groups}
            group_counts = Counter(strata[i] for i in reference)
            report = dict(source=source, sampling_seed=seed, task_count=len(quota),
                          unchanged_task_quota_full_action_coverage_feasible=not infeasible,
                          infeasible_full_coverage_tasks=infeasible,
                          proportional_support_needed_for_every_full_pool_action_family=minimum_support_same_proportions,
                          matched_reference_action_groups=len(groups),
                          eligible_action_groups=len(set(strata)),
                          reference_distinct_support=sum(group_quota.values()), candidates=[])
            for policy in ('random', 'min_nll'):
                indices = action_matched_schedule(tasks, actions, content, reference, seed, policy, scores)
                if len(indices) != 768 or len({(tasks[i], content[i]) for i in indices}) != 256:
                    raise ValueError('Candidate budget changed')
                if [strata[i] for i in indices] != [strata[i] for i in reference]:
                    raise ValueError('Candidate task/action positions differ')
                actual_quota = {g:len({content[i] for i in indices if strata[i] == g}) for g in groups}
                if actual_quota != group_quota:
                    raise ValueError('Candidate per-task/action distinct support changed')
                name = f'{policy}-{source}-s{seed}.json'
                recipe = dict(candidate_only=True, training_started=False, policy=policy, source=source,
                              sampling_seed=seed, data=str(data_path), data_sha256=sha(data_path),
                              score_sha256=sha(score_path) if policy == 'min_nll' else None,
                              reference_recipe=str(reference_path), reference_recipe_sha256=sha(reference_path),
                              indices=indices, schedule_sha256=digest(indices), presentations=768,
                              unique_task_content_examples=256,
                              reference_task_action_order_sha256=digest([strata[i] for i in reference]),
                              task_action_order_sha256=digest([strata[i] for i in indices]),
                              strata=[dict(task_id=t, action_family=a, presentations=group_counts[(t,a)],
                                           distinct_support=group_quota[(t,a)]) for t,a in groups],
                              note='Condition on frozen random-reference action coverage; absent action groups stay absent')
                plans.append((name, recipe))
                report['candidates'].append(dict(policy=policy, schedule_file=name, exact_controls_passed=True))
            results.append(report)
            provenance[str(reference_path)] = sha(reference_path)
    OUT.mkdir(parents=True)
    (OUT/'candidate-schedules').mkdir()
    for name, value in plans:
        (OUT/'candidate-schedules'/name).write_text(json.dumps(value, indent=2)+'\n')
    summary = dict(status='candidate_controls_feasible', candidate_schedules=len(plans),
                   official_tests_used=False, model_calls=0, gpu_used=False, formal_protocol_frozen=False,
                   stage3_unchanged=True, results=results, provenance=provenance,
                   note='Prepared after exploratory Stage 3 peek; no effect claims, no automatic training launch')
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('results','provenance')}), flush=True)


if __name__ == '__main__':
    main()
