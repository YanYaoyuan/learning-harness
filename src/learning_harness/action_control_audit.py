"""Audit action-matched content interventions before a worker may train."""
from collections import Counter
import hashlib
import json

from .action_matched_sampling import action_matched_schedule


def digest(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def audit_recipe(recipe, rows, scores, reference):
    """Regenerate selection from frozen training inputs, then verify all controls."""
    if recipe['policy'] not in ('random', 'min_nll'):
        raise ValueError('Unknown content policy')
    if any(row['partition'] != 'train' for row in rows):
        raise ValueError('Only training examples may enter the intervention')
    if recipe['sampling_seed'] != reference['sampling_seed'] or recipe['source'] != reference['source']:
        raise ValueError('Reference source or seed differs')
    if reference['policy'] != 'random':
        raise ValueError('Reference must be the already-frozen random control')
    tasks, families, content = [], [], []
    for row in rows:
        final = row['messages'][-1]
        if final['role'] != 'assistant':
            raise ValueError('Missing final action target')
        action = json.loads(final['content'])['action']
        if not isinstance(action, str) or not action.strip():
            raise ValueError('Missing canonical action command')
        tasks.append(row['task_id'])
        families.append('go' if action.startswith('go to ') else action.split()[0])
        content.append(hashlib.sha256(json.dumps(row['messages'], sort_keys=True, ensure_ascii=False).encode()).hexdigest())
    expected = action_matched_schedule(tasks, families, content, reference['indices'],
                                       recipe['sampling_seed'], recipe['policy'], scores)
    indices = recipe['indices']
    if indices != expected or recipe['schedule_sha256'] != digest(expected):
        raise ValueError('Stored schedule differs from regenerated intervention')
    strata = list(zip(tasks, families))
    order = [strata[i] for i in indices]
    reference_order = [strata[i] for i in reference['indices']]
    if (order != reference_order or recipe['task_action_order_sha256'] != digest(order)
            or recipe['reference_task_action_order_sha256'] != digest(reference_order)):
        raise ValueError('Task/action order mismatch')
    counts = Counter(order)
    actual = [dict(task_id=t, action_family=a, presentations=counts[(t,a)],
                   distinct_support=len({content[i] for i in indices if strata[i] == (t,a)}))
              for t,a in sorted(counts)]
    if actual != recipe['strata']:
        raise ValueError('Per-task/action support record differs')
    distinct = len({(tasks[i],content[i]) for i in indices})
    if (len(indices) != recipe['presentations'] or distinct != recipe['unique_task_content_examples']
            or distinct != len({(tasks[i],content[i]) for i in reference['indices']})):
        raise ValueError('Presentation or distinct-content budget mismatch')
    return dict(processed_examples=len(indices), unique_examples=len(set(indices)),
                unique_task_content_examples=distinct, distinct_content_examples=len({content[i] for i in indices}),
                unique_tasks=len(set(tasks[i] for i in indices)),
                task_exposures=dict(Counter(tasks[i] for i in indices)),
                task_order_sha256=digest([tasks[i] for i in indices]),
                task_action_order_sha256=digest(order), strata=actual,
                exact_reference_task_action_positions=True, exact_reference_stratum_support=True,
                regenerated_schedule_verified=True)
