"""Common public-action view and paired success selection for source comparisons."""
from collections import defaultdict
import json

from .common import digest
from .support import canonicalize_action, messages_for


def target_view(events, tasks, source):
    """Reconstruct prefixes: no actor examples, hints, thoughts, or synthetic OKs."""
    history=[];rows=[];initial=None
    if not events:return rows
    task_id=events[0]['task_id'];task=tasks[task_id]
    if task['partition']!='train' or task['official_split']!='train':
        raise ValueError('Only training games may create SFT labels')
    for e in sorted(events,key=lambda e:e['call']):
        if e['task_id']!=task_id or e['trial']!=1 or e['phase']!='act':
            raise ValueError('Expected a single collection attempt')
        if initial is None:initial=e['messages'][1]['content']
        if e.get('thought'):continue
        response=json.dumps({'action':canonicalize_action(e['action'])},ensure_ascii=False)
        if not e['invalid_format'] and e['observation'].strip()!='Nothing happens.':
            rows.append(dict(task_id=task_id,task_group=task['task_group'],task_type=task['task_type'],
                partition='train',source=source,rollout_seed=e['seed'],step=e['turn'],
                messages=messages_for(initial,history,history_turns=12)+[dict(role='assistant',content=response)]))
        # Failed actions remain public history, but are not training targets.
        history.append(dict(response=response,observation=e['observation']))
    return rows


def match_sources(sources,task_index,minimum_groups=8):
    grouped={s:defaultdict(list) for s in sources}
    for source,rows in sources.items():
        for r in rows:
            if r['partition']!='train':raise ValueError('Non-training row in matching')
            grouped[source][r['task_id'],r['rollout_seed']].append(r)
    common=sorted(set.intersection(*(set(g) for g in grouped.values())))
    group_count=len({task_index[t]['task_group'] for t,seed in common})
    counts=[];matched={s:[] for s in sources}
    for pair in common:
        n=min(len(grouped[s][pair]) for s in sources)
        counts.append(dict(task_id=pair[0],rollout_seed=pair[1],examples_each=n))
        for s in sources:
            rs=sorted(grouped[s][pair],key=lambda r:r['step'])
            indices=[len(rs)-1] if n==1 else [round(i*(len(rs)-1)/(n-1)) for i in range(n)]
            assert len(set(indices))==n
            matched[s].extend(dict(rs[j],matched_index=i) for i,j in enumerate(indices))
    for s in sources:matched[s].sort(key=lambda r:digest([20261111,r['task_id'],r['rollout_seed'],r['matched_index']]))
    keys=lambda rs:[(r['task_id'],r['rollout_seed'],r['matched_index']) for r in rs]
    if len({digest(keys(rs)) for rs in matched.values()})!=1:raise ValueError('Paired sample order differs')
    return matched,dict(common_task_seed_pairs=len(common),common_task_groups=group_count,
        matched_examples_each=len(next(iter(matched.values()))),pair_counts=counts,
        gate_passed=group_count>=minimum_groups and len(next(iter(matched.values())))>=4,
        conditioning='Mutual success on collection task/seed, then equal action-label count per pair; not a full-coverage causal effect')
