"""Content interventions conditional on an existing task/action/support schedule."""
from collections import Counter, defaultdict
import json
import math
import random


def action_matched_schedule(task_ids, action_families, content_keys, reference,
                            sampling_seed, selection, nll=None):
    """Preserve every reference stratum position and its distinct-content quota.

    Strata absent from the reference stay absent. This controls action coverage;
    it does not promise to restore all action families from the eligible pool.
    """
    size = len(task_ids)
    if not size or len(action_families) != size or len(content_keys) != size:
        raise ValueError('Aligned nonempty task/action/content arrays required')
    if selection not in ('random', 'min_nll'):
        raise ValueError('Unknown selection policy')
    if nll is not None and (len(nll) != size or not all(math.isfinite(v) for v in nll)):
        raise ValueError('Finite aligned NLL scores required')
    if selection == 'min_nll' and nll is None:
        raise ValueError('NLL scores required for min_nll')
    if not reference or any(type(i) is not int or not 0 <= i < size for i in reference):
        raise ValueError('Valid reference indices required')
    strata = list(zip(task_ids, action_families))
    copies, content_family = defaultdict(list), {}
    for index, (task, family) in enumerate(strata):
        content = (task, content_keys[index])
        if content in content_family and content_family[content] != family:
            raise ValueError('Same task/content assigned conflicting action families')
        content_family[content] = family
        copies[(task, family, content_keys[index])].append(index)
    candidates, scores = defaultdict(list), {}
    for (task, family, _), indices in copies.items():
        representative = indices[0]
        candidates[(task, family)].append(representative)
        if nll is not None:
            scores[representative] = sum(nll[i] for i in indices) / len(indices)
    counts = Counter(strata[i] for i in reference)
    quota = {s:len({content_keys[i] for i in reference if strata[i] == s}) for s in counts}
    sequences = {}
    for stratum in sorted(counts):
        scope = json.dumps(stratum, ensure_ascii=False)
        if selection == 'random':
            rng = random.Random(f'task-action-content-v1:{sampling_seed}:{scope}')
            chosen = rng.sample(candidates[stratum], quota[stratum])
        else:
            chosen = sorted(candidates[stratum], key=lambda i:(scores[i], i))[:quota[stratum]]
        rng = random.Random(f'task-action-order-v1:{sampling_seed}:{scope}')
        values = []
        while len(values) < counts[stratum]:
            order = list(chosen)
            rng.shuffle(order)
            values.extend(order[:counts[stratum] - len(values)])
        sequences[stratum] = iter(values)
    result = [next(sequences[strata[i]]) for i in reference]
    actual_quota = {s:len({content_keys[i] for i in result if strata[i] == s}) for s in counts}
    if [strata[i] for i in result] != [strata[i] for i in reference] or actual_quota != quota:
        raise ValueError('Task/action order or distinct-content quota changed')
    return result
