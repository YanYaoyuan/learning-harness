"""Control exact per-task exposure/order while varying within-task demonstration support."""
from collections import Counter, defaultdict
import math
import random


def allocate_support(capacities, total=256):
    """Capped largest-deficit allocation, at least one example per task."""
    if not capacities or min(capacities.values()) < 1 or not len(capacities) <= total <= sum(capacities.values()):
        raise ValueError("Infeasible support size")
    size=sum(capacities.values())
    ideal={k:total*v/size for k,v in capacities.items()}
    assigned={k:max(1,min(v,math.floor(ideal[k]))) for k,v in capacities.items()}
    while sum(assigned.values())<total:
        key=min((k for k in capacities if assigned[k]<capacities[k]),key=lambda k:(assigned[k]-ideal[k],k))
        assigned[key]+=1
    while sum(assigned.values())>total:
        key=min((k for k in capacities if assigned[k]>1),key=lambda k:(ideal[k]-assigned[k],k))
        assigned[key]-=1
    return assigned


def matched_schedule(task_ids, reference, sampling_seed, selection, nll=None, support_size=256, content_keys=None):
    if selection not in ("random","min_nll"):
        raise ValueError("Unknown within-task selection")
    if selection=="min_nll" and (nll is None or len(nll)!=len(task_ids) or not all(math.isfinite(x) for x in nll)):
        raise ValueError("Finite base-model NLL for every eligible example required")
    if content_keys is not None and len(content_keys)!=len(task_ids):
        raise ValueError('Content-key alignment mismatch')
    groups=defaultdict(list)
    content_groups=defaultdict(list)
    for index,task in enumerate(task_ids):
        key=(task,content_keys[index] if content_keys is not None else index)
        content_groups[key].append(index)
    score_by_rep={}
    for (task,key),copies in content_groups.items():
        representative=copies[0]
        groups[task].append(representative)
        if nll is not None:score_by_rep[representative]=sum(nll[i] for i in copies)/len(copies)
    counts=Counter(task_ids[i] for i in reference)
    if set(counts)!=set(groups):
        raise ValueError("Reference must expose every eligible task")
    support=allocate_support({k:min(len(v),counts[k]) for k,v in groups.items()},support_size)
    sequences={}
    for task in sorted(groups):
        rng=random.Random(f"task-matched-v1:{sampling_seed}:{task}")
        if selection=="random":
            chosen=rng.sample(groups[task],support[task])
        else:
            chosen=sorted(groups[task],key=lambda i:(score_by_rep[i],i))[:support[task]]
        # Reset ordering RNG so policies share the same ordering rule after selection.
        rng=random.Random(f"task-order-v1:{sampling_seed}:{task}")
        values=[]
        while len(values)<counts[task]:
            order=list(chosen)
            rng.shuffle(order)
            values.extend(order[:counts[task]-len(values)])
        sequences[task]=iter(values)
    result=[next(sequences[task_ids[i]]) for i in reference]
    if len(set(result))!=support_size or [task_ids[i] for i in result]!=[task_ids[i] for i in reference]:
        raise ValueError("Task order or support budget mismatch")
    return result
