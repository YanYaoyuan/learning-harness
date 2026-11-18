"""NLL range controls with the original task/action/content budget held fixed."""

from collections import defaultdict
import math
from statistics import median

from .action_matched_sampling import action_matched_schedule


def range_schedule(tasks, actions, content, reference, seed, policy, scores):
    if policy not in ("random", "low", "middle", "high"):
        raise ValueError("Unknown NLL range policy")
    if (
        len(scores) != len(tasks)
        or not scores
        or not all(math.isfinite(v) and v >= 0 for v in scores)
    ):
        raise ValueError("Finite nonnegative aligned NLL scores required")
    if policy in ("random", "low"):
        return action_matched_schedule(
            tasks,
            actions,
            content,
            reference,
            seed,
            "random" if policy == "random" else "min_nll",
            scores,
        )
    copies = defaultdict(list)
    for i, key in enumerate(zip(tasks, actions, content)):
        copies[key].append(i)
    means = {
        k: sum(scores[i] for i in indices) / len(indices)
        for k, indices in copies.items()
    }
    per_stratum = defaultdict(list)
    for (task, action, _), value in means.items():
        per_stratum[(task, action)].append(value)
    centers = {k: median(v) for k, v in per_stratum.items()}
    transformed = [None] * len(scores)
    for (task, action, key), indices in copies.items():
        value = means[(task, action, key)]
        rank = -value if policy == "high" else abs(value - centers[(task, action)])
        for i in indices:
            transformed[i] = rank
    return action_matched_schedule(
        tasks, actions, content, reference, seed, "min_nll", transformed
    )
