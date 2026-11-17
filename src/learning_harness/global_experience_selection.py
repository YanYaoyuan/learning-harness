"""Global comparators with the completed gradient selector's support and replay."""

from math import isfinite
from random import Random

from .experience_value_study import arrays


POLICIES = ("global_random", "global_high")


def global_schedule(rows, seed, policy, scores=None):
    if policy not in (*POLICIES, "gradient_global"):
        raise ValueError("Unknown global selection rule")
    tasks, _, content = arrays(rows)
    first = {}
    for index, key in enumerate(content):
        first.setdefault(key, index)
    representatives = list(first.values())
    if len(representatives) < 256:
        raise ValueError("At least 256 distinct contents are required")
    if policy == "global_random":
        chosen = Random(seed).sample(representatives, 256)
    else:
        if (
            scores is None
            or len(scores) != len(rows)
            or not all(isfinite(s) for s in scores)
        ):
            raise ValueError("Finite aligned scores are required")
        chosen = sorted(representatives, key=lambda i: (-scores[i], i))[:256]
    rng, result = Random(seed), []
    for _ in range(3):
        order = list(range(256))
        rng.shuffle(order)
        result.extend(chosen[i] for i in order)
    if len(result) != 768 or len({(tasks[i], content[i]) for i in result}) != 256:
        raise ValueError("Global replay support or presentation budget changed")
    return result


def matrix(
    initializations=(101, 202, 303, 404, 505, 606, 707, 808), sampling=(101, 202)
):
    cases = []
    block = 0
    for initialization in initializations:
        for sample in sampling:
            block += 1
            for policy in POLICIES:
                for source in ("api", "generic"):
                    cases.append(
                        dict(
                            name=f"b{block:02d}-{policy}-{source}-i{initialization}-s{sample}",
                            block=block,
                            policy=policy,
                            source=source,
                            initialization_seed=initialization,
                            sampling_seed=sample,
                            gpu=0,
                        )
                    )
    return cases
