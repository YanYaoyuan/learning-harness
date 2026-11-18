"""Frozen finite-panel replication contracts; no population-significance claims."""

from collections import Counter
from statistics import mean, stdev

from .experience_value_study import INITIALIZATIONS, SAMPLING, SOURCES

RULES = ("global_random", "global_high", "gradient_global")


def matrix():
    cases = [
        dict(
            name="base",
            policy="base",
            source="base",
            block=0,
            initialization_seed=None,
            sampling_seed=None,
            gpu=0,
        )
    ]
    block = 0
    for seed in INITIALIZATIONS:
        for sample in SAMPLING:
            block += 1
            for rule in RULES:
                for source in SOURCES:
                    cases.append(
                        dict(
                            name=f"b{block:02d}-{rule}-{source}-i{seed}-s{sample}",
                            policy=rule,
                            source=source,
                            block=block,
                            initialization_seed=seed,
                            sampling_seed=sample,
                            gpu=0,
                        )
                    )
    return cases


def validate_queries(rows, development, calibration):
    if len(rows) != 128 or Counter(r["task_type"] for r in rows) != Counter(
        pick_and_place_simple=64, pick_clean_then_place_in_recep=64
    ):
        raise ValueError("All 128 balanced confirmation tasks are required")
    groups = [r["task_group"] for r in rows]
    forbidden = {r["task_group"] for r in development + calibration}
    if (
        len(set(groups)) != 128
        or set(groups) & forbidden
        or len({r["task_id"] for r in rows}) != 128
        or any(
            r["fresh_role"] != "confirmation"
            or r["official_split"] != "train"
            or r["partition"] != "fresh_confirmation"
            for r in rows
        )
    ):
        raise ValueError("Confirmation role/group isolation failed")


def aggregate(records):
    expected = {c["name"]: c for c in matrix()}
    indexed = {r["case"]["name"]: r for r in records}
    if (
        len(records) != 97
        or len(indexed) != 97
        or set(indexed) != set(expected)
        or any(
            r["case"] != expected[n] or r["episodes"] != 128 for n, r in indexed.items()
        )
    ):
        raise ValueError("The full 97-model confirmation matrix is required")
    cells = {
        (
            r["case"]["source"],
            r["case"]["policy"],
            r["case"]["initialization_seed"],
            r["case"]["sampling_seed"],
        ): r
        for n, r in indexed.items()
        if n != "base"
    }
    endpoints = {}
    for source in SOURCES:
        endpoints[source] = {}
        for rule in RULES:
            values = {
                str(seed): mean(
                    cells[source, rule, seed, sample]["mean_success"]
                    for sample in SAMPLING
                )
                for seed in INITIALIZATIONS
            }
            endpoints[source][rule] = dict(
                mean_success=mean(values.values()),
                per_initialization=values,
                initialization_sd=stdev(values.values()),
            )
    contrasts = {}
    for comparator in RULES[:2]:
        per_source = {
            s: {
                str(i): endpoints[s]["gradient_global"]["per_initialization"][str(i)]
                - endpoints[s][comparator]["per_initialization"][str(i)]
                for i in INITIALIZATIONS
            }
            for s in SOURCES
        }
        pooled = {
            str(i): mean(per_source[s][str(i)] for s in SOURCES)
            for i in INITIALIZATIONS
        }
        contrasts[comparator] = dict(
            per_source=per_source,
            per_initialization=pooled,
            mean_difference=mean(pooled.values()),
            initialization_sd=stdev(pooled.values()),
            positive_initializations=sum(v > 0 for v in pooled.values()),
            negative_initializations=sum(v < 0 for v in pooled.values()),
        )
    return dict(
        baseline_success=indexed["base"]["mean_success"],
        endpoints=endpoints,
        contrasts=contrasts,
        inferential_tests_performed=False,
        scope="Prespecified finite-panel replication; SD describes initialization dispersion, not a confidence interval.",
    )
