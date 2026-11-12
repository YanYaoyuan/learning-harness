"""Explicit aggregate export: never copy trajectories, task IDs or free-form fields."""

import math
import re

from .controlled_study import POLICIES, SOURCES, paired_effects, validate_plan


def aggregate_report(report, frozen, freeze_sha):
    design = validate_plan(frozen)
    if (
        report["complete"] is not True
        or report["cases"] != len(frozen["cases"])
        or report["actual_evaluation_episodes"] != frozen["planned_episodes"]
        or report["freeze_sha256"] != freeze_sha
        or re.fullmatch(r"[0-9a-f]{64}", freeze_sha) is None
        or report["teacher_calls"] != 0
        or report["official_tests_used"] is not False
    ):
        raise ValueError("Only a complete, matching development report can be exported")
    planned = {c["name"]: c for c in frozen["cases"]}
    rows = report["results"]
    if len(rows) != len(planned) or {r["name"] for r in rows} != set(planned):
        raise ValueError("Report is missing or duplicating planned model results")
    for row in rows:
        case = planned[row["name"]]
        if any(
            row[k] != case[k]
            for k in (
                "source",
                "policy",
                "initialization_seed",
                "sampling_seed",
                "repeat",
                "gpu",
            )
        ):
            raise ValueError("Model result belongs to a different planned condition")
        if (
            not all(
                math.isfinite(row[k])
                for k in ("mean_success", "mean_nll", "cool_success", "heat_success")
            )
            or not all(
                0 <= row[k] <= 1
                for k in ("mean_success", "cool_success", "heat_success")
            )
            or row["mean_nll"] < 0
            or row["mean_success"] != (row["cool_success"] + row["heat_success"]) / 2
        ):
            raise ValueError("Invalid or inconsistent model outcome")
    effects = paired_effects(rows)
    groups, contrasts = {}, {}
    for source in SOURCES:
        groups[source] = {}
        for policy in POLICIES:
            group = [
                r
                for r in rows
                if r["repeat"] == 0 and r["source"] == source and r["policy"] == policy
            ]
            groups[source][policy] = {
                metric: sum(r[metric] for r in group) / len(group)
                for metric in ("mean_success", "mean_nll")
            }
        success = effects["mean_success"]["nll_minus_random"][source]
        nll = effects["mean_nll"]["nll_minus_random"][source]
        contrasts[source] = dict(
            success_difference_pp=100 * success["mean_difference"],
            nll_difference=nll["mean_difference"],
            wins=success["wins"],
            ties=success["ties"],
            losses=success["losses"],
        )
    return dict(
        complete=True,
        freeze_sha256=freeze_sha,
        models=len(rows),
        evaluation_episodes=frozen["planned_episodes"],
        primary_seed_cells=len(design["primary_seed_cells"]),
        initialization_levels=len({i for i, _ in design["primary_seed_cells"]}),
        sampling_levels=len({s for _, s in design["primary_seed_cells"]}),
        conditional_repeated_models=design["repeated_models"],
        groups=groups,
        min_nll_minus_random=contrasts,
        teacher_calls=0,
        official_tests_used=False,
        raw_records_uploaded=False,
        aggregation="Equal weights over primary paired cells; conditional repeats excluded from means.",
        uncertainty="Descriptive fixed-development-task results; no calibrated confidence intervals.",
    )
