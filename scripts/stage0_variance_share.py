#!/usr/bin/env python3
"""Stage 0 of the variance diagnostic: evaluation-repeat share of between-seed variance.

Specification: docs/VARIANCE_DIAGNOSTIC_STAGE0_SPEC.md. This script reads completed
evaluation directories and writes one JSON result. It never launches a model.

Estimator. For training seed s and evaluation replicate r, the paired difference is

    d[s, r] = mean over families of ( mean over the family's query tasks of
              success(API, s, r) - success(generic, s, r) )

with the two families weighted equally. With S seeds and R replicates:

    MSW = mean over s of the sample variance of d[s, :]        (S * (R - 1) df)
    MSB = R * sample variance over s of mean_r d[s, r]         (S - 1 df)

    replicate variance      = MSW
    seed variance           = max(0, (MSB - MSW) / R)
    single-evaluation total = replicate variance + seed variance
    share                   = replicate variance / single-evaluation total

The share is the fraction of the between-seed variance of a single evaluation's paired
difference that evaluation repeats alone would produce. It conditions on the fixed query
tasks. If the total is below TOTAL_FLOOR the share is undefined and the default branch
(Stage 1 in full) applies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FAMILIES = ["cool", "heat"]
SOURCES = ["api", "generic"]
B_SEEDS = [1101, 1202, 1303]
HISTORICAL_ROOT = Path("runs/transfer-factorial-v1")
STAGE0_ROOT = Path("runs/variance-diagnostic-v1/stage0")
TASKS_PER_FAMILY = 32
TOTAL_FLOOR = 1e-12
EVALUATION_DOMINATES = 0.50
MINIMUM_NEW_REPLICATES = 2


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def historical_directory(family, source, seed) -> Path:
    return HISTORICAL_ROOT / family / "evaluation" / "main" / f"full-{source}-seed{seed}" / "B" / "step-0000"


def replicate_directory(replicate, family, source, seed) -> Path:
    return STAGE0_ROOT / f"rep{replicate}" / family / f"full-{source}-seed{seed}"


def load_cell(directory: Path) -> dict:
    """Return {task_id: success} for one completed evaluation, or raise."""
    summary = json.loads((directory / "summary.json").read_text())
    rows = read_jsonl(directory / "episodes.jsonl")
    if not summary["complete"] or len(rows) != TASKS_PER_FAMILY:
        raise ValueError(f"Incomplete evaluation: {directory}")
    if any(row["status"] != "completed" or row["arm"] != "none" for row in rows):
        raise ValueError(f"Errored or assisted episode in {directory}")
    cell = {row["task_id"]: int(row["success"]) for row in rows}
    if len(cell) != len(rows):
        raise ValueError(f"Duplicate task in {directory}")
    return cell


def replicate_complete(replicate) -> bool:
    return all((replicate_directory(replicate, f, k, s) / "summary.json").exists()
               and json.loads((replicate_directory(replicate, f, k, s) / "summary.json").read_text())["complete"]
               for f in FAMILIES for k in SOURCES for s in B_SEEDS)


def compatibility(replicate) -> list:
    """Checks that make the historical evaluation exchangeable with a new replicate.

    Returns a list of failure descriptions; empty means compatible. None of these checks
    looks at task outcomes.
    """
    failures = []
    for family in FAMILIES:
        for source in SOURCES:
            for seed in B_SEEDS:
                old, new = historical_directory(family, source, seed), replicate_directory(replicate, family, source, seed)
                label = f"{family}/full-{source}-seed{seed}"
                old_run, new_run = (json.loads((d / "run.json").read_text()) for d in (old, new))
                for field in ("revision", "batch_protocol", "environment_backend", "planned_episodes"):
                    if old_run.get(field) != new_run.get(field):
                        failures.append(f"{label}: run.json {field} differs")
                old_config, new_config = dict(old_run["config"]), dict(new_run["config"])
                old_manifest, new_manifest = old_config.pop("manifest"), new_config.pop("manifest")
                if old_config != new_config:
                    failures.append(f"{label}: evaluation config differs beyond the manifest path")
                old_tasks = sorted(json.dumps(r, sort_keys=True) for r in read_jsonl(old_manifest))
                new_tasks = sorted(json.dumps(r, sort_keys=True) for r in read_jsonl(new_manifest))
                if old_tasks != new_tasks:
                    failures.append(f"{label}: manifests do not contain the same task records")
                old_summary = json.loads((old / "summary.json").read_text())
                if not old_summary["complete"] or old_summary["metrics"]["none"]["error_count"]:
                    failures.append(f"{label}: historical evaluation incomplete or errored")
    return failures


def paired_differences(cells_by_replicate: list, families=FAMILIES) -> np.ndarray:
    """Array d[s, r]; families weighted equally, tasks weighted equally within a family."""
    out = np.zeros((len(B_SEEDS), len(cells_by_replicate)))
    for r, cells in enumerate(cells_by_replicate):
        for s, seed in enumerate(B_SEEDS):
            per_family = []
            for family in families:
                api, generic = cells[family, "api", seed], cells[family, "generic", seed]
                if set(api) != set(generic):
                    raise ValueError("Sources were evaluated on different tasks")
                per_family.append(np.mean([api[t] - generic[t] for t in sorted(api)]))
            out[s, r] = np.mean(per_family)
    return out


def f_quantile(p, df1, df2, draws=2_000_000) -> float:
    """Seeded Monte Carlo F quantile; avoids a SciPy dependency."""
    rng = np.random.default_rng(20261111)
    ratio = (rng.chisquare(df1, draws) / df1) / (rng.chisquare(df2, draws) / df2)
    return float(np.quantile(ratio, p))


def variance_share(d: np.ndarray, level=0.90) -> dict:
    """Evaluation-repeat share from d[s, r]; see the module docstring."""
    n_seeds, n_reps = d.shape
    if n_seeds < 2 or n_reps < 2:
        raise ValueError("Need at least two seeds and two replicates")
    msw = float(d.var(axis=1, ddof=1).mean())
    msb = float(n_reps * d.mean(axis=1).var(ddof=1))
    seed_variance = max(0.0, (msb - msw) / n_reps)
    total = msw + seed_variance
    result = dict(seeds=n_seeds, replicates=n_reps, msw=msw, msb=msb,
                  df_within=n_seeds * (n_reps - 1), df_between=n_seeds - 1,
                  replicate_variance=msw, seed_variance=seed_variance, single_evaluation_total=total,
                  replicate_sd_pp=100 * msw ** 0.5, seed_sd_pp=100 * seed_variance ** 0.5,
                  seed_variance_truncated_at_zero=msb < msw, share=None, share_interval=None,
                  interval_level=level, interval_method="one-way random-effects F interval; normal-theory, approximate")
    if total < TOTAL_FLOOR:
        result["note"] = "No variance to apportion; share undefined."
        return result
    result["share"] = msw / total
    if msw == 0.0:
        result["share_interval"] = [0.0, 0.0]
        result["note"] = "Replicates identical within every seed; interval degenerate."
        return result
    df1, df2, tail = n_seeds - 1, n_seeds * (n_reps - 1), (1 - level) / 2
    f_obs = msb / msw
    f_low = f_obs / f_quantile(1 - tail, df1, df2)
    f_high = f_obs * f_quantile(1 - tail, df2, df1)
    icc_low = (f_low - 1) / (f_low + n_reps - 1)
    icc_high = (f_high - 1) / (f_high + n_reps - 1)
    result["share_interval"] = [float(np.clip(1 - icc_high, 0, 1)), float(np.clip(1 - icc_low, 0, 1))]
    return result


def flip_rate(cells_by_replicate: list) -> float:
    """Share of (family, source, seed, task) cells whose success differs between replicates."""
    flips = total = 0
    for key in cells_by_replicate[0]:
        for task in cells_by_replicate[0][key]:
            flips += len({cells[key][task] for cells in cells_by_replicate}) > 1
            total += 1
    return flips / total


def decide(share, new_replicates) -> str:
    if new_replicates < MINIMUM_NEW_REPLICATES:
        return "inconclusive: fewer than two complete new replicates; Stage 1 in full with evaluation repeats added"
    if share is None:
        return "share undefined: Stage 1 in full"
    if share >= EVALUATION_DOMINATES:
        return "evaluation dominates: Stage 1 at reduced size (2 x 2 seeds); plan repeated evaluation"
    return "Stage 1 in full"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--include-historical", action="store_true",
                        help="Use the historical evaluation as an additional replicate. Decided at freeze "
                             "time; refused if the compatibility checks fail.")
    parser.add_argument("--out", default=str(STAGE0_ROOT / "result.json"))
    args = parser.parse_args()

    new = [r for r in (1, 2, 3) if replicate_complete(r)]
    if not new:
        raise SystemExit("No complete new replicate")
    keys = [(f, k, s) for f in FAMILIES for k in SOURCES for s in B_SEEDS]
    replicates = [{key: load_cell(replicate_directory(r, *key)) for key in keys} for r in new]
    labels = [f"rep{r}" for r in new]

    failures = [failure for r in new for failure in compatibility(r)]
    historical = {key: load_cell(historical_directory(*key)) for key in keys}
    for cells in replicates:
        for key in keys:
            if set(cells[key]) != set(historical[key]):
                raise ValueError(f"Task set differs from the historical evaluation: {key}")
    if args.include_historical:
        if failures:
            raise SystemExit("Historical evaluation is not compatible:\n  " + "\n  ".join(failures))
        primary_cells, primary_labels = [historical] + replicates, ["historical"] + labels
    else:
        primary_cells, primary_labels = replicates, labels

    primary = variance_share(paired_differences(primary_cells)) if len(primary_cells) >= 2 else None
    report = dict(
        specification="docs/VARIANCE_DIAGNOSTIC_STAGE0_SPEC.md",
        exploratory=True,
        complete_new_replicates=new,
        primary_replicates=primary_labels,
        historical_compatibility_failures=failures,
        paired_difference_by_seed_and_replicate=(
            paired_differences(primary_cells).tolist() if primary else None),
        primary=primary,
        decision=decide(primary["share"] if primary else None, len(new)),
        decision_is_a_spending_rule_not_a_finding=True,
        secondary=dict(
            per_family={f: variance_share(paired_differences(primary_cells, [f])) for f in FAMILIES}
            if primary else None,
            new_replicates_only=variance_share(paired_differences(replicates)) if len(replicates) >= 2 else None,
            with_historical_regardless_of_checks=variance_share(paired_differences([historical] + replicates)),
            task_outcome_flip_rate=flip_rate(primary_cells) if primary else None,
        ),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(primary=primary, decision=report["decision"]), indent=2))


if __name__ == "__main__":
    main()
