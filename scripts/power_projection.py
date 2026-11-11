#!/usr/bin/env python3
"""Planning-only variance decomposition and power projection for paired source comparisons.

Model for the paired difference D[s, t] (source A minus source B, same training seed s,
same query task t), with seeds crossed with tasks and one observation per cell:

    D[s, t] = mu + a_s + b_t + e_st

    a_s  ~ seed component        (variance vs)
    b_t  ~ task component        (variance vt)
    e_st ~ seed x task interaction plus evaluation noise, not separable here (variance ve)

Method-of-moments estimates from the two-way layout without replication:

    vs = (MS_seed - MS_resid) / T      vt = (MS_task - MS_resid) / S      ve = MS_resid

    Var(grand mean) = vs/S + vt/T + ve/(S*T) = (MS_seed + MS_task - MS_resid) / (S*T)

Projections use a Satterthwaite degrees-of-freedom approximation for that linear combination
of mean squares, so small seed counts are penalised through the t distribution instead of
being treated as known-variance normal. Both minimum-detectable-difference columns are
planning approximations, not bounds on realised power.

--simulate N calibrates instead of approximating: it draws N data sets per design from the
planning components (normal components, so the discreteness of success differences is not
reproduced) and reports the rejection rate of two candidate analyses at a true difference of
zero (Type I error) and at --effect-pp (power):

    t     Satterthwaite t-test on the grand mean of the crossed layout
    boot  percentile bootstrap resampling seeds and tasks independently, as in the frozen reports

This is an exploratory planning tool. It reads existing results and writes nothing unless
--out is given. It does not alter any frozen analysis.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np


def load_cells(main_dir: Path) -> dict:
    """Return {case: {(step, task_group): success}} from <case>/B/step-XXXX/episodes.jsonl."""
    cells: dict = collections.defaultdict(dict)
    for path in sorted(main_dir.glob("*/B/step-*/episodes.jsonl")):
        case = path.parts[-4]
        step = int(path.parts[-2].split("-")[1])
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (step, row["task_group"])
            if key in cells[case]:
                raise ValueError(f"Duplicate cell {key} in {path}")
            cells[case][key] = int(row["success"])
    return cells


def success_array(cells, prefix, source, seeds, steps, tasks) -> np.ndarray:
    """Array of shape (seeds, checkpoints, tasks); raises on any missing cell."""
    return np.array(
        [[[cells[f"{prefix}{source}-seed{seed}"][(step, task)] for task in tasks]
          for step in steps] for seed in seeds],
        dtype=float,
    )


def trapezoid_weights(steps) -> np.ndarray:
    """Weights so that weights @ curve equals the normalised trapezoid AUC."""
    weights = np.zeros(len(steps))
    span = steps[-1] - steps[0]
    for i in range(len(steps) - 1):
        weights[i:i + 2] += (steps[i + 1] - steps[i]) / 2 / span
    return weights


def decompose(diff: np.ndarray) -> dict:
    """Two-way random-effects decomposition of a (seeds, tasks) matrix of paired differences."""
    n_seeds, n_tasks = diff.shape
    grand = diff.mean()
    seed_means = diff.mean(axis=1)
    task_means = diff.mean(axis=0)
    ms_seed = n_tasks * ((seed_means - grand) ** 2).sum() / (n_seeds - 1)
    ms_task = n_seeds * ((task_means - grand) ** 2).sum() / (n_tasks - 1)
    resid = diff - seed_means[:, None] - task_means[None, :] + grand
    ms_resid = (resid ** 2).sum() / ((n_seeds - 1) * (n_tasks - 1))
    return dict(
        mean=float(grand),
        seed_means=seed_means.tolist(),
        n_seeds=n_seeds,
        n_tasks=n_tasks,
        ms_seed=float(ms_seed),
        ms_task=float(ms_task),
        ms_resid=float(ms_resid),
        # Raw estimates may be negative; that is sampling error, not a negative variance.
        var_seed_raw=float((ms_seed - ms_resid) / n_tasks),
        var_task_raw=float((ms_task - ms_resid) / n_seeds),
        var_resid=float(ms_resid),
    )


def planning_components(fit: dict, negative: str, seed_multiplier: float, seed_floor: float):
    """Turn raw estimates into planning values.

    negative='truncate': set a negative component to zero and keep the residual as estimated.
        This is optimistic for that component.
    negative='pool': set a negative component to zero and re-estimate the residual by pooling
        that component's sum of squares into it, which is the usual reduced-model refit.
    seed_multiplier and seed_floor then inflate the seed component, because two degrees of
    freedom say very little about it.
    """
    vs, vt, ve = fit["var_seed_raw"], fit["var_task_raw"], fit["var_resid"]
    if negative == "pool":
        s, t = fit["n_seeds"], fit["n_tasks"]
        ss_resid, df_resid = fit["ms_resid"] * (s - 1) * (t - 1), (s - 1) * (t - 1)
        if vs < 0:
            ss_resid, df_resid = ss_resid + fit["ms_seed"] * (s - 1), df_resid + (s - 1)
        if vt < 0:
            ss_resid, df_resid = ss_resid + fit["ms_task"] * (t - 1), df_resid + (t - 1)
        ve = ss_resid / df_resid
        if vs >= 0:
            vs = (fit["ms_seed"] - ve) / t
        if vt >= 0:
            vt = (fit["ms_task"] - ve) / s
    vs, vt = max(vs, 0.0), max(vt, 0.0)
    vs = max(vs * seed_multiplier, seed_floor)
    return vs, vt, ve


def t_cdf(x: float, df: float) -> float:
    """Student t CDF by trapezoid integration of the density; no SciPy dependency."""
    grid = np.linspace(0.0, abs(x), 20001)
    log_norm = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
    density = np.exp(log_norm - (df + 1) / 2 * np.log1p(grid ** 2 / df))
    half = float(((density[1:] + density[:-1]) / 2).sum() * (grid[1] - grid[0])) if x else 0.0
    return 0.5 + math.copysign(half, x)


def t_quantile(p: float, df: float) -> float:
    """Deterministic Student t quantile by bisection on t_cdf."""
    if p < 0.5:
        return -t_quantile(1 - p, df)
    low, high = 0.0, 1.0
    while t_cdf(high, df) < p:
        high *= 2
    for _ in range(60):
        mid = (low + high) / 2
        low, high = (mid, high) if t_cdf(mid, df) < p else (low, mid)
    return (low + high) / 2


def mean_squares(diff: np.ndarray):
    """Vectorised mean squares for an array of shape (..., seeds, tasks)."""
    n_seeds, n_tasks = diff.shape[-2:]
    grand = diff.mean(axis=(-2, -1), keepdims=True)
    seed_means = diff.mean(axis=-1, keepdims=True)
    task_means = diff.mean(axis=-2, keepdims=True)
    ms_seed = n_tasks * ((seed_means - grand) ** 2).sum(axis=(-2, -1)) / (n_seeds - 1)
    ms_task = n_seeds * ((task_means - grand) ** 2).sum(axis=(-2, -1)) / (n_tasks - 1)
    resid = diff - seed_means - task_means + grand
    ms_resid = (resid ** 2).sum(axis=(-2, -1)) / ((n_seeds - 1) * (n_tasks - 1))
    return grand[..., 0, 0], ms_seed, ms_task, ms_resid


def simulate(vs, vt, ve, n_seeds, n_tasks, alpha, effect, n_sims, n_boot, rng) -> dict:
    """Rejection rates of the two candidate analyses at a true difference of 0 and of `effect`."""
    df_grid = np.exp(np.linspace(0.0, math.log(max((n_seeds - 1) * (n_tasks - 1), 2)), 48))
    crit_grid = np.array([t_quantile(1 - alpha / 2, df) for df in df_grid])
    reject = {("t", 0.0): 0, ("t", effect): 0, ("boot", 0.0): 0, ("boot", effect): 0}
    chunk = 20
    for start in range(0, n_sims, chunk):
        size = min(chunk, n_sims - start)
        noise = (rng.normal(0, vs ** 0.5, (size, n_seeds, 1))
                 + rng.normal(0, vt ** 0.5, (size, 1, n_tasks))
                 + rng.normal(0, ve ** 0.5, (size, n_seeds, n_tasks)))
        seed_counts = rng.multinomial(n_seeds, [1 / n_seeds] * n_seeds, (size, n_boot)).astype(float)
        task_counts = rng.multinomial(n_tasks, [1 / n_tasks] * n_tasks, (size, n_boot)).astype(float)
        for mu in (0.0, effect):
            diff = noise + mu
            grand, ms_seed, ms_task, ms_resid = mean_squares(diff)
            # A negative combination is floored at the residual-only variance of the mean.
            combo = np.maximum(ms_seed + ms_task - ms_resid, ms_resid)
            df = combo ** 2 / (ms_seed ** 2 / (n_seeds - 1) + ms_task ** 2 / (n_tasks - 1)
                               + ms_resid ** 2 / ((n_seeds - 1) * (n_tasks - 1)))
            critical = np.interp(np.log(np.clip(df, df_grid[0], df_grid[-1])), np.log(df_grid), crit_grid)
            reject["t", mu] += int((np.abs(grand) / np.sqrt(combo / (n_seeds * n_tasks)) > critical).sum())
            boot = ((seed_counts @ diff) * task_counts).sum(axis=-1) / (n_seeds * n_tasks)
            low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2], axis=1)
            reject["boot", mu] += int(((low > 0) | (high < 0)).sum())
    return dict(seeds=n_seeds, tasks=n_tasks, n_sims=n_sims, effect_pp=100 * effect,
                type1_t=reject["t", 0.0] / n_sims, power_t=reject["t", effect] / n_sims,
                type1_boot=reject["boot", 0.0] / n_sims, power_boot=reject["boot", effect] / n_sims,
                monte_carlo_se_at_5pct=(0.05 * 0.95 / n_sims) ** 0.5)


def project(vs, vt, ve, n_seeds, n_tasks, alpha, power) -> dict:
    """Standard error, Satterthwaite df and minimum detectable difference for a new design."""
    ems_seed = n_tasks * vs + ve
    ems_task = n_seeds * vt + ve
    variance = (ems_seed + ems_task - ve) / (n_seeds * n_tasks)
    df = (ems_seed + ems_task - ve) ** 2 / (
        ems_seed ** 2 / (n_seeds - 1)
        + ems_task ** 2 / (n_tasks - 1)
        + ve ** 2 / ((n_seeds - 1) * (n_tasks - 1))
    )
    t_alpha = t_quantile(1 - alpha / 2, max(df, 1.0))
    t_power = t_quantile(power, max(df, 1.0))
    normal = NormalDist()
    se = variance ** 0.5
    return dict(
        seeds=n_seeds, tasks=n_tasks, cells=n_seeds * n_tasks,
        se_pp=100 * se, df=float(df),
        # Both are approximations: central t ignores noncentrality and does not pool zero
        # components; the normal version treats the variance as known.
        mde_t_pp=100 * (t_alpha + t_power) * se,
        mde_z_pp=100 * (normal.inv_cdf(1 - alpha / 2) + normal.inv_cdf(power)) * se,
        share_seed=(vs / n_seeds) / variance,
        share_task=(vt / n_tasks) / variance,
        share_resid=(ve / (n_seeds * n_tasks)) / variance,
    )


def checkpoint_correlation(delta: np.ndarray) -> np.ndarray:
    """Correlation between checkpoints of the cell-level paired differences."""
    flat = delta.transpose(1, 0, 2).reshape(delta.shape[1], -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.corrcoef(flat)


def parse_designs(text: str):
    return [tuple(int(v) for v in item.split("x")) for item in text.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="runs/transfer-factorial-v1")
    parser.add_argument("--families", nargs="+", default=["cool", "heat"])
    parser.add_argument("--selections", nargs="+", default=["matched", "full"],
                        help="Case prefix before the source name, e.g. matched-api-seed1101")
    parser.add_argument("--source-a", default="api")
    parser.add_argument("--source-b", default="generic")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1101, 1202, 1303])
    parser.add_argument("--designs", default="3x32,3x128,6x64,8x64,12x32,6x128,12x64,12x128,24x64",
                        help="Comma-separated SEEDSxTASKS designs to project")
    parser.add_argument("--negative", choices=["truncate", "pool"], default="pool")
    parser.add_argument("--seed-var-multipliers", nargs="+", type=float, default=[1.0, 4.0])
    parser.add_argument("--seed-var-floor", type=float, default=0.0,
                        help="Lower bound on the seed variance used for planning, on the 0-1 scale squared")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--simulate", type=int, default=0, metavar="N",
                        help="Simulated data sets per design for Type I error and power; 0 disables")
    parser.add_argument("--effect-pp", type=float, default=5.0,
                        help="True difference, in percentage points, used for simulated power")
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--metrics", nargs="+", default=["raw_auc", "initial", "final", "gain_auc"])
    parser.add_argument("--out", help="Optional JSON output path")
    args = parser.parse_args()

    rng = np.random.default_rng(20261111)
    designs = parse_designs(args.designs)
    report = dict(arguments=vars(args), planning_only=True, conditions=[])

    for family in args.families:
        cells = load_cells(Path(args.root) / family / "evaluation" / "main")
        if not cells:
            raise SystemExit(f"No evaluation records under {args.root}/{family}/evaluation/main")
        for selection in args.selections:
            prefix = f"{selection}-"
            reference = cells[f"{prefix}{args.source_a}-seed{args.seeds[0]}"]
            steps = sorted({step for step, _ in reference})
            tasks = sorted({task for _, task in reference})
            delta = (success_array(cells, prefix, args.source_a, args.seeds, steps, tasks)
                     - success_array(cells, prefix, args.source_b, args.seeds, steps, tasks))
            weights = trapezoid_weights(steps)
            raw_auc = (delta * weights[None, :, None]).sum(axis=1)
            metrics = {
                "raw_auc": raw_auc,
                "initial": delta[:, 0, :],
                "final": delta[:, -1, :],
                "gain_auc": raw_auc - delta[:, 0, :],
            }
            corr = checkpoint_correlation(delta)
            print(f"\n=== {family} / {selection}: {args.source_a} - {args.source_b}, "
                  f"{len(args.seeds)} seeds x {len(tasks)} tasks, checkpoints {steps}")
            print("checkpoint correlation of cell-level differences:")
            print(np.array2string(corr, precision=2, suppress_small=True))
            condition = dict(family=family, selection=selection, steps=steps,
                             checkpoint_correlation=corr.tolist(), metrics={})
            for name in args.metrics:
                matrix = metrics[name]
                fit = decompose(matrix)
                print(f"\n{name}: mean {100 * fit['mean']:+.2f}pp   seed means "
                      f"{np.round(100 * np.array(fit['seed_means']), 1).tolist()}")
                print(f"  raw components   seed {fit['var_seed_raw']:+.4f}   task {fit['var_task_raw']:+.4f}"
                      f"   resid {fit['var_resid']:.4f}")
                entry = dict(fit=fit, projections=[])
                for multiplier in args.seed_var_multipliers:
                    vs, vt, ve = planning_components(fit, args.negative, multiplier, args.seed_var_floor)
                    print(f"  planning ({args.negative}, seed variance x{multiplier:g}, floor {args.seed_var_floor:g})"
                          f": seed {vs:.4f}  task {vt:.4f}  resid {ve:.4f}")
                    print("    seeds tasks cells   SE(pp)    df   MDE_t(pp) MDE_z(pp)   variance share seed/task/resid")
                    for n_seeds, n_tasks in designs:
                        row = project(vs, vt, ve, n_seeds, n_tasks, args.alpha, args.power)
                        row["seed_var_multiplier"] = multiplier
                        entry["projections"].append(row)
                        print(f"    {n_seeds:5d} {n_tasks:5d} {row['cells']:5d} {row['se_pp']:8.2f} {row['df']:5.1f}"
                              f" {row['mde_t_pp']:10.2f} {row['mde_z_pp']:9.2f}"
                              f"   {row['share_seed']:.2f}/{row['share_task']:.2f}/{row['share_resid']:.2f}")
                    if args.simulate:
                        print(f"    simulated rejection rates, nominal alpha {args.alpha:g}, "
                              f"true difference 0 and {args.effect_pp:g}pp, {args.simulate} data sets")
                        print("    seeds tasks   type1_t  power_t   type1_boot  power_boot")
                        for n_seeds, n_tasks in designs:
                            sim = simulate(vs, vt, ve, n_seeds, n_tasks, args.alpha, args.effect_pp / 100,
                                           args.simulate, args.bootstrap_draws, rng)
                            sim["seed_var_multiplier"] = multiplier
                            entry.setdefault("simulations", []).append(sim)
                            print(f"    {n_seeds:5d} {n_tasks:5d} {sim['type1_t']:9.3f} {sim['power_t']:8.3f}"
                                  f" {sim['type1_boot']:12.3f} {sim['power_boot']:11.3f}")
                condition["metrics"][name] = entry
            report["conditions"].append(condition)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
