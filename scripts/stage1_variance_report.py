#!/usr/bin/env python3
"""Descriptive crossed-seed decomposition; sensitivity ranges are not confidence intervals."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("runs/variance-diagnostic-v1/stage1")
OUT = Path("output/variance-diagnostic-v1/stage1")
SEEDS = (101, 202, 303, 404)
SOURCES = ("api", "generic")


def components(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError("A finite crossed matrix with at least two levels per factor is required")
    ni, ns = values.shape
    grand = values.mean()
    # Reference centering keeps identical cells exactly constant.
    centered = values - values[0, 0]
    im = centered.mean(axis=1)
    sm = centered.mean(axis=0)
    mean = centered.mean()
    msi = float(ns * ((im - mean) ** 2).sum() / (ni - 1))
    mss = float(ni * ((sm - mean) ** 2).sum() / (ns - 1))
    interaction = centered - im[:, None] - sm[None, :] + mean
    mse = float((interaction ** 2).sum() / ((ni - 1) * (ns - 1)))
    raw_i, raw_s = (msi - mse) / ns, (mss - mse) / ni
    vi, vs = max(raw_i, 0), max(raw_s, 0)
    total = vi + vs + mse
    return dict(mean=float(grand), ms_initialization=msi, ms_sampling=mss, ms_residual=mse,
        initialization_variance_raw=raw_i, sampling_variance_raw=raw_s,
        initialization_variance=vi, sampling_variance=vs, interaction_plus_noise_variance=mse,
        sampling_share=vs / total if total >= 1e-12 else None,
        initialization_share=vi / total if total >= 1e-12 else None,
        df_initialization=ni - 1, df_sampling=ns - 1, df_interaction=(ni - 1) * (ns - 1))


def sensitivity(values):
    values = np.asarray(values)
    fits = []
    for axis in (0, 1):
        for level in range(values.shape[axis]):
            fit = components(np.delete(values, level, axis=axis))
            fits.append(dict(deleted_axis="initialization" if axis == 0 else "sampling", deleted_level=level, **fit))
    shares = [r["sampling_share"] for r in fits if r["sampling_share"] is not None]
    return dict(method="Leave-one-factor-level-out; not a confidence interval", fits=fits,
                sampling_share_range=[min(shares), max(shares)] if shares else None)


def allocation(success_fit, nll_fit):
    shares = [v["sampling_share"] for v in (success_fit, nll_fit) if v["sampling_share"] is not None]
    return "Complete-pass sampling follow-up recommended" if any(v >= .30 for v in shares) else "Sampling control not indicated by the point-estimate rule"


def main():
    freeze = json.loads((ROOT / "freeze.json").read_text())
    from learning_harness.common import file_sha
    for name, expected in freeze["files"].items():
        if file_sha(name) != expected:
            raise ValueError("Frozen Stage 1 file changed: " + name)
    rows, records, exposure_rows = [], {}, []
    actual_episodes = 0
    init_hashes, schedules = {}, {}
    for case in freeze["cases"]:
        path = ROOT / "cases" / case["name"] / "worker-result.json"
        if not path.exists():
            raise ValueError("Missing planned result: " + str(path))
        record = json.loads(path.read_text())
        if not record["complete"] or record["smoke"]:
            raise ValueError("Incomplete or smoke record in main study")
        if record["case"] != case:
            raise ValueError("Result case differs from the frozen plan")
        ctrl = record["control"]
        directory = path.parent
        if not ctrl["completed"] or ctrl["trained_adapter_sha256"] != file_sha(directory / "A/adapter_model.safetensors"):
            raise ValueError("Training is incomplete or checkpoint changed")
        if ctrl["initialization_seed"] != case["initialization_seed"] or ctrl["sampling_seed"] != case["sampling_seed"]:
            raise ValueError("Training seeds differ from frozen plan")
        for family in ("cool", "heat"):
            summary = json.loads((directory / "evaluation" / family / "summary.json").read_text())
            run = json.loads((directory / "evaluation" / family / "run.json").read_text())
            if (not summary["complete"] or summary["recorded"] != 32
                    or summary["metrics"]["none"]["error_count"]
                    or run["revision"] != record["composition_revision"]
                    or summary["metrics"]["none"]["success_rate"] != record["success"][family]):
                raise ValueError("Incomplete, errored or inconsistent evaluation")
            actual_episodes += summary["recorded"]
        key = case["initialization_seed"]
        if key in init_hashes and init_hashes[key] != ctrl["initial_parameter_sha256"]:
            raise ValueError("Initialization changed with sampling/source")
        init_hashes[key] = ctrl["initial_parameter_sha256"]
        key = (case["source"], case["sampling_seed"])
        if key in schedules and schedules[key] != ctrl["schedule_sha256"]:
            raise ValueError("Sampling changed with initialization")
        schedules[key] = ctrl["schedule_sha256"]
        row = dict(**case, mean_success=record["mean_success"], mean_nll=record["mean_nll"],
                   cool_success=record["success"]["cool"], heat_success=record["success"]["heat"])
        rows.append(row)
        exposure = record["exposure"]
        exposure_rows.append(dict(**case, processed_examples=exposure["processed_examples"],
            unique_examples=exposure["unique_examples"], unique_tasks=exposure["unique_tasks"],
            input_tokens=exposure["input_tokens"], target_tokens=exposure["target_tokens"],
            place_presentations=exposure["task_type_presentations"].get("pick_and_place_simple", 0),
            clean_presentations=exposure["task_type_presentations"].get("pick_clean_then_place_in_recep", 0),
            heat_presentations=exposure["task_type_presentations"].get("pick_heat_then_place_in_recep", 0)))
        records[case["name"]] = record
    if len(set(init_hashes.values())) != 4:
        raise ValueError("Changing initialization seed did not change initialization")
    fits, matrices = {}, {}
    for metric in ("mean_success", "mean_nll"):
        matrices[metric] = {}
        fits[metric] = {}
        for source in SOURCES:
            matrix = np.array([[next(r[metric] for r in rows if r["source"] == source and r["initialization_seed"] == i and r["sampling_seed"] == s and r["repeat"] == 0) for s in SEEDS] for i in SEEDS])
            matrices[metric][source] = matrix
        matrices[metric]["paired_api_minus_generic"] = matrices[metric]["api"] - matrices[metric]["generic"]
        for source, matrix in matrices[metric].items():
            fits[metric][source] = dict(components=components(matrix), sensitivity=sensitivity(matrix))
    repeats = []
    for source in SOURCES:
        pair = [r for r in rows if r["source"] == source and r["initialization_seed"] == 101 and r["sampling_seed"] == 101]
        if len(pair) != 2:
            raise ValueError("Training repeat is missing")
        a, b = sorted(pair, key=lambda r: r["repeat"])
        repeats.append(dict(source=source, success_difference_pp=100 * (b["mean_success"] - a["mean_success"]),
                            nll_difference=b["mean_nll"] - a["mean_nll"], conditional_single_pair=True))
    decision = allocation(fits["mean_success"]["paired_api_minus_generic"]["components"],
                          fits["mean_nll"]["paired_api_minus_generic"]["components"])
    result = dict(complete=True, exploratory=True, teacher_calls=0, official_tests_used=False,
        cases=len(rows), planned_evaluation_episodes=2176, actual_evaluation_episodes=actual_episodes,
        freeze_sha256=file_sha(ROOT / "freeze.json"), results=rows, exposure=exposure_rows,
        fits=fits, training_repeats=repeats,
        matrices={metric: {source: matrix.tolist() for source, matrix in group.items()} for metric, group in matrices.items()},
        allocation_decision=decision, allocation_is_not_a_mechanism_claim=True,
        uncertainty="All cell values and factor-deletion ranges; no calibrated confidence intervals",
        init_hash_invariance_verified=True, schedule_invariance_verified=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (OUT / "cell-scores.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (OUT / "training-exposure.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exposure_rows[0]))
        writer.writeheader()
        writer.writerows(exposure_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, metric in zip(axes, ("mean_success", "mean_nll")):
        matrix = matrices[metric]["paired_api_minus_generic"]
        scale = 100 if metric == "mean_success" else 1
        bound = max(float(np.abs(matrix).max()) * scale, .001)
        image = ax.imshow(matrix * scale, cmap="RdBu_r", vmin=-bound, vmax=bound)
        for i in range(4):
            for s in range(4):
                ax.text(s, i, f"{matrix[i, s] * scale:+.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(matrix[i, s] * scale) > .6 * bound else "black")
        ax.set_xticks(range(4), SEEDS); ax.set_yticks(range(4), SEEDS)
        ax.set_xlabel("Sampling seed"); ax.set_ylabel("Initialization + dropout seed")
        ax.set_title("Success difference (pp)" if metric == "mean_success" else "NLL difference (nats)")
        fig.colorbar(image, ax=ax, shrink=.8)
    fig.suptitle("Stage 1 — API minus generic, fixed development tasks")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"crossed-seeds.{ext}", dpi=180)
    plt.close(fig)
    lines = ["# Stage 1：初始化/dropout与样本抽取", "", decision, "",
             "描述性开发集诊断；不比较新API−ReAct方案，不作方法优越性检验。", "",
             "| 配对指标 | 初始化分量 | 抽样分量 | 交互及剩余噪声 | 抽样占比 |", "|---|---:|---:|---:|---:|"]
    for metric in fits:
        fit = fits[metric]["paired_api_minus_generic"]["components"]
        lines.append(f"| {metric} | {fit['initialization_variance']:.6g} | {fit['sampling_variance']:.6g} | {fit['interaction_plus_noise_variance']:.6g} | {fit['sampling_share']} |")
    lines += ["", "原始未截断分量、每个单元、来源单独分解及留一敏感性范围见summary.json。敏感性范围不是置信区间。",
              "训练重复仅各一个固定配置的条件比较，不是总体训练噪声估计。Stage 2不会自动启动。", ""]
    (OUT / "RESULTS.zh-CN.md").write_text("\n".join(lines))
    print(json.dumps(dict(complete=True, cases=len(rows), allocation_decision=decision)))


if __name__ == "__main__":
    main()
