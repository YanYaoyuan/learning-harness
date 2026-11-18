#!/usr/bin/env python3
"""Describe matched-compute sampling effects, factor components and exposure audits."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from learning_harness.adaptation import composition_revision
from learning_harness.bounded_queue import save
from learning_harness.common import file_sha
from stage1_variance_report import components, sensitivity, SEEDS
import run_stage2_sampling as controller
import run_stage1_crossed_seeds as previous

ROOT, OUT = controller.ROOT, controller.OUT
ARMS = ("partial_256", "full_pass", "subset_replay")
LABELS = ("Historical 256", "Full pool 768", "Subset replay 768")
SOURCES = ("api", "generic")


def collect(frozen):
    rows, exposure_rows, records = [], [], {}
    init_hashes, schedule_hashes = {}, {}
    new_episodes = 0
    groups = [("partial_256", previous.ROOT, previous.plan()),
              (None, ROOT, frozen["cases"])]
    for fixed_arm, root, cases in groups:
        for case in cases:
            arm = fixed_arm or case["arm"]
            directory = root / "cases" / case["name"]
            record = json.loads((directory / "worker-result.json").read_text())
            if not record["complete"] or record["smoke"] or record["case"] != case:
                raise ValueError("Incomplete, smoke, or wrong case result")
            ctrl = record["control"]
            if (not ctrl["completed"] or ctrl["trained_adapter_sha256"] != file_sha(directory / "A/adapter_model.safetensors")
                    or ctrl["initialization_seed"] != case["initialization_seed"]
                    or ctrl["sampling_seed"] != case["sampling_seed"]
                    or composition_revision(directory / "checkpoint") != record["composition_revision"]):
                raise ValueError("Checkpoint or seed mismatch")
            key = case["initialization_seed"]
            if key in init_hashes and init_hashes[key] != ctrl["initial_parameter_sha256"]:
                raise ValueError("Initialization changed across policy/source")
            init_hashes[key] = ctrl["initial_parameter_sha256"]
            key = (arm, case["source"], case["sampling_seed"])
            if key in schedule_hashes and schedule_hashes[key] != ctrl["schedule_sha256"]:
                raise ValueError("Schedule changed across initialization/repeat")
            schedule_hashes[key] = ctrl["schedule_sha256"]
            for family in ("cool", "heat"):
                summary = json.loads((directory / "evaluation" / family / "summary.json").read_text())
                run = json.loads((directory / "evaluation" / family / "run.json").read_text())
                if (not summary["complete"] or summary["recorded"] != 32
                        or summary["metrics"]["none"]["error_count"]
                        or run["revision"] != record["composition_revision"]
                        or summary["metrics"]["none"]["success_rate"] != record["success"][family]):
                    raise ValueError("Evaluation audit failed")
                if fixed_arm is None:
                    new_episodes += summary["recorded"]
            exposure = record["exposure"]
            if exposure["processed_examples"] != (256 if arm == "partial_256" else 768):
                raise ValueError("Unequal planned presentation budget")
            if arm == "full_pass" and exposure["unique_examples"] != (721 if case["source"] == "api" else 732):
                raise ValueError("Incomplete full-pool exposure")
            if arm == "subset_replay" and exposure["unique_examples"] != 256:
                raise ValueError("Replay control changed membership")
            nll = json.loads((directory / "diagnostic-nll.json").read_text())
            if not nll["checkpoint_unchanged"] or nll["composition_revision"] != record["composition_revision"]:
                raise ValueError("NLL checkpoint mismatch")
            if record["mean_success"] != sum(record["success"].values()) / 2 or record["mean_nll"] != sum(nll[f]["task_mean_nll"] for f in ("cool", "heat")) / 2:
                raise ValueError("Stored mean mismatch")
            row = dict(case, arm=arm, mean_success=record["mean_success"], mean_nll=record["mean_nll"],
                       cool_success=record["success"]["cool"], heat_success=record["success"]["heat"])
            rows.append(row)
            exposure_rows.append(dict(name=case["name"], arm=arm, source=case["source"],
                initialization_seed=case["initialization_seed"], sampling_seed=case["sampling_seed"], repeat=case["repeat"],
                **{k:exposure[k] for k in ("processed_examples", "unique_examples", "unique_tasks", "input_tokens", "target_tokens")},
                task_type_presentations=json.dumps(exposure["task_type_presentations"], sort_keys=True)))
            records[(arm, case["name"])] = record
    if len(set(init_hashes.values())) != 4 or new_episodes != 4352:
        raise ValueError("Initialization or episode totals changed")
    return rows, exposure_rows, new_episodes


def analyze(rows):
    matrices, fits = {}, {}
    for metric in ("mean_success", "mean_nll"):
        matrices[metric], fits[metric] = {}, {}
        for arm in ARMS:
            group = {source: np.array([[next(r[metric] for r in rows if r["arm"] == arm and
                r["source"] == source and r["initialization_seed"] == i and r["sampling_seed"] == s and r["repeat"] == 0)
                for s in SEEDS] for i in SEEDS]) for source in SOURCES}
            group["paired_api_minus_generic"] = group["api"] - group["generic"]
            matrices[metric][arm] = group
            fits[metric][arm] = {source: dict(components=components(matrix), sensitivity=sensitivity(matrix))
                                  for source, matrix in group.items()}
    effects = {}
    for metric, group in matrices.items():
        effects[metric] = {}
        for source in (*SOURCES, "paired_api_minus_generic"):
            delta = group["full_pass"][source] - group["subset_replay"][source]
            full_var = fits[metric]["full_pass"][source]["components"]["sampling_variance"]
            replay_var = fits[metric]["subset_replay"][source]["components"]["sampling_variance"]
            effects[metric][source] = dict(mean_difference=float(delta.mean()), cell_differences=delta.tolist(),
                full_pass_sampling_variance=full_var, subset_replay_sampling_variance=replay_var,
                sampling_variance_reduction_fraction=1-full_var/replay_var if replay_var >= 1e-12 else None)
    return matrices, fits, effects


def export_csv(name, rows):
    with (OUT / name).open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figure(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=180)
    plt.close(fig)


def main():
    frozen = controller.load_freeze()
    rows, exposure, episodes = collect(frozen)
    matrices, fits, effects = analyze(rows)
    repeats = []
    for arm in ARMS:
        for source in SOURCES:
            group = sorted([r for r in rows if r["arm"] == arm and r["source"] == source and
                            r["initialization_seed"] == 101 and r["sampling_seed"] == 101], key=lambda r:r["repeat"])
            if len(group) != 2:
                raise ValueError("Conditional training repeat missing")
            a,b = group
            repeats.append(dict(arm=arm, source=source, success_difference_pp=100*(b["mean_success"]-a["mean_success"]),
                                nll_difference=b["mean_nll"]-a["mean_nll"], conditional_single_pair=True))
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(complete=True, exploratory=True, teacher_calls=0, official_tests_used=False,
        cases=68, historical_reference_cases=34, actual_evaluation_episodes=episodes,
        freeze_sha256=file_sha(ROOT / "freeze.json"), results=rows, exposure=exposure,
        fits=fits, matched_compute_policy_effects=effects, training_repeats=repeats,
        matrices={m:{a:{s:x.tolist() for s,x in g.items()} for a,g in arms.items()} for m,arms in matrices.items()},
        primary_comparison="full_pass versus subset_replay at 768 presentations / 192 updates",
        interpretation="Sampling policy changes membership, frequency and order together; historical 256 is a dose reference",
        uncertainty="All crossed cells and factor-deletion sensitivity ranges; ranges are not confidence intervals")
    export_csv("cell-scores.csv", rows)
    export_csv("training-exposure.csv", exposure)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, source in zip(axes, SOURCES):
        for j, arm in enumerate(ARMS):
            values = matrices["mean_success"][arm][source] * 100
            ax.scatter(np.full(16,j)+np.tile(np.linspace(-.14,.14,4),4), values.ravel(), alpha=.55, s=22)
            ax.plot([j-.2,j+.2], [values.mean()]*2, color="black", linewidth=2)
        ax.set_xticks(range(3),LABELS,rotation=12)
        ax.set_title(source.upper())
        ax.set_ylabel("Post-A success (%)")
        ax.set_ylim(-2,102)
    fig.suptitle("Sampling policies — all 16 crossed cells; black line = mean")
    figure(fig,"sampling-policy-scores")
    fig, axes = plt.subplots(1,2,figsize=(11,4.4))
    for ax, metric in zip(axes,("mean_success","mean_nll")):
        bottom=np.zeros(3)
        for field,label in (("initialization_variance","Initialization + dropout"),("sampling_variance","Sampling"),("interaction_plus_noise_variance","Interaction + noise")):
            values=np.array([fits[metric][arm]["paired_api_minus_generic"]["components"][field] for arm in ARMS])
            ax.bar(range(3),values,bottom=bottom,label=label)
            bottom+=values
        ax.set_xticks(range(3),LABELS,rotation=12)
        ax.set_ylabel("Variance (success proportions squared)" if metric=="mean_success" else "Variance (NLL squared)")
        ax.set_title("API − generic / "+metric)
    axes[1].legend(fontsize=8)
    figure(fig,"sampling-variance-components")
    fig, axes=plt.subplots(1,3,figsize=(13,4))
    for ax,source in zip(axes,(*SOURCES,"paired_api_minus_generic")):
        delta=np.array(effects["mean_success"][source]["cell_differences"])*100
        bound=max(float(np.abs(delta).max()),.01)
        im=ax.imshow(delta,cmap="RdBu_r",vmin=-bound,vmax=bound)
        for i in range(4):
            for s in range(4):
                ax.text(s,i,f"{delta[i,s]:+.1f}",ha="center",va="center",color="white" if abs(delta[i,s])>.6*bound else "black",fontsize=9)
        ax.set_xticks(range(4),SEEDS)
        ax.set_yticks(range(4),SEEDS)
        ax.set_xlabel("Sampling seed")
        ax.set_ylabel("Initialization + dropout seed")
        ax.set_title(source)
        fig.colorbar(im,ax=ax,shrink=.75)
    fig.suptitle("Full pool − subset replay / matched 768 presentations / success difference (pp)")
    figure(fig,"matched-compute-policy-effects")
    lines=["# Stage 2：同预算采样干预","","68个新模型、4,352个新任务回合已完成。两个新组均192次更新、768次呈现。","",
           "| 新组 | API−generic平均成功率差（pp） | 抽样方差 | 抽样占比 |","|---|---:|---:|---:|"]
    for arm in controller.ARMS:
        fit=fits["mean_success"][arm]["paired_api_minus_generic"]["components"]
        lines.append(f"| {arm} | {100*fit['mean']:.3f} | {fit['sampling_variance']:.6g} | {fit['sampling_share']} |")
    lines += ["","同预算full_pass−subset_replay平均成功率差："]
    for source in (*SOURCES,"paired_api_minus_generic"):
        lines.append(f"- {source}: {100*effects['mean_success'][source]['mean_difference']:+.3f} pp")
    lines += ["","三组逐单元、NLL、实际覆盖/token、原始分量、留一敏感性范围和条件训练重复见summary.json与CSV。",
              "留一范围不是置信区间。采样策略同时改变示范覆盖、频率和顺序；历史256组同时改变优化剂量和学习率调度长度。","",
              "![策略分数](sampling-policy-scores.png)","","![方差分解](sampling-variance-components.png)","",
              "![同预算效应](matched-compute-policy-effects.png)",""]
    (OUT/"RESULTS.zh-CN.md").write_text("\n".join(lines))
    # Commit the summary last so observers never see a complete report before its figures exist.
    save(OUT / "summary.json", result)
    print(json.dumps(dict(complete=True,cases=68,actual_evaluation_episodes=episodes)),flush=True)


if __name__ == "__main__":
    main()
