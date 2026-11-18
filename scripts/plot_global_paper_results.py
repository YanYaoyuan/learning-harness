#!/usr/bin/env python3
"""Six paper figures from completed results; no partial confirmation outcomes."""

from collections import Counter
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from learning_harness.common import file_sha
from learning_harness.training_log_metrics import training_metrics
import run_global_selection_baselines as study

OUT = Path("output/paper-global-v2")
FIG = OUT / "figures"
RULES = ("global_random", "global_high", "gradient_global")
LABELS = ("Random", "High NLL", "Gradient")
COLORS = ("#65758b", "#df8054", "#6654a0")


def save_figure(fig, name):
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    study.validate()
    if FIG.exists():
        raise ValueError("Preserve any previous paper-figure snapshot")
    FIG.mkdir(parents=True)
    current = study.read(study.OUT / "aggregate-summary.json")
    original = study.read(study.parent.OUT / "aggregate-summary.json")
    if not current["complete"] or not original["complete"]:
        raise ValueError("Use completed matrices only")
    audit = study.read("output/research-operations/20261118/global-final-local-audit.json")
    if audit["summary_sha256"] != file_sha(study.OUT / "aggregate-summary.json"):
        raise ValueError("Source differs from raw-audited results")
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    public = dict(source_sha256={
        str(study.OUT / "aggregate-summary.json"): file_sha(study.OUT / "aggregate-summary.json"),
        str(study.parent.OUT / "aggregate-summary.json"): file_sha(study.parent.OUT / "aggregate-summary.json"),
    }, code_sha256=file_sha(Path(__file__)), confirmation_included=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, source in zip(axes, study.SOURCES):
        d = current["endpoints"][source]
        values = [100 * d[p]["mean_success"] for p in RULES]
        ax.bar(range(3), values, color=COLORS, width=0.65, alpha=0.9)
        for i, rule in enumerate(RULES):
            points = 100 * np.array(list(d[rule]["per_initialization"].values()))
            ax.scatter(i + np.linspace(-0.15, 0.15, 8), points, s=14, color="#152238", alpha=0.8)
            ax.text(i, max(points) + 1.8, f"{values[i]:.2f}", ha="center", fontsize=8)
        ax.axhline(100 * current["baseline_success"], linestyle="--", color="#7b8794", linewidth=1)
        ax.set(title=source.upper(), ylim=(0, 95), xticks=range(3), xticklabels=LABELS, ylabel="Success (%)")
        ax.grid(axis="y", alpha=0.12)
    save_figure(fig, "global-main")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    interaction = {}
    for ax, source in zip(axes, study.SOURCES):
        interaction[source] = {}
        for rule, matched, label, color in zip(RULES, ("random", "high", "gradient_high"), LABELS, COLORS):
            a = original["endpoints"]["sources"][source][matched]
            b = current["endpoints"][source][rule]
            values = [100 * a["mean_success"], 100 * b["mean_success"]]
            ax.plot([0, 1], values, "o-", label=label, color=color, linewidth=2, markersize=5)
            interaction[source][rule] = (values[1] - values[0])
        ax.set(title=source.upper(), xticks=[0, 1], xticklabels=["Coverage matched", "Global"],
               xlim=(-0.12, 1.12), ylim=(63, 85), ylabel="Success (%)")
        ax.grid(axis="y", alpha=0.15)
    axes[0].legend(frameon=False, fontsize=8)
    save_figure(fig, "selection-coverage-interaction")
    public["global_minus_matched_pp"] = interaction

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    columns = [(s, c) for c in ("global_random", "global_high") for s in study.SOURCES]
    values = np.array([[100 * current["contrasts"][c]["per_source"][s][str(seed)]
                        for s, c in columns] for seed in study.parent.INITIALIZATIONS])
    heat = ax.imshow(values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=12)
    ax.set(xticks=range(4), xticklabels=["API\nvs random", "Generic\nvs random", "API\nvs high NLL", "Generic\nvs high NLL"],
           yticks=range(8), yticklabels=[str(s) for s in study.parent.INITIALIZATIONS], ylabel="Initialization seed")
    for i in range(8):
        for j in range(4):
            ax.text(j, i, f"+{values[i,j]:.2f}", ha="center", va="center",
                    color="white" if values[i, j] > 7 else "#132c38", fontsize=8)
    fig.colorbar(heat, ax=ax, label="Gradient advantage (percentage points)")
    save_figure(fig, "paired-initializations")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    curves = {}
    for ax, source in zip(axes, study.SOURCES):
        curves[source] = {}
        for rule, label, color in zip(RULES, LABELS, COLORS):
            records = [r for r in current["models"] if r["case"]["source"] == source and r["case"]["policy"] == rule]
            trajectories = []
            for record in records:
                c = record["case"]
                root = study.parent.ROOT if rule == "gradient_global" else study.ROOT
                rows = training_metrics((root / "cases" / c["name"] / "training.log").read_text(errors="replace"))
                if [r["step"] for r in rows] != list(range(10, 191, 10)):
                    raise ValueError("Incomplete recorded loss history")
                trajectories.append([r["loss"] for r in rows])
            points = np.array(trajectories)
            avg, sd = points.mean(axis=0), points.std(axis=0, ddof=1)
            steps = list(range(10, 191, 10))
            ax.plot(steps, avg, label=label, color=color, linewidth=1.7)
            ax.fill_between(steps, np.maximum(0, avg - sd), avg + sd, color=color, alpha=0.12, linewidth=0)
            curves[source][rule] = dict(steps=steps, mean_loss=avg.tolist(), model_sd=sd.tolist(), models=len(trajectories))
        ax.set(title=source.upper(), xlabel="Optimizer update", ylabel="Recorded SFT loss")
        ax.grid(alpha=0.15)
    axes[0].legend(frameon=False, fontsize=8)
    save_figure(fig, "training-loss")
    public["training_loss"] = curves

    coverage, action_set = {}, set()
    for source in study.SOURCES:
        rows, nll, gradient, _ = study.parent.scores_for(source)
        tasks, actions, _ = study.parent.arrays(rows) if hasattr(study.parent, "arrays") else (None, None, None)
        if tasks is None:
            from learning_harness.experience_value_study import arrays
            tasks, actions, _ = arrays(rows)
        coverage[source] = {}
        for rule in RULES:
            sizes, entropies, mixes = [], [], []
            for seed in study.SAMPLING:
                root = study.parent.ROOT if rule == "gradient_global" else study.ROOT
                indices = study.read(root / "recipes" / f"{rule}-{source}-s{seed}.private.json")["indices"]
                counts = Counter(tasks[i] for i in indices)
                sizes.append(len(counts))
                entropies.append(-sum((n/768)*math.log(n/768) for n in counts.values()) / math.log(len(set(tasks))))
                mixes.append(Counter(actions[i] for i in indices))
            avg_mix = {a: sum(m.get(a, 0) for m in mixes) / (2 * 768) for a in set(actions)}
            action_set.update(avg_mix)
            coverage[source][rule] = dict(distinct_tasks_mean=float(np.mean(sizes)), normalized_task_entropy=float(np.mean(entropies)), action_fraction=avg_mix)
    public["coverage"] = coverage
    actions = sorted(action_set)
    labels = [source.upper() + " / " + label for source in study.SOURCES for label in LABELS]
    values = np.array([[100 * coverage[s][p]["action_fraction"].get(a, 0) for a in actions]
                       for s in study.SOURCES for p in RULES])
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    heat = ax.imshow(values, cmap="Blues", aspect="auto")
    ax.set(yticks=range(6), yticklabels=labels, xticks=range(len(actions)), xticklabels=actions)
    for i in range(6):
        for j in range(len(actions)):
            ax.text(j, i, f"{values[i,j]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if values[i, j] > values.max() * .55 else "#132c38")
    fig.colorbar(heat, ax=ax, label="Training presentations (%)")
    save_figure(fig, "action-coverage")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    scatter_data = []
    old_rules = ("random", "low", "middle", "high", "gradient_high", "gradient_low", "gradient_global")
    markers = ("o", "v", "s", "^", "D", "P", "X", "h", "*")
    names = (*old_rules, "global_random", "global_high")
    cmap = plt.get_cmap("tab10")
    for ax, source in zip(axes, study.SOURCES):
        rows, nll, _, _ = study.parent.scores_for(source)
        for j, rule in enumerate(names):
            for seed in study.SAMPLING:
                root = study.parent.ROOT if rule in old_rules else study.ROOT
                ids = study.read(root / "recipes" / f"{rule}-{source}-s{seed}.private.json")["indices"]
                records = (original if rule in old_rules else current)["models"]
                values = [r["mean_success"] for r in records if r["case"]["source"] == source and r["case"]["policy"] == rule and r["case"]["sampling_seed"] == seed]
                if len(values) != 8:
                    raise ValueError("Scatter must average exactly eight initializations")
                x, y = sum(nll[i] for i in ids) / 768, 100 * np.mean(values)
                ax.scatter(x, y, color=cmap(j), marker=markers[j], s=38, alpha=0.85,
                           label=rule.replace("gradient_", "grad ").replace("global_", "global ") if seed == 101 else None)
                scatter_data.append(dict(source=source, policy=rule, sampling_seed=seed, mean_selected_nll=x, success_percent=float(y)))
        ax.set(title=source.upper(), xlabel="Selected action NLL (presentation mean)", ylabel="Success (%)")
        ax.grid(alpha=0.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.23))
    save_figure(fig, "imitation-learning")
    public["imitation_learning_points"] = scatter_data
    public["figure_sha256"] = {p.name: file_sha(p) for p in FIG.iterdir() if p.is_file()}
    (OUT / "figure-data.json").write_text(json.dumps(public, indent=2) + "\n")
    print(json.dumps(dict(status="six_completed_result_figures_created", output=str(FIG), confirmation_included=False)))


if __name__ == "__main__":
    main()
