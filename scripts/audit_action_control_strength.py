#!/usr/bin/env python3
"""CPU-only audit of remaining content/NLL contrast after action matching."""

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import action_control_worker as worker
from learning_harness.common import file_sha, read_jsonl

ROOT = Path("output/variance-diagnostic-v1/task-action-feasibility")


def main():
    worker.parent.load_freeze()
    out = ROOT / "selection-strength"
    if out.exists():
        raise ValueError("Preserve existing preparation audit")
    results, provenance = [], {}
    for source in ("api", "generic"):
        data_path = Path(f"runs/transfer-factorial-v1/corpora/full-{source}.jsonl")
        score_path = worker.parent.ROOT / "base-nll" / (source + ".json")
        rows, scores = (
            read_jsonl(data_path),
            json.loads(score_path.read_text())["scores"],
        )
        provenance.update({str(p): file_sha(p) for p in (data_path, score_path)})
        content, groups = [], []
        score_by_content = defaultdict(list)
        eligible = defaultdict(set)
        for row, score in zip(rows, scores):
            digest = hashlib.sha256(
                json.dumps(row["messages"], sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            key = (row["task_id"], digest)
            content.append(key)
            command = json.loads(row["messages"][-1]["content"])["action"]
            group = (
                row["task_id"],
                "go" if command.startswith("go to ") else command.split()[0],
            )
            groups.append(group)
            eligible[group].add(key)
            score_by_content[key].append(score["mean_target_nll"])
        mean_nll = {
            key: sum(values) / len(values) for key, values in score_by_content.items()
        }
        for seed in (101, 202, 303, 404):
            schedules = {}
            for policy in ("random", "min_nll"):
                candidate_path = (
                    ROOT / "candidate-schedules" / f"{policy}-{source}-s{seed}.json"
                )
                candidate, _ = worker.load_recipe(candidate_path)
                old_path = (
                    worker.parent.ROOT
                    / "inputs/schedules"
                    / f"{policy}-{source}-s{seed}.json"
                )
                old = json.loads(old_path.read_text())
                schedules[("action_matched", policy)] = candidate["indices"]
                schedules[("task_matched", policy)] = old["indices"]
                provenance.update(
                    {str(p): file_sha(p) for p in (candidate_path, old_path)}
                )
            reference = json.loads(
                (
                    ROOT / "candidate-schedules" / f"random-{source}-s{seed}.json"
                ).read_text()
            )
            quotas = {
                (r["task_id"], r["action_family"]): r["distinct_support"]
                for r in reference["strata"]
            }
            choice_groups = {
                g for g, quota in quotas.items() if len(eligible[g]) > quota
            }
            candidate_indices = schedules[("action_matched", "random")]
            record = dict(
                source=source,
                sampling_seed=seed,
                represented_action_groups=len(quotas),
                groups_with_content_choice=len(choice_groups),
                presentations_in_groups_with_choice=sum(
                    groups[i] in choice_groups for i in candidate_indices
                )
                / len(candidate_indices),
                distinct_support_in_groups_with_choice=sum(
                    quotas[g] for g in choice_groups
                )
                / 256,
            )
            for condition in ("task_matched", "action_matched"):
                a, b = [schedules[(condition, p)] for p in ("random", "min_nll")]
                support_a, support_b = [
                    {content[i] for i in indices} for indices in (a, b)
                ]
                if (
                    len(a) != 768
                    or len(b) != 768
                    or len(support_a) != 256
                    or len(support_b) != 256
                ):
                    raise ValueError("Original/candidate comparison changed budget")
                overlap = len(support_a & support_b)
                record[condition + "_support_replaced_fraction"] = 1 - overlap / 256
                record[condition + "_different_content_position_fraction"] = sum(
                    content[i] != content[j] for i, j in zip(a, b)
                ) / len(a)
                for policy, indices, support in (
                    ("random", a, support_a),
                    ("min_nll", b, support_b),
                ):
                    record[f"{condition}_{policy}_presented_nll"] = sum(
                        mean_nll[content[i]] for i in indices
                    ) / len(indices)
                    record[f"{condition}_{policy}_support_nll"] = sum(
                        mean_nll[key] for key in support
                    ) / len(support)
                    record[f"{condition}_{policy}_target_tokens"] = sum(
                        scores[i]["target_tokens"] for i in indices
                    )
                record[condition + "_presented_nll_difference"] = (
                    record[f"{condition}_min_nll_presented_nll"]
                    - record[f"{condition}_random_presented_nll"]
                )
            results.append(record)
    means = {
        source: {
            key: sum(r[key] for r in results if r["source"] == source) / 4
            for key in results[0]
            if key not in ("source", "sampling_seed")
        }
        for source in ("api", "generic")
    }
    out.mkdir()
    with (out / "selection-strength.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = dict(
        status="candidate_contrast_measured",
        candidate_only=True,
        gpu_used=False,
        model_calls=0,
        formal_protocol_frozen=False,
        source_means=means,
        sampling_seed_rows=results,
        provenance=provenance,
        code_sha256=file_sha(__file__),
        definitions=dict(
            content="Task ID plus canonical student-message SHA; duplicate NLL values averaged before all comparisons.",
            support_replaced_fraction="One minus intersection/256; both supports have exactly 256 contents.",
            groups_with_content_choice="Represented task/action strata whose eligible distinct contents exceed the fixed quota.",
            interpretation="Training-input description only, not an outcome. Action matching may change both action distribution and the strength of NLL/content selection; shrinking a later performance gap alone does not identify mediation.",
        ),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for x, source in enumerate(("api", "generic")):
        for offset, condition, color in (
            (-0.18, "task_matched", "#2879ad"),
            (0.18, "action_matched", "#d46a30"),
        ):
            for ax, metric, scale in (
                (axes[0], "support_replaced_fraction", 100),
                (axes[1], "presented_nll_difference", 1),
            ):
                values = [
                    scale * r[f"{condition}_{metric}"]
                    for r in results
                    if r["source"] == source
                ]
                mean = sum(values) / len(values)
                ax.bar(
                    x + offset,
                    mean,
                    width=0.3,
                    color=color,
                    alpha=0.55,
                    label=condition if x == 0 else None,
                )
                ax.scatter(
                    [x + offset + 0.04 * (j - 1.5) for j in range(4)],
                    values,
                    s=25,
                    color=color,
                    edgecolors="white",
                    linewidths=0.4,
                    zorder=3,
                )
                ax.plot(
                    [x + offset - 0.1, x + offset + 0.1],
                    [mean, mean],
                    color="black",
                    linewidth=2,
                )
    for ax in axes:
        ax.set(xticks=[0, 1], xticklabels=["API source", "Generic source"])
        ax.grid(axis="y", alpha=0.2)
    axes[0].set(ylabel="Different support contents (%)", ylim=(0, 100))
    axes[0].legend(fontsize=8)
    axes[1].set(ylabel="Presented base NLL: min NLL minus random")
    axes[1].axhline(0, color="black", linewidth=0.8)
    fig.suptitle(
        "PREPARATION ONLY: remaining content-selection contrast after action matching",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.015,
        "Four fixed sampling recipes per source; no training, new inference or evaluation outcomes.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    for extension in ("png", "pdf"):
        fig.savefig(out / ("selection-strength." + extension), dpi=180)
    plt.close(fig)
    print(json.dumps(dict(output=str(out), source_means=means)), flush=True)


if __name__ == "__main__":
    main()
