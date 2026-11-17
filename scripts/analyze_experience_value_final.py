#!/usr/bin/env python3
"""Post-development descriptive analysis; never opens task or confirmation data."""

from collections import defaultdict
import json
from pathlib import Path

from learning_harness.common import file_sha
from learning_harness.experience_value_study import INITIALIZATIONS, POLICIES, SAMPLING, SOURCES


OUT = Path("output/experience-value-final-analysis-v1")
PARENT = Path("output/experience-value-development-v1/aggregate-summary.json")
AUDIT = Path("output/experience-value-completion-audit-v1/local-audit.json")


def paired_contrasts(models, initializations=INITIALIZATIONS, sampling=SAMPLING):
    cells = {}
    for record in models:
        case = record["case"]
        if case["name"] == "base":
            continue
        key = (
            case["source"], case["policy"],
            case["initialization_seed"], case["sampling_seed"],
        )
        if key in cells:
            raise ValueError("Duplicate experimental cell")
        cells[key] = record["mean_success"]
    expected = {
        (source, policy, seed, sample)
        for source in SOURCES for policy in POLICIES
        for seed in initializations for sample in sampling
    }
    if set(cells) != expected:
        raise ValueError("The complete crossed matrix is required")
    result = {}
    for comparator in ("random", "high", "gradient_low"):
        per_source = {}
        for source in SOURCES:
            per_source[source] = {
                str(seed): sum(
                    cells[source, "gradient_high", seed, sample]
                    - cells[source, comparator, seed, sample]
                    for sample in sampling
                ) / len(sampling)
                for seed in initializations
            }
        pooled = {
            str(seed): sum(per_source[s][str(seed)] for s in SOURCES) / len(SOURCES)
            for seed in initializations
        }
        result[comparator] = dict(
            per_source=per_source, equal_source_per_initialization=pooled,
            mean_difference=sum(pooled.values()) / len(pooled),
            positive_initializations=sum(v > 0 for v in pooled.values()),
            zero_initializations=sum(v == 0 for v in pooled.values()),
            negative_initializations=sum(v < 0 for v in pooled.values()),
        )
    return result


def main():
    audit = json.loads(AUDIT.read_text())
    if audit["final_summary_sha256"] != file_sha(PARENT):
        raise ValueError("Final report no longer matches the local integrity audit")
    summary = json.loads(PARENT.read_text())
    if not summary["complete"] or summary["interim"] or summary["completed_students"] != 224:
        raise ValueError("A completed final report is required")
    contrasts = paired_contrasts(summary["models"])
    for comparator, values in contrasts.items():
        if values["mean_difference"] != summary["endpoints"]["contrasts"][
            "gradient_high_minus_" + comparator
        ]["mean_difference"]:
            raise ValueError("Paired initialization reconstruction differs")
    budgets = defaultdict(list)
    family = defaultdict(list)
    for record in summary["models"]:
        case = record["case"]
        if case["name"] == "base":
            continue
        training = record["training"]
        if training["presentations"] != 768 or training["optimizer_steps"] != 192:
            raise ValueError("The common presentation budget changed")
        budgets[(case["source"], case["policy"])].append(training["input_tokens"])
        for name, counts in record["families"].items():
            family[(case["source"], case["policy"], name)].append(
                counts["successes"] / counts["tasks"]
            )
    result = dict(
        analysis_kind="post-development descriptive analysis",
        final_summary_sha256=file_sha(PARENT), local_audit_sha256=file_sha(AUDIT),
        models=225, students=224, evaluation_episodes=28800,
        independent_initialization_count=8, sampling_replicates_per_initialization=2,
        independent_experience_pool_count=2, inferential_tests_performed=False,
        confirmation_evaluations=0, contrasts=contrasts,
        input_tokens={
            source: {
                policy: dict(mean=sum(budgets[source, policy]) / 16,
                             minimum=min(budgets[source, policy]),
                             maximum=max(budgets[source, policy]))
                for policy in POLICIES
            } for source in SOURCES
        },
        family_success={
            source: {policy: {
                name: sum(values) / len(values)
                for (src, pol, name), values in family.items() if src == source and pol == policy
            } for policy in POLICIES} for source in SOURCES
        },
        scope="Eight initializations, two reused experience pools, one backbone/environment; no confirmatory significance or equivalence claim.",
    )
    if OUT.exists():
        raise ValueError("Preserve the previous analysis snapshot")
    OUT.mkdir()
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), sharey=True)
    labels = {"random": "Random", "high": "High NLL", "gradient_low": "Gradient low"}
    for ax, (comparator, values) in zip(axes, contrasts.items()):
        for i, source in enumerate((*SOURCES, "equal-source")):
            per_seed = (values["equal_source_per_initialization"] if source == "equal-source"
                        else values["per_source"][source])
            points = 100 * np.array(list(per_seed.values()))
            offsets = np.linspace(-0.14, 0.14, len(points))
            color = ("#147d92", "#c85b39", "#6554a4")[i]
            ax.scatter(i + offsets, points, color=color, s=24, alpha=0.8)
            ax.scatter(i, points.mean(), marker="D", color="black", s=40, zorder=5)
            ax.annotate(f"{points.mean():+.2f}", (i, points.mean()),
                        xytext=(8, 3), textcoords="offset points", fontsize=8)
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set(xticks=[0, 1, 2], xticklabels=["API", "Generic", "Equal-source"],
               title="Gradient high − " + labels[comparator], ylim=(-9, 13))
        ax.grid(axis="y", alpha=0.15)
    axes[0].set_ylabel("Success difference (percentage points)")
    fig.suptitle("Complete development study: each dot is one initialization, averaged over two sampling seeds")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / ("paired-initializations." + suffix), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(dict(status="descriptive_analysis_complete", output=str(OUT))))


if __name__ == "__main__":
    main()
