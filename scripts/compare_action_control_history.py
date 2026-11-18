#!/usr/bin/env python3
"""Post-hoc historical-reference audit of an immutable action-control snapshot."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from action_control_report import load_study
from learning_harness.common import file_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Preserve the existing historical comparison")
    root = Path("runs/variance-diagnostic-v1/stage4")
    load_study(root)
    manifest = json.loads((args.snapshot / "local-manifest.json").read_text())
    for name, expected in manifest.items():
        if file_sha(args.snapshot / name) != expected:
            raise ValueError("Immutable snapshot changed")
    snap = json.loads((args.snapshot / "aggregate-summary.json").read_text())
    provenance = json.loads((args.snapshot / "local-provenance.json").read_text())
    old_root = Path("runs/variance-diagnostic-v1/stage3")
    old_report = Path("output/variance-diagnostic-v1/stage3/summary.json")
    old = json.loads(old_report.read_text())
    lookup = {
        (r["source"], r["initialization_seed"], r["sampling_seed"], r["arm"]): r
        for r in old["results"]
        if not r["historical"] and r["repeat"] == 0
    }
    rows = []
    controls = []
    excluded = set()
    for name, expected in provenance["worker_results"].items():
        if file_sha(name) != expected:
            raise ValueError("Snapshot worker record changed")
        result = json.loads(Path(name).read_text())
        case = result["case"]
        key = (case["source"], case["initialization_seed"], case["sampling_seed"])
        if any((*key, p) not in lookup for p in ["random", "min_nll"]):
            excluded.add(key[1:])
            continue
        parent = lookup[(*key, case["policy"])]
        previous = json.loads(
            (old_root / "cases" / parent["name"] / "worker-result.json").read_text()
        )
        before = json.loads(
            (old_root / "cases" / parent["name"] / "A/experiment.json").read_text()
        )
        after = json.loads((Path(name).parent / "A/experiment.json").read_text())
        if (
            not previous["complete"]
            or previous["mean_success"] != parent["mean_success"]
            or previous["mean_nll"] != parent["mean_nll"]
        ):
            raise ValueError("Historical worker and completed parent report disagree")
        same = dict(
            initial_parameters=previous["control"]["initial_parameter_sha256"]
            == result["control"]["initial_parameter_sha256"],
            trainer=previous["control"]["trainer_sha256"]
            == result["control"]["trainer_sha256"],
            optimizer=before["config"] == after["config"],
            data=before["training_data_sha256"] == after["training_data_sha256"],
            presentations=before["processed_examples_rank0"]
            == after["processed_examples_rank0"]
            == 768,
            physical_gpu=previous["case"]["gpu"] == case["gpu"],
        )
        if not all(same.values()):
            raise ValueError("Historical budget/initialization/compute control differs")
        reference = lookup[(*key, "random")]
        ref_result = json.loads(
            (old_root / "cases" / reference["name"] / "worker-result.json").read_text()
        )
        recipe = json.loads(Path(case["schedule"]).read_text())
        if Path(recipe["reference_recipe"]) != Path(
            ref_result["case"]["schedule"]
        ) or recipe["reference_recipe_sha256"] != file_sha(
            ref_result["case"]["schedule"]
        ):
            raise ValueError(
                "Action matching does not use this historical random reference"
            )
        controls.append(same)
        for stage, score in [
            ("task_only", previous["mean_success"]),
            ("task_action", result["mean_success"]),
        ]:
            rows.append(
                dict(
                    source=key[0],
                    initialization_seed=key[1],
                    sampling_seed=key[2],
                    stage=stage,
                    policy=case["policy"],
                    success=score,
                )
            )
    groups = defaultdict(list)
    for r in rows:
        groups[(r["source"], r["stage"], r["policy"])].append(r["success"])
    means = {
        s: {
            stage: {
                p: sum(groups[s, stage, p]) / len(groups[s, stage, p])
                for p in ["random", "min_nll"]
            }
            for stage in ["task_only", "task_action"]
        }
        for s in ["api", "generic"]
    }
    changes = {}
    for s, g in means.items():
        random = 100 * (g["task_action"]["random"] - g["task_only"]["random"])
        low = 100 * (g["task_action"]["min_nll"] - g["task_only"]["min_nll"])
        changes[s] = dict(
            random_change_pp=random,
            min_nll_change_pp=low,
            contrast_change_pp=low - random,
            new_min_nll_minus_historical_random_pp=100
            * (g["task_action"]["min_nll"] - g["task_only"]["random"]),
        )
    args.out.mkdir(parents=True)
    result = dict(
        post_hoc=True,
        interim=True,
        source_snapshot=str(args.snapshot),
        snapshot_summary_sha256=file_sha(args.snapshot / "aggregate-summary.json"),
        historical_report_sha256=file_sha(old_report),
        script_sha256=file_sha(__file__),
        snapshot_completed_models=snap["completed_models"],
        paired_cells=len(
            {(r["initialization_seed"], r["sampling_seed"]) for r in rows}
        ),
        initializations=sorted({r["initialization_seed"] for r in rows}),
        excluded_cells_without_historical_reference=len(excluded),
        means=means,
        absolute_changes=changes,
        all_controls_verified=True,
        new_model_calls=0,
        scope="Historical cross-run descriptive comparison; content support and NLL intervention strength both change. No causal mediation or calibrated inference; original primary study remains unchanged.",
    )
    (args.out / "aggregate-summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    labels = [
        "Task-only\nrandom",
        "Task-only\nlow NLL",
        "Task/action\nrandom",
        "Task/action\nlow NLL",
    ]
    order = [
        ("task_only", "random"),
        ("task_only", "min_nll"),
        ("task_action", "random"),
        ("task_action", "min_nll"),
    ]
    for ax, source in zip(axes, ["api", "generic"]):
        for x, (stage, policy) in enumerate(order):
            values = groups[source, stage, policy]
            ax.scatter(
                [x + (i - (len(values) - 1) / 2) * 0.04 for i in range(len(values))],
                [100 * v for v in values],
                alpha=0.6,
                s=26,
            )
            avg = 100 * means[source][stage][policy]
            ax.plot([x - 0.18, x + 0.18], [avg, avg], color="black", linewidth=3)
            ax.text(x, avg + 3, f"{avg:.2f}%", ha="center", fontsize=9)
        ax.set(
            title=source.upper(),
            xticks=list(range(4)),
            xticklabels=labels,
            ylim=(0, 100),
            ylabel="Same development-task success (%)",
        )
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        f"POST HOC historical comparison: {len(result['initializations'])} initialization(s), {result['paired_cells']} paired cells",
        fontsize=11,
    )
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(args.out / ("historical-reference." + extension), dpi=180)
    plt.close(fig)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
