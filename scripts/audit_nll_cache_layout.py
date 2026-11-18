#!/usr/bin/env python3
"""CPU-only duplicate-score and frozen-selection sensitivity audit; no rescoring."""

from collections import defaultdict
import json
from pathlib import Path
import statistics
import time

from learning_harness.action_matched_sampling import action_matched_schedule
from learning_harness.common import digest, file_sha, read_jsonl
from learning_harness.task_matched_sampling import matched_schedule
from train_lora import fixed_schedule


def audit(source):
    data = Path(f"runs/transfer-factorial-v1/corpora/full-{source}.jsonl")
    cache = Path(f"runs/variance-diagnostic-v1/stage3/base-nll/{source}.json")
    rows = read_jsonl(data)
    record = json.loads(cache.read_text())
    scores = record["scores"]
    if record["data_sha256"] != file_sha(data) or len(scores) != len(rows):
        raise ValueError("Cache/corpus identity mismatch")
    tasks = [r["task_id"] for r in rows]
    content = [digest(r["messages"]) for r in rows]
    families, groups = [], defaultdict(list)
    for index, (row, score) in enumerate(zip(rows, scores)):
        if score["index"] != index or score["task_id"] != row["task_id"]:
            raise ValueError("NLL cache alignment changed")
        if row["partition"] != "train":
            raise ValueError("Only training pool metadata is permitted")
        action = json.loads(row["messages"][-1]["content"])["action"]
        families.append("go" if action.startswith("go to ") else action.split()[0])
        groups[content[index]].append(index)
    values = [s["mean_target_nll"] for s in scores]
    duplicates = []
    for content_sha, indices in groups.items():
        if len(indices) < 2:
            continue
        if (
            len(
                {
                    (scores[i]["input_tokens"], scores[i]["target_tokens"])
                    for i in indices
                }
            )
            != 1
        ):
            raise ValueError("Identical messages have inconsistent encoded lengths")
        lo, hi = (
            min(indices, key=values.__getitem__),
            max(indices, key=values.__getitem__),
        )

        def padded_length(index):
            return max(
                s["input_tokens"]
                for s in scores[(index // 8) * 8 : (index // 8 + 1) * 8]
            )

        duplicates.append(
            dict(
                content_sha256=content_sha,
                source=source,
                copies=len(indices),
                score_span=values[hi] - values[lo],
                extreme_pair_padding_gap=abs(padded_length(hi) - padded_length(lo)),
            )
        )
    spans = [r["score_span"] for r in duplicates]
    same = [r["score_span"] for r in duplicates if r["extreme_pair_padding_gap"] == 0]
    different = [
        r["score_span"] for r in duplicates if r["extreme_pair_padding_gap"] > 0
    ]
    variants = {
        name: [values[groups[k][which]] for k in content]
        for name, which in (("first_copy", 0), ("last_copy", -1))
    }
    sensitivity = []
    inputs = {str(data): file_sha(data), str(cache): file_sha(cache)}
    for stage in ("stage3", "stage4"):
        for seed in (101, 202, 303, 404):
            path = Path(
                f"runs/variance-diagnostic-v1/{stage}/inputs/schedules/min_nll-{source}-s{seed}.json"
            )
            recipe = json.loads(path.read_text())
            inputs[str(path)] = file_sha(path)
            if stage == "stage3":
                reference = fixed_schedule(len(rows), 768, seed)

                def regenerate(vector):
                    return matched_schedule(
                        tasks, reference, seed, "min_nll", vector, content_keys=content
                    )
            else:
                ref_path = Path(recipe["reference_recipe"])
                inputs[str(ref_path)] = file_sha(ref_path)
                reference = json.loads(ref_path.read_text())["indices"]

                def regenerate(vector):
                    return action_matched_schedule(
                        tasks, families, content, reference, seed, "min_nll", vector
                    )

            if regenerate(values) != recipe["indices"]:
                raise ValueError(
                    "The original frozen mean-score schedule did not regenerate"
                )
            original = {content[i] for i in recipe["indices"]}
            for variant, vector in variants.items():
                selected = {content[i] for i in regenerate(vector)}
                if len(selected) != len(original):
                    raise ValueError("Sensitivity changed distinct-support size")
                sensitivity.append(
                    dict(
                        stage=stage,
                        sampling_seed=seed,
                        variant=variant,
                        selected_content=len(original),
                        replaced_content=len(original - selected),
                        retained_fraction=len(original & selected) / len(original),
                    )
                )
    summary = dict(
        duplicate_message_groups=len(duplicates),
        score_span_mean=statistics.mean(spans),
        score_span_median=statistics.median(spans),
        score_span_max=max(spans),
        groups_with_span_above_001=sum(v > 0.01 for v in spans),
        groups_with_span_above_01=sum(v > 0.1 for v in spans),
        same_extreme_pair_padded_length_groups=len(same),
        same_extreme_pair_padded_length_max_span=max(same, default=0),
        different_extreme_pair_padded_length_groups=len(different),
        different_extreme_pair_padded_length_max_span=max(different, default=0),
        all_duplicate_encoded_lengths_identical=True,
        original_frozen_schedules_regenerated=True,
        selection_sensitivity=sensitivity,
    )
    return summary, duplicates, inputs


def render(out, rows, summaries):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    colors = dict(api="#2375B8", generic="#D67B20")
    for source in ("api", "generic"):
        selected = [r for r in rows if r["source"] == source]
        spans = [max(r["score_span"], 1e-10) for r in selected]
        axes[0].hist(
            np.log10(spans),
            bins=np.linspace(-10, 0, 26),
            histtype="step",
            linewidth=1.8,
            label=source.upper(),
            color=colors[source],
        )
        values = summaries[source]["selection_sensitivity"]
        for stage, x in (("stage3", 0), ("stage4", 1)):
            retained = [
                100 * r["retained_fraction"] for r in values if r["stage"] == stage
            ]
            offset = -0.12 if source == "api" else 0.12
            axes[1].scatter(
                x + offset + np.linspace(-0.04, 0.04, len(retained)),
                retained,
                color=colors[source],
                alpha=0.7,
                s=30,
                label=source.upper() if stage == "stage3" else None,
            )
    axes[0].set(
        xlabel="log10(score span for identical message sequences)",
        ylabel="Number of duplicate groups",
        title="A. Existing cache consistency",
    )
    axes[0].text(
        0.02,
        0.96,
        "Zero spans displayed at 1e-10",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8,
    )
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["Task matched", "Task + action matched"],
        ylabel="Selected content retained vs frozen mean (%)",
        ylim=(96.5, 100.4),
        title="B. First/last-copy score sensitivity",
    )
    for ax in axes:
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("CPU audit of cached base-NLL scores; no new model calls", fontsize=13)
    fig.savefig(out / "cache-consistency.png", dpi=180)
    fig.savefig(out / "cache-consistency.pdf")
    plt.close(fig)


def main():
    out = Path("output/nll-cache-layout-audit/20261113")
    if (out / "aggregate-summary.json").exists():
        raise ValueError("Preserve the existing cache audit")
    out.mkdir(parents=True, exist_ok=True)
    summaries, duplicates, inputs = {}, [], {}
    for source in ("api", "generic"):
        summary, rows, hashes = audit(source)
        summaries[source] = summary
        duplicates.extend(rows)
        inputs.update(hashes)
    local = Path("runs/nll-cache-layout-audit/20261113")
    local.mkdir(parents=True, exist_ok=True)
    (local / "duplicate-groups.json").write_text(
        json.dumps(duplicates, indent=2) + "\n"
    )
    result = dict(
        status="cache_duplicate_and_selection_sensitivity_audited",
        sources=summaries,
        checked_unix=time.time(),
        inputs_sha256=inputs,
        script_sha256=file_sha(__file__),
        gpu_used=False,
        actual_new_model_calls=0,
        teacher_calls=0,
        raw_records_uploaded=False,
        original_schedule_files_modified=False,
        interpretation="Layout association in existing cache only, not a causal diagnosis or GPU rescoring. First/last-copy sensitivity is not a bound on the effect of independently rescored NLL. Original frozen study remains unchanged.",
    )
    render(out, duplicates, summaries)
    (out / "aggregate-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                status=result["status"],
                duplicate_groups=len(duplicates),
                retained_fraction_min=min(
                    r["retained_fraction"]
                    for s in summaries.values()
                    for r in s["selection_sensitivity"]
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
