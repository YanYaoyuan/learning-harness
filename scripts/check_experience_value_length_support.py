#!/usr/bin/env python3
"""CPU-only length-matching feasibility on frozen historical training pools.

Never runs students, queries confirmation tasks, or publishes example identities.
Bin widths and gates are fixed here before any matched-length student outcome.
"""

import json
from pathlib import Path

from learning_harness.action_matched_sampling import action_matched_schedule
from learning_harness.common import file_sha
from learning_harness.experience_value_study import arrays
import run_experience_value_development as parent


OUT = Path("output/experience-value-length-feasibility-v1")
WIDTHS = (0, 8, 16, 32)
POLICIES = ("random", "high", "gradient_high", "gradient_low")


def schedules(rows, lengths, targets, nll, gradient, reference, seed, width):
    tasks, actions, content = arrays(rows)
    length_groups = [
        f"{action}|input={length if width == 0 else length // width}|target={target}"
        for action, length, target in zip(actions, lengths, targets)
    ]
    result = {}
    for policy in POLICIES:
        values = (
            [-v for v in gradient] if policy == "gradient_high" else gradient
            if policy == "gradient_low" else [-v for v in nll]
        )
        result[policy] = action_matched_schedule(
            tasks, length_groups, content, reference, seed,
            "random" if policy == "random" else "min_nll", values,
        )
        indices = result[policy]
        if (
            [tasks[i] for i in indices] != [tasks[i] for i in reference]
            or [actions[i] for i in indices] != [actions[i] for i in reference]
            or [length_groups[i] for i in indices] != [length_groups[i] for i in reference]
            or len({(tasks[i], content[i]) for i in indices}) != 256
            or sum(targets[i] for i in indices) != sum(targets[i] for i in reference)
        ):
            raise ValueError("Length matching changed the reference constraints")
    return result


def main():
    if OUT.exists():
        raise ValueError("Preserve the existing feasibility snapshot")
    parent.validate()
    from transformers import AutoTokenizer
    from train_lora import encode_example

    tokenizer = AutoTokenizer.from_pretrained(
        parent.engine.MODEL, local_files_only=True, trust_remote_code=False
    )
    records = []
    for source in parent.SOURCES:
        rows, nll, gradient, meta = parent.scores_for(source)
        content = arrays(rows)[2]
        encoded = [encode_example(tokenizer, row, 2048) for row in rows]
        if any(value is None for value in encoded):
            raise ValueError("Eligible content unexpectedly dropped")
        lengths = [len(row["input_ids"]) for row in encoded]
        targets = [sum(value != -100 for value in row["labels"]) for row in encoded]
        for seed in parent.SAMPLING:
            reference = parent.read(parent.recipe_path(source, "random", seed))["indices"]
            old_hi = parent.read(parent.recipe_path(source, "gradient_high", seed))["indices"]
            old_lo = parent.read(parent.recipe_path(source, "gradient_low", seed))["indices"]

            def score(indices):
                return sum(gradient[i] for i in indices) / len(indices)

            original_gap = score(old_hi) - score(old_lo)
            for width in WIDTHS:
                plans = schedules(rows, lengths, targets, nll, gradient, reference, seed, width)
                tokens = {p: sum(lengths[i] for i in indices) for p, indices in plans.items()}
                maximum_imbalance = (max(tokens.values()) - min(tokens.values())) / tokens["random"]
                retained = ((score(plans["gradient_high"]) - score(plans["gradient_low"]))
                            / original_gap if original_gap > 0 else 0)
                records.append(dict(
                    source=source, sampling_seed=seed, input_bin_width=width,
                    input_tokens=tokens, target_tokens=sum(targets[i] for i in reference),
                    max_input_token_imbalance=maximum_imbalance,
                    original_gradient_gap=original_gap, retained_gradient_gap_fraction=retained,
                    high_low_distinct_selected_content=len(
                        {content[i] for i in plans["gradient_high"]}
                        ^ {content[i] for i in plans["gradient_low"]}
                    ),
                    passes=maximum_imbalance <= 0.01 and retained >= 0.25,
                ))
    feasible = [width for width in WIDTHS if all(
        r["passes"] for r in records if r["input_bin_width"] == width
    )]
    result = dict(
        status="cpu_feasibility_complete", parent_freeze_sha256=file_sha(parent.ROOT / "freeze.json"),
        code_sha256=file_sha(Path(__file__)), source_rows=dict(api=721, generic=732),
        widths_checked=list(WIDTHS), thresholds=dict(max_input_imbalance=0.01, min_retained_score_gap=0.25),
        selection_rule="finest predeclared bin width passing every source and sampling condition",
        selected_bin_width=feasible[0] if feasible else None,
        feasible=bool(feasible), records=records,
        gpu_work_started=False, confirmation_evaluations=0, raw_records_uploaded=False,
        scope="Candidate support only; retained score gap is not a measured learning effect.",
    )
    OUT.mkdir()
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(feasible=result["feasible"], selected_bin_width=result["selected_bin_width"])))


if __name__ == "__main__":
    main()
