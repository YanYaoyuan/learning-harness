#!/usr/bin/env python3
"""Prepare candidate range controls without training or reading fresh outcomes."""

from collections import Counter
import json
from pathlib import Path

from canonical_action_controls import ROOT as PARENT, aligned_scores, digest_messages
from learning_harness.common import file_sha, read_jsonl
from learning_harness.bounded_queue import save
from learning_harness.nll_range_sampling import range_schedule

ROOT = Path("runs/nll-range-controls-v1")
OUT = Path("output/nll-range-controls-v1")


def main():
    if ROOT.exists() or OUT.exists():
        raise ValueError("Preserve the previous preparation; do not overwrite")
    ledger = json.loads((PARENT / "execution.json").read_text())
    if ledger["status"] != "completed" or ledger["open"]:
        raise ValueError("Canonical replication must be complete")
    provenance = {str(PARENT / "freeze.json"): file_sha(PARENT / "freeze.json")}
    report = []
    pending = []
    for source in ("api", "generic"):
        data = Path(f"runs/transfer-factorial-v1/corpora/full-{source}.jsonl")
        score_path = PARENT / "inputs/scores" / f"{source}.json"
        rows = read_jsonl(data)
        if any(r["partition"] != "train" for r in rows):
            raise ValueError("Only historical training data may be selected")
        scores = aligned_scores(
            json.loads(score_path.read_text()), rows, file_sha(data)
        )
        tasks = [r["task_id"] for r in rows]
        commands = [json.loads(r["messages"][-1]["content"])["action"] for r in rows]
        actions = ["go" if a.startswith("go to ") else a.split()[0] for a in commands]
        content = [digest_messages(r["messages"]) for r in rows]
        strata = list(zip(tasks, actions))
        provenance.update(
            {str(data): file_sha(data), str(score_path): file_sha(score_path)}
        )
        for seed in (101, 202, 303, 404):
            reference_path = (
                PARENT / "inputs/schedules" / f"random-{source}-s{seed}.json"
            )
            low_path = reference_path.with_name(f"min_nll-{source}-s{seed}.json")
            reference = json.loads(reference_path.read_text())["indices"]
            policies = {}
            budget = Counter((strata[i], content[i]) for i in reference)
            quota = Counter(s for s, key in budget)
            for policy in ("random", "low", "middle", "high"):
                indices = range_schedule(
                    tasks, actions, content, reference, seed, policy, scores
                )
                if (
                    [strata[i] for i in indices] != [strata[i] for i in reference]
                    or Counter(
                        s for s, key in set((strata[i], content[i]) for i in indices)
                    )
                    != quota
                    or len(indices) != 768
                    or sum(quota.values()) != 256
                ):
                    raise ValueError("Range control changed task/action/support budget")
                if policy in ("random", "low"):
                    original = json.loads(
                        (reference_path if policy == "random" else low_path).read_text()
                    )
                    if indices != original["indices"]:
                        raise ValueError(
                            "Existing canonical low/random control changed"
                        )
                selected = set((tasks[i], content[i]) for i in indices)
                policies[policy] = dict(
                    mean_selected_nll=sum(scores[i] for i in indices) / len(indices),
                    distinct_support=len(selected),
                )
                path = ROOT / "candidate-schedules" / f"{policy}-{source}-s{seed}.json"
                pending.append(
                    (
                        path,
                        dict(
                            candidate_only=True,
                            training_started=False,
                            source=source,
                            sampling_seed=seed,
                            policy=policy,
                            data=str(data),
                            data_sha256=file_sha(data),
                            score_file=str(score_path),
                            score_sha256=file_sha(score_path),
                            reference_recipe=str(reference_path),
                            reference_recipe_sha256=file_sha(reference_path),
                            indices=indices,
                            presentations=768,
                            distinct_support=256,
                            definition="Middle minimizes distance from the within-task/action median over distinct content; high maximizes mean canonical target NLL.",
                        ),
                    )
                )
            provenance.update(
                {
                    str(reference_path): file_sha(reference_path),
                    str(low_path): file_sha(low_path),
                }
            )
            report.append(
                dict(
                    source=source,
                    recipe=seed,
                    policies=policies,
                    all_budget_controls_passed=True,
                    random_and_low_identical_to_completed_canonical_study=True,
                )
            )
    for path, recipe in pending:
        save(path, recipe)
    provenance.update({str(p): file_sha(p) for p, _ in pending})
    for path in (
        "scripts/prepare_nll_range_controls.py",
        "src/learning_harness/nll_range_sampling.py",
        "tests/test_nll_range_sampling.py",
    ):
        provenance[path] = file_sha(path)
    save(ROOT / "preparation.private.json", dict(provenance=provenance))
    summary = dict(
        status="candidate_schedules_prepared_not_trained",
        schedules=len(pending),
        model_calls=0,
        training_runs=0,
        fresh_query_evaluations=0,
        rows=report,
        provenance_sha256=file_sha(ROOT / "preparation.private.json"),
        scope="Selection controls prepared on historical training pools; a separate protocol must fix fresh evaluation and the gradient baseline before training.",
    )
    save(OUT / "preparation-summary.json", summary)
    print(
        json.dumps(dict(status=summary["status"], schedules=len(pending))), flush=True
    )


if __name__ == "__main__":
    main()
