#!/usr/bin/env python3
"""Formalize a reviewed follow-up plan after the complete parent report exists."""

import argparse
import json
from pathlib import Path
import time

from action_control_report import REQUIRED_CODE, load_study
import action_control_worker as worker
from learning_harness.common import file_sha
from learning_harness.controlled_study import validate_plan


def finished_parent():
    frozen = worker.parent.load_freeze()
    ledger = json.loads((worker.parent.ROOT / "execution.json").read_text())
    report_path = worker.parent.OUT / "summary.json"
    if ledger["status"] != "completed" or not report_path.is_file():
        raise ValueError("Parent execution and complete local report must finish first")
    report = json.loads(report_path.read_text())
    freeze_sha = file_sha(worker.parent.ROOT / "freeze.json")
    if (
        report["complete"] is not True
        or report["cases"] != len(frozen["cases"])
        or report["actual_evaluation_episodes"] != frozen["planned_episodes"]
        or report["freeze_sha256"] != freeze_sha
        or ledger["freeze_sha256"] != freeze_sha
        or ledger.get("open")
        or any(c.get("status") != "completed" for c in ledger["cases"].values())
        or set(ledger["cases"]) != {c["name"] for c in frozen["cases"]}
    ):
        raise ValueError("Parent ledger, frozen design and complete report disagree")
    return frozen, report_path


def freeze(root, proposal_path, review_path):
    # All readiness checks precede any output-directory creation.
    parent, report_path = finished_parent()
    relative = root.resolve().relative_to(Path("runs").resolve())
    if not relative.parts or root.exists():
        raise ValueError(
            "Use a new study directory under runs; never overwrite a study"
        )
    root = Path("runs") / relative
    proposal = json.loads(proposal_path.read_text())
    design = validate_plan(proposal)
    if (
        proposal["proposal_state"] != "draft"
        or proposal["report_script"] != "scripts/action_control_report.py"
        or proposal["observer_script"] != "scripts/track_action_controls.py"
        or proposal["cloud_scope"] != "aggregate_only"
        or proposal["cap_gpu_hours"] > 18
    ):
        raise ValueError(
            "Explicit draft, aggregate-only observer and bounded plan required"
        )
    milestones = proposal["interim_cell_milestones"]
    if (
        not milestones
        or any(
            type(n) is not int or n < 1 or n > len(design["primary_seed_cells"])
            for n in milestones
        )
        or sorted(set(milestones)) != milestones
    ):
        raise ValueError(
            "Prespecified unique ascending paired-cell monitoring nodes required"
        )
    review = json.loads(review_path.read_text())
    if (
        review["decision"] != "run_task_action_controls"
        or review["parent_report_sha256"] != file_sha(report_path)
        or review["proposal_sha256"] != file_sha(proposal_path)
        or review["parent_freeze_sha256"]
        != file_sha(worker.parent.ROOT / "freeze.json")
        or review["complete_parent_results_reviewed"] is not True
    ):
        raise ValueError(
            "Record a review bound to the exact complete parent and proposed plan"
        )
    # Regenerate every recipe before creating a formal directory. Proposed
    # schedule paths are replaced with paths under this new study root.
    recipes, cases = {}, []
    for case in proposal["cases"]:
        key = (case["policy"], case["source"], case["sampling_seed"])
        name = f"{key[0]}-{key[1]}-s{key[2]}.json"
        candidate_path = (
            Path(
                "output/variance-diagnostic-v1/task-action-feasibility/candidate-schedules"
            )
            / name
        )
        recipe, _ = worker.load_recipe(candidate_path)
        if recipe.get("candidate_only") is not True or any(
            recipe[k] != case[k] for k in ("source", "policy", "sampling_seed")
        ):
            raise ValueError("Expected an immutable untrained candidate recipe")
        formal = dict(
            recipe,
            candidate_only=False,
            formalized_from=str(candidate_path),
            candidate_recipe_sha256=file_sha(candidate_path),
        )
        formal.pop("training_started", None)
        recipes[name] = (formal, candidate_path)
        cases.append(dict(case, schedule=str(root / "inputs/schedules" / name)))
    new_plan = {
        k: v
        for k, v in proposal.items()
        if k not in ("proposal_state", "status", "cases", "files")
    }
    new_plan.update(
        cases=cases,
        frozen_unix=time.time(),
        parent_stage3_freeze_sha256=file_sha(worker.parent.ROOT / "freeze.json"),
        proposal_sha256=file_sha(proposal_path),
        parent_review_sha256=file_sha(review_path),
        preparatory_gpu_seconds=0,
    )
    validate_plan(new_plan)
    root.mkdir(parents=True, exist_ok=False)
    (root / "inputs/schedules").mkdir(parents=True)
    for name, (recipe, _) in recipes.items():
        (root / "inputs/schedules" / name).write_text(
            json.dumps(recipe, indent=2) + "\n"
        )
    config_path = root / "a-lora.json"
    config_path.write_bytes((worker.parent.ROOT / "a-lora.json").read_bytes())
    paths = (
        set(parent["files"])
        | REQUIRED_CODE
        | {
            str(
                Path(__file__).relative_to(Path.cwd())
                if Path(__file__).is_absolute()
                else Path(__file__)
            ),
            str(worker.parent.ROOT / "freeze.json"),
            str(report_path),
            str(config_path),
            str(proposal_path),
            str(review_path),
        }
    )
    paths |= {str(root / "inputs/schedules" / name) for name in recipes}
    paths |= {str(candidate) for _, candidate in recipes.values()}
    new_plan["files"] = {p: file_sha(p) for p in sorted(paths)}
    with (root / "freeze.json").open("x") as stream:
        stream.write(json.dumps(new_plan, indent=2) + "\n")
    load_study(root)
    return dict(
        status="frozen_and_hash_validated",
        root=str(root),
        cases=len(cases),
        episodes=new_plan["planned_episodes"],
        freeze_sha256=file_sha(root / "freeze.json"),
        training_started=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.root, args.proposal, args.review)), flush=True)


if __name__ == "__main__":
    main()
