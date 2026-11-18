#!/usr/bin/env python3
"""Read-only paired snapshots for the future frozen action-control observer."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from action_control_report import collect, render_policies, load_study, output_path
from learning_harness.bounded_queue import save
from learning_harness.common import file_sha
from learning_harness.controlled_study import completed_policy_pairs
from learning_harness.research_aggregates import aggregate_report


def inventory(root, frozen):
    freeze_sha = file_sha(root / "freeze.json")
    records = []
    for case in frozen["cases"]:
        path = root / "cases" / case["name"] / "worker-result.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result["freeze_sha256"] != freeze_sha:
                raise ValueError("Interim worker belongs to a different frozen study")
            records.append(result)
    keys, selected = completed_policy_pairs(frozen, records)
    return freeze_sha, records, keys, selected


def create_snapshot(root, out, frozen, node):
    freeze_sha, records, keys, selected = inventory(root, frozen)
    if len(keys) < node:
        raise ValueError("Not enough fully paired cells for this node")
    folder = out / "interim" / f"node-{node:02}"
    if folder.exists():
        raise ValueError("Existing interim snapshot must be preserved")
    # Reuse the complete report's weight, schedule and raw-episode
    # audits, restricted to fully paired primary cases. No inference.
    subset = dict(frozen, cases=selected, planned_episodes=64 * len(selected))
    rows, _, tasks = collect(root, subset)
    local_subset = dict(
        complete=True,
        cases=len(rows),
        actual_evaluation_episodes=len(tasks),
        freeze_sha256=freeze_sha,
        teacher_calls=0,
        official_tests_used=False,
        results=rows,
    )
    aggregate = aggregate_report(local_subset, subset, freeze_sha)
    aggregate.update(
        complete=False,
        subset_complete=True,
        interim=True,
        final_study_complete=False,
        milestone=node,
        actual_paired_cells=len(keys),
        completed_models=len(records),
        planned_models=len(frozen["cases"]),
        monitoring_only=True,
        note="All fully paired cells at first observation of the prespecified node; no inference, early stop or scheduling change.",
    )
    folder.mkdir(parents=True, exist_ok=False)
    render_policies(
        rows,
        folder,
        f"INTERIM: {len(keys)} paired cells; {len(records)}/{len(frozen['cases'])} models complete",
    )
    save(folder / "aggregate-summary.json", aggregate)
    local_sources = {
        str(root / "cases" / c["name"] / "worker-result.json"): file_sha(
            root / "cases" / c["name"] / "worker-result.json"
        )
        for c in selected
    }
    save(
        folder / "local-provenance.json",
        dict(
            freeze_sha256=freeze_sha,
            worker_results=local_sources,
            primary_seed_cells=keys,
        ),
    )
    names = (
        "aggregate-summary.json",
        "action-matched-policies.png",
        "action-matched-policies.pdf",
    )
    save(
        folder / "local-manifest.json",
        {name: file_sha(folder / name) for name in names},
    )
    return folder


def observe_interim(root, out, frozen, run, wandb):
    nodes = frozen.get("interim_cell_milestones", [])
    if not nodes:
        return
    state_path = out / "interim/observer-state.json"
    state = (
        json.loads(state_path.read_text()) if state_path.exists() else dict(looks={})
    )
    pending = [
        n
        for n in nodes
        if state["looks"].get(str(n), {}).get("cloud_status") != "uploaded"
    ]
    if not pending:
        return
    freeze_sha, _, keys, _ = inventory(root, frozen)
    for node in pending:
        if len(keys) < node:
            continue
        folder = out / "interim" / f"node-{node:02}"
        if not folder.exists():
            # Plot in the existing model/analysis environment; the W&B runtime
            # deliberately has no numpy, matplotlib or model dependencies.
            with (out / f"interim-node-{node:02}.log").open("a") as log:
                subprocess.run(
                    [
                        str(Path(".venv-model/bin/python").absolute()),
                        "scripts/action_control_interim.py",
                        "--root",
                        str(root),
                        "--node",
                        str(node),
                    ],
                    env=dict(
                        os.environ,
                        CUDA_VISIBLE_DEVICES="",
                        PYTHONPATH=str(Path("src").absolute()),
                        MPLCONFIGDIR="/tmp/lh-mpl",
                    ),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=180,
                )
        aggregate = json.loads((folder / "aggregate-summary.json").read_text())
        manifest = json.loads((folder / "local-manifest.json").read_text())
        names = (
            "aggregate-summary.json",
            "action-matched-policies.png",
            "action-matched-policies.pdf",
        )
        if (
            aggregate["freeze_sha256"] != freeze_sha
            or set(manifest) != set(names)
            or any(file_sha(folder / name) != manifest[name] for name in names)
        ):
            raise ValueError("Cached interim snapshot changed; preserve it for review")
        try:
            run.log(
                {
                    f"interim/node_{node}/paired-policies": wandb.Image(
                        str(folder / "action-matched-policies.png")
                    )
                }
            )
            artifact = wandb.Artifact(
                f"task-action-interim-{freeze_sha[:12]}-node-{node}",
                type="descriptive-interim-results",
            )
            for name in names:
                artifact.add_file(str(folder / name), name=name)
            run.log_artifact(artifact).wait()
            run.summary["interim/latest_aggregate"] = aggregate
            state["looks"][str(node)] = dict(
                directory=str(folder),
                actual_paired_cells=aggregate["actual_paired_cells"],
                cloud_status="uploaded",
                raw_records_uploaded=False,
                run_id=run.id,
                run_url=run.url,
            )
        except Exception as error:
            # Retry only the same immutable upload on the next observer tick;
            # never restart a model or rebuild a snapshot from later outcomes.
            state["looks"][str(node)] = dict(
                directory=str(folder),
                cloud_status="failed",
                error_type=type(error).__name__,
                raw_records_uploaded=False,
            )
        save(state_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--node", type=int, required=True)
    args = parser.parse_args()
    frozen = load_study(args.root)
    if args.node not in frozen["interim_cell_milestones"]:
        raise ValueError("Node was not prespecified in the frozen plan")
    folder = create_snapshot(args.root, output_path(args.root), frozen, args.node)
    print(
        json.dumps(dict(snapshot=str(folder), interim=True, new_model_calls=0)),
        flush=True,
    )


if __name__ == "__main__":
    main()
