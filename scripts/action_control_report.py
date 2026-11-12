#!/usr/bin/env python3
"""Audit raw task outcomes and report a complete frozen action-control study."""

import argparse
import csv
import json
import os
from pathlib import Path

import action_control_worker as worker
from learning_harness.adaptation import composition_revision
from learning_harness.bounded_queue import save
from learning_harness.common import file_sha, read_jsonl
from learning_harness.controlled_study import (
    SOURCES,
    POLICIES,
    audit_episodes,
    paired_effects,
    validate_plan,
)

REQUIRED_CODE = {
    "scripts/action_control_worker.py",
    "scripts/action_control_report.py",
    "scripts/run_action_controls.py",
    "scripts/track_action_controls.py",
    "scripts/supervise_research.py",
    "scripts/action_control_interim.py",
    "src/learning_harness/action_control_audit.py",
    "src/learning_harness/action_matched_sampling.py",
    "src/learning_harness/controlled_study.py",
    "src/learning_harness/research_aggregates.py",
}


def output_path(root):
    relative = root.resolve().relative_to(Path("runs").resolve())
    return Path("output") / relative


def load_study(root, require_completed_parent=True):
    worker.parent.load_freeze()
    frozen = json.loads((root / "freeze.json").read_text())
    validate_plan(frozen)
    parent_freeze = worker.parent.ROOT / "freeze.json"
    parent_report = worker.parent.OUT / "summary.json"
    required = REQUIRED_CODE | {
        str(parent_freeze),
        str(parent_report),
        str(root / "a-lora.json"),
    }
    required |= {c["schedule"] for c in frozen["cases"]}
    if (
        not required.issubset(frozen["files"])
        or frozen["report_script"] != "scripts/action_control_report.py"
        or frozen["parent_stage3_freeze_sha256"] != file_sha(parent_freeze)
    ):
        raise ValueError("Study dependencies or parent identity were not frozen")
    for path, expected in frozen["files"].items():
        if file_sha(path) != expected:
            raise ValueError("Frozen study input changed: " + path)
    if json.loads((root / "a-lora.json").read_text()) != json.loads(
        (worker.parent.ROOT / "a-lora.json").read_text()
    ):
        raise ValueError(
            "Action-content intervention changed parent optimizer/model configuration"
        )
    if require_completed_parent:
        state = json.loads((worker.parent.ROOT / "execution.json").read_text())
        if (
            state["status"] != "completed"
            or not json.loads(parent_report.read_text())["complete"]
        ):
            raise ValueError(
                "Parent Stage 3 must finish before follow-up execution/reporting"
            )
    return frozen


def collect(root, frozen):
    rows, exposure, task_rows, initializations = [], [], [], {}
    config = json.loads((root / "a-lora.json").read_text())
    for case in frozen["cases"]:
        directory = root / "cases" / case["name"]
        result = json.loads((directory / "worker-result.json").read_text())
        if (
            not result["complete"]
            or result["smoke"]
            or result["case"] != case
            or result["freeze_sha256"] != file_sha(root / "freeze.json")
        ):
            raise ValueError("Incomplete or mismatched model result")
        recipe, expected = worker.load_recipe(Path(case["schedule"]))
        if recipe.get("candidate_only", True) or any(
            recipe[k] != case[k] for k in ("source", "policy", "sampling_seed")
        ):
            raise ValueError("Candidate recipe cannot enter a completed formal report")
        control = result["control"]
        meta = json.loads((directory / "A/experiment.json").read_text())
        if (
            not control["completed"]
            or control["initialization_seed"] != case["initialization_seed"]
            or control["sampling_seed"] != case["sampling_seed"]
            or control["selection_policy"] != case["policy"]
            or control["external_schedule_sha256"] != file_sha(case["schedule"])
            or control["schedule_sha256"] != recipe["schedule_sha256"]
            or control["trained_adapter_sha256"]
            != file_sha(directory / "A/adapter_model.safetensors")
            or meta["config"] != config
            or meta["training_data_sha256"] != recipe["data_sha256"]
            or meta["processed_examples_rank0"] != recipe["presentations"]
            or result["exposure"]["input_tokens"] != meta["input_tokens_rank0"]
            or result["exposure"]["target_tokens"] != meta["target_tokens_rank0"]
            or any(result["exposure"].get(k) != v for k, v in expected.items())
        ):
            raise ValueError("Training, weights or exact action-control audit failed")
        init = case["initialization_seed"]
        if (
            init in initializations
            and initializations[init] != control["initial_parameter_sha256"]
        ):
            raise ValueError("Initialization differs with policy, source or repeat")
        initializations[init] = control["initial_parameter_sha256"]
        point = directory / "checkpoint"
        composition = json.loads((point / "composition.json").read_text())
        zero_b = (
            worker.kernel.OLD
            / "heat/trajectories/full-api-seed1101/step-0000/B/adapter_model.safetensors"
        )
        if (
            composition["optimizer_step"] != 0
            or not composition["zero_b_reference_verified"]
            or Path(composition["a_adapter"]).resolve() != (directory / "A").resolve()
            or composition["a_adapter_sha256"] != control["trained_adapter_sha256"]
            or composition["base_revision"] != worker.engine.REV
            or file_sha(point / "B/adapter_model.safetensors") != file_sha(zero_b)
        ):
            raise ValueError("Evaluation did not use the fixed A plus zero-B artifact")
        revision = composition_revision(point)
        if revision != result["composition_revision"]:
            raise ValueError("Checkpoint composition changed")
        for family in ("cool", "heat"):
            folder = directory / "evaluation" / family
            episodes = read_jsonl(folder / "episodes.jsonl")
            rate = audit_episodes(
                episodes, read_jsonl(worker.kernel.OLD / family / "b_query.jsonl")
            )
            summary = json.loads((folder / "summary.json").read_text())
            run = json.loads((folder / "run.json").read_text())
            if (
                not summary["complete"]
                or summary["recorded"] != 32
                or summary["metrics"]["none"]["error_count"]
                or summary["metrics"]["none"]["success_rate"] != rate
                or result["success"][family] != rate
                or run["revision"] != revision
            ):
                raise ValueError("Raw episodes, summaries or served revision disagree")
            task_rows.extend(
                dict(
                    name=case["name"],
                    source=case["source"],
                    policy=case["policy"],
                    initialization_seed=init,
                    sampling_seed=case["sampling_seed"],
                    repeat=case["repeat"],
                    family=family,
                    task_id=e["task_id"],
                    task_group=e["task_group"],
                    success=int(e["success"]),
                )
                for e in episodes
            )
        nll = json.loads((directory / "diagnostic-nll.json").read_text())
        if (
            not nll["checkpoint_unchanged"]
            or nll["used_for_updates"]
            or nll["composition_revision"] != revision
            or any(
                nll[f]["data_sha256"]
                != file_sha(worker.kernel.OLD / f / "diagnostic.jsonl")
                for f in ("cool", "heat")
            )
            or result["mean_nll"]
            != sum(nll[f]["task_mean_nll"] for f in ("cool", "heat")) / 2
            or result["mean_success"] != sum(result["success"].values()) / 2
        ):
            raise ValueError("Read-only diagnostic or mean scores disagree")
        rows.append(
            dict(
                name=case["name"],
                source=case["source"],
                policy=case["policy"],
                initialization_seed=init,
                sampling_seed=case["sampling_seed"],
                repeat=case["repeat"],
                gpu=case["gpu"],
                mean_success=result["mean_success"],
                mean_nll=result["mean_nll"],
                cool_success=result["success"]["cool"],
                heat_success=result["success"]["heat"],
            )
        )
        exposure.append(dict(name=case["name"], **result["exposure"]))
    if len(task_rows) != frozen["planned_episodes"]:
        raise ValueError("Actual evaluation total differs from frozen budget")
    return rows, exposure, task_rows


def render_policies(rows, out, title, labels=None):
    effects = paired_effects(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), sharey=True)
    for ax, source in zip(axes, SOURCES):
        cells = effects["mean_success"]["nll_minus_random"][source]["cells"]
        for c in cells:
            pair = [
                next(
                    r
                    for r in rows
                    if r["source"] == source
                    and r["policy"] == p
                    and r["initialization_seed"] == c["initialization_seed"]
                    and r["sampling_seed"] == c["sampling_seed"]
                    and r["repeat"] == 0
                )
                for p in POLICIES
            ]
            ax.plot(
                [0, 1],
                [100 * r["mean_success"] for r in pair],
                "o-",
                alpha=0.45,
                linewidth=1,
                markersize=4,
            )
        means = [
            100
            * sum(
                r["mean_success"]
                for r in rows
                if r["source"] == source and r["policy"] == p and r["repeat"] == 0
            )
            / len(cells)
            for p in POLICIES
        ]
        for x, mean in enumerate(means):
            ax.plot([x - 0.14, x + 0.14], [mean, mean], color="black", linewidth=3)
        ax.set(
            title=source.upper(),
            ylabel="Post-A success (%)",
            ylim=(0, 100),
            xlim=(-0.5, 1.5),
            xticks=[0, 1],
            xticklabels=labels or ["Action-matched random", "Action-matched low NLL"],
        )
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(out / ("action-matched-policies." + extension), dpi=180)
    plt.close(fig)


def generate(root):
    frozen = load_study(root)
    ledger = json.loads((root / "execution.json").read_text())
    if ledger["freeze_sha256"] != file_sha(root / "freeze.json") or ledger[
        "status"
    ] not in ("evaluation_completed", "completed"):
        raise ValueError(
            "Durable execution ledger does not identify this completed study"
        )
    rows, exposure, task_rows = collect(root, frozen)
    effects = paired_effects(rows)
    out = output_path(root)
    out.mkdir(parents=True, exist_ok=True)
    for name, values in (
        ("cell-scores.csv", rows),
        ("task-outcomes.csv", task_rows),
        ("training-exposure.csv", exposure),
    ):
        with (out / name).open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(
                [
                    {
                        k: json.dumps(v, sort_keys=True)
                        if isinstance(v, (dict, list))
                        else v
                        for k, v in row.items()
                    }
                    for row in values
                ]
            )
    render_policies(
        rows,
        out,
        "Same task/action positions and within-stratum distinct-content quotas",
    )
    result = dict(
        complete=True,
        cases=len(rows),
        actual_evaluation_episodes=len(task_rows),
        freeze_sha256=file_sha(root / "freeze.json"),
        results=rows,
        exposure=exposure,
        policy_effects=effects,
        teacher_calls=0,
        official_tests_used=False,
        uncertainty="All paired seed cells and per-initialization values; no calibrated confidence intervals",
        aggregation="Equal weight per prespecified primary seed cell; equal cool/heat task families within each model",
        scope="Conditional on frozen random-reference action coverage; no full causal mediation claim",
    )
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# 任务与动作构成匹配：完整结果",
        "",
        f"完成{len(rows)}个模型、{len(task_rows)}个评测回合，逐任务原始成败与检查点、调度和诊断核对一致。",
        "",
        "| 来源 | 低NLL−随机成功率 | 胜/平/负 |",
        "|---|---:|---:|",
    ]
    for source in SOURCES:
        e = effects["mean_success"]["nll_minus_random"][source]
        lines.append(
            f"| {source} | {100 * e['mean_difference']:+.3f}pp | {e['wins']}/{e['ties']}/{e['losses']} |"
        )
    lines += [
        "",
        "条件重复分列；均值仅使用repeat=0的成对单元。固定开发任务上的描述性结果，不提供未经校准的置信区间。",
        "",
        "![成对结果](action-matched-policies.png)",
        "",
    ]
    (out / "RESULTS.zh-CN.md").write_text("\n".join(lines))
    # Also finish the durable state when the existing observer recovers this
    # report after the original pipeline has exited.
    ledger["status"] = "completed"
    save(root / "execution.json", ledger)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=os.environ.get("LH_CONTROL_STUDY_ROOT")
    )
    args = parser.parse_args()
    if args.root is None:
        parser.error(
            "Specify --root or bind LH_CONTROL_STUDY_ROOT for observer recovery"
        )
    result = generate(args.root)
    print(
        json.dumps(
            {k: result[k] for k in ("complete", "cases", "actual_evaluation_episodes")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
