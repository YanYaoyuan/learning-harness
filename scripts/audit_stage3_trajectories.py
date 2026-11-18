#!/usr/bin/env python3
"""Read-only, exploratory behavior audit of an immutable paired Stage 3 snapshot."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_stage3_task_controls as study
from learning_harness.common import file_sha
from learning_harness.controlled_study import audit_episodes
from learning_harness.trajectory_audit import episode_behavior

METRICS = (
    "success",
    "steps",
    "take_attempt",
    "move_attempt",
    "any_format_error",
    "format_error_fraction",
    "literal_no_effect_fraction",
    "hit_call_limit",
    "exact_command_repetition_fraction",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    freeze = study.load_freeze()
    snapshot_file = args.snapshot / "inputs.json"
    snapshot = json.loads(snapshot_file.read_text())
    if snapshot["freeze_sha256"] != file_sha(study.ROOT / "freeze.json"):
        raise ValueError("Snapshot belongs to a different frozen study")
    cells = {tuple(pair) for pair in snapshot["matched_seed_cells"]}
    records = [
        r
        for r in snapshot["records"]
        if r["case"]["repeat"] == 0
        and r["case"]["arm"] in ("random", "min_nll")
        and (r["case"]["initialization_seed"], r["case"]["sampling_seed"]) in cells
    ]
    expected = {
        (i, s, source, policy)
        for i, s in cells
        for source in ("api", "generic")
        for policy in ("random", "min_nll")
    }
    actual = {
        (
            r["case"]["initialization_seed"],
            r["case"]["sampling_seed"],
            r["case"]["source"],
            r["case"]["arm"],
        )
        for r in records
    }
    if not cells or actual != expected or len(records) != len(expected):
        raise ValueError("Snapshot must contain every unique source/policy mate")
    planned = {c["name"]: c for c in freeze["cases"]}
    provenance = {str(snapshot_file): file_sha(snapshot_file)}
    episode_rows, model_rows = [], []

    def read_json(path, lines=False):
        data = path.read_bytes()
        provenance[str(path)] = hashlib.sha256(data).hexdigest()
        return (
            [json.loads(line) for line in data.splitlines() if line]
            if lines
            else json.loads(data)
        )

    for record in records:
        case = record["case"]
        folder = study.ROOT / "cases" / case["name"]
        worker_path = folder / "worker-result.json"
        if (
            case != planned[case["name"]]
            or file_sha(worker_path) != snapshot["provenance"][str(worker_path)]
        ):
            raise ValueError("Worker or planned condition differs from snapshot")
        if read_json(worker_path) != record:
            raise ValueError("Worker content differs from immutable snapshot")
        rows, family_counts = [], Counter()
        identity = dict(
            case=case["name"],
            source=case["source"],
            policy=case["arm"],
            initialization_seed=case["initialization_seed"],
            sampling_seed=case["sampling_seed"],
        )
        for family in ("cool", "heat"):
            evaluation = folder / "evaluation" / family
            config = read_json(evaluation / "config.json")
            manifest_path = Path(config["manifest"])
            if file_sha(manifest_path) != freeze["files"][str(manifest_path)]:
                raise ValueError("Evaluation task manifest has changed")
            manifest = read_json(manifest_path, lines=True)
            episodes = read_json(evaluation / "episodes.jsonl", lines=True)
            if (
                len(episodes) != 32
                or audit_episodes(episodes, manifest) != record["success"][family]
            ):
                raise ValueError(
                    "Raw family outcomes disagree with frozen task protocol"
                )
            events = defaultdict(list)
            steps_path = evaluation / "steps.jsonl"
            digest = hashlib.sha256()
            with steps_path.open("rb") as stream:
                for line in stream:
                    digest.update(line)
                    if line.strip():
                        event = json.loads(line)
                        event.pop("messages", None)
                        events[event["attempt_id"]].append(event)
            provenance[str(steps_path)] = digest.hexdigest()
            if set(events) != {e["attempt_id"] for e in episodes}:
                raise ValueError(
                    "Orphaned or missing episode attempts require explicit review"
                )
            for episode in episodes:
                behavior = episode_behavior(
                    episode, events[episode["attempt_id"]], config["max_steps"]
                )
                counts = behavior.pop("action_family_counts")
                family_counts.update(counts)
                row = dict(
                    **identity,
                    family=family,
                    task_id=episode["task_id"],
                    attempt_id=episode["attempt_id"],
                    **behavior,
                )
                episode_rows.append(row)
                rows.append(row)
        model = dict(
            **identity, episodes=len(rows), action_family_counts=dict(family_counts)
        )
        model.update({m: sum(r[m] for r in rows) / len(rows) for m in METRICS})
        if model["success"] != record["mean_success"]:
            raise ValueError("Recomputed model success differs from snapshot")
        model_rows.append(model)

    out = (
        Path("output/variance-diagnostic-v1/stage3/trajectory-audit")
        / args.snapshot.name
    )
    out.mkdir(parents=True, exist_ok=False)
    groups = {}
    for source in ("api", "generic"):
        groups[source] = {}
        for policy in ("random", "min_nll"):
            group = [
                r for r in model_rows if r["source"] == source and r["policy"] == policy
            ]
            groups[source][policy] = {
                m: sum(r[m] for r in group) / len(group) for m in METRICS
            }
    summary = dict(
        exploratory=True,
        interim=True,
        final_study_complete=False,
        source_snapshot=str(args.snapshot),
        source_snapshot_sha256=file_sha(snapshot_file),
        freeze_sha256=snapshot["freeze_sha256"],
        matched_seed_cells=sorted(cells),
        models=len(model_rows),
        episodes=len(episode_rows),
        teacher_calls=0,
        new_model_calls=0,
        official_tests_used=False,
        groups=groups,
        model_rows=model_rows,
        definitions={
            "aggregation": "Fractions normalized within each episode; 32 episodes per family, 64 per model; equal weights across models and paired seed cells. Shared tasks and sampling cells are not independent initialization seeds.",
            "take_attempt": "At least one parsed canonical action beginning with take; attempt only, not verified acquisition.",
            "move_attempt": "At least one canonical move action, including public put aliases; attempt only, not verified placement.",
            "any_format_error": "JSON/action parser failure; does not measure hidden physical action validity.",
            "literal_no_effect_fraction": "Fraction of steps with observation exactly Nothing happens. after stripping outer whitespace; observable proxy, not an oracle of invalid actions.",
            "hit_call_limit": "Episode has exactly config.max_steps actions, including any success at the cap.",
            "exact_command_repetition_fraction": "1 - number of distinct canonical commands / step count; repeated navigation may be legitimate.",
            "scope": "Post hoc behavior description of existing outcomes; no causal attribution, inference, early stopping, or change to frozen training/evaluation.",
        },
        provenance=provenance,
        code_sha256={
            p: file_sha(p)
            for p in (
                "scripts/audit_stage3_trajectories.py",
                "src/learning_harness/trajectory_audit.py",
                "src/learning_harness/controlled_study.py",
                "src/learning_harness/support.py",
            )
        },
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for filename, rows in (("episodes.csv", episode_rows), ("models.csv", model_rows)):
        flat = [
            {k: v for k, v in row.items() if k != "action_family_counts"}
            for row in rows
        ]
        with (out / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    panels = (
        ("take_attempt", "Episodes attempting take (%)"),
        ("move_attempt", "Episodes attempting placement (%)"),
        ("any_format_error", "Episodes with a format error (%)"),
        ("literal_no_effect_fraction", "Mean literal no-effect steps per episode (%)"),
    )
    for ax, (metric, title) in zip(axes.flat, panels):
        for x, source in enumerate(("api", "generic")):
            for offset, policy, color in (
                (-0.18, "random", "#2879ad"),
                (0.18, "min_nll", "#d46a30"),
            ):
                group = sorted(
                    (
                        r
                        for r in model_rows
                        if r["source"] == source and r["policy"] == policy
                    ),
                    key=lambda r: (r["initialization_seed"], r["sampling_seed"]),
                )
                values = [100 * r[metric] for r in group]
                mean = 100 * groups[source][policy][metric]
                ax.bar(
                    x + offset,
                    mean,
                    width=0.32,
                    color=color,
                    alpha=0.5,
                    label=policy if x == 0 else None,
                )
                jitter = [
                    (0.17 * (j / max(1, len(values) - 1) - 0.5))
                    for j in range(len(values))
                ]
                ax.scatter(
                    [x + offset + j for j in jitter],
                    values,
                    c=color,
                    s=22,
                    edgecolors="white",
                    linewidths=0.4,
                    zorder=3,
                )
                ax.plot(
                    [x + offset - 0.12, x + offset + 0.12],
                    [mean, mean],
                    color="black",
                    linewidth=1.7,
                )
                ax.text(
                    x + offset,
                    max(values) + 3,
                    f"{mean:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        ax.set(
            xticks=[0, 1],
            xticklabels=["API source", "Generic source"],
            ylabel=title,
            ylim=(-2, 113),
            xlim=(-0.6, 1.6),
        )
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        f"EXPLORATORY INTERIM: {len(cells)} paired cells, {len(model_rows)} models, {len(episode_rows)} existing episodes",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.015,
        "Dots: model means on the same 64 tasks. Bars: equal-weight means. Attempts do not establish successful execution.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    for extension in ("png", "pdf"):
        fig.savefig(out / f"trajectory-behavior.{extension}", dpi=180)
    plt.close(fig)
    lines = [
        "# Stage 3 已有轨迹行为诊断",
        "",
        f"事后描述性分析：固定于原{len(cells)}单元快照，{len(model_rows)}个模型、{len(episode_rows)}个已完成回合。无新增模型调用或训练。",
        "",
        "| 来源/策略 | 成功率 | 尝试take回合 | 尝试move回合 | 有格式错误回合 | 回合内无效果反馈占比均值 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source, policies in groups.items():
        for policy, values in policies.items():
            lines.append(
                f"| {source}/{policy} | "
                + " | ".join(
                    f"{100 * values[m]:.2f}%"
                    for m in (
                        "success",
                        "take_attempt",
                        "move_attempt",
                        "any_format_error",
                        "literal_no_effect_fraction",
                    )
                )
                + " |"
            )
    lines += [
        "",
        "take/move统计尝试动作，不验证物体实际获取或放置。invalid_action字段只标记JSON/action解析错误；Nothing happens.是独立的字面环境反馈代理。无效果占比先在每回合归一化，再跨回合/模型平均，避免长失败轨迹支配统计。",
        "",
        f"行为与成功率来自相同轨迹，不能据此认定训练动作分布造成成功率差异。动作类型匹配训练仍是独立待检验干预。这里共{len({i for i, _ in cells})}个初始化，不把{len(cells)}个采样配对单元当作独立初始化。完整数值、模型点、回合与原始文件哈希见JSON/CSV。",
        "",
        "![行为诊断](trajectory-behavior.png)",
        "",
    ]
    (out / "RESULTS.zh-CN.md").write_text("\n".join(lines))
    print(
        json.dumps(
            dict(
                output=str(out),
                models=len(model_rows),
                episodes=len(episode_rows),
                groups=groups,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
