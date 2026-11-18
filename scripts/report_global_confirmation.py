#!/usr/bin/env python3
"""Final raw-verified fresh-task replication and development comparison."""

import json

from learning_harness.bounded_queue import save
from learning_harness.common import file_sha
from learning_harness.global_confirmation import RULES, aggregate
import run_global_confirmation as study


def main():
    frozen = study.validate()
    if (study.OUT / "aggregate-summary.json").exists():
        raise ValueError("Preserve any previous final confirmation report")
    records = [study.audit_result(case) for case in frozen["cases"]]
    result = aggregate(records)
    development = study.read(study.previous.OUT / "aggregate-summary.json")
    result.update(
        complete=True,
        models=97,
        episodes=12416,
        training_updates=0,
        freeze_sha256=file_sha(study.ROOT / "freeze.json"),
        development_summary_sha256=file_sha(
            study.previous.OUT / "aggregate-summary.json"
        ),
        confirmation_gpu_hours=study.read(study.ROOT / "execution.json")["gpu_seconds"]
        / 3600,
        prior_finishing_gpu_hours=frozen["prior_finishing_gpu_hours"],
        official_tests_used=False,
        raw_records_uploaded=False,
        case_receipt_sha256={
            c["name"]: file_sha(study.ROOT / "cases" / c["name"] / "worker-result.json")
            for c in frozen["cases"]
        },
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    colors = ["#6b7785", "#c85b39", "#8665a4"]
    for ax, source in zip(axes, ("api", "generic")):
        for k, (label, values) in enumerate(
            (("Development", development), ("Confirmation", result))
        ):
            positions = np.arange(3) + (k - 0.5) * 0.36
            means = [
                100 * values["endpoints"][source][p]["mean_success"] for p in RULES
            ]
            ax.bar(
                positions,
                means,
                0.34,
                color=colors,
                alpha=0.45 if k == 0 else 1,
                edgecolor="black",
                linewidth=0.4,
                label=label,
            )
        ax.set(
            title=source.upper(),
            ylim=(0, 100),
            ylabel="Success (%)",
            xticks=range(3),
            xticklabels=["Global random", "Global high NLL", "Global gradient"],
        )
        ax.tick_params(axis="x", labelrotation=20)
    axes[0].legend(frameon=False)
    fig.suptitle("Frozen checkpoint replication on disjoint task groups")
    fig.tight_layout()
    fig.savefig(
        study.OUT / "development-confirmation.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)
    lines = [
        "# 全局经验选样：新任务独立复现",
        "",
        "97个既有检查点、12,416回合，无重新训练。",
        "",
        "| 规则 | API开发 | API确认 | Generic开发 | Generic确认 |",
        "|---|---:|---:|---:|---:|",
    ]
    for rule in RULES:
        values = [
            100 * source["endpoints"][s][rule]["mean_success"]
            for s in ("api", "generic")
            for source in (development, result)
        ]
        lines.append(
            "| " + rule + " | " + " | ".join(f"{v:.3f}%" for v in values) + " |"
        )
    lines += [
        "",
        "主分析是有限任务面板的描述性复现。初始化SD不是置信区间；不报告p值、总体等效或官方测试成绩。",
        "",
        "![开发与确认](development-confirmation.png)",
    ]
    (study.OUT / "RESULTS.zh-CN.md").write_text("\n".join(lines) + "\n")
    save(study.OUT / "aggregate-summary.json", result)
    save(
        study.OUT / "handoff.json",
        dict(
            status="final_raw_audited_replication_ready_for_paper",
            summary_sha256=file_sha(study.OUT / "aggregate-summary.json"),
            required_next="Visually inspect figure, verify cloud files, then rebuild manuscript.",
            automatic_new_gpu_work=False,
        ),
    )
    print(
        json.dumps(
            dict(status="confirmation_complete_raw_audited", models=97, episodes=12416)
        )
    )


if __name__ == "__main__":
    main()
