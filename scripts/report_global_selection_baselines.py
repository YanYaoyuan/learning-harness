#!/usr/bin/env python3
"""Full equal-freedom global comparison; only complete endpoints are compared."""

import json

from learning_harness.bounded_queue import save
from learning_harness.common import file_sha
import run_global_selection_baselines as study


def main():
    frozen = study.validate()
    if (study.OUT / "aggregate-summary.json").exists():
        raise ValueError("Preserve the previous global report")
    records = [study.audit_result(c) for c in frozen["cases"]]
    reused = [study.parent.audit_result(c) for c in frozen["reused_cases"]]
    if len(records) != 64 or len(reused) != 33:
        raise ValueError("Incomplete global comparison")
    base = next(r for r in reused if r["case"]["name"] == "base")
    students = records + [r for r in reused if r["case"]["name"] != "base"]
    rules = ("global_random", "global_high", "gradient_global")
    cells = {}
    for r in students:
        c = r["case"]
        key = (c["source"], c["policy"], c["initialization_seed"], c["sampling_seed"])
        if key in cells:
            raise ValueError("Duplicate global comparison cell")
        if (
            r["training"]["initial_parameter_sha256"]
            != frozen["initialization_anchors"][str(c["initialization_seed"])]
        ):
            raise ValueError("Unpaired initialization")
        cells[key] = r
    expected = {
        (s, p, i, j)
        for s in study.SOURCES
        for p in rules
        for i in study.parent.INITIALIZATIONS
        for j in study.SAMPLING
    }
    if set(cells) != expected:
        raise ValueError("The complete 96-student matrix is required")
    endpoints = {}
    for source in study.SOURCES:
        endpoints[source] = {}
        for rule in rules:
            per_seed = {
                str(seed): sum(
                    cells[source, rule, seed, sample]["mean_success"]
                    for sample in study.SAMPLING
                )
                / len(study.SAMPLING)
                for seed in study.parent.INITIALIZATIONS
            }
            endpoints[source][rule] = dict(
                mean_success=sum(per_seed.values()) / 8, per_initialization=per_seed
            )
    contrasts = {}
    for comparator in rules[:2]:
        by_source = {}
        for source in study.SOURCES:
            d = endpoints[source]
            by_source[source] = {
                str(seed): d["gradient_global"]["per_initialization"][str(seed)]
                - d[comparator]["per_initialization"][str(seed)]
                for seed in study.parent.INITIALIZATIONS
            }
        pooled = {
            str(seed): sum(by_source[s][str(seed)] for s in study.SOURCES) / 2
            for seed in study.parent.INITIALIZATIONS
        }
        contrasts[comparator] = dict(
            per_source=by_source,
            per_initialization=pooled,
            mean_difference=sum(pooled.values()) / 8,
        )
    ledger = study.read(study.ROOT / "execution.json")
    public = [
        dict(
            case=r["case"],
            episodes=r["episodes"],
            successes=r["successes"],
            mean_success=r["mean_success"],
            training=r["training"],
            families=r["families"],
        )
        for r in students
    ]
    result = dict(
        complete=True,
        development_only=True,
        new_students=64,
        reused_students=32,
        new_episodes=8192,
        compared_models_including_base=97,
        compared_episodes=12416,
        baseline_success=base["mean_success"],
        endpoints=endpoints,
        contrasts=contrasts,
        models=public,
        freeze_sha256=file_sha(study.ROOT / "freeze.json"),
        added_gpu_hours=ledger["gpu_seconds"] / 3600,
        parent_development_gpu_hours=study.read(study.parent.ROOT / "execution.json")[
            "gpu_seconds"
        ]
        / 3600,
        confirmation_evaluations=0,
        inferential_tests_performed=False,
        raw_records_uploaded=False,
        scope="Supplemental development comparison after parent outcomes were known; fixed historical pools, equal selection freedom and presentations, not equal tokens/FLOPs or a new gradient algorithm.",
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, source in zip(axes, study.SOURCES):
        d = endpoints[source]
        values = [100 * d[p]["mean_success"] for p in rules]
        ax.bar(range(3), values, color=["#6b7785", "#c85b39", "#8665a4"])
        for i, p in enumerate(rules):
            points = [100 * v for v in d[p]["per_initialization"].values()]
            ax.scatter(
                i + np.linspace(-0.15, 0.15, 8), points, s=15, color="black", alpha=0.7
            )
        ax.axhline(100 * base["mean_success"], color="#555555", linestyle="--")
        ax.set(
            title=source.upper(),
            ylim=(0, 100),
            ylabel="Success (%)",
            xticks=[0, 1, 2],
            xticklabels=["Global random", "Global high NLL", "Global gradient"],
        )
        ax.tick_params(axis="x", labelrotation=20)
    fig.suptitle(
        "Equal selection freedom / eight initializations / development results"
    )
    fig.tight_layout()
    fig.savefig(study.OUT / "global-comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    lines = [
        "# 全局选样：同等选择自由度的开发对照",
        "",
        "64个新增学生，复用32个全局梯度学生及基础模型。八初始化、两采样；无确认集结果。",
        "",
        "| 规则 | API成功率 | Generic成功率 |",
        "|---|---:|---:|",
    ]
    for rule in rules:
        lines.append(
            f"| {rule} | {100 * endpoints['api'][rule]['mean_success']:.3f}% | {100 * endpoints['generic'][rule]['mean_success']:.3f}% |"
        )
    lines += [
        "",
        "同呈现预算和选择自由度，实际token与评分成本另计；不作未经校准的确认性检验。",
        "",
        "![全局比较](global-comparison.png)",
    ]
    (study.OUT / "RESULTS.zh-CN.md").write_text("\n".join(lines) + "\n")
    save(study.OUT / "aggregate-summary.json", result)
    print(json.dumps(dict(status="complete_global_development_report", students=96)))


if __name__ == "__main__":
    main()
