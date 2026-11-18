#!/usr/bin/env python3
"""Audit exact feature extraction and prepare untrained gradient-selection recipes."""

from collections import Counter
import json

import numpy as np

from learning_harness.action_matched_sampling import action_matched_schedule
from learning_harness.bounded_queue import save
from learning_harness.common import file_sha, read_jsonl
from learning_harness.gradient_value import combine_scores
from run_optimizer_aware_selection_fp32 import CASE, OUT, PARENT, ROOT, read, validate


def correlation(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from canonical_action_controls import aligned_scores
    from learning_harness.webshop_requests import summarize_requests

    frozen, plan = validate()
    directory = ROOT / "cases" / CASE
    result = read(directory / "worker-result.json")
    scores_record = read(directory / "scores.private.json")
    usage = summarize_requests(directory / "forwards.private")
    if (
        not result["complete"]
        or result["freeze_sha256"] != file_sha(ROOT / "freeze.json")
        or result["scores_sha256"] != file_sha(directory / "scores.private.json")
        or result["usage"] != usage
        or usage["attempted_requests"] != 3708
        or not usage["complete_usage"]
        or usage["error_records"]
    ):
        raise ValueError("Complete feature-pilot accounting failed")
    attempts = [
        r for r in read_jsonl(directory / "forwards.private") if r["kind"] == "attempt"
    ]
    expected_phases = Counter(
        warmup=128, support=160, candidate=3412, support_repeat=4, candidate_repeat=4
    )
    if Counter(r["phase"] for r in attempts) != expected_phases:
        raise ValueError("Missing or unexpected warmup/gradient forwards")
    pool = read_jsonl(ROOT / "pool.private.jsonl")
    if scores_record["content_sha256"] != [r["content_sha256"] for r in pool]:
        raise ValueError("Final feature scores lost candidate identity")
    checkpoints = []
    repeatability = []
    for step in plan["checkpoints"]:
        path = directory / "checkpoints" / f"step-{step:04d}"
        record = read(path / "features.private.json")
        if (
            result["checkpoint_receipts"][str(step)]
            != file_sha(path / "features.private.json")
            or record["adapter_sha256"] != file_sha(path / "adapter_model.safetensors")
            or record["optimizer_sha256"] != file_sha(path / "optimizer.private.pt")
            or not record["scoring_state_unchanged"]
            or len(record["scores"]) != 853
        ):
            raise ValueError("Checkpoint/optimizer feature identity changed")
        for diagnostic in record["repeatability"].values():
            if diagnostic["cosine"] < 0.9999 or diagnostic["relative_l2_error"] > 0.001:
                raise ValueError("Checkpoint repeatability gate failed")
            repeatability.append(diagnostic)
        checkpoints.append(record["scores"])
    if (
        combine_scores(checkpoints, plan["checkpoint_weights"])
        != scores_record["scores"]
    ):
        raise ValueError("Final weighted/max feature score is not reproducible")
    final = scores_record["scores"]
    summaries, controls, source_values = {}, [], {}
    recipes = {}
    for source, meta in plan["sources"].items():
        rows = read_jsonl(meta["data"])
        nll = aligned_scores(read(meta["score_file"]), rows, meta["data_sha256"])
        values = [final[i] for i in meta["unique_indices"]]
        content = [pool[i]["content_sha256"] for i in meta["unique_indices"]]
        tasks = [r["task_id"] for r in rows]
        actions = [json.loads(r["messages"][-1]["content"])["action"] for r in rows]
        families = ["go" if a.startswith("go to ") else a.split()[0] for a in actions]
        strata = list(zip(tasks, families))
        distinct = list(dict.fromkeys(meta["unique_indices"]))
        first = {index: meta["unique_indices"].index(index) for index in distinct}
        unique_nll = [nll[first[i]] for i in distinct]
        unique_values = [final[i] for i in distinct]
        summaries[source] = dict(
            rows=len(rows),
            unique_contents=len(distinct),
            mean_score=float(np.mean(unique_values)),
            standard_deviation=float(np.std(unique_values)),
            min_score=min(unique_values),
            max_score=max(unique_values),
            negative_score_fraction=float(np.mean(np.array(unique_values) < 0)),
            pearson_with_nll=correlation(unique_values, unique_nll),
        )
        source_values[source] = (unique_nll, unique_values)
        for seed in (101, 202, 303, 404):
            reference_path = (
                PARENT / "inputs/schedules" / f"random-{source}-s{seed}.json"
            )
            low_path = reference_path.with_name(f"min_nll-{source}-s{seed}.json")
            reference, low = read(reference_path)["indices"], read(low_path)["indices"]
            indices = action_matched_schedule(
                tasks,
                families,
                content,
                reference,
                seed,
                "min_nll",
                [-v for v in values],
            )
            quota = Counter(
                s for s, key in set((strata[i], content[i]) for i in reference)
            )
            actual = Counter(
                s for s, key in set((strata[i], content[i]) for i in indices)
            )
            if (
                len(indices) != 768
                or [strata[i] for i in indices] != [strata[i] for i in reference]
                or quota != actual
                or sum(quota.values()) != 256
            ):
                raise ValueError("Gradient selector changed matched budget")
            selected = set((tasks[i], content[i]) for i in indices)
            selected_low = set((tasks[i], content[i]) for i in low)
            selected_random = set((tasks[i], content[i]) for i in reference)
            common = dict(
                candidate_only=True,
                training_started=False,
                source=source,
                sampling_seed=seed,
                data=meta["data"],
                data_sha256=meta["data_sha256"],
                scores_sha256=file_sha(directory / "scores.private.json"),
                pilot_freeze_sha256=file_sha(ROOT / "freeze.json"),
                presentations=768,
                distinct_support=256,
            )
            recipes[f"gradient-matched-{source}-s{seed}.json"] = dict(
                common,
                policy="optimizer_aware_matched",
                indices=indices,
                reference_recipe=str(reference_path),
                reference_recipe_sha256=file_sha(reference_path),
            )
            # Standard ranking comparison: global top 256 distinct contents, then
            # repeat complete shuffled passes; explicitly no coverage constraint.
            representatives = list(first.values())
            top = sorted(representatives, key=lambda i: (-values[i], i))[:256]
            from train_lora import fixed_schedule

            global_indices = [top[i] for i in fixed_schedule(256, 768, seed)]
            recipes[f"gradient-global-{source}-s{seed}.json"] = dict(
                common,
                policy="optimizer_aware_global",
                indices=global_indices,
                coverage_matched=False,
                unique_task_contents=len(
                    {(tasks[i], content[i]) for i in global_indices}
                ),
            )
            controls.append(
                dict(
                    source=source,
                    recipe=seed,
                    budget_controls_passed=True,
                    overlap_with_low=len(selected & selected_low),
                    overlap_with_random=len(selected & selected_random),
                    mean_selected_nll=float(np.mean([nll[i] for i in indices])),
                    mean_selected_gradient_score=float(
                        np.mean([values[i] for i in indices])
                    ),
                )
            )
    recipe_hashes = {}
    for name, recipe in recipes.items():
        path = ROOT / "candidate-schedules" / name
        if path.exists() and read(path) != recipe:
            raise ValueError("Preserve a changed candidate schedule")
        save(path, recipe)
        path.chmod(0o600)
        recipe_hashes[name] = file_sha(path)
    warmup = read_jsonl(directory / "warmup.jsonl")
    if [r["step"] for r in warmup] != list(range(1, 33)):
        raise ValueError("Incomplete warmup history")
    OUT.mkdir(parents=True, exist_ok=True)
    aggregate = dict(
        complete=True,
        feature_pilot_only=True,
        freeze_sha256=file_sha(ROOT / "freeze.json"),
        unique_candidates=853,
        support_examples=40,
        checkpoints=4,
        warmup_updates=32,
        warmup_unique_contents=43,
        warmup_presentations=128,
        sources=summaries,
        usage=usage,
        forward_phases=dict(expected_phases),
        supervised_tokens_processed=sum(r["target_tokens"] for r in attempts),
        scoring_weights_optimizer_and_rng_unchanged=True,
        gradient_repeatability_min_cosine=min(r["cosine"] for r in repeatability),
        gradient_repeatability_max_relative_l2=max(
            r["relative_l2_error"] for r in repeatability
        ),
        candidate_schedules=len(recipes),
        candidate_schedule_sha256=recipe_hashes,
        matched_controls=controls,
        downstream_students_trained=0,
        query_evaluations=0,
        teacher_calls=0,
        official_tests_used=False,
        raw_records_uploaded=False,
        scope="Optimizer-aware feature extraction and untrained selector preparation; no downstream learning-value evidence yet.",
    )
    save(OUT / "aggregate-summary.json", aggregate)
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.6))
    axes[0, 0].plot(
        [r["step"] for r in warmup], [r["loss"] for r in warmup], color="#147D92"
    )
    axes[0, 0].set(
        xlabel="Warmup update",
        ylabel="Mean target loss",
        title="43 contents / 128 presentations",
    )
    for ax, source in zip((axes[0, 1], axes[1, 0]), ("api", "generic")):
        nll, values = source_values[source]
        order = np.argsort(nll)
        bins = np.array_split(order, 10)
        ax.plot(
            [float(np.mean(np.array(nll)[b])) for b in bins],
            [float(np.mean(np.array(values)[b])) for b in bins],
            "o-",
            color="#C85B39",
        )
        ax.set(
            xlabel="Mean canonical NLL (10 bins)",
            ylabel="Mean gradient score",
            title=source.upper(),
        )
    for i, source in enumerate(("api", "generic")):
        group = [r for r in controls if r["source"] == source]
        axes[1, 1].scatter(
            [i - 0.12] * 4,
            [r["overlap_with_low"] / 256 for r in group],
            marker="o",
            color="#147D92",
            label="with low-NLL" if i == 0 else None,
        )
        axes[1, 1].scatter(
            [i + 0.12] * 4,
            [r["overlap_with_random"] / 256 for r in group],
            marker="x",
            color="#C85B39",
            label="with random" if i == 0 else None,
        )
    axes[1, 1].set(
        xticks=[0, 1],
        xticklabels=["API", "Generic"],
        ylim=(0, 1.05),
        ylabel="Selected-content overlap",
        title="256-content controlled selector",
    )
    axes[1, 1].legend(fontsize=7)
    for ax in axes.flat:
        ax.grid(alpha=0.15)
    fig.suptitle(
        "Optimizer-aware feature pilot / no downstream learning evaluation", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(OUT / "feature-pilot.png", dpi=180)
    plt.close(fig)
    lines = [
        "# 优化器感知梯度选样：实现试验结果",
        "",
        "完成32次预热更新、4个检查点、853个不同候选和40个公开支持动作；总3708次forward/backward。没有训练最终学生，也没有新任务成功率。",
        "",
        "| 来源 | 不同内容 | 分数标准差 | 与NLL的Pearson相关 |",
        "|---|---:|---:|---:|",
    ]
    for source, summary in summaries.items():
        lines.append(
            f"| {source} | {summary['unique_contents']} | {summary['standard_deviation']:.6f} | {summary['pearson_with_nll']} |"
        )
    lines.extend(
        [
            "",
            "已准备8份任务/动作/内容预算匹配清单和8份全局top256清单，均未训练。分数与NLL相关或不同都不等于预测真实收益，下一步需要同预算学习对照。",
            "",
            "![梯度特征与选样图](feature-pilot.png)",
        ]
    )
    (OUT / "RESULTS.zh-CN.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            dict(
                status="feature_pilot_audited",
                candidate_schedules=len(recipes),
                query_evaluations=0,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
