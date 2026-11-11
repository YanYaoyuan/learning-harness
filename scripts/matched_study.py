#!/usr/bin/env python3
"""Build and analyze a training-only matched-success diagnostic."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

from learning_harness.analysis import export_sft, summarize
from learning_harness.common import digest, read_jsonl, write_jsonl

ROOT = Path("runs/dual5090-v2")


def merge_collection():
    target = ROOT / "collect"
    target.mkdir(exist_ok=True)
    metadata = [json.loads((ROOT / f"collect-gpu{g}/run.json").read_text()) for g in (0, 1)]
    assert len({(m["model"], m["revision"], m["batch_protocol"]) for m in metadata}) == 1
    comparable = [{k: v for k, v in m["config"].items() if k != "manifest"} for m in metadata]
    assert comparable[0] == comparable[1]
    for filename in ("steps.jsonl", "episodes.jsonl", "calls.jsonl"):
        combined = []
        for gpu in (0, 1):
            rows = read_jsonl(ROOT / f"collect-gpu{gpu}" / filename)
            if filename == "episodes.jsonl":
                assert len(rows) == metadata[gpu]["planned_episodes"] and all(r["status"] == "completed" for r in rows)
            combined.extend(rows)
        if filename == "episodes.jsonl":
            assert len({(r["task_id"], r["arm"], r["rollout_seed"]) for r in combined}) == len(combined)
        write_jsonl(target / filename, combined)
    (target / "run.json").write_text(json.dumps(dict(fingerprint=digest([m["fingerprint"] for m in metadata]),
        planned_episodes=sum(m["planned_episodes"] for m in metadata), source_workers=metadata), indent=2) + "\n")
    behavior = {}
    steps = read_jsonl(target / "steps.jsonl")
    successful_attempts = {r["attempt_id"] for r in read_jsonl(target / "episodes.jsonl") if r["success"]}
    for arm in ("none", "early"):
        records = [r for r in steps if r["arm"] == arm]
        previous, repeated = {}, 0
        for row in records:
            old = previous.get(row["attempt_id"])
            repeated += int(old is not None and row["action"] == old["action"] and row["observation"] == old["observation"])
            previous[row["attempt_id"]] = row
        behavior[arm] = dict(steps=len(records), invalid_json=sum(r["invalid_action"] for r in records),
            literal_nothing_happens=sum(r["observation"].strip() == "Nothing happens." for r in records),
            repeated_action_and_feedback=repeated,
            successful_trajectory_steps=sum(r["attempt_id"] in successful_attempts for r in records),
            successful_trajectory_nothing_happens=sum(r["attempt_id"] in successful_attempts and r["observation"].strip() == "Nothing happens." for r in records))
    (ROOT / "behavior.json").write_text(json.dumps(behavior, indent=2) + "\n")
    return summarize(target)


def evenly_spaced(rows, count):
    rows = sorted(rows, key=lambda row: row["step"])
    if not 0 < count <= len(rows):
        raise ValueError("Invalid matched count")
    if count == 1:
        return [rows[-1]]
    indices = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    assert len(set(indices)) == count
    return [rows[index] for index in indices]


def match_rows(corpora):
    grouped = {arm: defaultdict(list) for arm in ("none", "early")}
    for arm, rows in corpora.items():
        for row in rows:
            if row["partition"] != "train":
                raise ValueError("Matching cannot consume development or evaluation data")
            grouped[arm][(row["task_id"], row["rollout_seed"])].append(row)
    common = sorted(set(grouped["none"]) & set(grouped["early"]))
    matched = {arm: [] for arm in grouped}
    pair_counts = []
    for pair in common:
        count = min(len(grouped[arm][pair]) for arm in grouped)
        pair_counts.append(dict(task_id=pair[0], rollout_seed=pair[1], examples_per_arm=count))
        for arm in grouped:
            matched[arm].extend(dict(row, matched_index=index) for index, row in enumerate(evenly_spaced(grouped[arm][pair], count)))
    if not common:
        raise ValueError("No mutually successful training episodes")
    for arm, rows in matched.items():
        rows.sort(key=lambda r: digest([20261110, r["task_id"], r["rollout_seed"], r["matched_index"]]))
    keys = lambda rows: [(r["task_id"], r["rollout_seed"], r["matched_index"]) for r in rows]
    assert keys(matched["none"]) == keys(matched["early"])
    return matched, pair_counts


def build():
    if (ROOT / "jobs.json").exists():
        raise ValueError("Job list already frozen; do not overwrite training corpora")
    merge_collection()
    from transformers import AutoTokenizer
    from train_lora import encode_example
    tokenizer = AutoTokenizer.from_pretrained("data/models/Qwen3-4B-Instruct-2507", trust_remote_code=False)
    corpus_dir = ROOT / "corpora"
    corpus_dir.mkdir(exist_ok=True)
    source, metadata, dropped = {}, {}, {}
    for arm in ("none", "early"):
        path = corpus_dir / f"raw-{arm}.jsonl"
        export_sft(ROOT / "collect", arm, path)
        rows = read_jsonl(path)
        source[arm] = [r for r in rows if encode_example(tokenizer, r, 2048) is not None]
        dropped[arm] = len(rows) - len(source[arm])
        write_jsonl(path, source[arm])
        meta_path = Path(str(path) + ".meta.json")
        meta = json.loads(meta_path.read_text())
        meta.update(examples=len(source[arm]), overlength_dropped_before_matching=dropped[arm], regime="raw")
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        metadata[arm] = meta
    matched, counts = match_rows(source)
    for arm in ("none", "early"):
        path = corpus_dir / f"matched-{arm}.jsonl"
        write_jsonl(path, matched[arm])
        meta = dict(metadata[arm], regime="matched", examples=len(matched[arm]),
            successful_episodes=len(counts), matching_counts=counts,
            matching="Common successful task/seed pairs; equal action-example counts per pair; deterministic quantiles")
        Path(str(path) + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    selection = dict(raw_examples={arm: len(rows) for arm, rows in source.items()},
        matched_examples={arm: len(rows) for arm, rows in matched.items()}, pairs=counts,
        raw_successful_tasks={arm: len({r["task_id"] for r in rows}) for arm, rows in source.items()},
        matched_successful_tasks=len({r["task_id"] for r in matched["none"]}),
        overlength_dropped=dropped, probe_results_used_for_matching=False)
    types = {r["task_id"]: r["task_type"] for r in read_jsonl(ROOT / "train.jsonl")}
    selection["raw_task_type_coverage"] = {arm: dict(Counter(types[task] for task in {r["task_id"] for r in rows})) for arm, rows in source.items()}
    selection["matched_task_type_coverage"] = dict(Counter(types[task] for task in {r["task_id"] for r in matched["none"]}))
    (ROOT / "data-selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    jobs = [dict(name=f"{regime}-{arm}-seed{seed}", regime=regime, arm=arm, seed=seed,
                 data=str(corpus_dir / f"{regime}-{arm}.jsonl"), method="target_sft")
            for seed in (101, 202, 303) for regime in ("raw", "matched") for arm in ("none", "early")]
    random.Random(20261110).shuffle(jobs)
    jobs.insert(0, dict(name="base-repeat"))
    jobs.insert(0, dict(name="base"))
    (ROOT / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
    frozen = [ROOT / name for name in ("protocol.json", "amendments.json", "jobs.json", "lora.json", "train.jsonl", "probe.jsonl", "fresh-probe.jsonl", "support.jsonl", "data-selection.json")]
    frozen.extend(corpus_dir.glob("*.jsonl*"))
    (ROOT / "inputs.sha256.json").write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen}, indent=2) + "\n")
    return selection


def crossed_interval(matrix, draws=4000):
    """Resample paired training seeds and paired task groups on separate axes."""
    rng = random.Random(20261110)
    seeds, tasks = len(matrix), len(matrix[0])
    samples = []
    for _ in range(draws):
        picked_seeds = rng.choices(range(seeds), k=seeds)
        picked_tasks = rng.choices(range(tasks), k=tasks)
        samples.append(statistics.mean(matrix[s][t] for s in picked_seeds for t in picked_tasks))
    samples.sort()
    return [samples[int(draws * 0.025)], samples[int(draws * 0.975)]]


def analyze_subset(indices, evaluations):
    matrices = {}
    cells = {}
    for regime in ("raw", "matched"):
        for arm in ("none", "early"):
            rows = [[int(evaluations[f"{regime}-{arm}-seed{seed}"][task]["success"]) for task in indices]
                    for seed in (101, 202, 303)]
            cells[f"{regime}-{arm}"] = dict(wins_per_seed=[sum(row) for row in rows],
                training_seeds=[101, 202, 303], tasks_per_seed=len(indices), mean_success=statistics.mean(sum(row) / len(row) for row in rows))
            matrices[f"{regime}-{arm}"] = rows
    contrasts = {}
    deltas = {}
    for regime in ("raw", "matched"):
        delta = [[a - b for a, b in zip(e, n)] for e, n in zip(matrices[f"{regime}-early"], matrices[f"{regime}-none"])]
        deltas[regime] = delta
        contrasts[regime] = dict(early_minus_none=statistics.mean(v for row in delta for v in row),
            per_seed_delta=[statistics.mean(row) for row in delta], crossed_bootstrap_95=crossed_interval(delta))
    difference = [[r - m for r, m in zip(raw, matched)] for raw, matched in zip(deltas["raw"], deltas["matched"])]
    return dict(n_tasks=len(indices), n_training_seeds=3,
        base_wins=sum(evaluations["base"][task]["success"] for task in indices),
        cells=cells, contrasts=contrasts,
        raw_gap_minus_matched_gap=dict(estimate=statistics.mean(v for row in difference for v in row),
            crossed_bootstrap_95=crossed_interval(difference)))


def report():
    for path, expected in json.loads((ROOT / "inputs.sha256.json").read_text()).items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, f"Frozen input changed: {path}"
    jobs = json.loads((ROOT / "jobs.json").read_text())
    manifest = read_jsonl(ROOT / "probe.jsonl")
    task_ids = sorted(r["task_id"] for r in manifest)
    fresh = sorted(r["task_id"] for r in read_jsonl(ROOT / "fresh-probe.jsonl"))
    fresh_set = set(fresh)
    evaluations = {}
    training = {}
    for job in jobs:
        directory = ROOT / "evaluation" / job["name"]
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["complete"] and summary["recorded"] == 40
        rows = read_jsonl(directory / "episodes.jsonl")
        assert all(r["status"] == "completed" and not r["support_injected"] and r["partition"] == "probe" for r in rows)
        assert len(rows) == len({r["task_id"] for r in rows}) == 40
        evaluations[job["name"]] = {r["task_id"]: r for r in rows}
        assert set(evaluations[job["name"]]) == set(task_ids)
        if job.get("data"):
            t = json.loads((ROOT / "checkpoints" / job["name"] / "experiment.json").read_text())
            assert t["overlength_dropped"] == 0 and t["training_seed"] == job["seed"]
            assert t["processed_examples_rank0"] == t["scheduled_examples"] == 256
            assert t["training_data_sha256"] == hashlib.sha256(Path(job["data"]).read_bytes()).hexdigest()
            training[job["name"]] = t
    for seed in (101, 202, 303):
        assert training[f"matched-none-seed{seed}"]["schedule_sha256"] == training[f"matched-early-seed{seed}"]["schedule_sha256"]
    repeat_differences = [task for task in task_ids if evaluations["base"][task]["success"] != evaluations["base-repeat"][task]["success"]]
    results = dict(protocol="dual5090-v2", complete=True, protocol_tests_passed=15,
        environment_execution_validation=json.loads((ROOT / "parallel-environment-validation.json").read_text()),
        processed_examples_each=256, matched_sampling_schedules_aligned=True,
        baseline_repeat_outcome_differences=repeat_differences,
        fresh_primary=analyze_subset(fresh, evaluations),
        all40_secondary=analyze_subset(task_ids, evaluations),
        old12_secondary=analyze_subset([task for task in task_ids if task not in fresh_set], evaluations),
        data_selection=json.loads((ROOT / "data-selection.json").read_text()), training=training,
        collection_behavior=json.loads((ROOT / "behavior.json").read_text()),
        collection=json.loads((ROOT / "collect/summary.json").read_text()),
        uncertainty="Crossed paired bootstrap over 3 training seeds and held-out task groups; three seeds still give limited precision",
        final_benchmarks_used=False, teacher_api_used=False,
        limitations=["One model family and weak generic support", "Common-success conditioning is a diagnostic selected population", "Matching count and coverage does not match exact input/target tokens or trajectory quality", "No future-task learning-speed or online-RL result"])
    by_type = {}
    for family in sorted({r["task_type"] for r in manifest}):
        ids = sorted(r["task_id"] for r in manifest if r["task_id"] in fresh_set and r["task_type"] == family)
        by_type[family] = dict(n_tasks=len(ids), base_wins=sum(evaluations["base"][t]["success"] for t in ids))
        for regime in ("raw", "matched"):
            for arm in ("none", "early"):
                by_type[family][f"{regime}-{arm}"] = [sum(evaluations[f"{regime}-{arm}-seed{seed}"][t]["success"] for t in ids) for seed in (101, 202, 303)]
    results["fresh_by_type_descriptive"] = by_type
    (ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["# 双 5090 第二轮：覆盖与数量匹配、三个训练种子", "",
        "训练：96 个任务 × 两种来源 × 两个采样种子，共 384 个回合。原始和匹配两种 corpus，各训练 seed 101/202/303，普通 LoRA SFT 64 次更新。评测每个 checkpoint 相同 40 个开发任务，主结果仅使用此前未评测的 28 个。正式 seen/unseen 测试未运行。", "",
        "支持仍为固定公开策略，不代表前沿教师或强代办式 harness。本轮全部重新批量采集，不与上一轮不同采样实现混合。", "",
        f"基础模型在两张卡各重复一次相同的 40 任务 H0 评测，结果不同的任务数为 {len(repeat_differences)}。", "",
        "后续评测使用独立进程执行环境；转换前重放基础模型全部 40 个任务、1,444 次动作，初始观察、每步公开反馈和成败逐条一致。推理批量、槽位顺序、模型与预算不变；各 run 记录实际环境后端。", "",
        "## 训练数据", "",
        f"原始示范数量：{results['data_selection']['raw_examples']}；成功任务数量：{results['data_selection']['raw_successful_tasks']}。",
        f"匹配后每个来源 {len(read_jsonl(ROOT / 'corpora/matched-none.jsonl'))} 条示范，共同成功任务 {results['data_selection']['matched_successful_tasks']} 个；匹配精确到任务/采样种子，每对动作示范数量相同。按时间等距取样并保留可用首尾步骤。", "",
        "匹配规则只读训练轨迹；共同成功条件会选择子群体，因此匹配对照不是总体因果效应的无偏估计。", ""]
    lines.extend(["正的来源差距表示早期支持来源训练后更好；负值才与其即时优势方向相反。排序冲突应依据符号与重复性判断，不能用来源差距的绝对值是否缩小代替。", ""])
    for key, title in (("fresh_primary", "主结果：28 个新 probe"), ("all40_secondary", "次要结果：全部 40 个 probe"), ("old12_secondary", "次要结果：原来的 12 个 probe")):
        subset = results[key]
        lines.extend([f"## {title}", "", f"基础模型：{subset['base_wins']}/{subset['n_tasks']}。各单元格按训练 seed 101/202/303 报告成功数，不能把三次重复的同一任务当成三个独立任务。", "",
            "| 数据条件 | 无支持来源：三个 seed | 早期支持来源：三个 seed | 平均差距及交叉 bootstrap 95% 区间（百分点） |", "| --- | --- | --- | --- |"])
        for regime in ("raw", "matched"):
            c = subset["contrasts"][regime]
            interval = c["crossed_bootstrap_95"]
            lines.append(f"| {regime} | {subset['cells'][regime + '-none']['wins_per_seed']} | {subset['cells'][regime + '-early']['wins_per_seed']} | {c['early_minus_none'] * 100:+.2f} [{interval[0] * 100:+.2f}, {interval[1] * 100:+.2f}] |")
        contrast = subset["raw_gap_minus_matched_gap"]
        lines.extend(["", f"原始差距减匹配差距：{contrast['estimate'] * 100:+.2f} 个百分点；区间 {[round(v * 100, 2) for v in contrast['crossed_bootstrap_95']]}。仅三个训练种子，区间不是稳定性或等效性的保证。", ""])
    lines.extend(["## 解释边界", "", "若匹配后差距改变，说明结果对样本选择/覆盖/数量敏感，不能把变化全部归因于覆盖。若仍有差距，也不能排除轨迹质量、长度或有限样本影响。零差距不等于证明等效。", "",
        "本轮未比较直接动作帮助与技能反馈，未测新任务学习速度、持续学习或在线 RL。是否继续此主张，应先结合新 probe 和跨 seed 方向，再设计更强干预；不能仅凭旧 12 题分数推进 meta-RL。", "",
        "原始记录：`runs/dual5090-v2/results.json`、`data-selection.json`、`protocol.json`、`evaluation/*/episodes.jsonl` 和 `checkpoints/*/experiment.json`。"])
    Path("docs/DUAL5090_MATCHED_RESULTS.zh-CN.md").write_text("\n".join(lines) + "\n")
    return results["fresh_primary"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "report"])
    args = parser.parse_args()
    print(json.dumps(build() if args.command == "build" else report(), indent=2))
