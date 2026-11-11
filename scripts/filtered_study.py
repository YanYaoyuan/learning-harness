#!/usr/bin/env python3
"""Auxiliary public-feedback quality control, prespecified from training data."""
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import statistics

from learning_harness.common import read_jsonl, write_jsonl
from matched_study import match_rows, crossed_interval

SOURCE = Path("runs/dual5090-v2")
ROOT = Path("runs/dual5090-v2-filtered")


def build():
    if (ROOT / "jobs.json").exists():
        raise ValueError("Already frozen")
    episodes = read_jsonl(SOURCE / "collect/episodes.jsonl")
    valid = {(r["task_id"], r["arm"], r["rollout_seed"]): r["attempt_id"] for r in episodes if r["success"]}
    feedback = {}
    for row in read_jsonl(SOURCE / "collect/steps.jsonl"):
        if row["attempt_id"] == valid.get((row["task_id"], row["arm"], row["rollout_seed"])):
            feedback[(row["task_id"], row["arm"], row["rollout_seed"], row["step"])] = row["observation"].strip()
    cleaned, dropped = {}, {}
    for arm in ("none", "early"):
        rows = read_jsonl(SOURCE / f"corpora/raw-{arm}.jsonl")
        cleaned[arm] = [r for r in rows if feedback[(r["task_id"], arm, r["rollout_seed"], r["step"])] != "Nothing happens."]
        dropped[arm] = len(rows) - len(cleaned[arm])
    matched, counts = match_rows(cleaned)
    original = json.loads((SOURCE / "data-selection.json").read_text())
    pairs = lambda rows: {(r["task_id"], r["rollout_seed"]) for r in rows}
    assert pairs(counts) == pairs(original["pairs"])
    (ROOT / "corpora").mkdir(exist_ok=True)
    for arm in ("none", "early"):
        path = ROOT / f"corpora/filtered-{arm}.jsonl"
        write_jsonl(path, matched[arm])
        meta = json.loads((SOURCE / f"corpora/matched-{arm}.jsonl.meta.json").read_text())
        meta.update(regime="filtered_matched", examples=len(matched[arm]), matching_counts=counts,
            filter_rule="Resulting public feedback strips to exactly Nothing happens.",
            raw_dropped_before_matching=dropped[arm])
        Path(str(path) + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    for name in ("lora.json", "probe.jsonl", "fresh-probe.jsonl"):
        shutil.copyfile(SOURCE / name, ROOT / name)
    selection = dict(dropped_from_raw=dropped, matched_examples={arm: len(rows) for arm, rows in matched.items()},
        common_successful_tasks=len({r["task_id"] for r in matched["none"]}), same_success_pairs_as_unfiltered=True,
        pairs=counts, probe_used_for_selection=False)
    (ROOT / "data-selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    jobs = [dict(name=f"filtered-{arm}-seed{seed}", arm=arm, seed=seed,
        data=str(ROOT / f"corpora/filtered-{arm}.jsonl"), method="target_sft") for seed in (101, 202, 303) for arm in ("none", "early")]
    random.Random(20261111).shuffle(jobs)
    (ROOT / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
    files = [ROOT / name for name in ("prespec.json", "lora.json", "probe.jsonl", "fresh-probe.jsonl", "data-selection.json", "jobs.json")]
    files.extend((ROOT / "corpora").glob("*.jsonl*"))
    (ROOT / "inputs.sha256.json").write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2) + "\n")
    return selection


def analyze(task_ids, evaluations):
    matrices, cells = {}, {}
    for regime in ("raw", "matched", "filtered"):
        for arm in ("none", "early"):
            name = f"{regime}-{arm}"
            matrix = [[int(evaluations[f"{name}-seed{seed}"][task]["success"]) for task in task_ids] for seed in (101, 202, 303)]
            matrices[name] = matrix
            cells[name] = dict(wins_per_seed=[sum(row) for row in matrix], tasks_per_seed=len(task_ids))
    gaps, contrasts = {}, {}
    for regime in ("raw", "matched", "filtered"):
        delta = [[a - b for a, b in zip(e, n)] for e, n in zip(matrices[f"{regime}-early"], matrices[f"{regime}-none"])]
        gaps[regime] = delta
        contrasts[regime] = dict(early_minus_none=statistics.mean(v for row in delta for v in row),
            per_seed_delta=[statistics.mean(row) for row in delta], crossed_bootstrap_95=crossed_interval(delta))
    difference = [[m - f for m, f in zip(mm, ff)] for mm, ff in zip(gaps["matched"], gaps["filtered"])]
    return dict(n_tasks=len(task_ids), n_training_seeds=3, cells=cells, contrasts=contrasts,
        base_wins=sum(evaluations["base"][task]["success"] for task in task_ids),
        matched_gap_minus_filtered_gap=dict(estimate=statistics.mean(v for row in difference for v in row), crossed_bootstrap_95=crossed_interval(difference)))


def report():
    for path, expected in json.loads((ROOT / "inputs.sha256.json").read_text()).items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    assert json.loads((SOURCE / "results.json").read_text())["complete"]
    jobs = json.loads((SOURCE / "jobs.json").read_text()) + json.loads((ROOT / "jobs.json").read_text())
    task_ids = sorted(r["task_id"] for r in read_jsonl(ROOT / "probe.jsonl"))
    fresh = sorted(r["task_id"] for r in read_jsonl(ROOT / "fresh-probe.jsonl"))
    evaluations, training = {}, {}
    for job in jobs:
        root = ROOT if job["name"].startswith("filtered-") else SOURCE
        directory = root / "evaluation" / job["name"]
        assert json.loads((directory / "summary.json").read_text())["complete"]
        rows = read_jsonl(directory / "episodes.jsonl")
        assert len(rows) == 40 and all(r["status"] == "completed" and not r["support_injected"] for r in rows)
        evaluations[job["name"]] = {r["task_id"]: r for r in rows}
        assert set(evaluations[job["name"]]) == set(task_ids)
        if root == ROOT:
            t = json.loads((ROOT / "checkpoints" / job["name"] / "experiment.json").read_text())
            assert t["processed_examples_rank0"] == t["scheduled_examples"] == 256
            assert t["overlength_dropped"] == 0 and t["training_seed"] == job["seed"]
            assert t["training_data_sha256"] == hashlib.sha256(Path(job["data"]).read_bytes()).hexdigest()
            training[job["name"]] = t
    for seed in (101, 202, 303):
        assert training[f"filtered-none-seed{seed}"]["schedule_sha256"] == training[f"filtered-early-seed{seed}"]["schedule_sha256"]
    result = dict(complete=True, diagnostic="Public-feedback filtering followed by task/seed/action-count rematching",
        fresh_primary=analyze(fresh, evaluations), all40_secondary=analyze(task_ids, evaluations),
        data_selection=json.loads((ROOT / "data-selection.json").read_text()), training=training,
        prespec=json.loads((ROOT / "prespec.json").read_text()),
        caveat="Auxiliary diagnostic on the same development probes; common-success selection and rematching prevent a pure population causal attribution")
    contrasts = result["fresh_primary"]["contrasts"]
    result["interpretation"] = dict(
        all_source_gap_intervals_include_zero=all(c["crossed_bootstrap_95"][0] <= 0 <= c["crossed_bootstrap_95"][1] for c in contrasts.values()),
        raw_direction_changes_across_training_seeds=min(contrasts["raw"]["per_seed_delta"]) < 0 < max(contrasts["raw"]["per_seed_delta"]),
        negative_learning_order_seeds={regime: sum(delta < 0 for delta in c["per_seed_delta"]) for regime, c in contrasts.items()},
        repair_mechanism_confirmed=False,
        improved_future_task_learning_speed_measured=False)
    (ROOT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# 第二轮附加对照：过滤公开无效动作反馈", "",
        "本对照因训练数据中无效动作标签比例不同而加入；方法选择未使用第二轮训练模型的 probe 分数。保持共同成功任务/采样种子集合相同，过滤公开反馈恰为 `Nothing happens.` 的步骤后重新匹配每对示范数量。不是新的独立确认测试。", "",
        f"每个来源保留 {result['data_selection']['matched_examples']['none']} 条示范，覆盖 {result['data_selection']['common_successful_tasks']} 个共同成功任务。每个模型实际处理 256 条示范、更新 64 次，seed 为 101/202/303。", ""]
    if result["interpretation"]["all_source_gap_intervals_include_zero"]:
        lines.extend(["## 主结果判断", "", "三种来源差距的 95% 交叉 bootstrap 区间均覆盖零。原始数据的差距在三个训练种子之间换方向，不能把单个种子的反转称为稳定规律；区间覆盖零也不证明两种来源等效。", ""])
    negative = result["interpretation"]["negative_learning_order_seeds"]
    lines.extend([f"相对即时优势方向出现负向学习排序的种子数：raw {negative['raw']}/3，matched {negative['matched']}/3，filtered {negative['filtered']}/3。过滤后没有再出现本次观察到的单种子反转，但原反转本就不稳定，不能据此确认普遍的修复机制。", "",
        "正的来源差距表示早期支持来源训练后更好；负值才与其即时优势方向相反。对照是否修复排序冲突，应看方向与重复性，不能用来源差距的绝对值是否缩小来代替判断。本轮不足以推进 meta-RL 或声称可预测规律。", ""])
    for key, title in (("fresh_primary", "28 个新 probe：主指标"), ("all40_secondary", "全部 40 个 probe：次指标")):
        sub = result[key]
        lines.extend([f"## {title}", "", f"基础模型：{sub['base_wins']}/{sub['n_tasks']}。每格按 seed 101/202/303 列成功数。", "",
            "| 数据制度 | 无支持来源 | 早期支持来源 | 平均来源差距及交叉 bootstrap 95% 区间（百分点） |", "| --- | --- | --- | --- |"])
        for regime in ("raw", "matched", "filtered"):
            c = sub["contrasts"][regime];lo, hi = c["crossed_bootstrap_95"]
            lines.append(f"| {regime} | {sub['cells'][regime + '-none']['wins_per_seed']} | {sub['cells'][regime + '-early']['wins_per_seed']} | {100*c['early_minus_none']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] |")
        diff = sub["matched_gap_minus_filtered_gap"]
        lines.extend(["", f"匹配差距减过滤后差距：{100*diff['estimate']:+.2f} 个百分点；区间 {[round(100*v,2) for v in diff['crossed_bootstrap_95']]}。该差值描述来源差距大小的变化，不是修复排序冲突的直接指标。", ""])
    lines.extend(["## 解释边界", "", "此过滤只使用动作后的公开文本，不查专家、可用动作列表或隐藏状态。`Nothing happens.` 不是所有失败动作的完备检测器。", "",
        "过滤与重新匹配共同改变了训练示范分布，不能将来源差距的变化全部归因于无效标签。仍只有一个小模型家族、13 个共同成功任务和三个训练种子；零差距不证明等效。没有新任务学习速度、强教师或在线 RL 的结果。", "",
        "记录位于 `runs/dual5090-v2-filtered/`；主匹配实验位于 `runs/dual5090-v2/`。全部为开发实验，正式 seen/unseen benchmark 未使用，付费 API 调用为 0。"])
    Path("docs/DUAL5090_FILTERED_RESULTS.zh-CN.md").write_text("\n".join(lines) + "\n")
    return result["fresh_primary"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "report"])
    args = parser.parse_args()
    print(json.dumps(build() if args.command == "build" else report(), indent=2))
