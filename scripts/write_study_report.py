#!/usr/bin/env python3
"""Generate the first-round experiment report from completed results."""
import json
from pathlib import Path

ROOT = Path("runs/dual5090-v1")


def main():
    result = json.loads((ROOT / "results.json").read_text())
    if not result["grid_complete"]:
        raise ValueError("Finish all six checkpoints before writing the completed report")
    pilot = result["pilot"]["metrics"]
    comparisons = result["learning_comparisons"]
    selected = json.loads((ROOT / "selection.json").read_text())["arm"]
    base = comparisons["target_sft"]["base_wins"]
    n = comparisons["target_sft"]["n"]
    rows = []
    for field, label in (("none_wins", "无支持"), ("support_wins", selected)):
        cells = [f"{comparisons[m][field]}/{n}" for m in ("target_sft", "mixed_sft", "context_distill")]
        rows.append("| " + " | ".join([label] + cells) + " |")
    gap_rows = []
    for method, values in comparisons.items():
        interval = values["task_bootstrap_interval_95"]
        gap_rows.append(f"| {method} | {values['source_arm_delta'] * 100:+.2f} | [{interval[0] * 100:+.2f}, {interval[1] * 100:+.2f}] |")
    immediate = pilot[selected]["success_rate"] - pilot["none"]["success_rate"]
    target_gap = comparisons["target_sft"]["source_arm_delta"]
    verdict = ("出现即时排序与学习排序相反的候选现象，尚未证明稳定。"
               if immediate * target_gap < 0 else "当前没有观察到普通 SFT 下即时排序与学习排序相反的现象。")
    resources = []
    for job in result["trained_evaluation"]:
        t = job["training"]
        resources.append(f"| {job['checkpoint']} | {t['used_examples']} | {t['hinted_examples']} | {t['target_tokens_rank0']} | {t['teacher_input_tokens_rank0']} | {t['peak_cuda_allocated_bytes'] / 1024**3:.2f} |")
    coverage = result["training_data_coverage"]
    text = f"""# 双 5090 首轮实测

模型：Qwen3-4B-Instruct-2507，revision `cdbee75f17c01a7cc42f958dc650907174af0554`。本地 BF16 LoRA，64 次更新、global batch 4、训练 seed 101。两张 RTX 5090 各自执行独立任务，不要求跨卡模型并行。

## 即时表现

12 个 pilot 任务，固定公开策略支持：none {pilot['none']['success_rate'] * 12:.0f}/12，early {pilot['early']['success_rate'] * 12:.0f}/12，delayed {pilot['delayed']['success_rate'] * 12:.0f}/12，random_timing {pilot['random_timing']['success_rate'] * 12:.0f}/12。按预定规则选择 {selected}，即时差距 {immediate * 100:+.2f} 个百分点。pilot 参与选择，分数不是确认性证据。

## 参数更新后的独立 H0 表现

以下全部评估相同的 {n} 个开发 probe；没有支持消息、教师缓存或测试时参数更新。未训练基础模型为 {base}/{n}。

| 训练轨迹来源 | 普通 SFT | 上下文混合 | 本地上下文蒸馏 |
| --- | --- | --- | --- |
{chr(10).join(rows)}

同一方法下，支持来源减去无支持来源的配对差距：

| 方法 | 差距（百分点） | 任务组 bootstrap 95% 区间（百分点） |
| --- | --- | --- |
{chr(10).join(gap_rows)}

区间仅描述这一个训练 seed 的任务差异，不包含训练随机性；零差距或区间含零不能证明等效。

## 结果分析

{verdict}

混合和蒸馏均未缩小本轮来源差距的点估计。单训练种子和有限任务数不足以判断方法的稳定性或可迁移性。

后续对照扩大成功任务覆盖，重复独立训练种子，并匹配任务类型和示范数量，以区分覆盖与内容差异。跨模型、跨环境验证需要单独设计。

## 数据与限制

- 无支持来源：{coverage['none']['successful_episodes']} 个成功回合，覆盖 {coverage['none']['successful_task_groups']} 个任务组；支持来源：{coverage[selected]['successful_episodes']} 个成功回合，覆盖 {coverage[selected]['successful_task_groups']} 个任务组。全部由 16 个训练任务、每个条件 2 个采样种子采集。来源覆盖不同，不能把总差距直接归因于支持依赖。
- 每个来源的三个训练分支共用同一 corpus、标签、训练 seed、更新次数；本轮没有超长示范被丢弃。上下文蒸馏额外执行冻结本地基础模型的 teacher forward，不是同 FLOP 对照，也不是前沿模型教师。
- pilot/train/probe 的任务组互不重叠，全部来自官方训练 split 的预留开发任务；正式 seen/unseen benchmark 未运行。12 个 probe 的分辨率为每任务 8.33 个百分点。
- 测的是更新后的任务能力，未测未来新任务的学习速度、持续学习或在线 RL。不声称复现 SkillScaffold、EvoIn、Harness-Zero 原论文。
- 支持只有一种通用策略，改变的是出现时机；没有覆盖答案代办、替代推理或逐步技能反馈等更强干预。因此没有排序反转不能否定更广义的研究命题。
- 付费 API 调用为 0。接口修正前 calibration 回合与启动器依赖错误日志均排除于研究统计之外。

## 训练资源实测

显存是 PyTorch 记录的本轮样本峰值，不是占满 2,048 上下文时的上界。

| checkpoint | 示范数 | 混合中有支持示范数 | 目标 token | 额外 teacher 输入 token | 峰值显存 GiB |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(resources)}

原始数据：`runs/dual5090-v1/results.json`、各阶段 `episodes.jsonl`/`steps.jsonl`、`checkpoints/*/experiment.json` 与 `trained-eval/*/summary.json`。协议见 `docs/DUAL5090_DISCOVERY.zh-CN.md`。
"""
    output = Path("docs/DUAL5090_RESULTS.zh-CN.md")
    output.write_text(text)
    print(output)


if __name__ == "__main__":
    main()
