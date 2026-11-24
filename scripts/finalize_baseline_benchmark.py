#!/usr/bin/env python3
"""Generate the benchmark report after all task records are complete."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('runs/baseline-benchmark-v1')
OUT=Path('output/baseline-benchmark-v1')
METHODS=['json_direct','act','react_retry','reflexion']
LABELS={'json_direct':'Direct JSON','act':'Act（相同示例、去掉思考）','react_retry':'ReAct + 重试','reflexion':'Reflexion'}


def ready():
    for method in METHODS:
        path=ROOT/'development'/method/'tasks.jsonl'
        if not path.exists():return False
        try:rows=[json.loads(s) for s in path.read_text().splitlines()]
        except json.JSONDecodeError:return False
        if len(rows)!=40:return False
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    started=time.monotonic()
    while not ready():
        if not args.watch:raise RuntimeError('Development suite incomplete')
        if time.monotonic()-started>6*3600:raise TimeoutError('Development suite did not finish')
        time.sleep(15)
    lock=json.loads((ROOT/'pre-run-lock.json').read_text())
    for filename,expected in {**lock['files'],**lock['configs']}.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=expected:
            raise ValueError('Original execution input changed: '+filename)
    env=dict(os.environ,PYTHONPATH=str(Path('src').absolute()),MPLCONFIGDIR='/tmp/lh-mpl')
    with (ROOT/'report.log').open('w') as log:
        subprocess.run(['.venv-model/bin/python','scripts/baseline_benchmark_report.py'],env=env,
            stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((OUT/'summary.json').read_text());assert result['complete']
    rows={r['method']:r for r in result['summaries']};clean=result['excluding_demo_overlap']
    lines=['# ALFWorld 基线开发实验：实测结果','',
        '日期：2026-11-10。Qwen3-4B-Instruct-2507；双卡 5090。本轮为重复使用的开发集、单个环境/采样种子，不能作为正式测试或论文主结论。',
        '', '## 完整 40 任务', '',
        '| 方法 | 首次成功 | 预算内累计成功 | 平均调用 | 平均输入 token | 平均输出 token |',
        '|---|---:|---:|---:|---:|---:|']
    for m in METHODS:
        r=rows[m]
        lines.append(f'| {LABELS[m]} | {r["first_successes"]}/40 ({100*r["first_rate"]:.1f}%) | '
            f'{r["cumulative_successes"]}/40 ({100*r["cumulative_rate"]:.1f}%) | {r["mean_calls"]:.1f} | '
            f'{r["mean_input_tokens"]:.0f} | {r["mean_output_tokens"]:.0f} |')
    lines += ['', '最多 3 次尝试，每次最多 50 回合，总计不超过 150 次模型调用。反思计入预算；成功即停止。'
        '同调用/输出预算上限不代表实际 token 消耗相同。', '', '## 剔除已识别示例重合任务', '',
        '| 方法 | 剩余任务 | 首次成功 | 累计成功 |', '|---|---:|---:|---:|']
    for m in METHODS:
        r=clean[m];n=r['tasks']
        lines.append(f'| {LABELS[m]} | {n} | {r["first_successes"]}/{n} ({100*r["first_successes"]/n:.1f}%) | '
            f'{r["cumulative_successes"]}/{n} ({100*r["cumulative_successes"]/n:.1f}%) |')
    lines += ['', '剔除规则是公开目标与可见编号容器集合匹配官方 actor/反思示例；四组统一剔除，不按结果选择。'
        '已确认一条生菜清洗任务可被官方示例动作直接解决。这个审计不能排除模型预训练污染。', '',
        '## 配对效果与能说明的范围', '']
    for name,label in [('reasoning_increment','ReAct − Act'),('reflection_increment','Reflexion − ReAct retry')]:
        for key,title in [('first_success','首次'),('success','累计')]:
            r=result['exploratory_paired_effects'][name][key]
            lines.append(f'- {label}，{title}成功率差：{r["difference_pp"]:+.1f} pp；任务组配对 bootstrap 95% CI '
                f'[{r["ci95_pp"][0]:+.1f}, {r["ci95_pp"][1]:+.1f}] pp。')
    pair=result['prompt_audit']['first_request_pairing']
    lines += ['',f'ReAct 与 Reflexion 的 {pair["paired_tasks"]} 个首请求具有相同 prompt 和 request seed；'
        f'其中 {pair["identical_raw_outputs"]} 个原始输出完全相同。反思在首次失败之后才发生，首次差异不应归因于反思机制。',
        '', '本轮可以校准旧执行基线，并区分示例/思考/同任务重试的候选收益。它没有更新模型参数，'
        '因此不能证明学习速度、参数内化、跨任务持续学习或 RSI。只看旧 action-only 基线不足以证明对成熟 agent harness 的优势；'
        '但这也不自动推翻旧实验内部的收集和特定任务迁移比较。', '',
        '## 下一步', '',
        '1. 在统一协议下重评已登记的 9 个 SFT 起点，不能直接复用旧分数。',
        '2. 增加 ReAct 自采成功轨迹 → 相同预算 SFT，作为原教师辅助采集的强数据来源对照；'
        '按相同任务覆盖、成功筛选和更新预算匹配，再考察后续适应曲线。',
        '3. 对正式协议完成示例来源审计、明确 few-shot 与目标类型零示例迁移的区别后，'
        '再冻结 seen 140 / unseen 134 的测试。正式测试本轮未执行。', '',
        '## 产物与检查', '',
        '- `summary.json`：真实计费、配对效果、示例审计；`metrics.csv` 与剔除后 CSV。',
        '- `success-and-cost.png/pdf`：首次、累计、调用数/输出 token 预算曲线。',
        '- `task-family.png/pdf` 与 `excluding-demo-overlap.png/pdf`。',
        '- 全部原始任务记录通过 calls/events/token 一致性和预算检查；运行输入冻结校验通过。',
        '- `partition-audit.json`：train/dev/probe/pilot/seen/unseen 的 task_id、task_group、game SHA256 均无交集。',
        '- 模型及 tokenizer 实体文件校验值、9 个 adapter 权重校验值和软件版本已记录。',
        '- `wandb.json` 保存本轮云端运行地址；云端完成状态以该文件及服务端验证为准。','']
    (OUT/'RESULTS.zh-CN.md').write_text('\n'.join(lines))
    (OUT/'finalization.json').write_text(json.dumps(dict(complete=True,task_method_pairs=160,
        execution_lock_verified=True,final_test_executed=False,report_sha256=hashlib.sha256((OUT/'RESULTS.zh-CN.md').read_bytes()).hexdigest()),indent=2)+'\n')
    print('Finalized 160 task-method pairs, figures and measured report.',flush=True)


if __name__=='__main__':main()
