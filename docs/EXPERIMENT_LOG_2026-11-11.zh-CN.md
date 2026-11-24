# 实验记录快照：2026-11-11

客户端日期：2026-11-11。记录采集时间：2026-11-11T09:15:31.994704+08:00。

| 阶段 | 状态 | 主要记录 |
|---|---|---|
| 双 5090 首轮/匹配/过滤 | 已完成 | 直接行动、提示、SFT/上下文方法；见对应 docs 与 runs |
| learning-speed-v1 | 已完成 | 九条学习曲线；没有稳定一般来源优势 |
| api-harness-v1 | 已完成 | 辅助采集提高成功率，学习优势仍不确定 |
| transfer-factorial-v1 | 已完成 | 30 条轨迹、7,200 次主评测；整体交互不稳定，任务族差异需保留 |
| baseline-benchmark-v1 | 已完成 | 40 题 × 4 条件；执行基线偏弱；另报去除示例重合后的 39 题 |
| source-comparison-v1 | running | 新的候选方案与 ReAct 自采→SFT 对照，尚未完成 |

## 快照时的核心对照

| 来源 | 新采集成功/尝试 | 合格动作标签（匹配前） |
|---|---:|---:|
| 候选：教师辅助 | 69/144 | 726 |
| 强基线：ReAct | 110/144 | 981 |

共同成功的 task/seed 对：58；任务组：34；每源匹配标签：452。每个 A 模型实际预算 256 次样本呈现、64 更新、3 个训练种子。
快照时新 A 模型完成 6/6，新 B 轨迹完成 1/12，统一评测已提交 95/4284 个任务结果。已有 6 条无 A 参考权重会重新评测，不将旧分数混入新表。
采集成功率不等于学习优势；在适应与评测完成前，不对候选方法优劣作最终判断。

## 设计说明

- 执行基线的四条方法均为对照，属于已有执行方法。
- 官方示例与开发任务发生过重合：执行基线统一剔除一题后另报结果；本轮在启动前排除 A 查询中一条匹配 ReAct 示例的加热任务。
- 当前研究仍在 ALFWorld 文本环境中，采用文字观测与离散命令。
- 原始 AUC、起点、gain AUC、终点分别统计；不能把更强起点直接叫作更快学习。
- 官方最终 seen/unseen 保留，本轮复用开发查询，只作探索性证据。

## 查阅路径

- [历史结果审计](EXPERIMENT_AUDIT_2026-11-10.zh-CN.md)
- [执行基线结果及图](../output/baseline-benchmark-v1/RESULTS.zh-CN.md)
- [当前设计](../runs/source-comparison-v1/design.json)与[数据选择记录](../runs/source-comparison-v1/selection.json)
- [W&B 当前运行](https://forge.coreweave.com/wandb/yanyaoyuan-jawaharlal-nehru-university/learning-harness-lab/runs/557cc84a609d)
- GitHub 文件是上传时的快照；后续进度以运行机器与 W&B 为准。
