# ALFWorld 基线与评测协议 v1

日期：2026-11-10。状态：开发协议已冻结并开始执行；正式测试配置为候选，尚未执行。

## 这轮回答什么

补齐现有研究的执行基线和计费口径，确认相同模型在更成熟的推理/重试机制下的表现。它不直接检验“更好的 harness 培养更好的学习者”，也不替代后续参数学习对照。旧实验及其收集收益、任务特定迁移证据保持有效。

本轮只进行本地推理，无新增训练、无教师 API 消费。40 个 development 任务曾用于开发，不能作为新鲜确认性证据；所有结论标记 exploratory。不将开发结果与论文排行榜分数直接比较。

## 基线及代码来源

| 条件 | 实现与用途 | 预算 |
|---|---|---|
| Direct JSON | 保留旧 action-only 提示，零示例；连接旧实验 | 最多 3 次尝试 |
| Act | 官方 ReAct 示例轨迹，删除 think 行及其 OK 反馈；不生成 think | 同上 |
| ReAct + retries | 官方两条示例，允许 think/action；失败后重新开始，不保存失败经验 | 同上 |
| Reflexion | 与 ReAct 相同 actor；失败后生成计划，下一次同任务尝试携带计划 | 同上；反思计入总调用预算 |

ReAct 来源：https://github.com/ysymyth/ReAct ，commit `6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9`。

Reflexion 来源：https://github.com/noahshinn/reflexion ，commit `218cf0ef1df84b05ce379dd4a8e47f17766733a0`。

已下载官方示例、相应实现和 MIT 许可证至 `third_party/`。只使用审阅后的提示文本，自有 runner 执行环境，不运行上游脚本。

这是 **controlled local port（受控本地移植）**，不是论文原始分数复现。与上游不同：Qwen3-4B 聊天模型、temperature=0.7、50 回合、最多 3 次尝试、最近 12 回合上下文、128 token 响应上限、统一预算、无重复动作提前退出。上游历史上下文和模型等设置不同；Reflexion 原实现遇到重复动作可提前结束。本轮统一取消该启发式，避免它单独改变重试预算。反思同样只读取 reset 与最近 12 回合公开历史，不能假装完整历史复现。

示例按**公开任务指令**中的任务类型选择，不读环境隐状态/专家轨迹/合法动作列表。两条示例索引固定为 1、0。Act 删除同一轨迹的思考内容，动作与观察序列保持一致；不直接拿上游另一组不匹配的 Act 示例作因果对照。直接行动与 Act 的差别还包括示例、输出格式，不能全部归因于思考。

## 模型、环境与隔离

- 模型：`Qwen/Qwen3-4B-Instruct-2507`，revision `cdbee75f17c01a7cc42f958dc650907174af0554`；BF16，本地双卡 5090，各一独立服务。
- ALFWorld 0.4.2，TextWorld 1.7.0；文本环境，不包含图像、机械臂控制或机器人轨迹。
- 仅公开 reset/observation 和任务结束成功信号；不暴露 facts、expert、admissible_commands。对象编号固定不打乱，记录 game 文件 SHA256。
- 失败重试重置到同一任务、同一环境种子；reset 文本不一致直接报错。反思记忆仅在同一任务尝试间保存，不跨任务，不更新模型权重。
- ReAct 与 Reflexion 第一次尝试使用相同 messages、逐请求随机种子；BF16/动态 batch 的数值差异意味着不承诺逐位一致。报告首次表现差异作为实现检查，不暗中强制相等。
- 既有运行目录不改写；新目录 `runs/baseline-benchmark-v1/`。旧冻结 runner/server 保持原样，新增独立执行器。

## 预算与成功口径

每任务/种子：最多 3 次尝试、每次最多 50 回合、总共最多 150 次模型调用、每次最多 128 输出 token。因此输出 token 上界 19,200；**反思也占一个调用、占真实输入输出 token**。Reflexion 最坏情况下前两次各 50 回合，加 2 次反思后，第三次最多 48 回合。

think 和非法动作均消耗一个环境回合，与原 ReAct 的计步方式对齐；think 的公开反馈替换为 OK。文本协议仅执行首行，完整生成内容的 token 仍计费。`put ... in/on ...` 统一规范化为本环境使用的 `move ... to ...`。生成多行不允许一次执行多个动作。成功后停止，不再生成反思/额外尝试。

这是相同**最大调用/输出预算**，不是相同实际耗时或输入 token；示例和反思会增大输入 token。必须同时报告实际 input/output tokens、调用数和预算曲线。禁止把 Reflexion 累计成功率与其他方法单次成功率混在一列。

主报告：首次尝试成功率、相同预算内累计成功率；后者称 cumulative success，不能称独立采样 pass@k。附成功率对调用数、输出 token 上界曲线，各任务类型结果，以及每任务平均真实 token。

## 划分与正式测试

| 划分 | 数量 | 用途 |
|---|---:|---|
| 本轮 dev | 40（place/clean/cool/heat 各 10） | 工程与基线检查，seed 11；已重复使用 |
| 官方 valid_seen | 140，六类 | 候选正式 seen 测试；未执行 |
| 官方 valid_unseen | 134，六类 | 候选正式 unseen 测试；未执行 |

官方 final manifest、源代码、示例均记录 SHA256。正式候选每个条件 seeds=11/23/37，同一任务种子配对；最终方案冻结后才能运行。`learning_harness.benchmark` 检测正式任务即要求 `--freeze`，验证配置 digest 和源文件/manifest 校验和；`final-freeze.template.json` 的状态为 `development_only`，不能启动正式测试。这是避免边看测试边调参的技术约束，不是额外申请用户授权。

正式报告需同时给出全部六类、原四类及两类训练未见目标。**使用官方任务类型示例的 Act/ReAct/Reflexion，对 light/two-object 也属于 few-shot 暴露；不能将这部分称为零示例目标迁移。** 零示例目标迁移只能在无该类型示例/反思示范暴露的独立条件下评估，正式配置冻结前须明确这两种问题。

## 统计与可追溯性

开发图采用 Wilson 95% 区间说明小样本不确定性。方法差异使用同任务配对，task_group 聚类 bootstrap 5,000 次；不把动作、重试当成独立样本。不以开发集显著性挑选正式主假设。

正式确认性主比较建议限定为 Reflexion − ReAct retry 的预算内成功差异，并另行预注册学习实验的主比较。正式多种子分析应以任务组聚类、保留同组多种子相关性，报告种子明细；多个次级比较需标记探索性或校正，seen/unseen 不合并挑最高值。

每次调用记录 messages、原始输出、公开观察、动作、think/非法格式、trial、seed、逐请求 seed、真实 token；每任务记录尝试序列和累计预算。中断重跑采用新 run_id，分析仅使用已完成任务提交的 run_id，不把残留半条轨迹重复计入。恢复须匹配配置、模型 revision、源代码/示例/manifest 指纹。报告核对 calls/events/token 总和和预算上限，不只信日志汇总。

## 参数学习基线的边界与后续准入

已登记旧 API-harness 的 matched-success SFT：none/generic/api 三种收集来源 × 101/202/303 三种训练种子，共 9 个 adapter，记录权重 SHA256。登记文件 `checkpoint-registry.json` 明确它们**尚未在本轮新协议重评**。旧分数不能直接拼进新表格，因为 horizon、sampling 等有变化。

后续若继续检验“培养更好的学习者”，至少需要：

1. 用同一冻结评测协议重评这 9 个起点，区分初始能力、额外适应收益与最终水平；旧三种子全部保留，不挑最佳 checkpoint。
2. 在相同任务覆盖/成功筛选、训练样本与更新预算下，对照成功轨迹 SFT 和教师蒸馏。已有 success-SFT 不是 on-policy distillation，不能重新命名冒充。
3. 若主张 agentic RL 增益，再实现并验证 GRPO/GiGPO 等对应训练对照；当前未执行，不能写成已击败 SOTA。
4. 本轮先检验强一点的执行机制是否已解决原失败。如果是，优先追问哪些收益必须参数化、哪些只是执行策略；这不自动否定此前数据收集与特定任务迁移收益。

## 运行与产物

```bash
PYTHONPATH=src .venv-model/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python scripts/run_baseline_benchmark.py
PYTHONPATH=src .venv/bin/python scripts/run_baseline_benchmark.py --run
PYTHONPATH=src MPLCONFIGDIR=/tmp/lh-mpl .venv-model/bin/python scripts/baseline_benchmark_report.py
.venv-wandb/bin/python scripts/track_baseline_benchmark.py --watch
```

执行前 `prepare` 检查并冻结本轮 inputs/configs，已存在且内容变化会报错。冻结是版本标记，不代表所有科学问题已预注册完成。产物为 `output/baseline-benchmark-v1/summary.json`、CSV、PNG/PDF 和 W&B 链接。原始轨迹仅存本地；云端上传明确列出的实验指标、任务结果表、图和协议，不上传密钥文件或整个工作区。
