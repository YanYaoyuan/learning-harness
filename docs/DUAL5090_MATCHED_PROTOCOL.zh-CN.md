# 第二轮：成功数据的覆盖与数量对照

目标：检查第一轮支持来源的学习差距能否在更大训练池、独立训练种子与新开发 probe 上重复，以及该差距对成功数据选择和数量是否敏感。

## 冻结设计

- Qwen3-4B-Instruct-2507，沿用固定模型 revision、系统提示、公开观测和命令兼容层。
- 96 个预留训练任务，四种类型各 24 个；无支持和早期支持各采样 seed 11/23，共 384 个采集回合。
- 支持内容仍为第一轮的固定手写公开策略。第 0–3 个决策可见，此后从上下文移除。不使用隐藏 facts、专家计划或 admissible-action oracle。
- 两种数据制度 × 两种来源 × 训练 seed 101/202/303，共 12 个 LoRA checkpoint。全部普通目标视角 SFT，global batch 4、64 次更新、上下文上限 2,048。
- 每个模型明确安排 256 条训练示范：按固定 seed 对 corpus 逐轮打乱，够用时截取，数量不足时重复新一轮。再由统一 Trainer 采样。此控制避免数据集尾部不足 4 条造成的实际梯度批量变化；匹配条件的两个来源使用相同索引计划。记录实际处理条数并在结果生成时核验。
- 40 个开发 probe，全部 H0、40 步上限。28 个此前未评测的任务是主指标，旧的 12 个单独列为次指标，不混作新证据。正式 seen/unseen benchmark 不运行。
- 基础模型在两张卡上各重复评测一次相同的 40 任务，检查批量协议的结果重复性。此校准项在任何第二轮 probe 结果产生之前加入，记录于 `amendments.json`，不改变训练网格。
- 不使用付费 API，不运行 meta-RL、强教师或后续新任务学习速度实验。

## 两种数据制度

**raw**：保留各来源的成功训练轨迹，去掉显式支持消息。先统一按 tokenizer 过滤超长示范并记录数量。

**matched**：取两种来源都成功的同一 task_id/rollout_seed 对。每对在两边保留相同数量的动作示范，即两边可用步骤数的较小值；较长轨迹按时间等距抽取并保留可用首尾。两个来源的 corpus 顺序按 task/seed/matched_index 对齐，使同一训练种子使用相同的任务配额和采样顺序。

匹配控制任务/采样种子、成功回合数及动作示范数，不控制确切 token、轨迹质量和行动顺序。共同成功是干预后的选择条件，因此该对照只诊断选定子群体，不能作为总体 harness 因果效应的无偏估计。raw 与 matched 的差距变化也不能全部归因为任务覆盖。

## 推理与校准

第二轮统一使用固定 8 槽位的本地批量协议。采样为每个请求维护独立 torch.Generator，temperature=0.7、top_p=0.9；评测采用 greedy。按实际非 padding 输入和至 EOS 的输出计 token，同时记录物理 padding 量。

采集阶段使用串行环境后端；恢复旧采集时给 `learning_harness.batch_runner` 加 `--serial-env`。后续评测改为独立进程环境执行，转换前以相同动作重放基础模型全部 40 个任务、1,444 次动作，初始观察、公开反馈和成败逐条一致。模型推理批量与槽位顺序保持不变，run 元数据记录实际后端；校验记录见 `parallel-environment-validation.json`，执行转换见 `runtime-transition.json`。

已用非退化随机分布检查请求随机流独立性，并核对真实任务。发现 BF16 在不同批次形状下可能改变相同提示的 greedy 输出，因此第二轮完全重新采集，并重跑本轮基础模型。不拼接第一轮轨迹或将第一轮绝对分数当作第二轮的控制组。两轮的批量/逐条差异也不解释为学习效应。

出错立即退出，不默默跳过失败请求。恢复时核对运行指纹，以已完成 episode 为准；SFT 导出通过 attempt_id 排除中断留下的步骤。

## 分析

分别在 28/40/12 个任务子集报告每个训练种子的成功数和来源差距。对支持减无支持的差距、raw 差距减 matched 差距，联合配对重采样训练种子和任务组。三个训练种子的精度仍有限，区间含零不构成等效检验。

新任务结果用于检查第一轮差距能否重复；matched 结果用于区分数据选择与来源效应。轨迹质量、支持依赖及跨模型迁移仍需独立对照。

## 依据训练数据加入的附加对照

384 回合采集完成后，成功轨迹中无支持来源有 89/405 个动作收到公开反馈 `Nothing happens.`，早期支持有 73/953 个。为检查普通 SFT 学入无效标签的解释，在读取训练模型的 probe 结果前冻结 `runs/dual5090-v2-filtered/prespec.json`。

该对照仅过滤结果文本恰为 `Nothing happens.` 的动作，再按相同规则匹配；共同成功任务/采样种子集合保持一致，13 个任务各来源 255 条示范。新增两个来源 × 三个训练种子，仍每模型实际处理 256 条样本、更新 64 次。主网格结束后串行接续，不与其争用 GPU。

此附加分析沿用同一开发 probe，不是新的独立确认数据。过滤和重新匹配同时改变样本分布，不能将来源差距的变化全部归因于无效标签。

## 运行产物

位于 `runs/dual5090-v2/`：`protocol.json`、`batch-validation.json`、`collect-gpu*/`、`corpora/`、`data-selection.json`、`jobs.json`、`checkpoints/`、`evaluation/`、`job-progress.json` 和 `results.json`。

采集完成后：

```bash
PYTHONPATH=src .venv-model/bin/python scripts/matched_study.py build
# 回收本实验的基础模型采集服务，释放显存后运行。
PYTHONPATH=src .venv-model/bin/python scripts/run_local_jobs.py
PYTHONPATH=src .venv/bin/python scripts/matched_study.py report
```

冻结后不要重新运行 prepare 或覆盖已有 corpus。结果报告由 `matched_study.py report` 在全部 checkpoint 和评测完成后生成。
