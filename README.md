# Learning-Oriented Harness Lab

研究目标：检验提高即时求解表现的 harness，是否也最有利于模型形成持久、可迁移的能力。

**当前状态（2026-11-13）**：Stage 3完整结果及检查点复测已归档，Stage 4按冻结方案训练。独立监督器每30分钟核对真实进程/GPU，每30秒监听结束事件；中途成对图在固定完成数量节点上传。训练进度以[W&B实时记录](https://forge.coreweave.com/wandb/yanyaoyuan-jawaharlal-nehru-university/learning-harness-lab/runs/76f7e74a6c3b-progress)为准。

| 实验 | 状态 | 结果与入口 |
|---|---|---|
| API辅助 vs ReAct自采，共同成功匹配 | 完成4,284次评测 | 主AUC差−3.56pp；三种子−9.77/−3.42/+2.49pp。[完整解读](docs/SOURCE_COMPARISON_V1_RESULTS_2026-11-11.zh-CN.md) · [曲线](output/source-comparison-v1/adaptation-curves.png) |
| Stage 0：重复评测诊断 | 完成1,248回合 | 旧来源比较中，批布局重复分量占比约0.46%。[结果](docs/VARIANCE_DIAGNOSTIC_STAGE0_RESULTS_2026-11-12.zh-CN.md) |
| Stage 1：初始化×采样 | 完成34模型/2,176回合 | 抽样分量占比估计50.8%；逐单元与敏感性见[结果](output/variance-diagnostic-v1/stage1/RESULTS.zh-CN.md)。 |
| Stage 2：同预算采样干预 | 完成68模型/4,352回合 | 全池相对子集复用：API +5.18pp、generic +1.27pp。[最终结果与图](output/variance-diagnostic-v1/stage2/RESULTS.zh-CN.md) |
| Stage 3：任务与不同内容预算控制 | 完成70模型/4,480回合，云端归档完成 | 低NLL−随机：API −17.77pp、generic −7.81pp；四初始化平均差异方向一致。同任务位置/次数、192步/768次呈现及256条不同内容。[最终结果与图](output/variance-diagnostic-v1/stage3/RESULTS.zh-CN.md) · [汇总数据](output/variance-diagnostic-v1/stage3/aggregate-summary.json) |
| Stage 3固定检查点复测 | 完成2检查点/128新回合 | 逐任务成败与步数均与原记录一致；不混入原主分析。[结果与图](output/variance-diagnostic-v1/stage3-reliability-check/RESULTS.zh-CN.md) |
| Stage 4：任务×动作构成控制 | 2026-11-13 01:28启动，冻结100模型/6,400回合 | 六初始化×四采样配方，另列4个条件重训；两策略逐位置任务/动作类型与不同内容配额匹配。18累计GPU小时上限。[协议](docs/STAGE4_ACTION_CONTROL_SPEC_2026-11-13.zh-CN.md) · [预设中途结果与图](docs/STAGE4_INTERIM_MONITORING_2026-11-13.zh-CN.md) |

Stage 0—3使用旧API−generic经验池，不能替代API−ReAct强基线。原冻结分析与区间保留，少种子bootstrap尚未完成离散结果校准；采样分量的敏感性范围不是置信区间。Stage 2中途6/8/12/16组图保留为历史记录，主结果以完整16单元为准。

Stage 4执行和中途绘图：[16份调度设计](docs/TASK_ACTION_CONTROL_CANDIDATE_2026-11-12.zh-CN.md) · [共用队列、报告与汇总云端记录](docs/CONTROLLED_STUDY_EXECUTION_2026-11-12.zh-CN.md)。半小时监督器已覆盖Stage 2/3、固定检查点复测与Stage 4；不按观察到的效果提前停止。其余准备：[API/ReAct全合格成功经验预算审计](docs/FULL_SOURCE_COMPARISON_PREPARATION_2026-11-12.zh-CN.md) · [WebShop独立CPU环境与协议检查](docs/WEBSHOP_PREPARATION_2026-11-12.zh-CN.md)。WebShop完整118万商品库、索引和12,087个目标身份已固定，公开接口与目标池隔离已通过完整目录train机械检查。[32任务×两公开提示的校准入口](docs/WEBSHOP_DEVELOPMENT_PILOT_PROTOCOL_2026-11-13.zh-CN.md)已实现并通过CPU/本机HTTP/替身进程生命周期测试；真实模型校准等待Stage 4完成，正式效果实验尚未启动。[论文工作提纲](docs/PAPER_WORKING_OUTLINE_2026-11-12.zh-CN.md) · [研究决策及相关工作](docs/RESEARCH_DECISIONS_2026-11-11.zh-CN.md)。

研究入口：[方案与思路](docs/RESEARCH_OVERVIEW.zh-CN.md) · [历史实验日志](docs/EXPERIMENT_LOG_2026-11-11.zh-CN.md) · [核心对照协议](docs/SOURCE_COMPARISON_V1.zh-CN.md) · [GitHub快照说明](docs/GITHUB_SNAPSHOT.zh-CN.md)。

[独立NLL评分诊断](docs/NLL_SCORING_DIAGNOSTIC_READINESS_2026-11-13.zh-CN.md)已完成853条不同输入的CPU编码、原16份选样配方精确再生及16项CPU测试，首次GPU评分前冻结67份输入。600秒GPU1方案检查填充、position IDs、精度和logits窗口；真实运行入口已验证会在Stage 4仍运行时拒绝启动。尚未加载GPU模型或产生新评分，不改旧缓存、训练配方和主实验队列。

[后续实验只读观察器](docs/FOLLOWUP_MONITORING_2026-11-13.zh-CN.md)已在真实主机启动，统一跟踪评分诊断与WebShop校准，每30秒更新汇总进度、每半小时追加进程检查；目前两项仍为未启动。它使用独立W&B进度run，不启动或重启任何GPU任务，不更改主实验监控。

## CPU 开发检查

```bash
python3 -m venv .venv-dev
.venv-dev/bin/python -m pip install -e '.[dev]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-dev/bin/python -m pytest -q
```

独立开发环境已通过216项CPU测试，包含种子控制、采样干预、预算预留、失败记账、方差分解、存储迁移、轨迹日志一致性、后续研究的冻结/中途观察及空闲GPU上下文核验。GPU训练和真实环境依赖仍使用下述专用环境。若工作机的ROS等工具设置了全局`PYTHONPATH`，运行独立测试时先清除该变量。[CPU测试工作流模板](docs/ci/cpu-tests.yml)使用同一依赖入口；当前GitHub登录不含workflow权限，远端CI尚未启用。

[实验协议：环境、baseline、benchmark 与统计口径](docs/EXPERIMENT_PLAN.zh-CN.md) · [前期调研](output/pdf/agent-research/README.md)

[准备阶段的实测记录](docs/READINESS.json) · [固定数据清单摘要](docs/data_manifest.snapshot.json)

双 5090 的实际探索实验已完成[首轮实测](docs/DUAL5090_RESULTS.zh-CN.md)，使用[本地验证协议](docs/DUAL5090_DISCOVERY.zh-CN.md)：Qwen3-4B、固定公开策略支持、独立 pilot/train/probe，以及普通 SFT、上下文混合、本地上下文蒸馏六个 checkpoint。逐阶段原始日志位于 `runs/dual5090-v1/`。这组实验不需要教师 API，最终 benchmark 保持留出。

[第二轮实测](docs/DUAL5090_FILTERED_RESULTS.zh-CN.md)已完成 18 个模型：96 个训练任务，原始数据、成功任务/数量匹配、公开反馈过滤后三种对照，各重复三个训练种子。28 个此前未评测的开发 probe 是主指标。[冻结协议](docs/DUAL5090_MATCHED_PROTOCOL.zh-CN.md)、[主匹配结果](docs/DUAL5090_MATCHED_RESULTS.zh-CN.md)和[RL 奖励组诊断](docs/AGENTIC_RL_READINESS.zh-CN.md)分别记录设计、结果和边界；原始数据位于 `runs/dual5090-v2/` 与 `runs/dual5090-v2-filtered/`。

[实测图与论文图注](output/figures/harness-v2/README.zh-CN.md)包含逐种子配对、效应区间、数据组成、任务类型、奖励组和真实训练 loss 六组图，均导出 PNG/PDF/SVG、CSV 与独立重绘数据。使用 `python3 scripts/plot_study.py` 从原始结果核验并重绘。后续实验报告应同时交付逐种子图、不确定性和任务分解；有中途评测时再绘制真实学习曲线，不能用不同 checkpoint 的终点配对图代替。

[新任务学习速度实验](docs/LEARNING_SPEED_NEXT_PHASE.zh-CN.md)已完成九条主比较。early 相对 none 的原始学习曲线 AUC 高 1.48 个百分点，95% 区间为 [-6.51, 8.85]；尚无稳定来源效应证据。真实曲线、参数冻结验证和结果位于 `runs/learning-speed-v1/`；初始 `design.json` 保持冻结状态。

API 后续实验也已完成：432 个采集回合、9 个 A 训练、9 条 B 适应轨迹及 1,944 个评测回合。无提示、通用提示、API 策略的采集成功率分别为 17.36%、34.03%、47.22%。在三来源共同成功的 13 个任务组上匹配示范后，API 相对通用提示的原始 AUC 高 5.60 个百分点（95% 区间 [-1.39, 14.06]），扣除各自起点后的 gain AUC 高 1.43 个百分点（[-7.90, 10.20]）。采集收益明确，学习效率优势仍不确定；不能把区间跨零解释为等效。教师 72 次调用按未折扣标价计 ¥0.650292，不含本地算力。结果和图位于 `runs/api-harness-v1/main-result.json` 与 `figures/main/`。

后续实验 `runs/transfer-factorial-v1/` 已完成全部 30 条轨迹与 7,200 个主评测回合。跨族主交互效应为 +1.38 个百分点，95% 区间 [-6.19, 8.06]；gain 交互为 −2.78，区间 [-19.58, 14.03]，均不足以支持稳定的筛选规则效应。实验设计为：通用/API 来源 × 共同成功匹配/全部合格成功经验，加基础模型对照，在冷却与加热两个任务族上各做三种子比较，共 30 条 B 轨迹。新增 6 个 A 训练，复用 6 个既有 A adapter；每条 B 使用相同的 512 条任务均衡示范，检查点为 0/8/16/32/64/128。新留出集各有 32 个 B query、12 个校准任务、8 个动作 NLL 诊断任务，另有共同的 24 个 A 保持任务。最终跨任务族的双物体/灯光任务仍封存。加热属于 A 已见任务族，不能声称是全新技能。

训练前已发现同种子 BF16 训练并非逐位可重复（同卡无诊断重跑也有差异），记录在 `protocol-amendment-001.json`。诊断现由训练结束后的独立进程读取检查点，已验证权重文件不变。新 query 要在两族基础模型都通过预设校准门槛后才开放；不因预实验失败放宽科学门槛。主分析是来源效应在“全经验”和“共同成功匹配”两种规则下的差异，不能将差异简单归因于纯可塑性或某一覆盖机制。实际状态见 `execution.json` 和各级 `*-progress.json`。

## W&B 云端记录

当前使用 [Weights & Biases 私有项目](https://forge.coreweave.com/wandb/yanyaoyuan-jawaharlal-nehru-university/learning-harness-lab)，按实验、条件与随机种子分组。`learning-speed-v1` 与 `api-harness-v1` 已完成并补传，已完成的 v2 对照和六组图另有历史汇总 run。本地链接索引位于 `output/wandb/links.json`。记录器曾因空采集日志的除零错误退出，已修复并通过回归测试；两轮模型进程不受影响，未重启训练。补传结束后记录器正常退出，新实验需启动下述 `--watch` 命令。

记录器是独立的只读进程，训练与环境虚拟环境保持原样。它补传实际 loss、学习率、梯度范数、完整检查点评测、逐任务表格、公开动作/反馈轨迹压缩包和 PNG 图。W&B 的原生 run 对应一个模型初始化，训练 A/B 与评测分别使用自己的 optimizer-step 轴。未完成的评测只进入进度表；已有评测点不补成虚构的中间点。回填的时间是上传时间，不能当作实验原始运行时间。

密钥由本机已有的 W&B 登录配置提供，不在实验配置或文件包内。上传范围使用明确字段及文件白名单，不包含 `.env`、请求认证头或模型权重。API 策略教师的百炼密钥与 W&B 登录凭据分别使用。

新轮冷却/加热观察器使用 `output/wandb-transfer/` 独立同步状态，在相同私有项目下按 `transfer-factorial-v1-cool` 和 `transfer-factorial-v1-heat` 分组，包含额外的 `diagnostic/task_mean_nll` 曲线。复用的 A loss 是原训练日志回填，不代表重复运行；两个族共享 A 训练，不能将云端重复展示的 A 记录重复计入算力。启动命令为：

```bash
PYTHONPATH=src .venv-wandb/bin/python scripts/track_wandb.py --watch \
  --studies runs/transfer-factorial-v1/cool runs/transfer-factorial-v1/heat \
  --out output/wandb-transfer
```

仅在现有观察器已退出时重启上述命令。模型流水线为 `PYTHONPATH=src .venv-model/bin/python scripts/run_transfer_factorial.py --stage all`；执行代码和输入哈希均已冻结，已完成阶段自动核验复用，失败的部分检查点需先核查，禁止盲目覆盖。

复现安装：

```bash
python3 -m venv .venv-wandb
.venv-wandb/bin/python -m pip install -r configs/wandb-requirements.txt
```

确认已有记录器进程结束后，使用同一同步索引恢复，避免并行启动两个上传进程：

```bash
PYTHONPATH=src .venv-wandb/bin/python scripts/track_wandb.py --watch
```

`output/wandb/sync-state.json` 保存 run ID、已同步事件和训练日志偏移；恢复时复用云端 run。记录器退出不停止训练。原实验进程继续由各自的 `*-progress.json` 和实际进程状态核验。

## 已准备

- 真实 ALFWorld 文本环境、官方数据和固定清单；隔离 CPU 环境 `.venv`。
- 无支持、早期、随机时机、延迟支持的配对采样器，公开观测教师缓存，费用 ledger。
- 目标视角 rejection SFT 导出、LoRA 脚本和统一 H0 评测。
- GitHub快照包含ReAct、Reflexion的固定提示来源。SkillScaffold、EvoIn、Harness-Zero为原工作机的调研缓存，原论文训练尚未复现；它们没有随初始代码快照上传。

密钥、模型权重、原始数据集和本地缓存不进入代码仓库。历史快照包含筛选后的实验记录；新完成实验的摘要、图表和逐任务CSV随代码保存，原始JSON记录压缩包归档到私有W&B，校验值见各实验的`archival.json`。没有设置教师API时，不会发出教师请求。

## 1. CPU 环境与清单

```bash
bash scripts/setup_cpu.sh
python3 scripts/download_alfworld.py
bash scripts/lh prepare
bash scripts/lh smoke
```

`smoke` 检查四类真实环境的 reset 与动作重放，不是模型性能评测。

## 2. 模型服务

在能访问宿主机 GPU 的终端执行。该环境复用现有支持 5090 的 PyTorch，安装轻量 overlay，不替换 CUDA 运行时：

```bash
bash scripts/setup_model.sh
HF_HOME="$PWD/.cache/huggingface" .venv-model/bin/python scripts/hf_server.py
```

这是单请求串行 HF 服务，供小规模 pilot。已有 vLLM GPU 环境时可以改用 `scripts/serve_student.sh`。不要用 CPU 模拟器虚拟环境启动 GPU 服务。

如果模型 Hub 客户端下载不稳定，可以使用校验/续传脚本，完成后从本地加载：

```bash
python3 scripts/download_student.py
.venv-model/bin/python scripts/hf_server.py \
  --model data/models/Qwen3-4B-Instruct-2507 \
  --served-model-name Qwen/Qwen3-4B-Instruct-2507
```

另一个终端配置 endpoint 和实际 checkpoint：

```bash
export LH_STUDENT_BASE_URL=http://127.0.0.1:8000/v1
export LH_STUDENT_MODEL=Qwen/Qwen3-4B-Instruct-2507
export LH_STUDENT_REVISION=cdbee75f17c01a7cc42f958dc650907174af0554
bash scripts/lh rollout --config configs/smoke_model.json --out runs/model-smoke
bash scripts/lh summarize runs/model-smoke
```

四任务、30 步 smoke 只用开发池，最终测试保持未运行。

也可用自动入口完成下载、临时服务、四题 B0 和服务回收：

```bash
/usr/bin/python3 scripts/first_baseline.py
```

已有下载进程时用 `--wait-for-download`，避免重复下载。临时服务默认端口 18080；任务结束会释放占用的 GPU。此入口最多等待模型下载 3 小时，不调用教师或付费 API。

## 3. 配置教师并缓存建议

教师使用 OpenAI 兼容 Chat Completions 接口。密钥放环境变量或被忽略的 `.env.local`，`.env.example` 只保留空模板。`scripts/generate_api_hints.py` 显式读取 `.env.local` 中三个 `LH_TEACHER_*` 字段，已导出的环境变量优先；它不执行 shell、不加载学生字段、不打印值。通用 `bash scripts/lh build-teacher` 入口仍需显式配置环境变量。

```bash
export LH_TEACHER_BASE_URL=你的兼容接口地址
export LH_TEACHER_MODEL=固定的教师模型名称
# 在终端安全设置 LH_TEACHER_API_KEY，不要把密钥写入实验 JSON。
```

复制 `configs/teacher.example.json` 为本地配置，填写该接口真实的人民币/百万 token 单价，并检查 `max_calls`、`max_cost_cny`。未知单价时程序拒绝调用。默认最多 200 次、单账本上限 ¥100；这不是自动授权使用整个月预算。所有失败请求保留预算预留，不自动重试。

```bash
bash scripts/lh build-teacher --manifest data/manifests/alfworld/pilot.jsonl \
  --config configs/teacher.local.json --out data/teacher/pilot.jsonl
bash scripts/lh rollout --config configs/pilot.json --out runs/pilot-base
bash scripts/lh summarize runs/pilot-base
```

40 个 pilot 任务 × 4 条件 × 3 rollout seed。费用和请求账本默认 `runs/teacher-budget.json`。probe 和最终测试不能生成教师建议。

当前 API 复现实验使用独立账本 `runs/api-harness-v1/teacher-budget.json`。生成已缓存的 72 张卡不再收费：

```bash
PYTHONPATH=src .venv/bin/python scripts/generate_api_hints.py
```

确认没有同一实验的活进程后才恢复 GPU 流水线；已有完成项会校验并复用，未完成的失败训练目录需先检查：

```bash
PYTHONPATH=src .venv-model/bin/python scripts/run_api_harness.py --stage all
```

流水线先出采集图，再在预设门槛通过时出学习曲线、原始/增益 AUC、旧任务保持与任务类型图，全部导出 PNG/PDF/SVG、逐种子/逐 episode CSV 和重绘 JSON。完成后可脱离 GPU 重绘：

```bash
PYTHONPATH=src python3 scripts/api_harness_report.py \
  --data runs/api-harness-v1/figures/main/measured-data.json --out output/figures/api-harness-v1
```

## 4. 短训练 baseline

```bash
bash scripts/lh build-teacher --manifest data/manifests/alfworld/train_pilot.jsonl \
  --config configs/teacher.local.json --out data/teacher/train_pilot.jsonl
bash scripts/lh rollout --config configs/collect_train.json --out runs/train-data
bash scripts/lh export-sft runs/train-data --arm none --out runs/sft/none.jsonl
```

每个支持条件分别导出。所有条件都从同一个基础 checkpoint 开始：

```bash
CUDA_VISIBLE_DEVICES=0 .venv-model/bin/python scripts/train_lora.py \
  --data runs/sft/none.jsonl --seed 101 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --out checkpoints/rft-none-seed101
```

另一张 5090 可以运行另一个独立 seed；不要默认两卡具备 NVLink。脚本同样支持 `torchrun`，全局 batch 保持固定。LoRA 输出是 adapter，需要在基础模型上加载或合并后服务，不能直接当成完整模型目录。

先用 `configs/probe.json --evaluation` 检查学习收益。`configs/eval.json` 保留历史配置；当前基线的正式测试入口、冻结要求见下方“基线与评测协议 v1”。将 `LH_STUDENT_REVISION` 改为训练产物的不可变标识，最终评测要求该字段存在。

本地 HF 服务可直接加载 adapter：

```bash
.venv-model/bin/python scripts/hf_server.py \
  --adapter checkpoints/rft-none-seed101 --served-model-name rft-none-seed101
```

服务 ready 输出包含基础 revision 和 adapter SHA-256，将该标识用于新的评测 run。

当前训练脚本是 rejection SFT，不是在线 agentic RL。RL 与原论文强 baseline 按实验协议另行接入，不混淆复现状态。

## 5. 日志、失败与验证

每个 run 保存 `run.json`、`steps.jsonl`、`episodes.jsonl`、`summary.json`。记录真实请求上下文和 token usage，默认不记录 API 密钥。支持会按窗口退出上下文。

服务错误会落盘并立即停止，不把错误静默过滤。恢复运行核对配置指纹并跳过已有 episode；若希望重跑错误样本，使用新的输出目录。意外中断留下的未完成步骤可能仍在 steps 日志，统计以完整 episodes 为准。

```bash
PYTHONPATH=src python3 -m pytest -q
```

单元检查覆盖数据隔离、教师预算、支持时机、恢复指纹和训练 loss mask。真实模型/环境的检查与模拟编排测试分开报告。

## 6. 基线与评测协议 v1（2026-11-10）

本轮保持 ALFWorld 文本任务，新增同一 Qwen3-4B 下的 Direct JSON、匹配示例的 Act、ReAct + 重试、Reflexion。40 个重复使用的开发任务用于工程检查，正式 seen/unseen 测试仍保留。每任务最多 150 次模型调用，反思计入 token 和调用预算。

- [协议、预算、来源与正式测试边界](docs/BASELINE_EVALUATION_PROTOCOL_V1.zh-CN.md)
- [官方示例重合审计补充](docs/BASELINE_PROTOCOL_V1_AMENDMENT_01.zh-CN.md)
- [实测结果与图表](output/baseline-benchmark-v1/RESULTS.zh-CN.md)（完整运行后生成）
- 原始运行：`runs/baseline-benchmark-v1/development/`；候选正式配置：`runs/baseline-benchmark-v1/final-configs/`。

```bash
PYTHONPATH=src .venv-model/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python scripts/run_baseline_benchmark.py --run
.venv/bin/python scripts/finalize_baseline_benchmark.py --watch
.venv-wandb/bin/python scripts/track_baseline_benchmark.py --watch
```

`learning_harness.benchmark` 对正式 manifest 要求匹配的 `--freeze` 文件；当前模板状态 `development_only` 无法执行正式测试。复跑会核对冻结输入与断点指纹，完成的任务不重跑。9 个既有 SFT adapter 已登记校验值，尚未在此新协议下重评；不将它们的旧分数混入新表。ReAct/Reflexion 是受控本地移植，未宣称复现论文排行榜分数。

## 7. 候选方案与强基线的学习对照（2026-11-11）

| 角色 | 完整流程 |
|---|---|
| 我们的候选方案 | API 教师策略辅助采集 → 共同公开动作视图 → matched-success SFT |
| 强基线 | ReAct 自采 → 相同公开动作视图 → 同预算 matched-success SFT |
| 参考起点 | 无 A 阶段训练的基础模型 |

本轮两源各重新采集 144 次，重新训练 6 个 A adapter；两族 B 任务上训练 12 条适应轨迹，复用 6 条已冻结的 no-A 参考权重，但所有评测重新运行。共计划 4,284 个评测任务运行。现有 9 个 SFT adapter 的数据匹配集合不同，保留为历史对照，不直接充当这轮候选模型。

[完整协议](docs/SOURCE_COMPARISON_V1.zh-CN.md)说明共同成功筛选、提示移除、三个种子、JSON/ReAct 统一执行和学习曲线分析。执行记录在 `runs/source-comparison-v1/`，云端链接在 `output/source-comparison-v1/wandb.json`；完整运行后自动生成结果报告和 PNG/PDF。

```bash
PYTHONPATH=src .venv-model/bin/python scripts/run_source_comparison.py --stage all
.venv-wandb/bin/python scripts/track_source_comparison.py --watch
```
