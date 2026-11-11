# Learning-Oriented Harness：实验协议 v1

冻结日期：2026-11-09。首轮只检验即时支持收益能否预测持久学习收益，后续再决定是否上 meta-RL。

## 1. 环境与模型

主环境为 ALFWorld 0.4.2 / TextWorld 1.7.0 文本版，基础动态和观测遵循官方任务文件。关闭 expert、隐藏 facts 和 admissible-action 列表；学生、教师都只能使用公开文本。评分读取环境 `won`，不使用模型 judge。

学生：Qwen3-4B-Instruct-2507，固定 revision `cdbee75f17c01a7cc42f958dc650907174af0554`。首轮 BF16，不混入量化差异。当前 2×5090 可分别承担推理与独立训练检查；4×A100 后续承担重复训练。CPU 环境与模型环境分离。

固定工具语法、系统提示、历史窗口及停止规则。标准开发轮最多 50 次动作，每次输出最多 128 token，每任务最多 6,400 生成 token，历史保留最近 12 个交互。输入 token 如实计量，尚未实现严格等输入 token 分配，因此本轮不能宣称所有计算量精确相同。

模型链路冒烟测试单独使用 30 步、4 个开发任务，不能作为最终 benchmark 分数。

## 2. 数据与 benchmark

已从官方发布数据识别 3,553 个训练任务、140 个 valid_seen 和 134 个 valid_unseen。文件与清单均有 SHA-256。

| 清单 | 数量 | 用途 |
|---|---:|---|
| train | 2,134 | 四类核心任务的可训练池 |
| train_pilot | 96 | 每核心类型 24 个，首轮短训练采集；为 train 的子集 |
| pilot | 40 | 每核心类型 10 个，只做冻结模型的即时支持检查 |
| dev | 40 | 支持选择与超参数开发 |
| probe | 40 | 不带支持的训练后能力检查，可用于开发，不是最终测试 |
| test_seen | 140 | 官方 seen-room 最终结果，H0，无教师 |
| test_unseen | 134 | 官方 unseen-room 最终结果，H0，无教师 |

四类训练任务：pick/place、clean/place、heat/place、cool/place。两类不进入训练/教师生成的任务：examine-in-light、pick-two/place。

unseen-room 的 134 个任务分为训练类型 99 个、保留类型 35 个；seen-room 对应 103/37。分别报告两组和整体，不能将混合总分都称为纯场景迁移。

pilot/dev/probe 按 `goal/scene task directory` 分组保留，每个保留组只选一个 trial，其余 trial 不回流训练。官方 seen 测试本来允许已见房间，不能将其称为完全场景不重叠。

本协议训练池与常见全量 3,553 任务设置不同，50 步与部分近邻 100 步评测也不同；不直接拿论文公开数字做胜负比较。原设置复现与本协议的同预算对照应分开。

外部确认环境选 ScienceWorld，但仅在 ALFWorld 出现稳定现象后接入，目前未安装或评测。

## 3. 首批 baseline

| ID | 支持/训练 | 状态 |
|---|---|---|
| B0 | 无支持，冻结基础模型 | 采样器已实现 |
| B1 | 第 0 步提供固定建议，连续 4 个决策可见 | 已实现 |
| B2 | 第 1–8 步按 task/seed 固定随机时机提供同一建议 | 已实现 |
| B3 | 第 5 步提供同一建议 | 已实现；是时机控制，不是新算法 |
| T0 | B0 成功训练轨迹的目标视角 rejection SFT | 导出及 LoRA 训练脚本已实现 |
| T1/T2/T3 | B1/B2/B3 成功轨迹去除显式支持消息后，同预算 SFT | 脚本已实现，参数实验待运行 |
| R0 | 无技能的 agentic RL | 原生 SkillScaffold 源码已准备，未运行 |
| R1 | 原生 SkillScaffold 自适应技能生成/注入/RL | 固定源码，未安装其 Megatron/Slime 训练栈 |
| D1/D2 | EvoIn / Harness-Zero 的强蒸馏对照 | 固定官方源码；跨环境移植未完成 |
| S1 | HarnessBandit 风格学习/迁移信号调度 | 论文核查完成，独立公开实现尚未定位 |

T1 只是简单的 target-view rejection fine-tuning，不能称为 EvoIn 或 Harness-Zero 的复现。原论文方法加入前，本轮只能建立现象和工程链路，不能宣称胜过这些方法。

## 4. 建议的控制

教师每个任务只生成一份基于 reset 公开观测的通用建议，B1/B2/B3 完全复用相同文本。没有隐藏位置、专家计划、最终答案或基于未来轨迹的信息。建议应在若干动作后仍成立；在 dev 上事先检查其适用性。

每个支持条件最多一个注入事件、4 个可见决策。早结束可能使延迟/随机组没有获得建议；这是干预的实际结果，必须报告实际 exposure，不能宣称实际信息量完全相等。

同任务、同 rollout seed 成组比较，组内支持条件的运行顺序固定随机化。每次重置环境、历史和支持状态。恢复运行会核对配置、模型、endpoint 哈希、teacher bank 与任务清单指纹。

核心比较首先是 B1/B2/B3 的支持时机；B0 是无支持的参照。第一轮不把时机实验包装成完整“帮助强度”或“决策所有权”机制研究。

## 5. 训练与指标

先完成固定基础模型的 40×4×3=480 个开发 episode；这里的 3 是 rollout seed，不是训练 seed。

再从相同基础 checkpoint，分别用 train_pilot 收集各条件轨迹，按预定成功过滤规则导出训练样本。失败与所有采集成本均保留。第一轮短训练采用 LoRA r=16、global batch=16、100 optimizer steps，三个独立训练 seed 为 101/202/303。开发集先选少量有区分度条件，正式重复优先保留 B0 和两个支持条件。

SFT 只对最后一个 assistant 输出计算 loss，观测、历史、提示和支持不作为生成目标；过长样本整条丢弃并记录。每个条件更新步数和全局 batch 相同，实际输入和目标 token 仍需审计。这是固定更新预算的比较，不自动等于固定总采集成本。

主要指标：即时成功率 S、训练后 H0 的成功率及相对基础模型的变化 L；任务类型宏平均；无效动作率；生成/输入 token；实际支持使用率；服务错误与预算终止。

本轮分析器提供单 checkpoint 下按任务组聚类的配对 bootstrap，不能将其当作训练种子置信区间。正式学习结论必须跨独立训练种子复现。

持续学习阶段再采用任务族 A→B→C，独立建立训练/保持 probe，考察学习新任务和旧能力保留。当前没有相应结果。

## 6. 进入下一阶段的条件

1. 链路通过真实 ALFWorld reset/action 和真实模型回合，JSON 格式失败率可接受。
2. 在 dev 上发现可重复的即时/持久收益排序差异；不能用最终测试来选择条件。
3. 加入强蒸馏、混合支持训练和奖励方差检查后，差异仍有解释价值。
4. 若仅由 token、分布匹配或 GRPO 全组同分解释，则收窄结论或转向。

## 7. 来源与代码

- [ALFWorld 官方源码](https://github.com/alfworld/alfworld)
- [SkillScaffold](https://github.com/Vaidurya00/SkillScaffold)
- [EvoIn](https://github.com/Fanshaoliu/EvoIn)
- [Harness-Zero](https://github.com/metaevo-ai/harness-zero)
- [HarnessBandit 论文](https://arxiv.org/abs/2609.13739)

精确源码 commit 见 `configs/sources.lock.json`。全部实验入口与启动步骤见根目录 README。
