# Agent 论文包与研究索引

调研截止：2026-11-09。资源：同机 4×A100 40GB + 2×RTX 5090；API 预算每月 ¥1,000–5,000。

**当前建议：[研究议题 V2：学习型 harness、可迁移改进算子与元强化学习](RESEARCH_AGENDA_V2.zh-CN.md)。**

进一步讨论：[Harness 如何影响持久学习：质疑与最小实验 V3](LEARNING_HARNESS_DEBATE_V3.zh-CN.md)。

[首轮调研记录](RESEARCH_BRIEF.zh-CN.md) · [固定版本与校验清单](manifest.json)

共 40 篇原始论文 PDF。每份已通过 PDF 解析、前两页标题核对及 SHA-256 校验。P0 表示直接近邻，P1 为重要扩展，P2 为背景。

本轮新增 29–40。优先阅读：30 SOAR → 29 MetaSkill-Evolve → 35 SkillScaffold → 36 HarnessBandit → 33 EvoIn → 31 Rethinking Continual Experience Internalization → 39 Selection Is Not Learning。

当前研究定位：学会更有效地改进自己；先验证当前求解收益和后续学习收益是否存在可解释的差别，再考虑训练跨学习者迁移的改进算子。候选差异仍需逐篇精读排重。

## 学习机制、元强化学习与能力内化（本轮重点）

| 编号/优先级 | 论文与本地 PDF | 固定版本 | 页数 | 阅读用途 |
|---|---|---|---:|---|
| 29 / P0 | [MetaSkill-Evolve: Recursive Self-Improvement of LLM Agents via Two-Timescale Meta-Skill Evolution](papers/learning-to-improve/29_2607.05297v1_MetaSkill_Evolve.pdf) | [2607.05297v1](https://arxiv.org/abs/2607.05297v1) | 14 | 两时间尺度演化改进算子；learn-to-improve 方向的直接近邻。 |
| 30 / P0 | [Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability](papers/learning-to-improve/30_2601.18778v3_SOAR_Meta_RL.pdf) | [2601.18778v3](https://arxiv.org/abs/2601.18778v3) | 39 | 以学生真实学习增益训练课程生成教师；元奖励已有直接先例。 |
| 31 / P0 | [Rethinking Continual Experience Internalization for Self-Evolving LLM Agents](papers/learning-to-improve/31_2606.04703v1_Continual_Experience_Internalization.pdf) | [2606.04703v1](https://arxiv.org/abs/2606.04703v1) | 14 | 多轮内化的能力坍塌、原则级经验与注入方式；参数化持续学习核心。 |
| 32 / P1 | [When Skills Remain But Access Fails: Factorized Skill Memory for Continual Skill Internalization](papers/learning-to-improve/32_2610.15492v1_Factorized_Skill_Memory.pdf) | [2610.15492v1](https://arxiv.org/abs/2610.15492v1) | 22 | 区分技能寻址遗忘与执行遗忘，避免把调用失败都当成能力抹除。 |
| 33 / P0 | [EvoIn: Bridging Evolution and Internalization for Agent Fine-Tuning](papers/learning-to-improve/33_2609.35290v2_EvoIn.pdf) | [2609.35290v2](https://arxiv.org/abs/2609.35290v2) | 36 | 演化决策流程再内化，撤除 harness 后检验能力保留。 |
| 34 / P1 | [ActiveSaddler: Automated Curriculum Learning for Agent Harness Optimization](papers/learning-to-improve/34_2610.00906v1_ActiveSaddler.pdf) | [2610.00906v1](https://arxiv.org/abs/2610.00906v1) | 37 | 课程与 harness 同步演化；简单按失败模式出题已被覆盖。 |
| 35 / P0 | [SkillScaffold: Scaffolding Agentic Reinforcement Learning with Policy-Adaptive Skill Credit Assignment and Selective Injection](papers/learning-to-improve/35_2610.18807v1_SkillScaffold.pdf) | [2610.18807v1](https://arxiv.org/abs/2610.18807v1) | 28 | 策略自适应技能注入与无技能推理；harness 促学方向直接近邻。 |
| 36 / P0 | [HarnessBandit: Joint Learnability-Transferability Scheduling for Multi-Harness Agentic Reinforcement Learning](papers/learning-to-improve/36_2609.13739v1_HarnessBandit.pdf) | [2609.13739v1](https://arxiv.org/abs/2609.13739v1) | 13 | 联合可学习性与跨 harness 迁移信号调度；避免把跨 harness 训练包装成新题。 |
| 37 / P1 | [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](papers/learning-to-improve/37_2505.03335v3_Absolute_Zero.pdf) | [2505.03335v3](https://arxiv.org/abs/2505.03335v3) | 52 | 可执行环境锚定的自生成任务与自博弈。 |
| 38 / P1 | [Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning](papers/learning-to-improve/38_2511.16043v2_Agent0.pdf) | [2511.16043v2](https://arxiv.org/abs/2511.16043v2) | 23 | 课程生成与工具执行者共演化。 |
| 39 / P0 | [Selection Is Not Learning: Choosing Model Futures](papers/learning-to-improve/39_2610.22029v1_Selection_Is_Not_Learning.pdf) | [2610.22029v1](https://arxiv.org/abs/2610.22029v1) | 23 | 任务看起来可学不代表更新有效；参数继承与经验继承分离。 |
| 40 / P1 | [When the Critic Falls Behind: Co-Evolving Rubrics and Harnesses for Continual Agent Self-Improvement](papers/learning-to-improve/40_2610.36826v1_RuHarness.pdf) | [2610.36826v1](https://arxiv.org/abs/2610.36826v1) | 21 | 监督 rubric 与 harness 共演化；验证器方向需进一步越过该近邻。 |

## Harness 与 RSI

| 编号/优先级 | 论文与本地 PDF | 固定版本 | 页数 | 阅读用途 |
|---|---|---|---:|---|
| 01 / P1 | [Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents](papers/harness-rsi/01_2505.22954v3_DGM.pdf) | [2505.22954v3](https://arxiv.org/abs/2505.22954v3) | 72 | 自修改代码与开放式候选档案；理解系统级 RSI 的基础。 |
| 02 / P0 | [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](papers/harness-rsi/02_2507.19457v2_GEPA.pdf) | [2507.19457v2](https://arxiv.org/abs/2507.19457v2) | 96 | 轨迹反思与 prompt 搜索；应纳入低算力对照。 |
| 03 / P0 | [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](papers/harness-rsi/03_2510.04618v3_ACE.pdf) | [2510.04618v3](https://arxiv.org/abs/2510.04618v3) | 32 | 增量 playbook 更新；记忆或技能演化的重要对照。 |
| 04 / P1 | [Recursive Harness Self-Improvement](papers/harness-rsi/04_2607.15524v1_RHI.pdf) | [2607.15524v1](https://arxiv.org/abs/2607.15524v1) | 88 | 直接连接 harness、上下文管理与持续改进。 |
| 05 / P0 | [ModularRSI: Modular and Generalizable Recursive Harness Self-Improvement](papers/harness-rsi/05_2609.14857v2_ModularRSI.pdf) | [2609.14857v2](https://arxiv.org/abs/2609.14857v2) | 28 | 模块演化、跨任务对比与 benchmark-disjoint 评估；避免重复。 |
| 06 / P0 | [RRSI: Regularized Recursive Self-Improvement of Agent Harnesses](papers/harness-rsi/06_2609.24972v3_RRSI.pdf) | [2609.24972v3](https://arxiv.org/abs/2609.24972v3) | 27 | 约束候选修改与选择，关注泛化和运行成本；使用 v3 而非早期数字。 |
| 07 / P1 | [Hyperagents](papers/harness-rsi/07_2603.19461v2_Hyperagents.pdf) | [2603.19461v2](https://arxiv.org/abs/2603.19461v2) | 63 | 允许修改 meta agent 自身；区分改进任务表现与改进改进机制。 |
| 08 / P2 | [Recursive self-improvement of AI research agents](papers/harness-rsi/08_2609.26457v1_AIDE2.pdf) | [2609.26457v1](https://arxiv.org/abs/2609.26457v1) | 28 | 研究 Agent 的递归代码改进；了解更大规模闭环。 |
| 24 / P0 | [Does Failure Feedback Direct Agent Self-Modification?](papers/harness-rsi/24_2610.14662v1_Failure_Feedback_Self_Modification.pdf) | [2610.14662v1](https://arxiv.org/abs/2610.14662v1) | 28 | 区分目标失败修复、相关任务迁移与 held-out 净提升；重要负面证据。 |

## 持续学习、记忆与技能

| 编号/优先级 | 论文与本地 PDF | 固定版本 | 页数 | 阅读用途 |
|---|---|---|---:|---|
| 09 / P0 | [Are Self-Evolving Agents Continual Learners?](papers/continual-learning/09_2610.25858v1_Self_Evolving_Continual_Learners.pdf) | [2610.25858v1](https://arxiv.org/abs/2610.25858v1) | 24 | 直接分析 task drift、全局编辑干扰、遗忘和 retrieval collision；首读。 |
| 10 / P0 | [STALE: Can LLM Agents Know When Their Memories Are No Longer Valid?](papers/continual-learning/10_2605.06527v1_STALE.pdf) | [2605.06527v1](https://arxiv.org/abs/2605.06527v1) | 37 | 旧记忆失效和隐式冲突；需区分事实记忆与可执行技能。 |
| 11 / P0 | [Learning While Acting: A Skill-Enhanced Test-Time Co-Evolution Framework for Online Lifelong Learning Agents](papers/continual-learning/11_2606.04815v2_LifeSkill.pdf) | [2606.04815v2](https://arxiv.org/abs/2606.04815v2) | 14 | 技能学习与参数内化已有先例；外部技能+RL 并非独立创新。 |
| 12 / P1 | [Memory-Centric Continual Learning for AI Agents](papers/continual-learning/12_2610.29357v1_MCCL.pdf) | [2610.29357v1](https://arxiv.org/abs/2610.29357v1) | 50 | 记忆以未来行动收益评价；提供持续学习评估视角。 |
| 13 / P1 | [From AutoResearch to Continual AutoResearch: Do Agents Learn Across Tasks?](papers/continual-learning/13_2611.05108v1_Continual_AutoResearch.pdf) | [2611.05108v1](https://arxiv.org/abs/2611.05108v1) | 27 | 持久 Agent 与 reset 对照；区分研究效率和最终解质量。 |
| 14 / P0 | [LifelongAgentBench: Evaluating LLM Agents as Lifelong Learners](papers/continual-learning/14_2505.11942v3_LifelongAgentBench.pdf) | [2505.11942v3](https://arxiv.org/abs/2505.11942v3) | 44 | DB、OS、KG 交互任务与自动验证；可作为复现实验入口。 |
| 25 / P0 | [ERRAND: Budgeted Maintenance of Agent Memory](papers/continual-learning/25_2609.29545v1_ERRAND.pdf) | [2609.29545v1](https://arxiv.org/abs/2609.29545v1) | 40 | 带预算的记忆重验证已经存在；仅提出何时重查不构成新意。 |
| 26 / P0 | [SkillCommit: Evolving Agent Skills through Behaviorally Validated Scope Expansion](papers/continual-learning/26_2608.15165v2_SkillCommit.pdf) | [2608.15165v2](https://arxiv.org/abs/2608.15165v2) | 16 | 局部 patch、跨实例回放与技能提交；首选方向的重要近邻。 |
| 27 / P0 | [Harness Continual Learning: Continual Adaptation Beyond Model Parameters](papers/continual-learning/27_2608.19013v2_HCL.pdf) | [2608.19013v2](https://arxiv.org/abs/2608.19013v2) | 20 | harness-level forgetting 与更新提交门控已经存在。 |
| 28 / P0 | [Recurrent Harness for Agentic Continual Learning](papers/continual-learning/28_2610.12668v1_ReHarness.pdf) | [2610.12668v1](https://arxiv.org/abs/2610.12668v1) | 20 | 严格流式条件下回溯修订已检索经验；需纳入对照。 |

## 小模型 Agent

| 编号/优先级 | 论文与本地 PDF | 固定版本 | 页数 | 阅读用途 |
|---|---|---|---:|---|
| 15 / P0 | [AgentCPM-Explore: Realizing Long-Horizon Deep Exploration for Edge-Scale Agents](papers/small-model-agents/15_2602.06485v1_AgentCPM_Explore.pdf) | [2602.06485v1](https://arxiv.org/abs/2602.06485v1) | 22 | 4B 主 Agent、训练与上下文问题；pass@64 不等同单次成功率。 |
| 16 / P2 | [Small Language Models are the Future of Agentic AI](papers/small-model-agents/16_2506.02153v1_SLM_Agentic_AI.pdf) | [2506.02153v1](https://arxiv.org/abs/2506.02153v1) | 17 | 小模型 Agent 的立场论文；不作为普遍能力优势的实验证明。 |

## Agentic RL

| 编号/优先级 | 论文与本地 PDF | 固定版本 | 页数 | 阅读用途 |
|---|---|---|---:|---|
| 17 / P1 | [RAGEN-2: Reasoning Collapse in Agentic RL](papers/agentic-rl/17_2604.06268v1_RAGEN2.pdf) | [2604.06268v1](https://arxiv.org/abs/2604.06268v1) | 44 | 多轮 RL 模板坍塌、信号质量与诊断。 |
| 18 / P0 | [Agent Lightning v1.0: Towards Harnessed Agentic RL](papers/agentic-rl/18_2608.17528v1_Agent_Lightning_v1.pdf) | [2608.17528v1](https://arxiv.org/abs/2608.17528v1) | 21 | 实际 harness 参与 RL 的实现与复现入口。 |

## 模型与 Harness 共演化

| 编号/优先级 | 论文与本地 PDF | 固定版本 | 页数 | 阅读用途 |
|---|---|---|---:|---|
| 19 / P0 | [VACE: Validation-Gated Alternating Co-Evolution of Agent Models and Harnesses](papers/co-evolution/19_2609.37105v1_VACE.pdf) | [2609.37105v1](https://arxiv.org/abs/2609.37105v1) | 12 | RL 与 harness 交替更新+验证门控已经存在；重点排重。 |
| 20 / P0 | [Co-Evolving Harnesses and Models: On-Policy Correction Helps Weaker Models Catch Up Where Imitation Fails](papers/co-evolution/20_2609.09134v1_On_Policy_Correction.pdf) | [2609.09134v1](https://arxiv.org/abs/2609.09134v1) | 10 | 强模型整轨迹模仿可能破坏弱模型与 harness 的匹配。 |
| 21 / P0 | [CoHARL: Co-Evolving Agent Policies and Executable Harnesses with Reinforcement Learning](papers/co-evolution/21_2610.36469v1_CoHARL.pdf) | [2610.36469v1](https://arxiv.org/abs/2610.36469v1) | 22 | 共享策略与可执行 harness 群体共演化；比单 harness 交替更接近前沿。 |
| 22 / P1 | [Self-Adapting Language Models](papers/co-evolution/22_2506.10943v2_SEAL.pdf) | [2506.10943v2](https://arxiv.org/abs/2506.10943v2) | 25 | 模型生成自身适配数据与指令，再用下游收益训练 self-edit。 |
| 23 / P1 | [HELIX: Model-Harness Co-evolution for Recursive Self-Improvement](papers/co-evolution/23_2608.13951v1_HELIX.pdf) | [2608.13951v1](https://arxiv.org/abs/2608.13951v1) | 27 | 干预可追踪、成功与回归轨迹，以及 model-harness-data 循环。 |

## 下载与复用

PDF 保持原文。manifest.json 保存来源、固定版本、文件大小、页数和 SHA-256。

在允许联网的环境执行 `python3 download_papers.py` 可补齐缺失文件，已存在且可解析的文件将复用。脚本只访问清单中的 arXiv PDF。

此次调研未运行作者实验，未调用付费模型 API。首轮 28 篇合集保留，本轮合集包含全部 40 篇及 V2 研究议题。
