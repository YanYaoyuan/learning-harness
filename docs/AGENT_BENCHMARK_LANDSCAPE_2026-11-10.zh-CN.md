# Agent 数据、基线与评测基准：与本项目的关系

调研日期：2026-11-10。依据为原论文、官方项目与发布仓库；这是一组有明确使用实例的代表性资源，不是对整个行业使用率的统计。本次只做调研，不启动新的训练或环境迁移。

## 1. ALFWorld 的定位

[ALFWorld](https://alfworld.github.io/) 将文本交互环境与 ALFRED 的具身任务对齐。本项目使用文字观测和离散文字命令，不运行视觉机械臂控制。它适合低成本、可重复的规划与反馈利用实验，但不能单独代表代码、浏览器、工具服务或机器人控制能力。

它并非孤立选择：[ReAct](https://arxiv.org/abs/2210.03629) 使用 ALFWorld/WebShop；[AgentGym](https://agentgym.github.io/) 收录 ALFWorld、ScienceWorld、WebShop 等不同环境；[GiGPO/verl-agent](https://github.com/langfengQ/verl-agent) 提供 ALFWorld/WebShop 的 RL 实现；2026 年的 [SkillScaffold](https://arxiv.org/html/2610.18807v1) 仍将 ALFWorld 与 WebShop、ScienceWorld、τ²-bench 联合评测。不能由这些实例推断全行业都采用它，也不能因为环境较早提出就判定没有研究价值。

## 2. 区分四种对象

- 训练数据：交互轨迹、专家纠正、任务指令或用于生成经验的训练任务。
- 环境：执行动作、维护状态并返回观测的程序。
- Benchmark：任务集、划分、观测/动作接口、预算和评分规则的组合。
- Baseline：在相同条件下对照的模型、学习算法或完整 agent 系统。

AgentGym/BrowserGym 是环境与评测基础设施；AgentTraj/AgentInstruct 是训练资源；ReAct/Reflexion 是执行方法；GRPO/GiGPO 是优化方法。不能把这些名称直接放在一个未解释的排名表里。

## 3. 代表性评测地图

| 领域 | Benchmark / 环境 | 主要检验 | 对本研究的关系 |
|---|---|---|---|
| 文本交互规划 | [ALFWorld](https://alfworld.github.io/)、[ScienceWorld](https://github.com/allenai/ScienceWorld) | 目标分解、状态跟踪、搜索、科学实验步骤 | 适合控制实验；同为文字操作，跨环境证据仍需解释能力差异 |
| 购物与约束满足 | [WebShop](https://arxiv.org/abs/2207.01206) | 搜索商品、理解属性、选项选择与购买 | 与家务环境不同的语义域；可用于检验目标条件化是否迁移 |
| 真实风格网页操作 | [WebArena / VisualWebArena](https://webarena.dev/)、[WorkArena / BrowserGym](https://github.com/ServiceNow/BrowserGym) | 多页面业务流程、网页状态与交互 | 更接近网页产品；DOM、可访问性树和截图配置不可混比 |
| 跨应用 API 工作流 | [AppWorld](https://appworld.dev/appworld/) | 编写代码调用多应用 API、维护状态、避免副作用 | 可重复的数字任务，适合作为后续现实性检验候选；不是立刻切换的决定 |
| 客服与工具协作 | [τ-bench 家族](https://github.com/sierra-research/tau2-bench) | 对话、工具、业务约束、用户协作 | 用户模拟器、裁判与版本必须固定，并记录额外调用成本 |
| 软件工程 | [SWE-bench](https://www.swebench.com/)、[SWE-bench Pro](https://arxiv.org/abs/2509.16941)、[Pro V2](https://arxiv.org/abs/2610.24206) | 仓库级修复、测试与长程代码修改 | 需要代码环境和更强编码能力；不宜只为“更热门”切换 |
| 终端任务 | [Terminal-Bench](https://www.tbench.ai/) | 终端工具、系统配置与端到端工作 | 必须固定容器、资源、版本、时间预算及 harness |
| 桌面计算机操作 | [OSWorld](https://github.com/xlang-ai/OSWorld) | 桌面软件、图形界面、多应用操作 | 需要 VM 与视觉动作接口；不能与文字 ALFWorld 分数直接比较 |
| 工具调用专项 | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) | 函数选择、参数与多轮工具调用 | 能补工具接口诊断；不替代完整长程任务测量 |
| 综合信息任务 | [GAIA](https://arxiv.org/abs/2311.12983) | 检索、工具及多模态信息组合 | 更贴近助理任务；仅有答案正确率时，对过程学习机制的定位有限 |

版本核查提醒：访问时 Terminal-Bench 官方首页已显示 4.1；不能把 2.0/2.1/4.x 结果拼成同一榜单。τ 仓库保留 tau2-bench 路径，但 README 已含后续 τ³ 与不同版本/模式的协议，目录名不能当版本标识。SWE-bench Pro V2 也应与早期 Pro 区分。这里不引用未核对条件的“当前第一名”分数。

## 4. 用来训练的公开资源

| 资源 | 类型与用途 | 注意事项 |
|---|---|---|
| [AgentInstruct / AgentTuning](https://github.com/THUDM/AgentTuning) | 1,866 条、六类任务的交互数据；agent SFT 基础对照 | 此处是 AgentTuning 的同名资源；公开训练数据不等于所有任务的新颖测试 |
| [AgentBank](https://arxiv.org/abs/2410.07706) | 超过 50k 条轨迹、16 项任务；多任务 agent tuning | 可用于研究经验分布；不是 50k 个互不相关独立环境 |
| [AgentTraj / AgentGym](https://agentgym.github.io/) | 多环境轨迹与 AgentEval 评测套件 | 同环境 held-in 与跨环境 held-out 需分别报告；套件子集未必等于原环境完整官方测试 |
| [Agent-FLAN](https://arxiv.org/abs/2403.12881) | agent tuning 的数据组织、能力分解及训练方法 | 与“格式学习、推理、幻觉”的拆分相关；是方法与数据配方，不只是一个下载包 |
| [AgentRefine](https://github.com/Fu-Dayuan/AgentRefine) | 合成带错误与自纠正过程的训练经验 | 对本项目的偏航恢复分析有直接相关性，需要区分轨迹纠正与普通成功 SFT |
| [SWE-smith](https://swesmith.com/getting_started/assets/) | 可执行代码任务、仓库环境及 rollout 资源 | 官方 50k+ 数字指任务实例，不能写成同数量成功训练轨迹 |
| [AgentGym-RL](https://agentgym-rl.github.io/) | 交互环境、任务与多轮 RL 训练框架 | RL 可以在线生成经验；不必把固定静态轨迹集当唯一数据来源 |

下载公共 agent 数据后，要先核对与本项目 train/dev/test 的重叠。训练数据越大并不自动解决当前研究问题；改变共同起点时须重做相关基线，不能把结果与现有小样本训练直接相减。

## 5. Baseline 应按研究问题选择

| 要检验的主张 | 合理基线 | 公平性约束 |
|---|---|---|
| 执行时帮助有效 | 原模型直接执行、[ReAct](https://arxiv.org/abs/2210.03629)、固定提示、[Reflexion](https://arxiv.org/abs/2303.11366) | 相同模型与观测；反思的尝试次数和额外 token 计费；Reflexion 不等于参数 RL |
| 经验可内化 | 成功筛选 SFT、混合上下文 SFT、教师动作蒸馏、适用的 on-policy distillation | 相同教师访问预算、目标模型与训练成本；定义测试时是否保留辅助 |
| Harness 知识可移入参数 | [Harness-Zero](https://arxiv.org/abs/2609.24974)、[HAD](https://arxiv.org/abs/2610.02858) | 两者部署假设不同：前者移除专用 harness，后者在固定 harness 下蒸馏额外教师能力；不能当完全相同处理 |
| 训练期帮助改善 RL | 无帮助 GRPO/GiGPO、固定帮助、[SkillScaffold](https://arxiv.org/html/2610.18807v1) | 相同 rollout/更新预算，分别测辅助与撤除后的性能；RL 方法不应与极短 SFT 直接当等预算比较 |
| Harness 分配改善学习 | 固定、均匀/随机、按即时成功率选择、[HarnessBandit](https://arxiv.org/html/2609.13739v1) | harness 定义、选择器开销、任务和模型迁移划分要匹配 |
| 软件 agent 系统改进 | [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent)、[OpenHands](https://github.com/OpenHands/OpenHands) | 完整系统榜与同骨干因果消融分开；不适用于给 ALFWorld 文本命令系统直接排名 |

近期论文的实际配置：

- GiGPO 官方实现报告 Qwen2.5-1.5B/7B 在 ALFWorld/WebShop 上的实验，且提供 GRPO、PPO、RLOO 等入口。这说明小模型 agent 有成熟的算法基线；本文不将论文分数与我们私有开发划分直接比较。
- AgentGym-RL 官方项目以 Qwen2.5-3B/7B 为主要骨干，覆盖 WebArena、SearchQA、TextCraft、BabyAI、SciWorld 的场景。
- SkillScaffold 的主实验用 Qwen3-8B，在四类 benchmark 上比较提示/记忆、无技能 RL 和技能 RL。它与“帮助如何影响学习”直接相关，不能只用普通提示作为唯一竞争者。
- HarnessBandit 使用 PinchBench 与 ClawEval 测任务及接口迁移；不是所有 harness 论文都用 ALFWorld。

上述较新工作是必须阅读和判断可比性的近邻，不代表已在本地复现，也不意味着每一项都要在双卡上完整重跑。

## 6. 我们现在位于哪里

本地核对 `runs/transfer-factorial-v1/cool/design.json`、`main-result.json` 与 A 训练元数据：骨干为 Qwen3-4B-Instruct-2507；主要训练为 target_sft，B 为冻结 A 后新增 LoRA 的固定数据适应；最新每族 32 个 B query、24 个 A query、40 步上限。官方最终测试未使用。

因此，目前结果是 ALFWorld 训练池内、任务组隔离的开发控制实验，不是正式 ALFWorld leaderboard 复现。早期 mixed/context distill 只有小规模探索，不能当作已经系统击败强蒸馏基线。也尚未完成 ReAct/Reflexion、GRPO/GiGPO 或近期 harness 学习方法在同协议下的比较。

研究判断：选 ALFWorld 有充分先例；更紧迫的缺口是明确主张、可对齐的正式评测与同骨干强基线。把主任务直接换成具身、软件或桌面，不能自动补上这些缺口。

## 7. 有约束的后续选择

保留当前数据审查和失败状态诊断。先确定要检验的是经验内化、执行恢复还是后续学习效率，再选择对应的强基线。若当前方向得到鉴别性支持，优先考虑一个语义和动作结构有区别的第二环境：WebShop 检验目标约束与选择，AppWorld 检验可执行工具状态，ScienceWorld 检验另一种多步骤文字任务。这里只是候选的功能映射，不是追加实验授权或立即安装计划。

最终测试必须在方法与预算冻结之后使用；不能为了“先对榜”打开保留测试再持续调参。默认成功率不能独自回答学习效率，需要另外冻结迁移任务、采集/训练预算、学习曲线和旧能力保持协议。统计需区分任务、训练种子和重复回合。

面向论文的目标配置可为：一个可解释的主环境、一个真正检验同一假设的第二环境、同骨干执行/训练强基线、一个额外模型规模或家族。它是研究设计建议，不是发表保证，也不是要求把上表全部 benchmark 跑一遍。
