# 全局经验选样论文：新版LaTeX

`main.tex`为入口，附带官方NeurIPS 2026 preprint样式、参考文献、新结果图、原稿的诊断图、两套算法及完整证据附录。可直接将目录压缩上传Overleaf；本地使用TeX Live的latexmk，或设置`TECTONIC_BIN`后运行`bash build.sh`。

先前8页版本过度压缩了原13页稿的细节；当前完整稿恢复相关工作、历史匹配选择的公式/算法、控制关系图、详细历史实验、实验与超参清单、逐初始化结果、条件复测、数值评分诊断、完成后的统一评分复现，以及WebShop起点校准。历史冷却/加热面板和当前新任务面板分开陈述，过时的“复现进行中”已更新。原13页稿及8页快照均保留；恢复来源与更新说明见`restoration-evidence.json`。

本版本包含完整开发矩阵和同等全局选择自由度的基线：288个学生训练，三种全局规则的97模型（含基础模型）比较。新任务独立复现已经在运行；其结果未完成前，正文明确标为进行中，不放局部胜负。

图表包括：全局主对照、规则与覆盖约束交互、八初始化配对热图、实际训练loss、动作构成、NLL与行为收益散点。loss曲线阴影是16条训练记录的描述性SD，不是任务成功率曲线或置信区间；散点的36份固定调度相互重叠，不是36个独立语料。

确认完成后，只允许从完整97检查点、12,416回合的已审计汇总更新`generated/confirmation.tex`及开发/确认对比图，重建PDF前保留旧快照。没有新训练、教师调用或正式benchmark测试。原始任务、对话、请求和模型权重均不在源码包内。

当前PDF：`output/pdf/global-selection-paper/global-selection-paper.pdf`。
源码包：`output/global-selection-paper-latex.zip`。
数据和图：`output/paper-global-v2/`。
