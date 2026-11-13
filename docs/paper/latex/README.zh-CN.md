# 正式排版 LaTeX 论文稿

入口为 `main.tex`。使用从NeurIPS官方作者包下载的2026样式，preprint模式；未使用已录用论文的final标记。作者暂匿名，不代表已投稿或录用。

正文包括研究问题、相关工作、形式化选样约束、目标函数与估计量、算法、完整实验、讨论和结论；附录保留配置、六初始化、重复训练、评分仪器、WebShop起点校准及原API/ReAct负向基线。

- `figures/architecture.tex`：可编辑的TikZ系统架构图。
- `figures/control-ladder.tex`：可编辑的三层实验控制图。
- `figures/*.pdf`：从已完成报告绘制或逐字节复制的矢量实验图。
- `generated/*.tex`：从报告自动生成的主结果和初始化表行。
- `references.bib`：核对过第一方论文页面的参考文献。
- `evidence.json`：完成结果、生成表和图的SHA256；不含原始任务或消息。
- `vendor/source.json`：官方样式下载来源与SHA256，样式文件保持原样。

上传整个源码包到Overleaf，选择 `main.tex`，使用pdfLaTeX编译。也可本地运行：

```bash
bash build.sh
# 或已有Tectonic时：
TECTONIC_BIN=/absolute/path/to/tectonic bash build.sh
```

完整研究仓库内重新生成图表：

```bash
MPLCONFIGDIR=/tmp/lh-paper-mpl python3 scripts/build_paper_assets.py
```

图表生成器先核验v0.2证据快照中的全部来源和图，再读取明确列出的已完成JSON报告。正在运行的统一评分重训不进入本文主结果，也不因中途结果改变文章已记录的数字。需要纳入新实验时应新建证据版本、审核结果后再更新稿件。

源码包包含编译所需的全部图、表和样式，但不包含模型权重、原始任务/消息、私有评分输入、密钥或TeX工具链缓存。新任务、强选样基线、第二环境学习对照、离散结果统计校准和会议专用清单仍需在实际投稿前完成。
