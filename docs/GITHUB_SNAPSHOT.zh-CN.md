# 仓库快照与资产恢复

仓库存放研究代码、协议、历史报告及选定实验产物。最初导出时，部分实验仍在运行；带日期的进度记录反映各自采集时点。已发表论文和项目首页汇总最终选样结果。

## 文件范围

src/、scripts/、tests/、configs/、requirements/ 和 docs/ 包含实现、依赖与文档。归档资产包括 ALFWorld 划分清单、选定 output/ 产物，以及 runs/ 下的 JSON、JSONL、CSV 和图表。

较大的公开轨迹文件 steps/events/calls/batches.jsonl 以 gzip 保存。最初导出时仍在变化的核心对照评测事件流未打包，已提交的逐任务结果保留。

模型与 adapter 权重、重复 tokenizer、原始仿真数据、下载的外部论文 PDF、虚拟环境、SDK 缓存和控制台日志保留在运行机器上。模型与环境下载脚本、权重校验值和实验配置用于重建这些资产。论文调研保留文字、链接和下载清单。

本机凭据与设置（.env.local、key.env、.aws/.codex/.agents）排除在导出范围外；示例环境文件仅含空密钥。导出器在写入前检查凭据模式。

SNAPSHOT_MANIFEST.json 记录初次导出文件的原始与上传 SHA-256、大小、导出时间，以及被丢弃的不完整 JSONL 尾行。后续文档修订由 Git 提交记录追踪，该清单不是每次提交后重建的完整索引。实验自身的冻结记录独立于这份快照清单。

## 恢复轨迹

```bash
python3 scripts/unpack_snapshot_traces.py --study baseline-benchmark-v1
```

脚本检查压缩和解压内容的校验值，拒绝覆盖内容冲突的已有文件。恢复训练或评测还需要对应权重、虚拟环境和协议指定的本地路径，详见[运行说明](REPRODUCIBILITY.md)。

## 导出工作区

```bash
python3 scripts/export_github_snapshot.py --out /tmp/learning-harness-new-snapshot
```

目标目录需为空。导出过程只读取工作区；上传在独立 Git 目录中进行。

## 第三方来源与许可

仓库保留 ReAct/Reflexion 提示及少量参考源码。固定 commit 见 configs/baseline-sources.lock.json，MIT 许可证位于 third_party/ReAct/LICENSE 和 third_party/Reflexion/LICENSE。其他参考方法通过来源锁和获取脚本管理，其获取状态与复现状态分别记录。本项目尚未另行授予公开开源许可。
