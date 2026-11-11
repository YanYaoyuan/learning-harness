# GitHub 研究快照说明

这是私有研究备份，包含方案、历史判断与修正、协议、代码、测试、实验元数据、逐任务结果、训练输入与已生成图表。当前核心对照仍在运行，快照中的进度以各文件的采集时点为准，不是全部训练完成后的原子备份。

## 包含与保留在运行机器上的内容

包含 `src/`、`scripts/`、`tests/`、`configs/`、`requirements/`、`docs/`、ALFWorld 划分 manifest、精选 `output/` 和实验 `runs/` 中的 JSON/JSONL/CSV/图表。公开轨迹的大型 `steps/events/calls/batches.jsonl` 使用 gzip 保存；正在变化的核心对照评测事件流不打包，逐任务提交记录照常保留。

模型与 adapter 权重、重复 tokenizer、原始仿真数据、下载的论文 PDF、虚拟环境、SDK 缓存、控制台日志及本机凭据不入库。模型和数据使用原下载/准备脚本获取，权重校验值及实验配置保留。论文调研保留文字、链接和下载清单。

`.env.local`、`key.env`、本机 `.aws/.codex/.agents` 等设置不会被读取或打包；示例环境变量文件保留空密钥。导出器检查高置信度凭据模式，发现匹配时停止。源码中的变量名和空示例不等于真实凭据。

`SNAPSHOT_MANIFEST.json` 记录逐文件原始快照 SHA256、上传文件 SHA256、大小、导出起止时间和任何被丢弃的不完整 JSONL 尾行。

## 验证与使用

常规 CPU 单元测试所需依赖参照 `requirements/`，部分统计测试需要 NumPy/Matplotlib；ALFWorld 环境与模型训练依赖分开安装。真实 GPU 复现需下载固定模型和环境数据，并按对应实验协议重建运行资产。

只还原某轮的已打包轨迹：

```bash
python3 scripts/unpack_snapshot_traces.py --study baseline-benchmark-v1
```

脚本验证压缩与解压后的校验值，已有同名但不同内容的文件不会覆盖。可用完整结果和图表直接查阅研究记录；不要在下载快照后立即执行 `--stage all` 恢复原机器进程，因为原机器的权重路径和虚拟环境并未打包。

重新导出当前工作区，不修改运行数据：

```bash
python3 scripts/export_github_snapshot.py --out /tmp/learning-harness-new-snapshot
```

导出目录必须为空。上传流程在独立 Git 工作区进行，运行机器原有数据、冻结代码和 GPU 进程保持原状。

## 第三方来源

仅随仓库存放运行所需的 ReAct/Reflexion 提示和少量参考源码，固定 commit 见 `configs/baseline-sources.lock.json`，各自 MIT 许可证保存在 `third_party/ReAct/LICENSE` 与 `third_party/Reflexion/LICENSE`。其他参考方法使用来源锁和获取脚本，不宣称已完成复现。此私有备份不另行授予本项目公开开源许可。
