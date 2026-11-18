# Stage 1 执行记录：2026-11-12

Stage 0选择完整交叉诊断。先运行真实GPU种子控制试跑、8个校准环境回合和只读NLL评分，再冻结主实验。98项CPU测试通过；随后新增的存储迁移3项检查通过。

冻结标识：`dc1dca75927cc3f362c26ac319b0b5222acf9673218221906685b71efa34fabd`。
输入、代码、分卡及上限在`runs/variance-diagnostic-v1/stage1/freeze.json`中。旧研究的冻结文件未修改。

- 4×4初始化/dropout种子与采样种子，每来源16个模型；各来源增加一次同配置训练重复，共34个。
- 每模型64次更新、256次示范呈现；后续没有B阶段优化。
- 每模型固定32个冷却、32个加热开发任务，加只读动作NLL诊断。共2,176个任务回合。
- 旧API与generic数据，不是新API与ReAct方法优越性比较。教师调用为0，正式测试未打开。
- 两张5090按冻结计划分配；累计上限6 GPU小时，包含控制和流程试跑。错误停止后续调度，未完成结果不填补。
- 完成后自动生成逐单元CSV、实际任务/标签/token暴露、交叉种子热图PNG/PDF、原始分量及留一因素敏感性范围。敏感性范围不是置信区间。

主进程启动PID为251680，独立观察器启动PID为251681，仅用于本次启动定位，不作为未来存活证据。实际状态以`execution.json`、进程和结果文件交叉核实。已经确认首4个模型完成训练、两族评测和NLL评分；其余继续排队。本记录不分析部分结果或改变预算。

[W&B进度](https://forge.coreweave.com/wandb/yanyaoyuan-jawaharlal-nehru-university/learning-harness-lab/runs/dc1dca75927c-progress)只报告完成数和资源。独立观察器在本地完整报告生成后自动归档原始JSON/JSONL、表格与图；上传失败不影响GPU实验，收据会标记失败。

## 存储

按用户指定，将旧实验及下载模型中超过16MiB的文件迁至
`/media/user/50D2BD30D2BD1AE4/WorkSpace/learning-harness-lab/`，保持相对目录结构。
先复制、fsync及SHA-256校验，再原子替换原路径为软链接。打开的文件句柄继续有效，新进程沿原路径访问同字节文件。
运行中的Stage 1目录不迁移。清单及每文件校验值位于
`output/research-operations/20261112/storage-relocation.json`。

迁移不改变冻结清单及其中的内容哈希；目标盘需要在后续实验时保持挂载。未验证的副本不会替换原件。

## 复核与恢复

```bash
PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-dev/bin/python -m pytest -q
cat runs/variance-diagnostic-v1/stage1/execution.json
cat output/variance-diagnostic-v1/stage1/progress-wandb.json
```

先核实`job.json`中的进程和本地服务器状态，禁止重复启动。中断恢复必须核对冻结哈希、记录未结算尝试耗时；已有半成品训练保留检查，不会被覆盖。
完整4×4诊断只决定下一步资源分配；即使触发30%规则，也不会自动启动Stage 2或据此认定抽样机制成立。
