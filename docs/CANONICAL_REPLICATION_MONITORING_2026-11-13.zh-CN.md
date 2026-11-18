# 统一评分选样复现：运行与监控

完成更新：96/96模型、6,144回合、零失败，14.350累计GPU小时；独立原始结果复核与W&B最终四文件字节核验均通过。训练控制器和观察器已结束。[最终结果和对照图](CANONICAL_REPLICATION_RESULTS_2026-11-13.zh-CN.md)。以下保留原启动记录，PID和“下一次”时间是历史信息。

2026-11-13 10:23（中国时间）启动。固定96新模型/6,144回合、六初始化×四配方×两来源×两策略，两张5090，18累计GPU小时上限。完整设计见[冻结前协议](CANONICAL_SELECTION_REPLICATION_SPEC_2026-11-13.zh-CN.md)。

- 研究根目录：`runs/variance-diagnostic-v1/stage5-canonical`；freeze SHA256：`3d16840d5eb187fd7d1b855ec103d279034813ba351a7465a00050cbf7302cd7`，287项冻结输入。
- 控制器启动PID4091316，W&B观察器4091398；首次两个worker为4091401/4091402，分别占用GPU0/GPU1。PID可能随生命周期退出，后续必须读取真实`/proc`身份和execution.json，不凭此历史记录判断运行。
- 新半小时监督器PID4091317，根目录`runs/research-supervision-v3`，30秒轮询、1800秒例行记录。首次记录10:23:29，下一次约10:53:29；原监督器1111147及v2配置不变。
- [W&B进度和中途图](https://forge.coreweave.com/wandb/yanyaoyuan-jawaharlal-nehru-university/learning-harness-lab/runs/3d16840d5eb1-progress)，配对节点1/4/8/12/16/20/24固定，不按方向修改队列。
- CPU检查37项通过；启动前旧Stage 4/NLL/WebShop分别237/67/61冻结输入重新核验。宿主GPU空闲背景进程与端口检查通过；工作盘可用约21.97GB，96个待训练模型及预留需约10.88GB。

旧worker、训练器、预算器及逐任务报告代码保留原样，新入口显式安装recipe-loader适配。新观察器修正W&B summary的调用形式，只上传汇总与图；原始任务和评分文件仍留本地。执行失败保留现场，不自动重试或覆盖。控制器结束且报告完整时，继续做原始结果复核、图表审核、云端字节核验及指定硬盘权重迁移。

估计约14.1累计GPU小时，双卡约7–8小时墙钟；以真实账本为准。此页是启动记录，实时数量以execution.json、进程及W&B汇总为准，不能把启动当成有效性结论。
