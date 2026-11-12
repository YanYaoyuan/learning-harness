# 后续短实验的独立只读监控

2026-11-13。主实验的半小时监督器和W&B观察器继续运行，不更换或修改其冻结代码。新增同一个观察器覆盖已冻结的NLL评分诊断与WebShop基础模型校准；它不启动、重启、恢复或停止任何实验，不查询模型服务。

## 实际验证

7项CPU测试通过，覆盖未启动、调用待返回/截断日志、私有内容过滤、完成报告缺失、进程缺失、WebShop累计时间及输入身份拒绝。使用实际W&B环境完成一次无网络CPU快照；真实主机上已启动`track_bounded_followups.py --watch --wandb`，随后核查PID/命令身份。主研究237份、WebShop 61份和评分诊断67份冻结输入仍全部一致。

当前两个后续实验均为`not_started`、0调用；创建监控run不代表GPU实验已开始。评分诊断仍须等待Stage 4完整报告审阅、GPU1空闲；WebShop仍须等待父研究完成和GPU0可用。

## 记录范围与时间

- 每30秒读取进程身份、执行账本和本地日志计数，更新本地状态及W&B数字曲线；每1,800秒及状态/异常变化追加本地检查记录。
- NLL只提取各阶段完成观测数、forward尝试/返回/未知用量数、已知token消耗及GPU存活时间估计；不上传输入指纹、token、逐例分数或forward原记录。
- WebShop通过原冻结恢复器核对已完成回合与请求用量，上传完成数和调用/token总数；不上传目标、商品、逐回合记录或提示。进行中的未知用量保留为未知，不记作0。
- 状态值直接区分失败、中断和预算停止；`needs_attention`专门指出进程身份/存活或完成证据不一致。不会由观察器自动改队列。
- `audited_complete`是完成证据字段：评分侧检查绑定freeze的生产者报告和权重后验核验标记；WebShop侧复用逐回合用量恢复器并检查停止记录。它不替代研究结束后的完整结果审阅。
- 观察器最多运行24小时；退出或网络故障不影响实验。恢复观察器前先核对其真实PID与锁，避免两个云端写入者。

CPU快照不创建云端run；持续模式使用一个独立的进度run，避免与主研究写入冲突。只记录明确白名单的汇总数字，并关闭代码、git、控制台及系统统计自动上传。

```bash
# 只读CPU快照（独立输出目录，不创建云端run）
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' .venv-wandb/bin/python \
  scripts/track_bounded_followups.py --out runs/bounded-followup-monitor-smoke-v1

# 已实际启动；不要在其仍存活时重复执行
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' .venv-wandb/bin/python \
  scripts/track_bounded_followups.py --watch --wandb
```

本地运行入口为`runs/bounded-followup-monitor-v1/launch.json`，运行中状态为同目录`state.json`。云端[后续实验进度](https://forge.coreweave.com/wandb/yanyaoyuan-jawaharlal-nehru-university/learning-harness-lab/runs/bounded-followups-20261113)与主研究分开；[云端读取核验](../output/research-operations/20261113/followup-monitor-cloud-check.json)记录实际查询结果。该页面是进度记录，最终科学图表和结论另行审阅后发布。
