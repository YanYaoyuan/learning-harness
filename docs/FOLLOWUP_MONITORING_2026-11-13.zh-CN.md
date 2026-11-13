# 后续短实验的独立只读监控

**完成更新（2026-11-13）**：NLL诊断与WebShop校准均已完成并独立复核；观察器在两者均为completed后正常退出，停止记录的signal_requested=false。正常结束不需要恢复进程。原半小时主监督器继续独立保留；主实验结果观察器的SDK收尾错误已通过独立CPU汇总发布入口恢复，冻结源码未修改。

- [NLL完整结果与图](../output/nll-scoring-diagnostic-v1/RESULTS.zh-CN.md)：1,194次forward、853/853统一评分、107.127 GPU秒。
- [WebShop完整校准与图](../output/webshop-development-pilot-v1/RESULTS.zh-CN.md)：64回合、463请求、243.564 GPU秒，全部用量已记账。
- [主实验完整报告与归档恢复](STAGE4_COMPLETE_RESULTS_2026-11-13.zh-CN.md)。

## 初始设计与部署记录

2026-11-13部署时，主实验半小时监督器和W&B观察器仍在运行，冻结代码保持原样。新增同一个观察器覆盖已冻结的NLL评分诊断与WebShop基础模型校准；它不启动、重启、恢复或停止任何实验，不查询模型服务。

## 实际验证

7项CPU测试通过，覆盖未启动、调用待返回/截断日志、私有内容过滤、完成报告缺失、进程缺失、WebShop累计时间及输入身份拒绝。使用实际W&B环境完成一次无网络CPU快照；真实主机上已启动`track_bounded_followups.py --watch --wandb`，随后核查PID/命令身份。主研究237份、WebShop 61份和评分诊断67份冻结输入仍全部一致。

部署时两个后续实验均为`not_started`、0调用；创建监控run本身不代表GPU实验开始。之后在Stage 4完整报告复核、对应GPU资源检查通过后，依次实际执行了评分诊断与WebShop校准。

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

# 历史启动命令；原观察器已在两项实验完成后正常退出
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' .venv-wandb/bin/python \
  scripts/track_bounded_followups.py --watch --wandb
```

本地运行入口为`runs/bounded-followup-monitor-v1/launch.json`，运行中状态为同目录`state.json`。云端[后续实验进度](https://forge.coreweave.com/wandb/yanyaoyuan-jawaharlal-nehru-university/learning-harness-lab/runs/bounded-followups-20261113)与主研究分开；[云端读取核验](../output/research-operations/20261113/followup-monitor-cloud-check.json)记录实际查询结果。该页面是进度记录，最终科学图表和结论另行审阅后发布。
