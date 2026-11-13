# Stage 4 完整报告审核与后续接续

100/100模型、6,400回合均完成，冻结输入和逐任务结果重新核对通过。主比较使用96模型、六初始化×四采样配方；四次预设重复单列。实际消耗14.663372累计GPU小时，上限18小时。每个来源×策略的24个主模型在两张GPU上各分配12个。

| 来源 | 随机成功率 | 低NLL成功率 | 低NLL−随机 | 胜/平/负 |
|---|---:|---:|---:|---:|
| API | 31.705729% | 26.888021% | −4.817708pp | 8/1/15 |
| Generic | 6.770833% | 3.776042% | −2.994792pp | 3/6/15 |

API逐初始化均值五负一零，Generic六个均负；全部正向配对保留。主比较与node-24逐项一致。固定开发任务上的描述性结果，不给未经校准的显著性或等效结论；两个来源属于同一ALFWorld环境。

## 预设重复

以下均为初始化101、采样配方101，原始与重复使用同一GPU及相同初始参数；四次最终训练权重均有变化。

| 来源 | 策略 | 原始成功率 | 重复成功率 | 重复−原始 |
|---|---|---:|---:|---:|
| api | random | 31.2500% | 31.2500% | +0.0000pp |
| api | min_nll | 34.3750% | 29.6875% | -4.6875pp |
| generic | random | 4.6875% | 6.2500% | +1.5625pp |
| generic | min_nll | 4.6875% | 3.1250% | -1.5625pp |

这些是预先指定单元的重复观察，不估计所有任务或初始化的总体方差，也不替换主分析。

## 最终归档恢复

原冻结观察器在结束状态提交时触发W&B 0.31.0的SummaryDict.update关键字参数TypeError；训练与本地完整报告已成功完成。独立CPU发布入口使用位置字典更新，保留原冻结源码及其SHA，仅上传既定四个汇总文件。四文件已下载与本地逐字节核验。

[恢复入口](../scripts/finalize_action_control_archive.py) · [完整本地复核](../output/research-operations/20261113/stage4-complete-local-review.json) · [云端四文件核验](../output/research-operations/20261113/stage4-complete-cloud-check.json) · [最终汇总](../output/variance-diagnostic-v1/stage4/aggregate-summary.json)

## 已接续的两项检查

- [独立NLL诊断](../output/nll-scoring-diagnostic-v1/RESULTS.zh-CN.md)：107.127 GPU秒，1,194次forward、853/853统一评分，仪器门槛通过；布局变化引起4–11/256条选样变动，尚未重训新名单。
- [WebShop校准](../output/webshop-development-pilot-v1/RESULTS.zh-CN.md)：64回合、463请求、243.564 GPU秒；基础学生两提示成功5/32、6/32，属于train-only起点校准。

后续优先补齐全合格经验下API−ReAct同预算来源比较、共同强执行内核与新任务评测，并制定第二环境学习/选样对照。保留原API−ReAct matched-success负向结果；独立名单变化、校准起点和跨环境学习效应分别核算。
