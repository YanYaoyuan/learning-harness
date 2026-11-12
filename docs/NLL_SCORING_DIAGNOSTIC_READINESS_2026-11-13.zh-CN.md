# NLL评分诊断：CPU验证与首次GPU运行前冻结

2026-11-13。当前状态为 **frozen_not_executed**，不包含真实GPU评分结果。研究问题与顺序见[执行协议](NLL_SCORING_DIAGNOSTIC_SPEC_2026-11-13.zh-CN.md)；[早期草案](NLL_SCORING_DIAGNOSTIC_DRAFT_2026-11-13.zh-CN.md)保留历史，不覆盖新冻结协议。

## 已实际完成

- 原两个train来源的1,453行引用重新编码并逐行检查缓存输入/目标长度、原始数据身份和监督后缀，得到853条不同messages。32个预先选择的压力输入对应256个可行布局条件及49份原批重放。准备过程显式隐藏CUDA，仅加载tokenizer。
- 使用原缓存精确再生Stage 3/4的全部16份min-NLL配方。另用原缓存均值构造的合成向量执行新名单比较路径；它只是CPU路径检查，不能当作统一重评分结果。
- 在实际模型Python环境、CUDA隐藏的条件下运行16项测试，全部通过。覆盖真实Torch的移位交叉熵、padding与截取logits对齐、短窗口拒绝、部分日志记账，以及显式CPU替身的完整worker成功/门槛失败路径。替身不初始化CUDA，不能证明真实4B模型可在时间预算内完成。
- Ruff静态检查通过；第二次CPU准备与第一次的编码文件SHA逐字节一致：`5cb074b9891b75eff6287b6ebde16d3ff55fe8c3d2dfadd2a8d4e1f980a61b2a`。
- 首次GPU评分前冻结67份输入，freeze SHA：`d07f9cfdaa743984b09503c74810c68c82c0342df0da9d09f5657ca731872e43`。随后实际CLI启动检查因Stage 4仍运行而拒绝，未创建execution或job，也未加载GPU模型。

[CPU检查汇总](../output/nll-scoring-diagnostic-v1/preparation/cpu-verification.json) · [顺序启动门槛检查](../output/nll-scoring-diagnostic-v1/preparation/parent-gate.json)。一次并行validate/run检查未匹配预期拒绝消息；两入口共用非阻塞锁，之后顺序重验得到明确的父研究运行中拒绝，没有执行账本。

## 父研究完成后的执行

先审阅Stage 4全部100模型/6,400回合的完整报告，确认训练、服务与读权重进程已结束，再核对GPU1占用、冻结文件及固定模型权重。该诊断独立于WebShop的GPU0校准，运行总上限600秒、只尝试一次；不重用父研究模型服务。

```bash
PYTHONPATH=src python3 scripts/run_nll_scoring_probe.py validate
# 仅在父研究完成、完整报告核对且GPU1可用后：
PYTHONPATH=src python3 scripts/run_nll_scoring_probe.py run
```

评分依次为独立加载的FP32参照、独立加载的BF16压力矩阵、原批重放。门槛通过且仍有预算时才尝试全部853条输入的统一评分；不足完整覆盖则保留部分记录，不生成新选样结论。首个真实GPU运行的吞吐、显存和门槛是否通过仍待验证。

只上传汇总与图表；编码文件、逐例分数和forward日志保留本地。诊断不自动训练后续模型，不替换旧缓存或既有结果，不把评分差异直接解释成训练性能差异。
