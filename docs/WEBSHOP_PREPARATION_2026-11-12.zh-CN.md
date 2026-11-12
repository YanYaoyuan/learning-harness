# WebShop 第二环境准备记录

这是环境准备与协议审计，不是第二环境上的方法效果实验。当前状态为 `development_ready / benchmark_not_ready`。ALFWorld Stage 3继续使用原冻结协议、GPU与预算；本准备使用独立CPU环境，没有教师调用或正式测试评测。

## 已执行与可复现入口

- 官方源码：`princeton-nlp/WebShop`，固定提交 `64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd`。源码包SHA256见 `output/webshop-preparation/20261112/source.json`。未执行会修改系统的上游 `setup.sh`。
- `.venv-webshop`独立于训练环境，Python 3.10、CPU Torch 1.11.0、spaCy 3.3.0、`en_core_web_sm` 3.3.0、Pyserini 0.17.0、Java 11。安装脚本为 `scripts/prepare_webshop_runtime.py`，版本配置在 `configs/webshop-*-requirements.txt`，实际包清单在 `pip-freeze-validated.txt`。
- Pyserini使用`--no-deps`，只验证本实验所需的稀疏Lucene检索；未安装NMSLIB等无关后端，不宣称完整Pyserini依赖集通过检查。
- `scripts/smoke_webshop_runtime.py --community-mirror`已完成1000商品预览、Lucene索引、文字观察与搜索→商品详情→模拟购买的完整机械检查。采用合成目标，0次学生/教师LLM调用。终止奖励0只说明预设动作未命中合成目标，不是agent效果。
- `scripts/check_webshop_runtime.py`运行上游手工构造的奖励/规范化单元测试；`scripts/diagnose_webshop_reward.py`记录词性与标题分数，保持奖励源码和期望值不变。

脚本具有不覆盖已有收据的保护。复现应使用干净检出/新的记录目录，不能在既有实验目录盲目重跑。第一次依赖检查失败（缺ONNX Runtime）及第一次官方Drive下载失败均保留，修复后的状态分列记录。

## 数据来源与范围

上游Drive的预览商品与属性下载链接实际返回404。改用两个公开社区镜像，逐文件独立下载并核对SHA256相同：

- `YWZBrandon/webshop-data`：`ce990fff5aee388db2706f07820c578ab68e0453`。
- `HongbangYuan/webshop`：`0129d4a81dbdb827e76afd20a1e2c38b61098613`。

商品1000条、属性1000条，另有10136项人类指令注释。上游加载器即使关闭人类目标仍会打开注释文件；本次仅使用合成目标，未执行人类测试目标。两个镜像一致不等于已验证与原作者发布文件逐字节一致。完整URL、文件大小、SHA256、实际机械动作见 `smoke.json`。

原始数据位于用户指定磁盘 `.../WorkSpace/learning-harness-lab/data/webshop/smoke1000/`，源码目录保留文件链接。GitHub仅上传代码和来源收据，排除商品数据、索引和虚拟环境。本次没有下载完整商品库，1000商品环境不能标为完整WebShop benchmark。

## 初次单元测试：当前上游SM版本5项通过，1项失败

实际运行上游6项测试，标题类型奖励测试失败，其余5项通过。首个差异：期望标题分数约0.333，当前`en_core_web_sm`得到0.4444。独立诊断中另外两组标题分数为0.2778（期望0.3）和0.1429（期望0.15），第四组为0。

诊断中四组的类型奖励依次为1、1、0.5、0；这些例子中的标题分数差异没有跨越对应奖励阈值。但这不能证明所有商品上的奖励等价，也不能声称原失败测试的后续断言已经执行。`upstream-tests.json/xml`、日志和`reward-diagnostic.json`保留完整记录。

源码实际加载`en_core_web_sm`，上游安装脚本却下载`en_core_web_lg`；此处严格按奖励源码使用sm，没有替换模型或修改测试以获得通过。正式协议冻结前，需要确认一个可复现的参考奖励环境，并对固定手工案例及训练/开发案例核对奖励分量。现阶段不调模型效果来选择奖励实现。

## 正式实验前必须固定的实现细节

1. **划分只对完整商品模式启用。** 上游`baseline_models/env.py:27`仅在`args.num is None`时设置test `[0,500)`、eval `[500,1500)`、train `[1500,N)`；缩小商品模式三个split均使用全部目标。直接把预览模式作为正式划分会造成重叠。
2. **默认路径不是完整商品库。** 当前提交`web_agent_site/utils.py`默认商品和属性文件指向1000商品文件。正式运行须审计实际路径、完整索引、加载数量与目标清单，不能只设置`num_products=None`就宣称完整环境。
3. **记录实际目标身份，不仅记录整数ID。** `SimServer`在生成目标之后用233打乱顺序；生成价格限制等操作也使用随机数。应固定加载前种子、目录顺序、文件哈希，并保存目标内容指纹与split映射。正式测试目标不进入调参。
4. **学生输入仅限可见信息。** 上游基线`get_valid_actions`会从隐藏目标读取`attributes/query`来构造搜索候选。这可作为注明额外信息的原始基线，但不能用于我们的无特权学生。学生可用可见指令、观察、页面按钮，自行生成搜索词；奖励内部信息不进入采集提示或训练标签。
5. **奖励与检索先冻结。** 保持上游动作、检索、价格/属性/选项/类型奖励语义，固定运行依赖、最大交互步数与主指标。成功率和连续任务分数分列，禁止把预览奖励与论文完整环境成绩直接横比。

## 下一步边界

Stage 3完成后按其中得到的机制假设设计复现实验：固定来源经验池，比较全池遍历、任务匹配随机子集、低NLL子集，分别核算训练量、任务暴露、不同内容数与学生自主表现。先做训练/开发集的环境和吞吐校准，冻结主要对比与样本量后才打开正式测试集。

本记录未预注册正式效应、阈值或样本量，也未声称第二环境已验证任何论文结论。

## 补充核验：原版LG奖励已通过6项测试与预览机械流程

21:04前完成的核验已经定位上述差异。官方[2024年提交](https://github.com/princeton-nlp/WebShop/commit/d149d0d7a5219c1ea4e60c80f2364a15eb324d7b)将奖励文件中的`en_core_web_lg`改为`en_core_web_sm`，奖励测试仍是2022初始版本。两版奖励文件除这一行完全相同，规范化依赖与两份测试文件也逐字节相同。

已另外下载并保留2022官方初始提交`61e56dbba107ae13ff49e6058d7d2aad23326929`，使用同一spaCy 3.3.0环境与官方`en_core_web_lg` 3.3.0：**6项原测试全部通过**。未修改奖励、测试或期望值。官方模型wheel实际400651786字节，SHA256为`6ce19d37dfe5280400f80a5954d41afca10cbc742b97bfcf4b0e452b6eb24273`；wheel和解压模型均存放于用户指定大盘的`learning-harness-lab/data/models/webshop-nlp/`，不进入GitHub。

原版源码还完成了相同1000商品预览的搜索→详情→模拟购买终止检查，使用6910个合成目标之一、0次学生/教师LLM调用、0个正式测试目标。代码前后哈希一致。这里的终止奖励0仍是预设动作的机械检查，不是agent成绩。

入口为`scripts/check_webshop_original_reward.py`与`scripts/smoke_webshop_original_reward.py`。收据、XML、原失败日志和提交历史全部保留在`output/webshop-preparation/20261112/`，最新奖励准备状态见`readiness-latest.json`。原版和当前上游版本并存，不能将单个奖励测试通过称为整个WebShop benchmark就绪。正式环境、完整目录、划分、学生可见信息与预算仍需另行固定；比较论文原始分数时优先采用原版奖励语义，并注明版本。
