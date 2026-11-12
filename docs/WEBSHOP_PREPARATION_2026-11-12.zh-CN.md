# WebShop 第二环境准备记录

这是环境准备与协议审计，不是第二环境上的方法效果实验。当前状态仍为 `development_ready / benchmark_not_ready`。2026-11-13已补齐完整商品/属性文件及流式结构审计，详见文末；完整索引、目标映射、公开信息接口与正式协议仍待完成。ALFWorld Stage 3已完成，Stage 4按独立冻结协议运行；本准备使用独立CPU环境，没有教师调用或正式测试评测。

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

预览原始数据位于用户指定磁盘 `.../WorkSpace/learning-harness-lab/data/webshop/smoke1000/`，源码目录保留文件链接。GitHub仅上传代码和来源收据，排除商品数据、索引和虚拟环境。2026-11-12的准备没有下载完整商品库，1000商品环境不能标为完整WebShop benchmark；完整文件于次日另外获取，旧预览保留。

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

## 2026-11-13：完整商品文件获取与流式结构审计

两份官方完整文件链接在本次HEAD探测中均返回404，原始地址和结果保存在`20261113/official-full-links-probe.json`。两个既定社区镜像的固定版本元信息声明相同的完整文件大小和LFS SHA256；实际从第一个镜像下载一次，并逐文件核对实际字节与两份元信息一致。没有据此声称已和原作者文件逐字节核验。

| 文件 | 实际字节数 | SHA256 |
|---|---:|---|
| items_shuffle.json | 5,479,720,229 | 2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8 |
| items_ins_v2.json | 186,295,270 | 1d36af476bdb8f82a5da62bd8acdabe54cd8de2fa84010d37da5c4890feb447e |

文件位于指定硬盘`learning-harness-lab/data/webshop/full-catalog-2ef591d65df3/`。下载入口`scripts/download_webshop_full_catalog.py`使用固定元信息、SHA256和容量预留；不覆盖已有收据/文件，不改现有预览路径，失败保留部分文件。获取收据为`output/webshop-preparation/20261113/full-catalog/download.json`。

随后用独立CPU环境的ijson 3.3.0/yajl2_c流式解析，而非把5.5GB商品库整体读入内存：原始商品1,181,436条，ASIN均唯一；按原上游ASIN过滤规则排除6条，剩1,181,430条。属性记录1,181,436条，全部可用商品有属性记录。本次检查的必需字段无缺失，字符串及首图片等有限类型核对未发现不兼容；这不替代完整环境加载、价格/选项规范化及检索核验。

本次审计约21秒，峰值RSS约287MiB，没有打开人类目标注释文件、生成目标映射或运行正式测试。完整来源/结构收据在`output/webshop-preparation/20261113/`，入口`scripts/audit_webshop_full_catalog.py`，固定新增CPU依赖在`configs/webshop-data-requirements.txt`。训练环境未安装此依赖。

完整检索索引、固定随机数下的目标身份与划分、仅使用可见信息的学生接口，以及正式训练/评测预算仍未完成。完整文件获取不被写作第二环境结果。

## 2026-11-13：完整索引及公开输入接口的独立检查

`prepare_webshop_full_index.py`先在全部1000条预览记录上核对精简文档的ID与检索文本，结果完全一致；随后对7个固定公开查询逐一比较原索引与新索引，前50结果的ID、顺序和分数全部一致（包括零命中查询）。环境检索源码仅从原始索引文档取`id`，所以新索引只存`id/contents`，不重复存整份商品及附带的目标字段。原商品转换中的字段、大小写、选项归一化和顺序保持一致。

完整转换生成1,181,430条文档，约1.05GB；Lucene原默认行为实际收录1,181,370条。初次验证误把“可用商品数”和“可索引文档数”设为相等而停止。原日志、`index.json`失败状态及当时完整脚本`attempt-01-preparer.py`均保留。后续遍历全部生成文档，确认恰好60条检索文本为空白，与Lucene的`empty=60`一致；`errors/unindexable/skipped`均为0。全部60个空文本ID在索引中缺失，首64个非空文档的ID/文本与输入一致，索引查询和文件哈希校验通过。没有重建索引、删除商品或修改奖励；完整商品集合仍为1,181,430条。

完成依据为`output/webshop-preparation/20261113/full-index/verification.json`，而非被保留的首次失败记录。索引约1.42GiB，位于指定大盘的`data/webshop/full-index-2022-v1/full/index/`，原1000商品索引与源码路径未改动。未来从头构建时，入口已按实际空文本数核算预期收录量；保留索引的只读重验入口为`--verify-existing`。本次构建和验证均没有打开人类目标注释或运行正式目标。

`webshop_public.py`提供明确的策略输入对象：`observation/can_search/clickables`，从环境公开文字与DOM控件构造；搜索词由策略自行产生，不调用上游依赖隐藏目标的`get_valid_actions`。奖励和终止标志留在runner侧；终止页面的调试内容不再送给策略。适配器先核对显式允许的目标索引，终止后拒绝继续操作。它是数据边界，不是进程隔离；底层完整环境仍须在第一次自动reset之前限制目标池。

新增8项边界检查后，完整独立CPU套件205项通过。真实原版LG奖励环境也通过了适配器的搜索→点击公开商品→购买机械流程，使用1000商品、合成目标、预定动作、0次模型调用。为了不让上游在合成模式下顺便载入人类注释，该进程把未使用的注释路径指向空JSON夹具，并用文件打开保护拒绝原注释路径；源码字节保持一致。终止奖励0只是机械动作结果，不是agent成绩。收据为`public-preview/mechanics.json`。

最新综合准备状态见`output/webshop-preparation/20261113/readiness-latest.json`。剩余是全目录下目标身份/随机数/划分的冻结、完整环境工厂和公开接口的集成，以及正式实验预算与方法协议。当前不声称第二环境已有模型效果或完整benchmark已就绪。
