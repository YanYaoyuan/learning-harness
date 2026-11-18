# WebShop公开基线：来源核验与待执行校准

状态：公开代码/提示已获取并核对，尚无模型调用或模型效果评测。原ALFWorld Stage 4继续运行；其模型服务与GPU预算不被此准备使用。

## 固定来源

作者仓库[ysymyth/ReAct的WebShop.ipynb](https://github.com/ysymyth/ReAct/blob/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9/WebShop.ipynb)，commit `6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9`。官方树中的文件名首字母大写。GitHub插件未返回该1.26MB文件的内容，因此直接下载固定版本原文件，并核对git blob SHA `67cb9f878de242951a7804a31bb5da4264111a29`。原文件SHA256为`fb8704ec5528e43589924ab33d14eb7f97e82f7cd9da5c02b0295a9be30f9ca6`。

`third_party/ReAct-WebShop/`仅保留来源收据、MIT许可证、去掉全部输出的source cells和通过AST字面量解析提取的两份提示：`prompt1`为1-shot ReAct、`prompt1_actonly`为1-shot Act-only。没有import/exec Notebook，也没有使用其中的测试轨迹输出。原文件留在本地下载目录，不作为新实验结果上传。

提示字符数分别1786和1462，SHA及对应cell索引见`output/webshop-preparation/20261113/react-baseline/source-audit.json`。提示中完整指令在当前固定价格约束下，未与已封存目标的可见指令指纹精确匹配；这只检查包含价格约束的完整字符串，不证明语义、商品或不同价格版本互斥。

## 执行语义审计

原Notebook使用temperature=0、每次最多100输出token、换行stop、总输入6400字符的前缀加历史后缀。`think[...]`的公开反馈为`OK.`。

原循环15次包含一次初始reset，因此至多执行14个模型决定；若到最后仍未终止，还请求一次不会执行的completion。准备的chat移植应明确执行14个决定并核账实际调用，避免把未执行的请求当成额外动作机会。

原HTTP观察只显示前三个商品链接及相邻摘要，并人为禁用翻页；现有本地原版环境的文本/DOM控件不同。后续在本地Qwen3-4B指令模型上使用这些提示时，需要固定chat包装、公开观察渲染、动作解析与步骤计数，称为公开提示的本地移植，不能直接当成原远程服务器成绩复现。两数据来源和所有训练后模型使用同一固定执行方式。

## 下一步校准草案

先做完整目录train目标1500—1531的32任务、两提示方式，使用固定基础学生模型，无LoRA训练、教师或test/eval回合。比较自主成功率、连续得分、格式/公开动作错误、实际token与耗时；先确认底座及执行接口可用，再花预算采集训练池和训练adapter。

配置为`configs/webshop-development-pilot-v1.json`，实际执行入口和[校准协议](WEBSHOP_DEVELOPMENT_PILOT_PROTOCOL_2026-11-13.zh-CN.md)现已实现。它不自动请求当前正在运行的Stage 4模型服务，不把当前病例adapter当作基础模型。正式模型试运行在原队列完成、完整报告核对及GPU释放后启动；当前没有启动新模型或调用教师API。

## 已实现的CPU运行器组件

`webshop_agent.py`已实现本地chat端口的核心：从公开view构造文本与按钮标签，使用固定系统说明及原提示前缀，将主体前缀/历史限制在6400字符内（系统说明另列）；换行stop、每次100 token、14个可执行决定。可选`Action:`前缀被解析，思考反馈`OK.`且占一次调用，不执行环境动作；格式或不可见控件错误占调用但不借隐藏目标修复。全部实际输入/输出token由客户端计数，运行器拒绝超预算响应并关闭回合。

最初5项CPU检查覆盖思考与物理动作分别记账、错误/不可见动作、无多余末尾请求、字符预算和端点输出越界，原216项测试收据保留。现增加请求发出前fsync、丢失响应/异常/超预算用量恢复，以及独立基础模型服务的严格请求和身份检查。原HF服务不支持这份公开基线要求的换行stop，因此新增`serve_webshop_base.py`，不修改冻结训练服务。新服务在生成中停止换行，并计入实际生成的完整token。

新`run_webshop_pilot.py`和隔离CPU worker覆盖：先验证train后加载专用基础模型、绑定进程/权重/运行时、包含加载失败和清理的3600秒GPU预算、进程组终止、原始消息本地保留和仅模式级汇总。新增三条生命周期检查以显式CPU替身进程验证成功、失败、到期停止和未知请求恢复；完整套件248项通过。这些测试未加载CUDA模型，不是语言模型效果，也未查询Stage 4服务。真实模型表现和服务CUDA加载仍待父队列完成后核对。
