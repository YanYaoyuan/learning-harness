# Stage 0：批布局与数值敏感性

预算分支：Stage 1 in full

对象是旧的 API−generic 全量经验模型，不是刚完成的 API−ReAct 对照。
改变任务排列、保持温度0及环境种子不变；不能据此估计所有执行随机性。

完整新重复：[1, 2, 3]；累计 GPU 时间：2.031 小时。
主估计纳入历史重复：False。

| 任务族／来源 | 发生结果翻转的 checkpoint-task 单元 |
|---|---:|
| cool / api | 5/96 |
| cool / generic | 4/96 |
| heat / api | 4/96 |
| heat / generic | 3/96 |

方差分量及近似区间见 summary.json；只有三个训练种子，不能将预算分支当作机制结论。
原配置重跑与历史结果的差异也可能包含历史环境变化，单独列于 determinism_checks。
