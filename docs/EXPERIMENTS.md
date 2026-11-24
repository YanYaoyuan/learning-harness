# Experiment index

The completed selection study is reported in [From Acting to Learning](https://arxiv.org/abs/2611.27159). Earlier experiments below document the progression from assisted data collection to experience selection. Their splits, budgets, and evaluation protocols differ, so absolute success rates should be read within each study.

## Paper study

The paper compares matched-coverage and global selection rules on two fixed experience pools. It reports 288 trained students and a separate reserved-panel evaluation of 97 checkpoints. The reserved evaluation comprises 128 task groups and 12,416 episodes. The main result table and study scope are summarized in the [README](../README.md#results).

The historical snapshot on the default branch contains the collection, SFT, adaptation, and analysis code listed below. It is not a complete export of the later selection-study workspace: later model artifacts and experiment-specific selection files must not be inferred to exist from the paper link. The paper describes that study's full protocol.

## Archived studies

| Study | Question | Report or protocol | Code entry point |
|---|---|---|---|
| `dual5090-v1` | Does early assistance change the value of successful training data? | [Results](DUAL5090_RESULTS.zh-CN.md), [design](DUAL5090_DISCOVERY.zh-CN.md) | `scripts/train_and_evaluate_study.py` |
| `dual5090-v2` | How do matched task coverage and sample counts affect source comparisons? | [Results](DUAL5090_MATCHED_RESULTS.zh-CN.md), [design](DUAL5090_MATCHED_PROTOCOL.zh-CN.md) | `scripts/matched_study.py` |
| `dual5090-v2-filtered` | What changes after filtering public invalid-action feedback? | [Results](DUAL5090_FILTERED_RESULTS.zh-CN.md) | `scripts/filtered_study.py` |
| `learning-speed-v1` | Do different A-stage experience sources change B-stage adaptation? | [Design and results](LEARNING_SPEED_NEXT_PHASE.zh-CN.md) | `scripts/run_learning_speed.py` |
| `api-harness-v1` | Does a task-specific teacher card improve collection and adaptation? | [Audit](EXPERIMENT_AUDIT_2026-11-10.zh-CN.md) | `scripts/run_api_harness.py` |
| `transfer-factorial-v1` | Does success-data matching change the API-versus-generic source effect? | [Design and results](LEARNING_SPEED_NEXT_PHASE.zh-CN.md) | `scripts/run_transfer_factorial.py` |
| `baseline-benchmark-v1` | How do direct execution, Act, ReAct, and Reflexion compare? | [Protocol](BASELINE_EVALUATION_PROTOCOL_V1.zh-CN.md), [overlap amendment](BASELINE_PROTOCOL_V1_AMENDMENT_01.zh-CN.md) | `scripts/run_baseline_benchmark.py` |
| `source-comparison-v1` | How does teacher-assisted collection compare with ReAct self-collection after matched SFT? | [Protocol](SOURCE_COMPARISON_V1.zh-CN.md), [dated log](EXPERIMENT_LOG_2026-11-11.zh-CN.md) | `scripts/run_source_comparison.py` |

## Supporting analyses

- [Completed-experiment audit](EXPERIMENT_AUDIT_2026-11-10.zh-CN.md): reconstruction of source effects and analysis of final-task failures.
- [Reward-group diagnostic](AGENTIC_RL_READINESS.zh-CN.md): binary reward variation in two samples per task.
- [Benchmark and baseline survey](AGENT_BENCHMARK_LANDSCAPE_2026-11-10.zh-CN.md): resources and comparable evaluation settings, as of its recorded date.
- [Initial experiment plan](EXPERIMENT_PLAN.zh-CN.md): original environment, task splits, and support-timing conditions.

## Reading historical results

Reports distinguish collection success, post-SFT ability, initial adaptation ability, raw learning-curve AUC, and gain AUC. A stronger starting point can raise raw AUC without increasing subsequent learning gains.

Several early analyses used a percentile bootstrap over only three training seeds. Later variance diagnostics found that this procedure can under-cover in seed-dominated settings. Those intervals remain part of the historical reports; they should be read alongside per-seed values. The final paper uses paired descriptive comparisons on finite evaluation panels for its main result.

Frozen protocol and source hashes identify the versions used by each experiment. Original results, prompts, task manifests, and training schedules are retained.
