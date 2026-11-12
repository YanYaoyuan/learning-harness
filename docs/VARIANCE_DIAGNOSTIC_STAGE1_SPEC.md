# Stage 1: initialization/dropout crossed with sampling

Date: 2026-11-12. Exploratory diagnostic; freeze independently before main training.

## Question and design

Stage 0 found little batching variance on the legacy full API/generic cohort. Stage 1
separates initialization plus dropout from the selection/order of training examples.
It does not compare the newer API/ReAct candidate, establish a new method, or open official tests.

- Sources: the fixed full-api and full-generic corpora of transfer-factorial-v1, 721/732 examples.
- Initialization/dropout seeds: 101, 202, 303, 404.
- Sampling seeds: 101, 202, 303, 404, fully crossed with initialization.
- Each source has 16 main adapters; repeat initialization=101/sampling=101 once per source.
- Total: 34 newly trained adapters, each with 64 updates × batch 4 = 256 presentations.
- Training kernel/configuration: the frozen train_lora.py, invoked through the tested
  split-seed wrapper; dropout remains 0.05. No original trainer file is changed.
- Primary measurement: post-A success on the existing 32 cool and 32 heat development tasks,
  equally weighted across families. Common historical JSON execution, seed 11, temperature 0,
  batch 8, 40 calls, 128 output tokens/call. Total planned task episodes: 2,176.
- Continuous co-outcome: task-mean action NLL on each family's fixed diagnostic examples,
  equally weighted across families. No optimizer exists in the scoring process.

A fixed, hash-verified zero B branch is added to every A adapter for compatibility with the
historical serving kernel. It is never trained. Actual initialization and schedule hashes
must show: same initialization across all sampling seeds/sources, same sampling order across
initialization seeds within each source. Repeated training may have different final hashes.

## Execution and budget

The four one-update control checks already passed. Before freezing, an end-to-end plumbing
check reuses one of these tiny adapters, evaluates four preselected calibration tasks per
family, and scores the diagnostic NLL. No main 64-update adapter or query outcome is used
to tune the design. This check is excluded from the analysis.

Main source assignments are counterbalanced across physical GPUs 0/1 by the parity of the
initialization/sampling grid indices. Training repeats use the same GPU as their originals.
The plan and all relevant code/data hashes are stored in freeze.json.

The cap is six cumulative GPU-hours including the existing control smoke and pipeline
smoke. An adapter job reserves at most 1,200 seconds and starts only with at least 600
unreserved seconds remaining. Two jobs may run concurrently. Each worker enforces its own
timeout; the supervisor also terminates its process group on timeout. Shutdown can add up to
20 seconds per active job. Failed and interrupted attempts are charged. A recovered attempt
with unknown duration is conservatively charged its entire reservation; live workers block
resumption. The cap is not increased after outcomes are observed.

Model/rollout errors stop further scheduling and are never counted as task failure.
Committed evaluations can be resumed under unchanged hashes. Partial training is retained
for inspection and is never overwritten. Raw control/episode/token metadata is retained.

## Analysis and decision

For each source and for API−generic, form a 4×4 matrix of the equally weighted family score.
Use the crossed two-way method-of-moments decomposition:

    v_init_raw = (MS_init − MS_residual) / n_sampling
    v_sample_raw = (MS_sample − MS_residual) / n_initialization
    v_interaction = MS_residual

The interaction also contains remaining training/evaluation noise: one run per main cell
cannot separate those terms. The counterbalanced GPU assignment can also contribute to this
interaction; hardware effects are not independently estimated. Report raw estimates; truncate negative main components only
for the budget share. Denominator <1e-12 gives an undefined share.

For uncertainty, publish all cell values and leave-one-initialization/sampling-level-out
sensitivity ranges. These ranges are not confidence intervals. Four levels per factor do
not justify claiming calibrated variance-component intervals or method superiority. The
two training repeats are descriptive conditional checks, not population noise estimates.

The allocation rule is the previously proposed point-estimate rule: recommend a complete-pass
sampling follow-up if the sampling component accounts for at least 30% of summed nonnegative
components on paired success or paired NLL. Undefined components do not trigger it. Otherwise
prioritize initialization/dropout or unresolved noise. Report whether sensitivity ranges
cross the threshold. No Stage 2 is launched automatically and no winner is selected.

All queries are reused development data. Future confirmatory sizing still requires discrete
outcome power calibration and a new frozen protocol.

## Outputs

Per-case initial/final/schedule hashes, actual label/task-type/token exposure, success records,
NLL records, repeats, the 4×4 matrices, variance table, sensitivity ranges, PNG/PDF figures,
GPU ledger and W&B records. Partial results are labelled incomplete; no missing outcomes
are filled in.
