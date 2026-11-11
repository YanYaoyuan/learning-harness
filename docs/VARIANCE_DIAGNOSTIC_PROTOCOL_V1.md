# Variance diagnostic protocol v1 (draft for freezing)

Date: 2026-11-11. Status: **draft, not frozen**. Values marked *proposed* are placeholders the
team must confirm or replace before freezing. Nothing here changes the endpoints, code or
inputs of `source-comparison-v1` or any earlier frozen study.

## Purpose

Decide the design of the next main study from evidence instead of assumptions. The diagnostic
answers three questions and makes no claim about which collection source is better.

1. Does the planned analysis hold its nominal Type I error, and what power does it have?
2. Where does the variance of post-A performance come from: example sampling, initialisation,
   or evaluation?
3. Does any prespecified experience predictor vary enough within a task, and reliably enough,
   to make a predictor study feasible?

All results are exploratory. Query sets are the reused development sets. The official
seen/unseen sets stay closed. No teacher API calls are made.

## Part 1. Power calibration

Computational reproduction is done: the team reproduced the heating and cooling outputs of
commit `39e9b9e` to floating-point precision. That confirms the arithmetic, not the power.

- The minimum-detectable-difference columns of `scripts/power_projection.py` are planning
  approximations and are not treated as bounds.
- Calibration uses `--simulate`, which draws data from the planning components and reports
  rejection rates at a true difference of zero and at the effect of interest, for each
  candidate analysis (Satterthwaite t on the crossed layout; seed-and-task percentile
  bootstrap as in the frozen reports).
- Known limit: the simulation uses normal components. Before the main study is frozen it must
  be repeated with a generator for binary outcomes fitted to the Part 2 data.

Decision rule (*proposed*): an analysis is admissible for the main study only if its simulated
Type I error is at most 0.065 at nominal 0.05 under both the estimated and the four-fold
inflated seed variance. The main-study design is the cheapest one giving simulated power of at
least 0.80 for the admissible analysis at the practically meaningful effect. That effect is
chosen by the team before Part 2 results are seen (*proposed*: 5pp on post-A success).

## Part 2. Variance attribution

### Design

Paired corpora: the full-experience API and generic corpora of `transfer-factorial-v1`
(721 and 732 eligible examples), where the large post-A seed variance was observed.

| Factor | Levels |
|---|---|
| Source | API, generic |
| Sampling arm | (a) current: 256 presentations; (b) 256 presentations with fixed task-type exposure; (c) complete pass: budget equal to the larger corpus, same for both sources |
| Initialisation seed | 4 (*proposed*) |
| Sampling seed | 4 (*proposed*), fully crossed with initialisation seed |

That is 2 x 3 x 16 = 96 A adapters. Each is evaluated once after A on the 32 cooling and 32
heating query tasks with the common JSON execution, giving 6,144 episodes. No B phase is run.

Repeats that separate nondeterminism from the seed factors:

- **Evaluation repeat:** the four diagonal adapters of every source-by-arm cell are evaluated a
  second time (24 adapters, 1,536 episodes).
- **Training repeat:** one cell per source-by-arm is trained twice with identical seeds
  (6 extra adapters, 384 episodes), because BF16 training is known not to be bit-reproducible.

Total: 102 adapters and 8,064 episodes. Reduced option if the budget below is exceeded: drop
arm (b), giving 68 adapters and 5,376 episodes.

A cheap continuous co-outcome is recorded for every adapter: mean action NLL on the existing
diagnostic tasks. It needs no rollouts and is expected to be more sensitive than success.

### Control of randomness

`scripts/train_lora.py` takes one `--seed` that drives the example schedule, LoRA
initialisation and dropout together, and its hash is frozen by three studies. The diagnostic
therefore uses a separate copy of the trainer with two arguments and leaves the frozen file
untouched.

- **Sampling seed** determines only the schedule returned by `fixed_schedule`. In arm (c) it
  changes order, not membership.
- **Initialisation seed** determines LoRA initialisation and the dropout stream. Dropout stays
  at the studied value of 0.05, so this factor is "initialisation plus dropout" and is reported
  under that name.
- The same sampling seed gives both sources the same schedule indices where corpus sizes allow;
  realised schedules are hashed and stored.

Recorded per adapter: unique examples seen, repetitions, examples per task type and per target
family, input and target tokens, schedule hash, both seeds, adapter hash.

### Analysis

For each sampling arm, on the paired API minus generic post-A difference and on each source
separately: variance components for initialisation, sampling, their interaction, task, and the
evaluation and training repeat terms, each with an interval. Components are reported even when
the interval is wide; three degrees of freedom per seed factor is stated as the limit.

### Reading the result

The gate is diagnostic, not directional. No outcome is required for the work to continue.

| Observation | Consequence for the main study |
|---|---|
| Sampling dominates under arm (a) and shrinks under (b) or (c) | Adopt the arm that shrinks it; re-run Part 1 with the new components |
| Sampling dominates and no arm shrinks it | Budget many sampling seeds, or average several adapters per condition |
| Initialisation plus dropout dominates | Budget initialisation seeds; sampling control is not the lever |
| Evaluation repeat variance dominates | Spend on repeated evaluation or more environment seeds per task, not training seeds |
| Training repeat variance is comparable to the seed components | Seed labels are not a usable blocking factor; treat runs as exchangeable |
| Nothing dominates, or intervals are too wide to say | Size the main study on the pessimistic component values |

## Part 3. Predictor feasibility

Predictors are fixed before any new data are examined (*proposed* list, exact formulas to be
written into this section before freezing): base-model action surprisal in context, path
length relative to the shortest observed success on the task, invalid or no-effect action
rate, number of recovery segments.

Computed on existing successful collection episodes only, with no training:

- **Within-task variation:** share of each predictor's variance that lies within task group,
  across sources and rollout seeds.
- **Reliability:** agreement of the episode-level predictor across rollout seeds of the same
  task and source.

Feasibility rule (*proposed*): a predictor is feasible if its within-task share is at least
0.20 and its reliability at least 0.60. Passing establishes feasibility only. Predictive value
beyond collection success, task coverage and target-task exposure, validation grouped by
collection condition, and the selection intervention are separate later tests.

If no predictor is feasible in ALFWorld, a second environment precedes the predictor study.

## Budget limits

- Teacher API calls: 0. Official test sets: unopened.
- Adapters: at most 102. Evaluation episodes: at most 8,064.
- GPU time: *to be set by the team from measured throughput before freezing*.
- If a limit would be exceeded, the reduced option above is used; the design is not otherwise
  trimmed after results are seen.

## Outputs

Per-adapter records, per-episode results, schedule hashes, the variance-component table with
intervals, the Part 1 simulation tables, and the predictor feasibility table. One short report
states which row of the Part 2 table applies and what main-study design follows.
