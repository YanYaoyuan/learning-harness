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
- First result (team run, 2,000 simulated data sets, 4,000 bootstrap draws, Gaussian planning
  model fitted to heating / full experience / post-A, 3 seeds x 32 tasks, nominal 5%):
  null rejection 19.35% for the percentile bootstrap and 5.70% for the Satterthwaite t-test.
  The Monte Carlo standard errors are about 0.9 and 0.5 points, so the bootstrap excess is
  far outside simulation noise and the t-test figure is compatible with nominal.
- What this does and does not show. It is one seed-dominated scenario under normal
  components. It is a reason to check calibration, not a validation of either procedure for
  binary outcomes. It also bears on the percentile-bootstrap intervals of the frozen
  reports, which resample three seeds: their endpoints and primary analyses stay as
  preregistered, and any t-based interval added to them is labelled supplementary.

Calibration under discrete outcomes is required before the main-study analysis is frozen:

- A generator for binary success with task, seed and seed-by-task effects on the logit scale,
  fitted to the Part 2 data.
- Scenarios fixed in advance: low base rates as seen after A (roughly 5% to 20%), mid-range,
  and near ceiling; seed variance at the estimate and at four times the estimate; each of the
  three variance-dominance patterns seen so far (seed, task, residual).
- Every rejection rate is reported with its binomial standard error, and final calibration
  runs use at least 5,000 simulated data sets per scenario.

Decision rule (*proposed*): an analysis is admissible for the main study only if its simulated
Type I error is at most 0.065 at nominal 0.05 in every scenario above. The main-study design
is the cheapest one giving simulated power of at least 0.80 for the admissible analysis at the
practically meaningful effect. That effect is chosen by the team before Part 2 results are
seen (*proposed*: 5pp on post-A success).

## Part 2. Variance attribution

### Staged design

Paired corpora throughout: the full-experience API and generic corpora of
`transfer-factorial-v1` (721 and 732 eligible examples), where the large post-A seed variance
was observed. Evaluation is always post-A, common JSON execution, on the 32 cooling and 32
heating query tasks (64 episodes per adapter per evaluation). No B phase is run. Mean action
NLL on the existing diagnostic tasks is recorded for every adapter as a continuous co-outcome
that needs no rollouts.

Each stage has a fixed cost and a rule, written here before any stage is run, that decides
whether the next stage happens. The rules use point estimates and are known to be noisy; they
exist to bound spending, not to establish which factor matters.

**Stage 0. Repeat evaluation of existing checkpoints. No training.**

Executable specification: `VARIANCE_DIAGNOSTIC_STAGE0_SPEC.md`. Where the two differ, the
specification governs Stage 0. It adds a 64-episode determinism check and defines replicates
as task-order permutations.

- The six stored full-experience A adapters (API and generic, three seeds), each evaluated
  three more times: 1,152 episodes. The original recorded evaluation is a fourth replicate.
- Adapter hashes are checked against the recorded values first. If the weights are no longer
  available, Stage 0 is skipped, this is reported, and the evaluation repeats are added to
  Stage 1 instead.
- Quantity: the share of the between-seed variance of the mean paired difference that
  evaluation repeats alone would produce (repeat variance of a seed mean, divided by the
  observed variance of seed means).
- Rule (*proposed*): share of 0.50 or more, evaluation dominates: run Stage 1 at reduced size
  (2 x 2 seeds, 10 adapters) only to confirm, and plan repeated evaluation for the main study.
  Otherwise run Stage 1 in full.

**Stage 1. Crossed initialisation by sampling seeds under the current sampler.**

- Sampling arm (a) only: 256 presentations, as in the completed studies.
- 4 initialisation seeds x 4 sampling seeds x 2 sources = 32 adapters, plus one cell per
  source trained twice with identical seeds = 34 adapters, 2,176 episodes. The training repeat
  is needed because BF16 training is known not to be bit-reproducible.
- Quantity: the sampling seed's share of the summed initialisation, sampling and interaction
  components, for the paired difference, on success and on NLL.
- Rule (*proposed*): share of 0.30 or more on either outcome, run Stage 2. Below that, stop:
  sampling control is not the lever, and effort goes to initialisation or evaluation
  variability according to the table below.

**Stage 2. Controlled sampling. Conditional on Stage 1.**

- Arm (c), complete pass: budget equal to the larger corpus, same for both sources. Same
  4 x 4 seed grid: 34 adapters, 2,176 episodes.
- Arm (b), 256 presentations with fixed task-type exposure, is run only if the budget below
  still allows it after arm (c): a further 34 adapters and 2,176 episodes.

| Stage | Adapters | Episodes | Cumulative episodes |
|---|---:|---:|---:|
| 0 | 0 | 1,152 | 1,152 |
| 1 | 34 | 2,176 | 3,328 |
| 2, arm (c) | 34 | 2,176 | 5,504 |
| 2, arm (b), optional | 34 | 2,176 | 7,680 |

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

For each sampling arm that was run, on the paired API minus generic post-A difference and on
each source separately: variance components for initialisation plus dropout, sampling, their
interaction, task, and the evaluation-repeat and training-repeat terms, each with an interval.
Components are reported even when the interval is wide; three degrees of freedom per seed
factor is stated as the limit. Stage rules are applied exactly as written above, and the
report says which branch was taken.

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

Computed on existing successful collection episodes only, with no training.

The measurement target is the **corpus-level predictor**: a fixed aggregate (*proposed*: the
mean over action labels) of the episode-level quantity across a training corpus of the size
planned for the predictor study. The corpus is the unit that receives training, so reliability
is defined there. Differences between rollout seeds of the same task are genuine trajectory
variation, not measurement error, and are not penalised.

Three quantities, kept separate:

- **Scoring repeatability.** The same episode scored twice, for predictors that depend on a
  model forward pass (BF16 and batching can change values). This is the only term treated as
  measurement error. Rule (*proposed*): repeat correlation of at least 0.95.
- **Within-task variation.** Share of the episode-level predictor's variance that lies within
  task group, across sources and rollout seeds. This is the signal a selection intervention
  needs once task coverage is held fixed. Rule (*proposed*): at least 0.20.
- **Corpus-level reliability.** Split the episodes of a collection condition into random
  halves at the planned corpus size and compare the corpus-level predictor between halves,
  relative to its spread across the corpora the predictor study would build. Rule
  (*proposed*): at least 0.60. A low value means corpora of that size cannot be told apart on
  this predictor, whatever the episode-level variation.

A predictor is feasible if it passes all three. Passing establishes feasibility only.
Predictive value beyond collection success, task coverage and target-task exposure, validation
grouped by collection condition, and the selection intervention are separate later tests.

If no predictor is feasible in ALFWorld, a second environment precedes the predictor study.

## Budget limits

- Teacher API calls: 0. Official test sets: unopened.
- Adapters: at most 102. Evaluation episodes: at most 7,680. Spending follows the stage table
  and stops where a stage rule says to stop.
- GPU time: *to be set by the team from measured throughput before freezing*.
- A stage is not enlarged, and a skipped stage is not reinstated, after its results are seen.

## Outputs

Per-adapter records, per-episode results, schedule hashes, the variance-component table with
intervals, the Part 1 simulation tables with Monte Carlo standard errors, and the predictor
feasibility table. One short report states which stage rules fired, which row of the Part 2
table applies, and what main-study design follows.
