# Variance diagnostic, Stage 0: executable specification

Date: 2026-11-11. Status: **ready to freeze; not yet frozen**. Stage 0 freezes independently
of the rest of `VARIANCE_DIAGNOSTIC_PROTOCOL_V1.md`.

The runner and the analysis script were written without access to a GPU and have not been
executed end to end. The estimator has unit tests in `tests/test_stage0_variance.py`.

Stage 0 is exploratory. It trains nothing, makes no teacher calls and opens no official test
set. It writes only under `runs/variance-diagnostic-v1/stage0/`.

## 1. Evaluation configuration and the historical replicate

### What is evaluated

The post-A state of the six stored full-experience A adapters, on both task families:
12 cells = {cool, heat} x {API, generic} x B seeds {1101, 1202, 1303}.

The historical post-A evaluation served the step-0 composition (frozen A plus the
zero-initialised B adapter, no merging), not the A adapter alone. Stage 0 serves the same
artifact: `runs/transfer-factorial-v1/<family>/trajectories/full-<source>-seed<seed>/step-0000`.
The runner recomputes the composition revision, which verifies both adapter hashes, and
refuses to run if it differs from the revision recorded in the historical `run.json`. This
needs the step-0 B adapter weights as well as the six A adapters.

### Configuration

Each new run copies the historical `config.json` of its cell (temperature 0, rollout seed 11,
40 steps, 128 new tokens per call, history of 12 turns, batch size 8, no support arm) and
changes one field, the manifest path. It uses the same server script, batch runner, model
alias and environment backend, launched through the same helper as the historical pipeline.
One GPU is used for all of Stage 0.

### What differs between replicates

At temperature 0 an unmodified rerun may be bit-identical, in which case it carries no
information. Each new replicate therefore evaluates the same 32 task records per family in a
different fixed order (replicates 1 to 3; permutation seeds are in the runner). Task order
decides which episodes share a batch, so this exposes sensitivity to batch co-membership under
BF16. The historical evaluation is the same procedure with the original order.

This measures numerical and batching sensitivity only. It does not vary the environment seed
or the sampling stream, and it says nothing about evaluation variance from those sources.

A separate determinism check reruns two cells (heat, seed 1101, both sources) with the
unmodified historical configuration: 64 episodes, run last, descriptive only, never in the
estimator. It reports whether outcomes and action traces match the historical run exactly.

Total: 36 runs and 1,152 episodes for the estimator, plus 2 runs and 64 episodes for the
determinism check.

### Compatibility checks for the historical replicate

The historical evaluation enters the estimator as a fourth replicate only if every check
holds. None of them looks at task outcomes.

1. The served composition revision equals the historical one for all 12 cells.
2. The evaluation configuration is identical apart from the manifest path.
3. The new manifests contain exactly the historical task records.
4. `batch_protocol`, `environment_backend` and `planned_episodes` in `run.json` are equal.
5. The historical runs are complete with zero errors.
6. Every file listed in the historical study's `inputs.sha256.json`, `curricula.sha256.json`
   and `execution-code.sha256.json` still has its recorded hash (checked by the runner).
7. The team attests in `freeze.json` that the model virtual environment, GPU driver and GPU
   model are unchanged since the historical run. The repository holds no historical record
   of these, so this check cannot be automated.

Checks 1 to 5 are implemented in `stage0_variance_share.py`, which stops if the frozen
choice includes the historical replicate and any of them fails. If check 7 cannot be
attested, the historical replicate is left out, the primary estimate uses the new replicates
only, and the four-replicate estimate is reported as a labelled sensitivity result.

### Known hash mismatch in the historical study

`scripts/transfer_factorial_report.py` no longer matches `execution-code.sha256.json`
(recorded `50e95362...`, current `4ff9cdea...`). The current hash equals the
`report_script_sha256` recorded in `runs/transfer-factorial-v1/postrun-figure-audit.json`,
which describes the change as "Share y labels between combined-figure panels to avoid
overlap" and states that numerical results are unchanged. That statement is the audit
record's own; the earlier version of the file is not in the repository, so the difference
has not been re-inspected here. The frozen checksum is left as it is.

The runner handles this as a documented exception, not by narrowing the check:

- Every file in the three historical hash lists is still verified.
- An exception is accepted only for a report, plot or tracking script, only when given with
  its provenance at freeze time, and only while the file keeps the hash recorded in the
  exception. A changed file on the evaluation path always stops the runner.
- The report script is not imported or executed by Stage 0.

## Freezing

Order: unit tests, smoke check, freeze, run.

1. **Smoke check** (`--smoke --gpu N`): one evaluation of one cell (heat, API, seed 1101)
   with the unmodified historical configuration, written to `smoke/`. It is allowed only
   before the freeze, never enters the estimator, and its time is charged against the cap.
   Its outcomes are the same kind of information as the determinism check and are not used
   to change the design.
2. **Freeze** (`--freeze --gpu N ...`): writes `freeze.json` and refuses to overwrite one.
   It records the SHA-256 of this file, the runner and the analysis script; the GPU; the
   cap; whether the historical evaluation is a replicate, with the attestation of check 7;
   the replicate design; and each documented hash exception with both hashes and its reason.
3. **Run** (no mode flag): takes the GPU, the cap and every other choice from `freeze.json`.
   It stops before any evaluation if a frozen file changed, if a run-time argument
   contradicts the freeze, or if the historical hash lists show an undocumented or changed
   mismatch. The analysis script validates the same freeze and takes the historical-replicate
   choice from it; it has no flag for that choice.

## 2. Variance-share estimator

Let `y` be success (0 or 1) for source `k`, training seed `s`, family `f`, task `t`,
replicate `r`. The paired difference of one evaluation is

    d[s, r] = mean over f of ( mean over the 32 tasks of f of  y[API] - y[generic] )

Tasks are weighted equally within a family and the two families equally, matching the
equal-weight aggregation of the frozen primary analyses. With `S = 3` seeds and `R`
replicates (3, or 4 with the historical one):

    MSW = mean over s of the sample variance of d[s, .]          df = S (R - 1)
    MSB = R x sample variance over s of the replicate mean        df = S - 1

    replicate variance       = MSW
    seed variance            = max(0, (MSB - MSW) / R)
    single-evaluation total  = replicate variance + seed variance
    share                    = replicate variance / single-evaluation total

The share is the fraction of the between-seed variance of a single evaluation's paired
difference that evaluation repeats alone would produce. It conditions on the fixed query
tasks.

Boundary handling:

- **Near-zero denominator.** If the total is below 1e-12, every `d[s, r]` is the same; the
  share is undefined, recorded as null, and the default branch applies.
- **Negative seed variance estimate.** Truncated at zero, which makes the share 1; the
  truncation is flagged in the output.
- **Identical replicates within every seed.** The share is 0 with a degenerate interval.

Reported with the share, and kept separate from the rule: MSW, MSB and their degrees of
freedom, both standard deviations in percentage points, and a 90% interval from the one-way
random-effects F distribution. The interval is normal-theory and approximate for means of
binary outcomes; with 2 between-seed degrees of freedom it will be wide.

Secondary, never used for the rule: the same estimator per family, with new replicates only,
and with the historical replicate regardless of the checks; the share of task outcomes that
flip between replicates; the determinism check.

### Decision

| Condition | Branch |
|---|---|
| Fewer than two complete new replicates | Inconclusive. Stage 1 in full, with evaluation repeats added |
| Share undefined | Stage 1 in full |
| Share at least 0.50 | Stage 1 at reduced size (2 x 2 seeds, 10 adapters); plan repeated evaluation for the main study |
| Share below 0.50 | Stage 1 in full |

The threshold is a spending rule. Crossing it is not evidence about mechanism, and the
component estimates and interval are reported whichever branch is taken.

## 3. GPU-time cap and stopping rule

Measured throughput: the 12 historical evaluations of these cells took 155 to 208 seconds
each for 32 episodes (mean 190 s, total 2,274 s) on one RTX 5090 at batch size 8, excluding
model loading. The 38 planned runs therefore need about 2.0 GPU-hours plus loading.

- **Cap:** 4.0 GPU-hours of cumulative wall time on one GPU, fixed in `freeze.json`. Every
  attempt is charged from server start to server stop: the smoke check, completed runs, and
  failed, interrupted and timed-out runs alike.
- **Order:** replicate 1 for all 12 cells, then replicate 2, then replicate 3, then the
  determinism check. A stop therefore leaves whole replicates.
- **Termination:** a run starts only if at least 300 seconds of the cap remain. Once the
  server is up, the rollout is given exactly the remaining time as its timeout, so the cap
  can be exceeded only by server shutdown (up to about 20 seconds). A run cut off this way
  is recorded as stopped at the cap and is not completed later.
- **Interruption:** Ctrl-C and SIGTERM stop the server, charge the elapsed time and mark the
  ledger interrupted. An attempt that ends without accounting (kill -9, power loss) is
  charged on the next start, from its recorded start time to the newest file written in its
  directory.
- **Estimator input:** only complete replicates. The cap is not raised after results are seen.
- **Errors:** a server or rollout error stops the runner, is charged, and is not counted as
  task failure. A resumed run verifies and skips completed evaluations, and the batch runner
  continues an interrupted one under its fingerprint check. No evaluation is repeated
  because of its outcome.
- **Concurrency:** the runner refuses to start if a model server is already listening for the
  chosen GPU. It must not share a GPU with `source-comparison-v1` while that study runs.

## Commands

```bash
PYTHONPATH=src .venv-model/bin/python -m pytest -q tests/test_stage0_variance.py
PYTHONPATH=src .venv-model/bin/python scripts/run_stage0_repeat_eval.py --smoke --gpu 0
PYTHONPATH=src .venv-model/bin/python scripts/run_stage0_repeat_eval.py --freeze --gpu 0 \
    --cap-gpu-hours 4 \
    --hash-exception "scripts/transfer_factorial_report.py=<provenance and impact>"
PYTHONPATH=src .venv-model/bin/python scripts/run_stage0_repeat_eval.py
PYTHONPATH=src python3 scripts/stage0_variance_share.py
```

Add `--include-historical --attestation "<text>"` to the freeze command if check 7 is attested.

Outputs: `runs/variance-diagnostic-v1/stage0/rep<r>/...` (one directory per evaluation, same
files as the historical runs), `determinism/...`, `smoke/...`, `manifests/`, `freeze.json`, `execution.json` (the time ledger), and
`result.json` with the components, interval, secondary results and the branch taken.
