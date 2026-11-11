# Stage 0 local validation and scheduling

Date: 2026-11-11. Base: `34b72c2655bd13527e4d0e49872b1481b1ad24c5` on `power-projection`.

The candidate includes the local numerical fix before any freeze: subtract a row-wise
reference before computing repeat variance, so identical floating-point repeats have zero
spread while genuine small differences remain measurable. A seventh estimator test covers
the latter. Historical hash manifests must also exist; missing manifests are errors.

Thirty-one CPU tests cover the estimator, the smoke/full execution plans, completion/failure/
timeout/interruption accounting, freeze and budget changes, documented hash exceptions,
missing manifests, and the source-study/resource scheduling guards. No real model is launched
by these tests. GPU smoke and end-to-end acceptance remain pending.

Local model requests explicitly bypass inherited HTTP proxies through `NO_PROXY` and
`no_proxy`, preserving existing bypasses. This addresses observed HTTP 502 failures when
loopback inference requests were sent through the inherited proxy.

`scripts/queue_stage0_after_source_comparison.py` waits for all of:

- `source-comparison-v1` reports execution completed;
- its report contains all 4,284 planned task evaluations and passes its token/budget audit;
- local finalization completed;
- both model-server ports are closed, and the selected GPU has sufficient free memory and
  low utilization on two consecutive checks.

The queue then runs smoke, freeze, Stage 0, and analysis in that order. A smoke failure stops
before freezing. Source failure also stops the queue. Code hashes are pinned while waiting.
The queue is single-instance and never launches Stage 1. Waiting is limited to 24 hours.

Execution choices: GPU 0; four GPU-hours for Stage 0 including smoke; historical replicate
excluded from the primary estimate because historical environment identity cannot be
attested. The known transfer-report hash exception cites the existing post-run figure audit;
the historical frozen checksum remains unchanged.

```bash
PYTHONPATH=src .venv-model/bin/python -m pytest -q \
    tests/test_stage0_variance.py tests/test_stage0_runner.py tests/test_stage0_queue.py
PYTHONPATH=src .venv-model/bin/python scripts/queue_stage0_after_source_comparison.py --gpu 0
```

Queue status and commands are recorded under `runs/variance-diagnostic-v1/stage0/queue/`.
The Stage 0 specification and code are frozen only after the real smoke check succeeds.
