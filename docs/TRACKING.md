# W&B tracking

Experiment records are stored in the private [learning-harness-lab W&B project](https://forge.coreweave.com/wandb/yanyaoyuan-jawaharlal-nehru-university/learning-harness-lab). Runs are grouped by study, condition, and training initialization. Access depends on the project's W&B permissions.

## Install and start

```bash
python3 -m venv .venv-wandb
.venv-wandb/bin/python -m pip install -r configs/wandb-requirements.txt
PYTHONPATH=src .venv-wandb/bin/python scripts/track_wandb.py --watch
```

Authentication uses the machine's W&B login configuration. Local index files under output/wandb/ store run IDs, uploaded events, and training-log offsets. Reuse the same index when restarting an observer so it resumes the existing runs. Run one observer per index.

The transfer study uses a separate index:

```bash
PYTHONPATH=src .venv-wandb/bin/python scripts/track_wandb.py --watch \
  --studies runs/transfer-factorial-v1/cool runs/transfer-factorial-v1/heat \
  --out output/wandb-transfer
```

## Recorded fields

- Training loss, learning rate, gradient norm, and optimizer step.
- Checkpoint evaluation results, per-task outcomes, and task-family summaries.
- Public action/feedback traces and generated figures where enabled by the study.
- Completion state, artifact checksums, and synchronization progress.

A run corresponds to a model initialization; A- and B-stage training use separate step axes. Reused A-stage histories may appear in both transfer-family groups but count once in the compute budget. Backfilled records carry upload timestamps, which differ from the original execution times.

Only completed evaluation points are plotted. Intermediate progress tables can contain partial counts. The observer reads experiment files independently of training, so its exit does not stop a training process. W&B uploads exclude credentials and model weights.
