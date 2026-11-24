# Reproducibility

Run commands from the repository root. The simulator uses .venv; inference and training use .venv-model. The supplied setup scripts require uv. GPU setup also requires a host PyTorch build compatible with the installed driver and GPU.

## CPU tests

```bash
python3 -m venv .venv-dev
.venv-dev/bin/python -m pip install -e . numpy matplotlib pytest
PYTHONPATH=src .venv-dev/bin/python -m pytest -q
```

These tests cover data handling, protocol checks, analysis, and tracking. Environment smoke checks and model runs require their respective assets.

## ALFWorld

```bash
bash scripts/setup_cpu.sh
python3 scripts/download_alfworld.py
bash scripts/lh prepare
bash scripts/lh smoke
```

The environment uses text observations and discrete text commands. Public observations are available to the policy; hidden facts and expert plans are excluded. Task manifests distinguish training, development, and the official seen/unseen splits.

## Model service

The student is Qwen/Qwen3-4B-Instruct-2507, revision cdbee75f17c01a7cc42f958dc650907174af0554.

```bash
bash scripts/setup_model.sh
python3 scripts/download_student.py
.venv-model/bin/python scripts/hf_server.py \
  --model data/models/Qwen3-4B-Instruct-2507 \
  --served-model-name Qwen/Qwen3-4B-Instruct-2507
```

In a second terminal:

```bash
export LH_STUDENT_BASE_URL=http://127.0.0.1:8000/v1
export LH_STUDENT_MODEL=Qwen/Qwen3-4B-Instruct-2507
export LH_STUDENT_REVISION=cdbee75f17c01a7cc42f958dc650907174af0554
bash scripts/lh rollout --config configs/smoke_model.json --out runs/model-smoke
bash scripts/lh summarize runs/model-smoke
```

The model smoke configuration uses four development tasks and a 30-step limit. Other experiments specify their own budgets. The batch server in scripts/hf_batch_server.py is used by larger historical runs; batch layout and BF16 numerical behavior are part of their recorded configuration.

## Teacher interface

The teacher accepts an OpenAI-compatible Chat Completions endpoint. Configure the base URL and fixed model name in the environment; set LH_TEACHER_API_KEY in the terminal or ignored .env.local file.

```bash
export LH_TEACHER_BASE_URL=https://your-provider.example/v1
export LH_TEACHER_MODEL=your-fixed-model-name
```

Copy configs/teacher.example.json to configs/teacher.local.json and enter the provider's input/output prices and a spending cap. The collector requires known prices and records reservations for failed requests as well as successful ones.

```bash
bash scripts/lh build-teacher \
  --manifest data/manifests/alfworld/pilot.jsonl \
  --config configs/teacher.local.json \
  --out data/teacher/pilot.jsonl
```

The API experiments used cached strategy cards and separate budget ledgers. Their teacher sees the public initial observation. CPU tests, public-reference runs, and inspection of archived results do not require a teacher key.

## Collection and SFT

These commands show the generic collection and training interface. Study-specific splits, budgets, and sampling schedules are listed in the [experiment index](EXPERIMENTS.md).

```bash
bash scripts/lh rollout --config configs/collect_train.json --out runs/train-data
bash scripts/lh export-sft runs/train-data --arm none --out runs/sft/none.jsonl
CUDA_VISIBLE_DEVICES=0 .venv-model/bin/python scripts/train_lora.py \
  --data runs/sft/none.jsonl --seed 101 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --out checkpoints/rft-none-seed101
```

Training produces a LoRA adapter. Load it with the base model for evaluation:

```bash
.venv-model/bin/python scripts/hf_server.py \
  --adapter checkpoints/rft-none-seed101 \
  --served-model-name rft-none-seed101
```

The server reports the base revision and adapter checksum. Use the trained artifact's immutable identifier in evaluation metadata. Held-out evaluation requires the study's frozen input record; the generic rollout command alone does not establish the paper's evaluation protocol.

## Restoring an archived run

Weights, virtual environments, simulator downloads, and local credentials are excluded from the snapshot. Restore the assets required by the selected study before resuming its pipeline. Completion markers and input hashes distinguish reusable outputs from interrupted work.

```bash
python3 scripts/unpack_snapshot_traces.py --study baseline-benchmark-v1
```

The unpacker verifies compressed and uncompressed checksums and rejects conflicting existing files. See [Snapshot contents](GITHUB_SNAPSHOT.zh-CN.md) for the archive layout, and [W&B tracking](TRACKING.md) for cloud records.
