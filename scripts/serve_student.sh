#!/usr/bin/env bash
# Run this in a GPU-enabled vLLM environment, not in the CPU simulator venv.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${TASK_ROOT}"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is not visible in this session"'
exec python -m vllm.entrypoints.openai.api_server \
  --model "${LH_STUDENT_MODEL:-Qwen/Qwen3-4B-Instruct-2507}" \
  --revision "${LH_STUDENT_REVISION:-cdbee75f17c01a7cc42f958dc650907174af0554}" \
  --host 127.0.0.1 --port "${LH_STUDENT_PORT:-8000}" \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 8192 --gpu-memory-utilization 0.85
