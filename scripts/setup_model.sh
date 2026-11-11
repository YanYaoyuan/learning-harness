#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${TASK_ROOT}"
export UV_CACHE_DIR="${TASK_ROOT}/.cache/uv"
TASK_PYTHON="${LH_MODEL_PYTHON:-/usr/bin/python3}"
"${TASK_PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "Run in a GPU-enabled session with working torch"; print(torch.__version__)'
if [[ ! -x .venv-model/bin/python ]]; then
  uv venv .venv-model --python "${TASK_PYTHON}" --system-site-packages
fi
uv pip install --python .venv-model/bin/python --no-deps -r requirements/model-overlay.txt
.venv-model/bin/python -c 'import torch,transformers,peft; from transformers import Qwen3ForCausalLM; print(torch.__version__,transformers.__version__,peft.__version__)'
