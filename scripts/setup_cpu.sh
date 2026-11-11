#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${TASK_ROOT}"
export UV_CACHE_DIR="${TASK_ROOT}/.cache/uv"
if [[ ! -x .venv/bin/python ]]; then
  uv venv .venv --python "${LH_ENV_PYTHON:-/usr/bin/python3}"
fi
uv pip install --python .venv/bin/python -r requirements/alfworld.lock.txt
PYTHONPATH=src .venv/bin/python -m learning_harness doctor
