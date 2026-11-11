"""Load literal credentials without sourcing shell code or logging values."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shlex


def read_literal_env(path):
    values = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        match = re.fullmatch(r'\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)', line)
        if not match:
            raise ValueError(f'Invalid environment assignment at line {number}')
        key, literal = match.groups()
        parts = shlex.split(literal, comments=True, posix=True)
        if len(parts) > 1:
            raise ValueError(f'Quote whitespace in environment assignment at line {number}')
        if key in values:
            raise ValueError(f'Duplicate environment assignment at line {number}')
        values[key] = parts[0] if parts else ''
    return values


def load_teacher_env(path='.env.local'):
    """An exported environment takes precedence; only teacher variables are loaded."""
    allowed = {'LH_TEACHER_BASE_URL', 'LH_TEACHER_MODEL', 'LH_TEACHER_API_KEY'}
    values = read_literal_env(path)
    loaded = []
    for key in allowed:
        if values.get(key) and not os.environ.get(key):
            os.environ[key] = values[key]
            loaded.append(key)
    return sorted(loaded)
