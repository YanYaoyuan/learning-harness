from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def game_path(root, row):
    root = Path(root).resolve()
    path = (root / row["game_relpath"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Game path escapes data root")
    if file_sha(path) != row["game_sha256"]:
        raise ValueError(f"Game checksum changed: {row['task_id']}")
    return path

