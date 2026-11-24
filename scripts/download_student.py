#!/usr/bin/env python3
"""Download pinned model weights with range requests and SHA-256 verification."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request

REPO = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
TEXT_FILES = {"config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
              "merges.txt", "vocab.json", "model.safetensors.index.json"}


def sha256(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def fetch(row, output):
    path = output / row["path"]
    expected = row["size"]
    expected_hash = row.get("lfs", {}).get("oid")
    if path.exists() and path.stat().st_size == expected:
        if not expected_hash or sha256(path) == expected_hash:
            return dict(path=row["path"], bytes=expected, sha256=sha256(path))
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists() and partial.stat().st_size > expected:
        raise ValueError("Oversized partial file")
    failures = 0
    while not partial.exists() or partial.stat().st_size < expected:
        start = partial.stat().st_size if partial.exists() else 0
        end = min(expected - 1, start + 128 * 1024 * 1024 - 1)
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{row['path']}?download=true"
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                if response.status == 206:
                    received_range = response.headers.get("Content-Range", "")
                    if not received_range.startswith(f"bytes {start}-"):
                        raise ValueError("Unexpected HTTP content range")
                elif response.status != 200 or start != 0:
                    raise ValueError("Server ignored resume range")
                with partial.open("ab") as handle:
                    while chunk := response.read(1024 * 1024):
                        if handle.tell() + len(chunk) > expected:
                            raise ValueError("Model response exceeds expected size")
                        handle.write(chunk)
            failures = 0
            print(f"{row['path']}: {partial.stat().st_size / 1024**2:.1f}/{expected / 1024**2:.1f} MiB", flush=True)
        except (OSError, TimeoutError):
            failures += 1
            if failures >= 3:
                raise
            time.sleep(failures)
    actual_hash = sha256(partial)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(f"Model SHA-256 mismatch: {row['path']}")
    partial.replace(path)
    return dict(path=row["path"], bytes=expected, sha256=actual_hash)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/models/Qwen3-4B-Instruct-2507"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}", timeout=30) as response:
        rows = json.load(response)
    selected = [row for row in rows if row["path"] in TEXT_FILES or row["path"].endswith(".safetensors")]
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(lambda row: fetch(row, args.out), selected))
    (args.out / "download-manifest.json").write_text(json.dumps(dict(
        repo=REPO, revision=REVISION, files=records), indent=2) + "\n")
    print(f"Verified model: {args.out}", flush=True)


if __name__ == "__main__":
    main()
