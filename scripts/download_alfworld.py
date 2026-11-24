#!/usr/bin/env python3
"""Download official ALFWorld text-environment assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ASSETS = [
    ("0.2.2", "json_2.1.1_json.zip", 72018818),
    ("0.2.2", "json_2.1.1_pddl.zip", 34881784),
    ("0.4.2", "json_2.1.3_tw-pddl.zip", 36507267),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/alfworld"))
    parser.add_argument("--cache", type=Path, default=Path("data/downloads"))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    records = []
    for tag, name, expected_bytes in ASSETS:
        path = args.cache / name
        url = f"https://github.com/alfworld/alfworld/releases/download/{tag}/{name}"
        if not path.exists() or path.stat().st_size != expected_bytes:
            partial = path.with_suffix(".part")
            print(f"Downloading {name} ({expected_bytes / 1024**2:.1f} MiB)", flush=True)
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            if partial.stat().st_size != expected_bytes:
                raise ValueError(f"Size mismatch: {name}")
            partial.replace(path)
        extracted = 0
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir() or Path(member.filename).name not in {
                    "traj_data.json", "game.tw-pddl", "initial_state.pddl"
                }:
                    continue
                target = (args.root / member.filename).resolve()
                if not target.is_relative_to(args.root.resolve()):
                    raise ValueError("Archive path escapes data root")
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.write_bytes(archive.read(member))
                extracted += 1
        records.append(dict(url=url, bytes=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), files=extracted))
        print(f"Ready: {name}; {extracted} task files", flush=True)
    (args.root / "sources.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"Data root: {args.root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
