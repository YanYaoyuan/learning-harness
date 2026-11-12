#!/usr/bin/env python3
"""Copy and verify immutable large files, then atomically replace originals with links."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

WORK = Path(__file__).absolute().parents[1]
ROOTS = ("data/models", "data/downloads", "runs/transfer-factorial-v1", "runs/api-harness-v1",
         "runs/source-comparison-v1", "runs/learning-speed-v1", "runs/dual5090-v1",
         "runs/dual5090-v2", "runs/dual5090-v2-filtered")
RECEIPT = WORK / "output/research-operations/20261112/storage-relocation.json"
THRESHOLD = 16 * 1024 ** 2


def signature(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 ** 2), b""):
            result.update(chunk)
    return result.hexdigest()


def save(value):
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    temporary = RECEIPT.with_suffix(".json.part")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(RECEIPT)


def relocate(source, target):
    before = signature(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".relocation-part")
    if target.is_symlink():
        raise ValueError("Destination must be a regular file: " + str(target))
    if target.exists():
        digest = sha(source)
        if target.stat().st_size != before[2] or sha(target) != digest:
            raise ValueError("Destination collision; no overwrite: " + str(target))
    else:
        if temporary.exists():
            raise ValueError("Partial copy retained for inspection: " + str(temporary))
        digest = hashlib.sha256()
        with source.open("rb") as origin, temporary.open("xb") as destination:
            for chunk in iter(lambda: origin.read(8 * 1024 ** 2), b""):
                digest.update(chunk)
                destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        digest = digest.hexdigest()
        if temporary.stat().st_size != before[2] or sha(temporary) != digest:
            raise ValueError("Copied file failed SHA-256 verification")
        temporary.rename(target)
    if signature(source) != before:
        raise ValueError("Source changed while copying; original retained: " + str(source))
    link = source.with_name(source.name + ".relocation-link")
    if link.exists() or link.is_symlink():
        raise ValueError("Staged link exists; inspect before continuing")
    link.symlink_to(target)
    # Open readers retain their old inode. New readers see either the original or verified copy.
    link.replace(source)
    if not source.is_symlink() or source.resolve() != target.resolve():
        raise RuntimeError("Atomic link replacement failed")
    return dict(source=str(source.relative_to(WORK)), destination=str(target), bytes=before[2],
                sha256=digest, verified=True, original_path_preserved=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    destination = Path(args.destination).absolute() / "learning-harness-lab"
    candidates = sorted((p for root in ROOTS for p in (WORK / root).rglob("*")
        if p.is_file() and not p.is_symlink() and p.stat().st_size >= THRESHOLD),
        key=lambda p: p.stat().st_size, reverse=True)
    total = sum(p.stat().st_size for p in candidates)
    print(json.dumps(dict(files=len(candidates), bytes=total, destination=str(destination))), flush=True)
    if not args.execute:
        return
    destination.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination).free < total + 1024 ** 3:
        raise ValueError("Destination lacks planned capacity plus 1 GiB reserve")
    prior = json.loads(RECEIPT.read_text()) if RECEIPT.exists() else {}
    if prior and prior.get("destination") != str(destination):
        raise ValueError("Existing receipt points to another destination")
    receipt = dict(status="running", destination=str(destination), planned_files=len(candidates),
                   planned_bytes=total, started_unix=time.time(), files=prior.get("files", []),
                   active_stage1_excluded=True)
    save(receipt)
    try:
        for source in candidates:
            record = relocate(source, destination / source.relative_to(WORK))
            receipt["files"].append(record)
            receipt["verified_bytes"] = sum(row["bytes"] for row in receipt["files"])
            save(receipt)
            print(json.dumps(record), flush=True)
        receipt.update(status="completed", finished_unix=time.time(),
                       source_available_bytes=shutil.disk_usage(WORK).free)
        save(receipt)
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        save(receipt)
        raise


if __name__ == "__main__":
    main()
