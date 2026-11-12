#!/usr/bin/env python3
"""Acquire pinned public product files without loading goals or changing an env."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time
import urllib.request


def save(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    repositories = metadata["repositories"]
    if len(repositories) != 2 or len({r["repository"] for r in repositories}) != 2:
        raise ValueError("Two separately pinned repository metadata records required")
    files = []
    for name, opening in [("items_shuffle.json", "["), ("items_ins_v2.json", "{")]:
        records = [
            next(x for x in r["files"] if x["path"] == name) for r in repositories
        ]
        first, second = records
        if (first["size"], first["lfs"]["oid"]) != (
            second["size"],
            second["lfs"]["oid"],
        ):
            raise ValueError("Mirrors disagree on the full-file checksum or size")
        if first["size"] != first["lfs"]["size"] or len(first["lfs"]["oid"]) != 64:
            raise ValueError("Invalid expected LFS size or SHA256")
        files.append(
            dict(
                name=name,
                bytes=first["size"],
                sha256=first["lfs"]["oid"],
                opening=opening,
            )
        )
    receipt = args.out / "download.json"
    if receipt.exists() or any((args.destination / f["name"]).exists() for f in files):
        raise ValueError(
            "Existing download must be inspected; do not overwrite or restart"
        )
    args.destination.mkdir(parents=True, exist_ok=True)
    total = sum(f["bytes"] for f in files)
    # Keep room for the running study's eventual weights and a future index.
    reserve = 14 * 1024**3
    if shutil.disk_usage(args.destination).free < total + reserve:
        raise ValueError("Need catalog bytes plus a 14 GiB shared-disk reserve")
    args.out.mkdir(parents=True, exist_ok=True)
    os.nice(10)
    state = dict(
        status="downloading",
        pid=os.getpid(),
        started_unix=time.time(),
        metadata_sha256=hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
        destination=str(args.destination),
        expected_bytes=total,
        files=[],
        mirrors=[
            {k: r[k] for k in ["repository", "revision", "metadata_url"]}
            for r in repositories
        ],
        original_author_byte_identity_verified=False,
        official_full_links_returned_404=True,
        human_goals_read=False,
        official_test_evaluations=0,
        model_calls=0,
        gpu_used=False,
        full_benchmark_ready=False,
        existing_environment_or_preview_modified=False,
    )
    save(receipt, state)
    deadline = time.monotonic() + 2 * 3600

    def interrupted(signum, frame):
        raise InterruptedError("Download interrupted; partial file retained")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for item in files:
            repo = repositories[0]
            url = f"https://huggingface.co/datasets/{repo['repository']}/resolve/{repo['revision']}/{item['name']}"
            target = args.destination / item["name"]
            temporary = target.with_suffix(".json.part")
            if temporary.exists():
                raise ValueError(
                    "Partial file retained for inspection; no implicit retry"
                )
            current = dict(
                name=item["name"],
                url=url,
                expected_bytes=item["bytes"],
                expected_sha256=item["sha256"],
                downloaded_bytes=0,
                status="downloading",
            )
            state["files"].append(current)
            save(receipt, state)
            digest = hashlib.sha256()
            request = urllib.request.Request(
                url, headers={"Accept-Encoding": "identity"}
            )
            with (
                urllib.request.urlopen(request, timeout=45) as response,
                temporary.open("xb") as handle,
            ):
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != item["bytes"]:
                    raise ValueError("Response length differs from pinned metadata")
                last_record = time.monotonic()
                while True:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Two-hour acquisition cap reached")
                    chunk = response.read(8 * 1024**2)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    current["downloaded_bytes"] += len(chunk)
                    if current["downloaded_bytes"] > item["bytes"]:
                        raise ValueError("Response exceeds expected file size")
                    if time.monotonic() - last_record >= 5:
                        save(receipt, state)
                        last_record = time.monotonic()
                handle.flush()
                os.fsync(handle.fileno())
            if (
                current["downloaded_bytes"] != item["bytes"]
                or digest.hexdigest() != item["sha256"]
            ):
                raise ValueError("Actual bytes fail pinned size or SHA256 verification")
            with temporary.open("rb") as handle:
                if handle.read(128).lstrip()[:1] != item["opening"].encode():
                    raise ValueError("Unexpected JSON container opening")
            temporary.replace(target)
            current.update(
                status="checksum_verified",
                actual_sha256=digest.hexdigest(),
                path=str(target),
            )
            save(receipt, state)
        state.update(
            status="catalog_files_checksum_verified",
            finished_unix=time.time(),
            remaining=[
                "Streaming catalog/schema audit",
                "Full retrieval index",
                "Goal mapping and disjoint split protocol",
                "Public-only student interface",
            ],
        )
        save(receipt, state)
        print(
            json.dumps(
                {k: v for k, v in state.items() if k not in ("files", "mirrors")}
            ),
            flush=True,
        )
    except BaseException as error:
        state.update(
            status="failed", error_type=type(error).__name__, stopped_unix=time.time()
        )
        save(receipt, state)
        # Never print a redirected signed CDN URL from a network exception.
        print(
            json.dumps(
                dict(
                    status="failed",
                    error_type=type(error).__name__,
                    partial_files_retained=True,
                )
            ),
            flush=True,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
