#!/usr/bin/env python3
"""Download pinned baseline source archives."""
from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
import urllib.request


def main():
    lock = json.loads(Path("configs/sources.lock.json").read_text())
    root = Path("third_party")
    root.mkdir(exist_ok=True)
    for baseline in lock["baselines"]:
        if "repository" not in baseline:
            continue
        target = root / baseline["name"]
        stamp = target / ".source.json"
        if stamp.exists():
            if json.loads(stamp.read_text())["commit"] != baseline["commit"]:
                raise ValueError(f"Source version changed: {target}")
            print(f"Cached: {baseline['name']}", flush=True)
            continue
        if target.exists():
            raise ValueError(f"Incomplete checkout; inspect before reusing: {target}")
        repo = baseline["repository"].removeprefix("https://github.com/")
        url = f"https://codeload.github.com/{repo}/tar.gz/{baseline['commit']}"
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read(100 * 1024 * 1024 + 1)
        if len(payload) > 100 * 1024 * 1024:
            raise ValueError("Source archive exceeded 100 MiB")
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            members = archive.getmembers()
            for member in members:
                relative = Path(*Path(member.name).parts[1:])
                destination = (target / relative).resolve()
                if not destination.is_relative_to(target.resolve()) or member.issym() or member.islnk():
                    raise ValueError("Unsafe source archive member")
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(archive.extractfile(member).read())
        stamp.write_text(json.dumps(baseline, indent=2) + "\n")
        print(f"Ready: {baseline['name']} @ {baseline['commit']}", flush=True)


if __name__ == "__main__":
    main()
