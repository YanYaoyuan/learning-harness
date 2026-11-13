#!/usr/bin/env python3
"""Repair only missing W&B host resolution; preserve the frozen evaluator."""

from pathlib import Path
import time

from learning_harness.bounded_queue import save
from learning_harness.common import file_sha
import track_fresh_executor_calibration as original


def main():
    original.validate()
    receipt = original.OUT / "observer-recovery.json"
    if receipt.exists():
        raise ValueError("Inspect the existing observer recovery before restarting")
    import wandb

    reader = original.read
    target = reader("output/wandb/connection.json")
    host = target.get("base_url") or target.get("host") or wandb.Settings().base_url
    if not isinstance(host, str) or not host.startswith("https://"):
        raise ValueError("W&B requires an explicit resolved HTTPS API host")

    def read(path):
        value = reader(path)
        if Path(path) == Path("output/wandb/connection.json"):
            return dict(value, base_url=host)
        return value

    save(
        receipt,
        dict(
            started_unix=time.time(),
            freeze_sha256=file_sha(original.ROOT / "freeze.json"),
            evaluator_unchanged=True,
            original_observer_unchanged=True,
            reason="Connection file has no host key; use W&B settings default as prior observers did.",
            api_host=host,
            recovery_script_sha256=file_sha(__file__),
            raw_records_uploaded=False,
        ),
    )
    original.read = read
    original.main()


if __name__ == "__main__":
    main()
