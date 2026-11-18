#!/usr/bin/env python3
"""Watch both frozen follow-ups without launching, recovering or changing any job."""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import time
from zoneinfo import ZoneInfo

from learning_harness.bounded_queue import save
from learning_harness.followup_monitor import STUDIES, cloud_metrics, snapshot
from supervise_research import process


def observe(now):
    observations, errors = {}, {}
    for kind, spec in STUDIES.items():
        try:
            observations[kind] = snapshot(kind, spec, now, process)
        except Exception as error:
            # Keep exception messages and local data out of cloud output.
            errors[kind] = type(error).__name__
    return observations, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=Path("runs/bounded-followup-monitor-v1")
    )
    args = parser.parse_args()
    args.out.resolve().relative_to(Path("runs").resolve())
    if args.wandb and not args.watch:
        raise ValueError("A single CPU snapshot must not create cloud runs")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Check all frozen scientific inputs once before entering the read-only loop.
        from learning_harness.webshop_pilot import validate_frozen
        from run_nll_scoring_probe import validate

        validate(Path(STUDIES["nll"]["root"]))
        validate_frozen(Path(STUDIES["webshop"]["root"]))
        save(
            args.out / "job.json",
            dict(pid=os.getpid(), script=__file__, watch=args.watch, cloud=args.wandb),
        )
        stopping = False

        def stop(*_):
            nonlocal stopping
            stopping = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stop)
        cloud = None
        cloud_error = None
        if args.wandb:
            import wandb

            target = json.loads(Path("output/wandb/connection.json").read_text())
            if target.get("base_url", target.get("host")):
                os.environ["WANDB_BASE_URL"] = target.get(
                    "base_url", target.get("host")
                )
            cloud = wandb.init(
                entity=target["entity"],
                project=target["project"],
                id="bounded-followups-20261113",
                name="Frozen NLL/WebShop follow-ups / progress only",
                group="bounded-followup-progress",
                job_type="read-only-progress",
                resume="allow",
                mode="online",
                dir=str(args.out.absolute()),
                save_code=False,
                settings=wandb.Settings(
                    console="off",
                    disable_code=True,
                    disable_git=True,
                    save_code=False,
                    disable_job_creation=True,
                    x_disable_stats=True,
                    init_timeout=60,
                ),
            )
            save(
                args.out / "wandb.json",
                dict(
                    run_id=cloud.id,
                    run_url=cloud.url,
                    raw_records_uploaded=False,
                    progress_only=True,
                ),
            )
        deadline = time.monotonic() + 24 * 3600
        next_periodic, prior = 0, None
        try:
            while not stopping:
                now = time.time()
                observations, errors = observe(now)
                public = {kind: value["public"] for kind, value in observations.items()}
                changed = (
                    {
                        kind: (value["state"], value["needs_attention"])
                        for kind, value in public.items()
                    },
                    errors,
                )
                periodic = now >= next_periodic
                record = dict(
                    checked_unix=now,
                    time_china=datetime.fromtimestamp(
                        now, ZoneInfo("Asia/Shanghai")
                    ).isoformat(),
                    periodic=periodic,
                    studies=public,
                    local_processes={
                        k: v["local_processes"] for k, v in observations.items()
                    },
                    observation_error_types=errors,
                    cloud_error_type=cloud_error,
                    observer_pid=os.getpid(),
                    automatic_gpu_restart=False,
                    automatic_launch=False,
                )
                if periodic or changed != prior:
                    with (args.out / "checks.jsonl").open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                    if periodic:
                        next_periodic = now + 1800
                    print(
                        json.dumps(
                            dict(
                                time_china=record["time_china"],
                                studies=public,
                                observation_error_types=errors,
                            )
                        ),
                        flush=True,
                    )
                save(
                    args.out / "state.json",
                    dict(record, next_periodic_unix=next_periodic),
                )
                if cloud is not None:
                    try:
                        cloud.log(cloud_metrics(observations))
                        cloud.summary.update(
                            dict(
                                studies=public,
                                checked_unix=now,
                                observation_error_types=errors,
                                raw_records_uploaded=False,
                                progress_only=True,
                            )
                        )
                    except Exception as error:
                        cloud_error = type(error).__name__
                prior = changed
                if not args.watch or (
                    len(public) == len(STUDIES)
                    and not errors
                    and all(v["terminal"] for v in public.values())
                ):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Observer deadline; inspect its process before resuming"
                    )
                for _ in range(30):
                    if stopping:
                        break
                    time.sleep(1)
        finally:
            if cloud is not None:
                cloud.finish()
            save(
                args.out / "observer-stop.json",
                dict(
                    stopped_unix=time.time(),
                    observer_pid=os.getpid(),
                    signal_requested=stopping,
                    automatic_gpu_restart=False,
                ),
            )


if __name__ == "__main__":
    main()
