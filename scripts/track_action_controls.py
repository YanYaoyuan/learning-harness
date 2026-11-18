#!/usr/bin/env python3
"""Independent follow-up observer; cloud artifacts contain only aggregates/figures."""

import argparse
import os
from pathlib import Path
import subprocess
import time

from action_control_report import load_study, output_path
from action_control_interim import observe_interim
from learning_harness.bounded_queue import save
from learning_harness.common import file_sha
from learning_harness.research_aggregates import aggregate_report
from supervise_research import process, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frozen = load_study(args.root)
    out = output_path(args.root)
    if args.out.resolve() != out.resolve():
        raise ValueError("Observer output must match the frozen study root")
    out.mkdir(parents=True, exist_ok=True)
    freeze_sha = file_sha(args.root / "freeze.json")
    target = read(Path("output/wandb/connection.json"))
    host = target.get("base_url", target.get("host"))
    if host:
        os.environ["WANDB_BASE_URL"] = host
    import wandb

    settings = wandb.Settings(
        console="off",
        disable_code=True,
        disable_git=True,
        save_code=False,
        disable_job_creation=True,
        x_disable_stats=True,
        init_timeout=60,
    )
    progress = wandb.init(
        entity=target["entity"],
        project=target["project"],
        id=freeze_sha[:12] + "-progress",
        name=frozen["name"] + " / progress and descriptive interim",
        group="task-action-controls-v1",
        job_type="progress",
        resume="allow",
        mode="online",
        dir=str(out.absolute()),
        save_code=False,
        settings=settings,
    )
    save(
        out / "progress-wandb.json",
        dict(
            status="watching",
            run_id=progress.id,
            run_url=progress.url,
            progress_only=False,
            progress_and_interim=True,
            raw_records_uploaded=False,
        ),
    )
    deadline = time.monotonic() + 18 * 3600
    try:
        while True:
            ledger = read(args.root / "execution.json")
            state = ledger.get("status", "not_started")
            completed = sum(
                c.get("status") == "completed" for c in ledger.get("cases", {}).values()
            )
            elapsed = ledger.get("gpu_seconds", 0) + sum(
                max(0, time.time() - item["started_unix"])
                for item in ledger.get("open", {}).values()
            )
            progress.log(
                {
                    "progress/completed_adapters": completed,
                    "progress/planned_adapters": len(frozen["cases"]),
                    "progress/completed_episodes": 64 * completed,
                    "progress/elapsed_gpu_hours": elapsed / 3600,
                    "progress/cap_gpu_hours": frozen["cap_gpu_hours"],
                    "progress/state": state,
                }
            )
            observe_interim(args.root, out, frozen, progress, wandb)
            if state == "evaluation_completed":
                pipeline = read(args.root / "job.json").get("pipeline", {})
                pid = pipeline.get("pid")
                alive = (
                    process(pid, ["run_action_controls.py", str(args.root)])
                    if pid
                    else {}
                )
                if not alive.get("alive") or not alive.get("identity_verified"):
                    with (out / "report-recovery.log").open("a") as log:
                        subprocess.run(
                            [
                                str(Path(".venv-model/bin/python").absolute()),
                                "scripts/action_control_report.py",
                                "--root",
                                str(args.root),
                            ],
                            env=dict(
                                os.environ,
                                CUDA_VISIBLE_DEVICES="",
                                PYTHONPATH=str(Path("src").absolute()),
                                MPLCONFIGDIR="/tmp/lh-mpl",
                            ),
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=180,
                        )
                    continue
            if state in ("completed", "failed", "interrupted", "stopped_at_cap"):
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "Observer deadline; inspect existing processes before resuming"
                )
            time.sleep(30)
        progress.summary.update(
            termination_state=state,
            completed_adapters=completed,
            raw_records_uploaded=False,
        )
        save(
            out / "progress-wandb.json",
            dict(
                status=state,
                run_id=progress.id,
                run_url=progress.url,
                completed_adapters=completed,
                progress_only=False,
                progress_and_interim=True,
                raw_records_uploaded=False,
            ),
        )
    finally:
        progress.finish()
    if state != "completed":
        save(
            out / "finalization.json",
            dict(status="stopped", execution_state=state, raw_records_uploaded=False),
        )
        return
    save(
        out / "finalization.json",
        dict(
            local_report_complete=True,
            cloud_status="pending",
            raw_records_uploaded=False,
        ),
    )
    result_run = None
    try:
        load_study(args.root)
        report = read(out / "summary.json")
        aggregate = aggregate_report(report, frozen, freeze_sha)
        save(out / "aggregate-summary.json", aggregate)
        names = (
            "aggregate-summary.json",
            "action-matched-policies.png",
            "action-matched-policies.pdf",
            "RESULTS.zh-CN.md",
        )
        # Explicit names only: no output-directory glob, raw JSON package,
        # task-outcomes.csv, training-exposure.csv or source upload.
        result_run = wandb.init(
            entity=target["entity"],
            project=target["project"],
            id=freeze_sha[:12],
            name=frozen["name"] + " / aggregate results",
            group="task-action-controls-v1",
            job_type="aggregate-results",
            resume="allow",
            mode="online",
            dir=str(out.absolute()),
            save_code=False,
            settings=settings,
        )
        result_run.summary.update(aggregate)
        result_run.log(
            {
                "figures/action-matched-policies": wandb.Image(
                    str(out / "action-matched-policies.png")
                )
            }
        )
        artifact = wandb.Artifact(
            "task-action-controls-" + freeze_sha[:12] + "-aggregates",
            type="research-aggregates",
        )
        for name in names:
            artifact.add_file(str(out / name), name=name)
        result_run.log_artifact(artifact).wait()
        save(
            out / "archival.json",
            dict(
                status="uploaded",
                run_id=result_run.id,
                run_url=result_run.url,
                aggregate_only=True,
                raw_records_uploaded=False,
                uploaded_sha256={name: file_sha(out / name) for name in names},
            ),
        )
        save(
            out / "finalization.json",
            dict(
                local_report_complete=True,
                cloud_status="uploaded",
                raw_records_uploaded=False,
            ),
        )
    except Exception as error:
        save(
            out / "finalization.json",
            dict(
                local_report_complete=True,
                cloud_status="failed",
                error_type=type(error).__name__,
                raw_records_uploaded=False,
            ),
        )
        raise
    finally:
        if result_run is not None:
            result_run.finish()


if __name__ == "__main__":
    main()
