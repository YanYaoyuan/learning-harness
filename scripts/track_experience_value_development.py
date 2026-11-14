#!/usr/bin/env python3
"""Progress, complete balanced-block snapshots, and aggregate-only cloud files."""

import os
import subprocess
import time

from learning_harness.bounded_queue import save
from learning_harness.common import file_sha
from learning_harness.experience_value_study import NODES
from run_experience_value_development import (
    OUT,
    ROOT,
    engine,
    environment,
    read,
    validate,
)


def completed_lines(path):
    if not path.exists():
        return 0
    return sum(
        bool(line.strip())
        for line in path.read_bytes().splitlines(keepends=True)
        if line.endswith(b"\n")
    )


def main():
    frozen = validate()
    fingerprint = file_sha(ROOT / "freeze.json")
    connection = read("output/wandb/connection.json")
    host = connection.get("base_url") or connection.get("host")
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
    run = wandb.init(
        entity=connection["entity"],
        project=connection["project"],
        id=fingerprint[:12],
        name=frozen["name"],
        group="experience-value-development-v1",
        job_type="student-learning",
        mode="online",
        resume="allow",
        dir=str(OUT.absolute()),
        save_code=False,
        settings=settings,
    )
    save(
        OUT / "wandb.json",
        dict(
            status="watching",
            run_id=run.id,
            run_url=run.url,
            raw_records_uploaded=False,
        ),
    )
    nodes_path = OUT / "interim-tracking.json"
    nodes = read(nodes_path) if nodes_path.exists() else {}
    logged = set()
    deadline = time.monotonic() + 100 * 3600
    try:
        while True:
            ledger = (
                read(ROOT / "execution.json")
                if (ROOT / "execution.json").exists()
                else {}
            )
            state = ledger.get("status", "not_started")
            completed = {
                name
                for name, c in ledger.get("cases", {}).items()
                if c["status"] == "completed"
            }
            pending_episodes = sum(
                completed_lines(ROOT / "cases" / name / "evaluation/tasks.jsonl")
                for name in ledger.get("open", {})
            )
            elapsed = ledger.get("gpu_seconds", 0) + sum(
                max(0, time.time() - c["started_unix"])
                for c in ledger.get("open", {}).values()
            )
            values = {
                "progress/completed_models": len(completed),
                "progress/planned_models": 225,
                "progress/completed_episodes": 128 * len(completed) + pending_episodes,
                "progress/planned_episodes": 28800,
                "progress/gpu_hours": elapsed / 3600,
                "progress/cap_gpu_hours": 96,
                "progress/state": state,
            }
            for name in ledger.get("open", {}):
                p = ROOT / "cases" / name / "progress.json"
                if p.exists():
                    values["progress/active_phase"] = read(p).get("phase")
            run.log(values)
            # These are fully completed model endpoints; no partial task success.
            for name in sorted(completed - logged):
                result = read(ROOT / "cases" / name / "worker-result.json")
                case = result["case"]
                run.log(
                    {
                        f"completed_model/{case['source']}/{case['policy']}/success": result[
                            "mean_success"
                        ],
                        "completed_model/count": len(logged) + 1,
                    }
                )
                logged.add(name)
            for n in NODES:
                if str(n) in nodes:
                    continue
                required = {
                    c["name"]
                    for c in frozen["cases"]
                    if c["name"] == "base" or c["block"] <= n
                }
                if not required <= completed:
                    continue
                nodes[str(n)] = dict(status="running", started_unix=time.time())
                save(nodes_path, nodes)
                target = OUT / "interim" / f"blocks-{n:02d}"
                try:
                    with (OUT / f"interim-{n:02d}.log").open("x") as log:
                        subprocess.run(
                            [
                                engine.MODEL_PY,
                                "scripts/report_experience_value_development.py",
                                "--blocks",
                                str(n),
                            ],
                            env=dict(
                                environment(0),
                                CUDA_VISIBLE_DEVICES="",
                                MPLCONFIGDIR="/tmp/lh-value-interim-mpl",
                            ),
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=600,
                        )
                    aggregate = read(target / "aggregate-summary.json")
                    run.log(
                        {
                            f"figures/complete-blocks-{n:02d}": wandb.Image(
                                str(target / "selection-learning.png")
                            ),
                            "balanced/complete_blocks": n,
                            "balanced/gradient_high_minus_random": aggregate[
                                "endpoints"
                            ]["contrasts"]["gradient_high_minus_random"][
                                "mean_difference"
                            ],
                        }
                    )
                    artifact = wandb.Artifact(
                        "experience-value-" + fingerprint[:12] + f"-blocks-{n:02d}",
                        type="research-aggregates",
                    )
                    names = (
                        "aggregate-summary.json",
                        "selection-learning.png",
                        "RESULTS.zh-CN.md",
                    )
                    for filename in names:
                        artifact.add_file(str(target / filename), name=filename)
                    run.log_artifact(artifact).wait()
                    nodes[str(n)].update(
                        status="uploaded",
                        uploaded_sha256={v: file_sha(target / v) for v in names},
                    )
                except Exception as error:
                    nodes[str(n)].update(
                        status="failed_no_automatic_retry",
                        error_type=type(error).__name__,
                    )
                save(nodes_path, nodes)
            if state in ("completed", "failed", "interrupted", "stopped_at_cap"):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Observer deadline; preserve the running pipeline")
            time.sleep(30)
        if state == "completed":
            validate()
            aggregate = read(OUT / "aggregate-summary.json")
            if not aggregate["complete"] or aggregate["freeze_sha256"] != fingerprint:
                raise ValueError("Final aggregate identity differs")
            run.summary.update(aggregate)
            run.log(
                {
                    "figures/final-development": wandb.Image(
                        str(OUT / "selection-learning.png")
                    )
                }
            )
            names = (
                "aggregate-summary.json",
                "selection-learning.png",
                "RESULTS.zh-CN.md",
            )
            artifact = wandb.Artifact(
                "experience-value-" + fingerprint[:12] + "-final",
                type="research-aggregates",
            )
            for filename in names:
                artifact.add_file(str(OUT / filename), name=filename)
            run.log_artifact(artifact).wait()
            save(
                OUT / "archival.json",
                dict(
                    status="uploaded",
                    run_id=run.id,
                    run_url=run.url,
                    raw_records_uploaded=False,
                    uploaded_sha256={v: file_sha(OUT / v) for v in names},
                ),
            )
        run.summary.update(dict(termination_state=state, raw_records_uploaded=False))
        save(
            OUT / "wandb.json",
            dict(
                status=state, run_id=run.id, run_url=run.url, raw_records_uploaded=False
            ),
        )
    finally:
        run.finish()


if __name__ == "__main__":
    main()
