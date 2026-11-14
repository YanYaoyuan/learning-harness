#!/usr/bin/env python3
"""Supplementary read-only native model runs; never changes training or evaluation."""

import fcntl
import hashlib
import os
import time

from learning_harness.bounded_queue import save
from learning_harness.common import file_sha
from learning_harness.training_log_metrics import training_metrics
from run_experience_value_development import OUT, ROOT, read, validate


def main():
    frozen = validate()
    lock = (OUT / "native-training-wandb.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = OUT / "native-training-wandb.json"
    if state_path.exists():
        raise ValueError("Inspect existing telemetry before restarting it")
    fingerprint = file_sha(ROOT / "freeze.json")
    state = dict(
        status="watching",
        freeze_sha256=fingerprint,
        source_sha256=file_sha(__file__),
        parser_sha256=file_sha("src/learning_harness/training_log_metrics.py"),
        step_axis="Derived from immutable Trainer logging_steps=10, max_steps=192; no resume.",
        raw_records_uploaded=False,
        runs={},
    )
    save(state_path, state)
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
    current, current_name = None, None
    deadline = time.monotonic() + 100 * 3600
    try:
        while True:
            ledger = read(ROOT / "execution.json")
            terminal = ledger["status"] in (
                "completed",
                "failed",
                "interrupted",
                "stopped_at_cap",
            )
            if current is None:
                case = next(
                    (
                        c
                        for c in frozen["cases"]
                        if c["name"] != "base"
                        and not state["runs"].get(c["name"], {}).get("finished")
                        and (ROOT / "cases" / c["name"] / "training.log").exists()
                    ),
                    None,
                )
                if case is not None:
                    current_name = case["name"]
                    run_id = (
                        fingerprint[:10]
                        + "-"
                        + hashlib.sha256(current_name.encode()).hexdigest()[:12]
                    )
                    current = wandb.init(
                        entity=connection["entity"],
                        project=connection["project"],
                        id=run_id,
                        name=current_name,
                        group="experience-value-development-v1/models",
                        job_type="student-training-and-endpoint",
                        mode="online",
                        resume="allow",
                        dir=str(OUT.absolute()),
                        save_code=False,
                        settings=settings,
                        config=dict(
                            case,
                            study_freeze_sha256=fingerprint,
                            logging_steps=10,
                            updates=192,
                            telemetry_from_local_log=True,
                            cloud_log_time_is_not_original_training_time=True,
                        ),
                    )
                    current.define_metric("train/step")
                    current.define_metric("train/*", step_metric="train/step")
                    state["runs"][current_name] = dict(
                        run_id=current.id,
                        run_url=current.url,
                        last_logged_step=0,
                        finished=False,
                    )
                    save(state_path, state)
            if current is not None:
                folder = ROOT / "cases" / current_name
                record = state["runs"][current_name]
                rows = training_metrics(
                    (folder / "training.log").read_text(errors="replace")
                )
                for row in rows:
                    if row["step"] > record["last_logged_step"]:
                        current.log(
                            {
                                "train/step": row["step"],
                                **{
                                    "train/" + k: v
                                    for k, v in row.items()
                                    if k != "step"
                                },
                            }
                        )
                        record["last_logged_step"] = row["step"]
                if (folder / "A/experiment.json").exists():
                    meta = read(folder / "A/experiment.json")
                    current.summary.update(
                        dict(
                            training_presentations=meta["processed_examples_rank0"],
                            training_input_tokens=meta["input_tokens_rank0"],
                            training_target_tokens=meta["target_tokens_rank0"],
                        )
                    )
                complete = (folder / "worker-result.json").exists()
                stopped = (
                    current_name in ledger.get("cases", {})
                    and ledger["cases"][current_name]["status"] != "completed"
                )
                if complete:
                    result = read(folder / "worker-result.json")
                    if result["freeze_sha256"] != fingerprint or not result["complete"]:
                        raise ValueError("Model telemetry endpoint identity differs")
                    current.summary.update(
                        dict(
                            evaluation_success=result["mean_success"],
                            evaluation_episodes=128,
                            invalid_format_decisions=result["invalid_format_decisions"],
                            raw_records_uploaded=False,
                        )
                    )
                if complete or stopped or terminal:
                    current.summary.update(
                        dict(
                            termination_state="completed" if complete else "stopped",
                            raw_records_uploaded=False,
                        )
                    )
                    current.finish()
                    record.update(
                        finished=True, status="completed" if complete else "stopped"
                    )
                    current, current_name = None, None
                save(state_path, state)
            if terminal and current is None:
                waiting = any(
                    c["name"] != "base"
                    and (ROOT / "cases" / c["name"] / "training.log").exists()
                    and not state["runs"].get(c["name"], {}).get("finished")
                    for c in frozen["cases"]
                )
                if not waiting:
                    break
            if time.monotonic() > deadline:
                raise TimeoutError("Training telemetry deadline; preserve GPU work")
            time.sleep(15)
        state["status"] = ledger["status"]
        save(state_path, state)
    finally:
        if current is not None:
            current.finish()


if __name__ == "__main__":
    main()
