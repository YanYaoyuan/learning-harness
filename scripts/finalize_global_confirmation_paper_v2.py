#!/usr/bin/env python3
"""Read-only wait, final cloud check, then CPU-only paper update; never starts GPU work."""

import fcntl
import os
from pathlib import Path
import subprocess
import time

from learning_harness.bounded_queue import save
from learning_harness.common import file_sha
import run_global_confirmation as study

ROOT = Path("runs/global-confirmation-paper-finalization-v2")
OUT = Path("output/global-confirmation-paper-finalization-v2")
MANIFEST = Path("output/pdf/global-selection-paper/build-manifest.json")
BUILDER = Path("scripts/build_global_selection_paper.py")


def main():
    if ROOT.exists():
        raise ValueError("Inspect any prior paper-finalization attempt")
    study.validate()
    manuscript = study.read(MANIFEST)
    ROOT.mkdir()
    OUT.mkdir()
    lock = (ROOT / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    files = dict(manuscript["source_files"])
    files.update(manuscript.get("restoration", {}).get("original_source_files", {}))
    for path in (
        BUILDER,
        Path(__file__),
        Path("scripts/verify_global_confirmation_cloud.py"),
        Path("scripts/restore_global_paper_sections.py"),
        study.ROOT / "freeze.json",
    ):
        files[str(path)] = file_sha(path)
    save(
        ROOT / "freeze.json",
        dict(
            files=files,
            wait_hours=48,
            automatic_gpu_restart=False,
            parent_freeze_sha256=file_sha(study.ROOT / "freeze.json"),
        ),
    )
    save(ROOT / "job.json", dict(pid=os.getpid(), started_unix=time.time()))
    deadline = time.monotonic() + 48 * 3600
    try:
        while True:
            ledger = study.read(study.ROOT / "execution.json")
            status = ledger["status"]
            if status in ("failed", "interrupted", "stopped_at_cap"):
                save(
                    ROOT / "state.json",
                    dict(
                        status="parent_needs_review",
                        parent_status=status,
                        automatic_gpu_restart=False,
                        checked_unix=time.time(),
                    ),
                )
                return
            archive = study.OUT / "archival.json"
            if (
                status == "completed"
                and archive.exists()
                and study.read(archive).get("status") == "uploaded"
            ):
                break
            save(
                ROOT / "state.json",
                dict(
                    status="waiting_for_complete_confirmation_and_archive",
                    parent_status=status,
                    completed_models=len(ledger.get("cases", {})),
                    checked_unix=time.time(),
                    automatic_gpu_restart=False,
                ),
            )
            if time.monotonic() > deadline:
                raise TimeoutError("Paper-finalization observer deadline")
            time.sleep(30)
        for path, sha in files.items():
            if file_sha(path) != sha:
                raise ValueError(
                    "Paper input changed; preserve user edits instead of overwriting: "
                    + path
                )
        result = study.read(study.OUT / "aggregate-summary.json")
        if (
            not result.get("complete")
            or result.get("models") != 97
            or result.get("episodes") != 12416
        ):
            raise ValueError("Incomplete final confirmation")
        for name, sha in result["case_receipt_sha256"].items():
            if file_sha(study.ROOT / "cases" / name / "worker-result.json") != sha:
                raise ValueError("Audited final receipt changed")
        env = dict(
            study.environment(0),
            CUDA_VISIBLE_DEVICES="",
            MPLCONFIGDIR="/tmp/lh-paper-postrun-mpl",
        )
        with (OUT / "cloud-check.log").open("x") as log:
            subprocess.run(
                [
                    str(Path(".venv-wandb/bin/python").absolute()),
                    "scripts/verify_global_confirmation_cloud.py",
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )
        with (OUT / "paper-build.log").open("x") as log:
            subprocess.run(
                [study.engine.MODEL_PY, str(BUILDER), "--update"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=300,
            )
        manifest = study.read(MANIFEST)
        if not manifest["confirmation_complete"]:
            raise ValueError("Paper did not include the complete confirmation")
        render = OUT / "render"
        render.mkdir()
        subprocess.run(
            ["pdftoppm", "-png", "-r", "80", manifest["pdf"], str(render / "page")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
            timeout=120,
        )
        save(
            ROOT / "state.json",
            dict(
                status="paper_updated_pending_visual_review_and_publication",
                checked_unix=time.time(),
                pdf=manifest["pdf"],
                pdf_sha256=manifest["pdf_sha256"],
                cloud_verification=str(study.OUT / "cloud-verification.json"),
                auto_updated_tables=True,
                automatic_new_gpu_work=False,
            ),
        )
    except BaseException as error:
        save(
            ROOT / "state.json",
            dict(
                status="finalization_needs_attention",
                error_type=type(error).__name__,
                checked_unix=time.time(),
                automatic_gpu_restart=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
