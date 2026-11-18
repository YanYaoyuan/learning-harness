#!/usr/bin/env python3
"""Archive completed raw records and upload measured results to the private W&B project."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile


def archive_candidates(root):
    """Only experiment JSON records; no weights, logs, hidden files or symlinks."""
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (not path.is_file() or path.is_symlink() or path.suffix not in (".json", ".jsonl")
                or any(part.startswith(".") for part in relative.parts)
                or any(word in path.name.lower() for word in ("credential", "secret", "key.env"))):
            continue
        yield path


def package(root, output):
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "raw-records.tar.gz"
    manifest = []
    with tarfile.open(archive.with_suffix(".gz.part"), "w:gz", compresslevel=3) as handle:
        for path in archive_candidates(root):
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest.append(dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=sha))
            handle.add(path, arcname=str(path.relative_to(root)), recursive=False)
    archive.with_suffix(".gz.part").replace(archive)
    (output / "archive-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return archive, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("source", "stage0", "stage1", "stage2"), required=True)
    args = parser.parse_args()
    if args.study == "source":
        root = Path("runs/source-comparison-v1")
        output = Path("output/source-comparison-v1")
        measured = json.loads((output / "summary.json").read_text())
        if not measured.get("complete") or not measured.get("token_and_budget_audit_passed"):
            raise ValueError("Source study is not complete and audited")
        identity = json.loads((output / "wandb.json").read_text())["run_id"]
        group, title = "source-comparison-v1", "API-scaffold vs ReAct / matched SFT and adaptation"
    elif args.study == "stage0":
        root = Path("runs/variance-diagnostic-v1/stage0")
        output = Path("output/variance-diagnostic-v1/stage0")
        measured = json.loads((output / "summary.json").read_text())
        if not measured.get("complete"):
            raise ValueError("Stage 0 local report is incomplete")
        identity = hashlib.sha256((root / "freeze.json").read_bytes()).hexdigest()[:12]
        group, title = "variance-diagnostic-v1", "Stage 0 / legacy API-generic batching sensitivity"
    else:
        root = Path("runs/variance-diagnostic-v1") / args.study
        output = Path("output/variance-diagnostic-v1") / args.study
        measured = json.loads((output / "summary.json").read_text())
        if not measured.get("complete") or measured.get("actual_evaluation_episodes") != (2176 if args.study == "stage1" else 4352):
            raise ValueError("Local sampling diagnostic report is incomplete")
        identity = hashlib.sha256((root / "freeze.json").read_bytes()).hexdigest()[:12]
        group = "variance-diagnostic-v1"
        title = ("Stage 1 / crossed initialization and sampling seeds" if args.study == "stage1" else
                 "Stage 2 / full-pass versus subset-replay at matched compute")
    archive, manifest = package(root, output)
    connection = json.loads(Path("output/wandb/connection.json").read_text())
    base = connection.get("base_url", connection.get("host"))
    if base:
        os.environ["WANDB_BASE_URL"] = base
    import wandb
    run = wandb.init(entity=connection["entity"], project=connection["project"], id=identity,
        name=title, group=group, job_type="measured-results-archive", mode="online", resume="allow",
        dir=str(output.absolute()), save_code=False,
        settings=wandb.Settings(console="off", disable_code=True, disable_git=True, save_code=False,
            disable_job_creation=True, x_disable_stats=True, init_timeout=60))
    try:
        for path in sorted(output.glob("*.png")):
            run.log({"figures/" + path.stem: wandb.Image(str(path))})
        coverage = output / "supplementary/coverage/coverage-and-exposure.png"
        if coverage.exists():
            run.log({"figures/coverage-and-exposure": wandb.Image(str(coverage))})
        if args.study == "stage0":
            run.summary.update(dict(complete=True, stage0_measured_result=measured,
                                    official_tests_used=False, teacher_calls=0))
        elif args.study in ("stage1", "stage2"):
            run.summary.update({"complete": True, args.study + "_measured_result": measured,
                                "official_tests_used": False, "teacher_calls": 0})
        run.summary.update(dict(raw_records_archived=True, archived_json_files=len(manifest)))
        artifact_name = group + ("-" + args.study if args.study in ("stage1", "stage2") else "") + "-raw-records"
        artifact = wandb.Artifact(artifact_name, type="research-results")
        artifact.add_file(str(archive))
        artifact.add_file(str(output / "archive-manifest.json"))
        for path in sorted(output.iterdir()):
            if path.is_file() and (path.suffix in (".png", ".pdf", ".csv", ".md") or path.name == "summary.json"):
                artifact.add_file(str(path))
        run.log_artifact(artifact).wait()
        receipt = dict(status="uploaded", run_id=run.id, run_url=run.url, archived_json_files=len(manifest),
            archive=str(archive), archive_bytes=archive.stat().st_size,
            archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        (output / "archival.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
