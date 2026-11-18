import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

import pytest


def load(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name + "_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trace_comparison_uses_committed_attempts_only(tmp_path):
    module = load("report_stage0")
    episode = dict(task_id="t", attempt_id="committed", success=True, status="completed", arm="none")
    (tmp_path / "episodes.jsonl").write_text(json.dumps(episode) + "\n")
    traces = []
    for attempt, action in [("interrupted", "wrong"), ("committed", "correct")]:
        traces.append(dict(task_id="t", attempt_id=attempt, step=0, action=action,
                           observation="observed", success=True, invalid_action=False))
    (tmp_path / "steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in traces))
    cells, actual = module.committed_cell(tmp_path)
    assert cells["t"]["success"] is True
    assert actual["t"] == [(0, "correct", "observed", True, False)]


def test_flip_counts_track_task_identity():
    module = load("report_stage0")
    cells = [{"a": {"success": True}, "b": {"success": False}},
             {"b": {"success": False}, "a": {"success": False}}]
    assert module.flip_counts(cells) == (1, 2)
    with pytest.raises(ValueError, match="different task sets"):
        module.flip_counts([cells[0], {"c": {"success": True}}])


def test_archive_excludes_weights_hidden_files_credentials_and_symlinks(tmp_path):
    module = load("upload_research_results")
    root = tmp_path / "study"
    root.mkdir()
    for name in ["run.json", "episodes.jsonl", "adapter_model.safetensors", ".env.local", "credentials.json", ".hidden.json"]:
        (root / name).write_text("record")
    outside = tmp_path / "outside.json"
    outside.write_text("private")
    (root / "linked.json").symlink_to(outside)
    assert {p.name for p in module.archive_candidates(root)} == {"run.json", "episodes.jsonl"}


def test_archive_manifest_matches_roundtrip_bytes(tmp_path):
    module = load("upload_research_results")
    root = tmp_path / "study"
    root.mkdir()
    data = b'{"success":true}\n'
    (root / "episodes.jsonl").write_bytes(data)
    archive, manifest = module.package(root, tmp_path / "out")
    with tarfile.open(archive) as handle:
        restored = handle.extractfile("episodes.jsonl").read()
    assert restored == data
    assert manifest == [dict(path="episodes.jsonl", bytes=len(data), sha256=hashlib.sha256(data).hexdigest())]
