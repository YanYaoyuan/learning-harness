import importlib
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def parent_fixture(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    tool = importlib.import_module("freeze_action_controls")
    root, out = tmp_path / "parent", tmp_path / "output"
    root.mkdir()
    out.mkdir()
    (root / "freeze.json").write_text("{}")
    sha = tool.file_sha(root / "freeze.json")
    frozen = dict(cases=[dict(name="a")], planned_episodes=64)
    ledger = dict(
        status="completed",
        freeze_sha256=sha,
        open={},
        cases=dict(a=dict(status="completed")),
    )
    report = dict(
        complete=True, cases=1, actual_evaluation_episodes=64, freeze_sha256=sha
    )
    monkeypatch.setattr(tool.worker.parent, "ROOT", root)
    monkeypatch.setattr(tool.worker.parent, "OUT", out)
    monkeypatch.setattr(tool.worker.parent, "load_freeze", lambda: frozen)
    return tool, root, out, ledger, report


def write_parent(root, out, ledger, report):
    (root / "execution.json").write_text(json.dumps(ledger))
    (out / "summary.json").write_text(json.dumps(report))


def test_parent_ready_requires_completed_ledger_and_report(parent_fixture):
    tool, root, out, ledger, report = parent_fixture
    write_parent(root, out, ledger, report)
    frozen, report_path = tool.finished_parent()
    assert frozen["planned_episodes"] == 64
    assert report_path == out / "summary.json"


@pytest.mark.parametrize(
    "change",
    ["running", "incomplete_report", "wrong_hash", "open_worker", "missing_case"],
)
def test_parent_not_ready_never_creates_followup_directory(
    parent_fixture, change, tmp_path
):
    tool, root, out, ledger, report = parent_fixture
    if change == "running":
        ledger["status"] = "running"
    elif change == "incomplete_report":
        report["complete"] = False
    elif change == "wrong_hash":
        report["freeze_sha256"] = "wrong"
    elif change == "open_worker":
        ledger["open"] = dict(a=dict(pid=123))
    else:
        ledger["cases"] = {}
    write_parent(root, out, ledger, report)
    next_root = tmp_path / "followup"
    with pytest.raises(ValueError):
        tool.freeze(
            next_root,
            tmp_path / "unread-proposal.json",
            tmp_path / "unread-review.json",
        )
    assert not next_root.exists()
