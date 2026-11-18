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


def test_successful_freeze_preserves_candidates_and_locks_formal_inputs(
    parent_fixture, tmp_path, monkeypatch
):
    tool, parent_root, parent_out, ledger, report = parent_fixture
    write_parent(parent_root, parent_out, ledger, report)
    (parent_root / "a-lora.json").write_text(
        json.dumps(dict(max_steps=192, global_batch_size=4))
    )
    parent = tool.worker.parent.load_freeze()
    parent["files"] = {}
    actual_freezer_source = Path(tool.__file__).read_bytes()
    monkeypatch.chdir(tmp_path)
    for name in tool.REQUIRED_CODE | {"scripts/freeze_action_controls.py"}:
        path = Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            actual_freezer_source
            if name == "scripts/freeze_action_controls.py"
            else b"isolated file-lock fixture\n"
        )
    monkeypatch.setattr(
        tool, "__file__", str(tmp_path / "scripts/freeze_action_controls.py")
    )
    cases = [
        dict(
            name=f"{source}-{policy}",
            source=source,
            policy=policy,
            initialization_seed=101,
            sampling_seed=101,
            repeat=0,
            gpu=gpu,
            schedule="candidate.json",
        )
        for source, gpu in (("api", 0), ("generic", 1))
        for policy in ("random", "min_nll")
    ]
    candidates = Path(
        "output/variance-diagnostic-v1/task-action-feasibility/candidate-schedules"
    )
    candidates.mkdir(parents=True)
    original = {}
    for case in cases:
        path = candidates / f"{case['policy']}-{case['source']}-s101.json"
        path.write_text(
            json.dumps(
                dict(
                    source=case["source"],
                    policy=case["policy"],
                    sampling_seed=101,
                    candidate_only=True,
                    training_started=False,
                )
            )
        )
        original[path] = path.read_bytes()
    # Recipe regeneration has separate tests and real-input checks; this test
    # exercises actual filesystem formalization and the real hash loader.
    monkeypatch.setattr(
        tool.worker, "load_recipe", lambda p: (json.loads(p.read_text()), {})
    )
    proposal = dict(
        name="isolated freeze fixture",
        proposal_state="draft",
        status="Proposal only; no freeze",
        protocol="task-action-controls-v1",
        teacher_calls=0,
        official_tests_used=False,
        primary_metric="mean_success",
        primary_contrast="min_nll_minus_random",
        gpus=[0, 1],
        cap_gpu_hours=1,
        allocation_seconds=1200,
        minimum_allocation_seconds=600,
        planned_episodes=256,
        cases=cases,
        report_script="scripts/action_control_report.py",
        observer_script="scripts/track_action_controls.py",
        cloud_scope="aggregate_only",
        interim_cell_milestones=[1],
    )
    proposal_path, review_path = Path("proposal.json"), Path("review.json")
    proposal_path.write_text(json.dumps(proposal))
    review_path.write_text(
        json.dumps(
            dict(
                decision="run_task_action_controls",
                complete_parent_results_reviewed=True,
                parent_report_sha256=tool.file_sha(parent_out / "summary.json"),
                parent_freeze_sha256=tool.file_sha(parent_root / "freeze.json"),
                proposal_sha256=tool.file_sha(proposal_path),
            )
        )
    )
    root = Path("runs/followup")
    result = tool.freeze(root, proposal_path, review_path)
    assert result["training_started"] is False
    assert not (root / "execution.json").exists()
    frozen = tool.load_study(root)
    assert "proposal_state" not in frozen and "status" not in frozen
    for case in frozen["cases"]:
        recipe = json.loads(Path(case["schedule"]).read_text())
        assert recipe["candidate_only"] is False and "training_started" not in recipe
        assert recipe["candidate_recipe_sha256"] == tool.file_sha(
            recipe["formalized_from"]
        )
    assert all(path.read_bytes() == data for path, data in original.items())
    before = (root / "freeze.json").read_bytes()
    with pytest.raises(ValueError, match="never overwrite"):
        tool.freeze(root, proposal_path, review_path)
    assert (root / "freeze.json").read_bytes() == before
    path = Path(frozen["cases"][0]["schedule"])
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Frozen study input changed"):
        tool.load_study(root)
