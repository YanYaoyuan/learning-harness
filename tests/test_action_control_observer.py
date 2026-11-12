import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def observer_fixture(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    observer = importlib.import_module("track_action_controls")
    root, out = tmp_path / "study", tmp_path / "output"
    root.mkdir()
    out.mkdir()
    (root / "freeze.json").write_text("{}")
    (root / "job.json").write_text(json.dumps(dict(pipeline=dict(pid=123))))
    (out / "summary.json").write_text("{}")
    for name in (
        "action-matched-policies.png",
        "action-matched-policies.pdf",
        "RESULTS.zh-CN.md",
        "task-outcomes.csv",
        "training-exposure.csv",
        "private-steps.jsonl",
    ):
        (out / name).write_text("test fixture")
    runs, artifacts = [], []

    class Run:
        def __init__(self, **kwargs):
            self.id = kwargs["id"]
            self.url = "https://example.invalid/" + self.id
            self.summary, self.logs = {}, []
            self.finished = False
            runs.append(self)

        def log(self, item):
            self.logs.append(item)

        def finish(self):
            self.finished = True

        def log_artifact(self, item):
            return SimpleNamespace(wait=lambda: None)

    class Artifact:
        def __init__(self, *args, **kwargs):
            self.files = []
            artifacts.append(self)

        def add_file(self, path, name):
            assert Path(path).is_file()
            self.files.append(name)

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            init=lambda **kwargs: Run(**kwargs),
            Settings=lambda **kwargs: kwargs,
            Artifact=Artifact,
            Image=lambda path: path,
        ),
    )
    real_read = observer.read
    monkeypatch.setattr(
        observer,
        "read",
        lambda p: (
            dict(entity="fixture", project="fixture")
            if p == Path("output/wandb/connection.json")
            else real_read(p)
        ),
    )
    monkeypatch.setattr(
        observer,
        "load_study",
        lambda root: dict(name="fixture", cases=[{}] * 4, cap_gpu_hours=1),
    )
    monkeypatch.setattr(observer, "output_path", lambda root: out)
    monkeypatch.setattr(
        observer,
        "aggregate_report",
        lambda *args: dict(complete=True, raw_records_uploaded=False),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["track_action_controls.py", "--root", str(root), "--out", str(out)],
    )
    return SimpleNamespace(
        observer=observer,
        root=root,
        out=out,
        runs=runs,
        artifacts=artifacts,
        monkeypatch=monkeypatch,
    )


def ledger(fixture, status):
    (fixture.root / "execution.json").write_text(
        json.dumps(
            dict(
                status=status,
                cases={"a": dict(status="completed")},
                gpu_seconds=12,
                open={},
            )
        )
    )


def test_completed_observer_uploads_only_the_explicit_aggregate_files(observer_fixture):
    f = observer_fixture
    ledger(f, "completed")
    f.observer.main()
    assert len(f.artifacts) == 1
    assert set(f.artifacts[0].files) == {
        "aggregate-summary.json",
        "action-matched-policies.png",
        "action-matched-policies.pdf",
        "RESULTS.zh-CN.md",
    }
    assert all(run.finished for run in f.runs)
    receipt = json.loads((f.out / "archival.json").read_text())
    assert receipt["aggregate_only"] is True
    assert receipt["raw_records_uploaded"] is False


def test_failed_study_is_not_published_as_a_complete_result(observer_fixture):
    f = observer_fixture
    ledger(f, "failed")
    f.observer.main()
    assert len(f.runs) == 1 and not f.artifacts
    assert not (f.out / "archival.json").exists()
    assert json.loads((f.out / "finalization.json").read_text())["status"] == "stopped"


def test_report_recovery_runs_cpu_report_with_explicit_root_and_never_restarts_gpu(
    observer_fixture,
):
    f = observer_fixture
    ledger(f, "evaluation_completed")
    f.monkeypatch.setattr(
        f.observer, "process", lambda *args: dict(alive=False, identity_verified=False)
    )
    commands = []

    def recover(argv, **kwargs):
        commands.append(argv)
        assert argv[1:] == ["scripts/action_control_report.py", "--root", str(f.root)]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        ledger(f, "completed")

    f.monkeypatch.setattr(f.observer.subprocess, "run", recover)
    f.observer.main()
    assert len(commands) == 1
    assert (
        json.loads((f.out / "finalization.json").read_text())["cloud_status"]
        == "uploaded"
    )
