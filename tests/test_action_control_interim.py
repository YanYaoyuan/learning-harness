import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def interim_fixture(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    tool = importlib.import_module("action_control_interim")
    root, out = tmp_path / "study", tmp_path / "output"
    root.mkdir()
    out.mkdir()
    cases = [
        dict(
            name=f"{init}-{source}-{policy}",
            initialization_seed=init,
            sampling_seed=101,
            source=source,
            policy=policy,
            repeat=0,
            gpu=gpu,
            schedule="fixture.json",
        )
        for init in (101, 202)
        for source, gpu in (("api", 0), ("generic", 1))
        for policy in ("random", "min_nll")
    ]
    frozen = dict(
        protocol="task-action-controls-v1",
        teacher_calls=0,
        official_tests_used=False,
        primary_metric="mean_success",
        primary_contrast="min_nll_minus_random",
        gpus=[0, 1],
        cap_gpu_hours=1,
        allocation_seconds=1200,
        minimum_allocation_seconds=600,
        planned_episodes=512,
        cases=cases,
        interim_cell_milestones=[1],
    )
    (root / "freeze.json").write_text(json.dumps(frozen))
    sha = tool.file_sha(root / "freeze.json")

    def complete_case(index):
        case = cases[index]
        path = root / "cases" / case["name"] / "worker-result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(dict(case=case, complete=True, smoke=False, freeze_sha256=sha))
        )

    for i in range(7):
        complete_case(i)
    collections, artifacts = [], []

    def collect(root, subset):
        collections.append(len(subset["cases"]))
        rows = [
            dict(
                c,
                mean_success=0.2 if c["policy"] == "random" else 0.3,
                cool_success=0.2 if c["policy"] == "random" else 0.3,
                heat_success=0.2 if c["policy"] == "random" else 0.3,
                mean_nll=0.5,
            )
            for c in subset["cases"]
        ]
        return rows, [], [{}] * subset["planned_episodes"]

    def render(rows, folder, title):
        for suffix in ("png", "pdf"):
            (folder / f"action-matched-policies.{suffix}").write_text(title)

    def cpu_snapshot(argv, **kwargs):
        assert argv[1:] == [
            "scripts/action_control_interim.py",
            "--root",
            str(root),
            "--node",
            "1",
        ]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        tool.create_snapshot(root, out, frozen, 1)

    monkeypatch.setattr(tool, "collect", collect)
    monkeypatch.setattr(tool, "render_policies", render)
    monkeypatch.setattr(tool.subprocess, "run", cpu_snapshot)

    class Artifact:
        def __init__(self, *args, **kwargs):
            self.files = []
            artifacts.append(self)

        def add_file(self, path, name):
            self.files.append(name)

    upload = dict(fail=True)

    def wait():
        if upload["fail"]:
            raise TimeoutError("fixture upload interruption")

    run = SimpleNamespace(
        id="fixture",
        url="https://example.invalid",
        summary={},
        log=lambda item: None,
        log_artifact=lambda artifact: SimpleNamespace(wait=wait),
    )
    wandb = SimpleNamespace(Artifact=Artifact, Image=lambda path: path)
    return SimpleNamespace(
        tool=tool,
        root=root,
        out=out,
        frozen=frozen,
        complete_case=complete_case,
        collections=collections,
        artifacts=artifacts,
        run=run,
        wandb=wandb,
        upload=upload,
    )


def test_upload_retry_keeps_original_snapshot_when_more_models_finish(interim_fixture):
    f = interim_fixture
    f.tool.observe_interim(f.root, f.out, f.frozen, f.run, f.wandb)
    folder = f.out / "interim/node-01"
    before = (folder / "aggregate-summary.json").read_bytes()
    initial = json.loads(before)
    assert initial["actual_paired_cells"] == 1
    assert initial["models"] == 4 and initial["completed_models"] == 7
    assert initial["complete"] is False and initial["subset_complete"] is True
    f.complete_case(7)
    f.upload["fail"] = False
    f.tool.observe_interim(f.root, f.out, f.frozen, f.run, f.wandb)
    f.tool.observe_interim(f.root, f.out, f.frozen, f.run, f.wandb)
    assert (folder / "aggregate-summary.json").read_bytes() == before
    assert f.collections == [4]
    assert len(f.artifacts) == 2
    assert all(
        set(a.files)
        == {
            "aggregate-summary.json",
            "action-matched-policies.png",
            "action-matched-policies.pdf",
        }
        for a in f.artifacts
    )
    state = json.loads((f.out / "interim/observer-state.json").read_text())
    assert state["looks"]["1"]["cloud_status"] == "uploaded"


def test_modified_cached_snapshot_is_not_silently_uploaded(interim_fixture):
    f = interim_fixture
    f.tool.observe_interim(f.root, f.out, f.frozen, f.run, f.wandb)
    path = f.out / "interim/node-01/action-matched-policies.png"
    path.write_text("changed cached figure")
    f.upload["fail"] = False
    with pytest.raises(ValueError, match="snapshot changed"):
        f.tool.observe_interim(f.root, f.out, f.frozen, f.run, f.wandb)
    assert len(f.artifacts) == 1
