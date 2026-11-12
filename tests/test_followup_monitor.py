import json
from pathlib import Path

import pytest

from learning_harness.common import file_sha
from learning_harness.followup_monitor import cloud_metrics, safe_counts, snapshot
from learning_harness.webshop_requests import RequestJournal


def setup_study(tmp_path, monkeypatch, kind="nll"):
    monkeypatch.chdir(tmp_path)
    root = Path("runs/test-study")
    root.mkdir(parents=True)
    config = dict(
        config=dict(prompt_modes=["prompt1", "prompt1_actonly"]),
        cases=[dict(name="private-goal-123", mode="prompt1")],
    )
    (root / "freeze.json").write_text(json.dumps(config))
    spec = dict(
        root=str(root),
        freeze_sha256=file_sha(root / "freeze.json"),
        controller="frozen_runner.py",
        cap_gpu_seconds=600,
    )
    return root, spec


def live(pid, expected):
    return dict(pid=pid, alive=True, identity_verified=True)


def save(root, name, value):
    (root / name).write_text(json.dumps(value))


@pytest.mark.parametrize("kind", ["nll", "webshop"])
def test_no_ledger_is_waiting_and_does_not_create_an_execution(
    tmp_path, monkeypatch, kind
):
    root, spec = setup_study(tmp_path, monkeypatch, kind)
    result = snapshot(
        kind, spec, 100, lambda *_: pytest.fail("No process was launched")
    )
    assert result["public"]["state"] == "not_started"
    assert result["public"]["gpu_seconds_estimate"] == 0
    assert not result["public"]["started"] and not result["public"]["needs_attention"]
    assert not (root / "execution.json").exists()


def test_nll_pending_usage_counts_and_cloud_projection_hide_private_records(
    tmp_path, monkeypatch
):
    root, spec = setup_study(tmp_path, monkeypatch)
    save(
        root,
        "execution.json",
        dict(
            status="running",
            freeze_sha256=spec["freeze_sha256"],
            gpu_seconds=5,
            open={"private-case-name": dict(pid=22, started_unix=90)},
        ),
    )
    save(root, "job.json", dict(pipeline=dict(pid=11)))
    path = root / "cases/base-model-readonly-scoring/forwards.private"
    with RequestJournal(path) as journal:
        journal(dict(kind="attempt", turn=0, private_prompt="DO_NOT_UPLOAD"))
        journal(
            dict(
                kind="response",
                turn=0,
                response=dict(
                    phase="stress",
                    scores=[dict(content_sha256="PRIVATE_FINGERPRINT", nll=0.2)],
                    input_tokens=5,
                    output_tokens=0,
                ),
            )
        )
        journal(dict(kind="attempt", turn=1))
    with path.open("ab") as handle:
        handle.write(b'{"kind":"response"')
    result = snapshot("nll", spec, 100, live)
    public = result["public"]
    assert public["gpu_seconds_estimate"] == 15
    assert public["counts"]["stress_observations"] == 1
    assert public["counts"]["unknown_usage_requests"] == 1
    assert public["counts"]["pending_requests"] == 1
    assert public["counts"]["known_input_tokens"] == 5
    text = json.dumps(public)
    assert all(
        s not in text
        for s in (
            "DO_NOT_UPLOAD",
            "PRIVATE_FINGERPRINT",
            "private-case-name",
            "private-goal-123",
            'nll":',
        )
    )
    assert public["live_verified_processes"] == 2
    result["local_processes"]["private"] = "DO_NOT_UPLOAD"
    metrics = cloud_metrics(dict(nll=result))
    assert metrics["progress/nll/counts/stress_observations"] == 1
    assert "DO_NOT_UPLOAD" not in json.dumps(metrics)


def test_completed_ledger_without_parent_report_is_not_audited_complete(
    tmp_path, monkeypatch
):
    root, spec = setup_study(tmp_path, monkeypatch)
    save(
        root,
        "execution.json",
        dict(status="completed", freeze_sha256=spec["freeze_sha256"]),
    )
    public = snapshot("nll", spec, 100, live)["public"]
    assert public["terminal"] and not public["audited_complete"]
    assert public["needs_attention"]


def test_missing_live_worker_is_flagged_without_restart(tmp_path, monkeypatch):
    root, spec = setup_study(tmp_path, monkeypatch)
    save(
        root,
        "execution.json",
        dict(
            status="running",
            freeze_sha256=spec["freeze_sha256"],
            open={"case": dict(pid=22, started_unix=90)},
        ),
    )

    def dead(pid, expected):
        return dict(pid=pid, alive=False, identity_verified=False)

    assert snapshot("nll", spec, 100, dead)["public"]["needs_attention"]
    assert json.loads((root / "execution.json").read_text())["status"] == "running"


def test_webshop_active_elapsed_and_partial_call_accounting(tmp_path, monkeypatch):
    root, spec = setup_study(tmp_path, monkeypatch, "webshop")
    save(
        root,
        "execution.json",
        dict(
            status="running",
            pid=11,
            freeze_sha256=spec["freeze_sha256"],
            gpu_seconds=2,
            gpu_started_unix=90,
            children={},
        ),
    )
    with RequestJournal(root / "local/private-goal-123/requests.jsonl") as journal:
        journal(dict(kind="attempt", turn=0, prompt="PRIVATE_GOAL"))
    result = snapshot("webshop", spec, 100, live)["public"]
    assert result["gpu_seconds_estimate"] == 10
    assert result["counts"]["completed_episodes"] == 0
    assert result["counts"]["unknown_usage_requests"] == 1
    assert "PRIVATE_GOAL" not in json.dumps(result)
    ledger = json.loads((root / "execution.json").read_text())
    ledger.update(status="failed", stopped_unix=95, gpu_seconds=5)
    save(root, "execution.json", ledger)
    assert snapshot("webshop", spec, 1000, live)["public"]["gpu_seconds_estimate"] == 5


def test_changed_freeze_and_unrecognized_status_are_rejected(tmp_path, monkeypatch):
    root, spec = setup_study(tmp_path, monkeypatch)
    save(
        root,
        "execution.json",
        dict(status="PRIVATE_UNTRUSTED_TEXT", freeze_sha256=spec["freeze_sha256"]),
    )
    with pytest.raises(ValueError, match="execution state"):
        snapshot("nll", spec, 100, live)
    (root / "freeze.json").write_text("{}")
    with pytest.raises(ValueError, match="identity changed"):
        snapshot("nll", spec, 100, live)
    for invalid in (True, -1, 0.5, float("nan")):
        with pytest.raises(ValueError, match="operational count"):
            safe_counts(dict(attempted_requests=invalid))
