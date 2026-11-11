"""CPU-only checks: never start a model or touch existing experiment directories."""
import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("stage0_runner_under_test", scripts / "run_stage0_repeat_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "stage0"
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "FREEZE", root / "freeze.json")
    monkeypatch.setattr(module, "LEDGER", root / "execution.json")
    monkeypatch.setattr(module, "SOURCE_ROOT", tmp_path / "historical")
    return module


@pytest.fixture
def fake_evaluation(runner, tmp_path, monkeypatch):
    historical = tmp_path / "cell"
    historical.mkdir()
    (historical / "run.json").write_text(json.dumps({"revision": "test-revision", "model": "test-model"}))
    monkeypatch.setattr(runner, "historical", lambda *args: historical)
    monkeypatch.setattr(runner, "composition_revision", lambda path: "test-revision")
    monkeypatch.setattr(runner, "is_complete", lambda *args: False)
    clock = SimpleNamespace(now=100.0, entered=0, exited=0)
    monkeypatch.setattr(runner, "time", SimpleNamespace(time=lambda: clock.now))

    @contextmanager
    def server(*args):
        clock.entered += 1
        clock.now += 7
        try:
            yield {}
        finally:
            clock.exited += 1
            clock.now += 3

    monkeypatch.setattr(runner.engine, "server", server)
    monkeypatch.setattr(runner, "summarize", lambda directory: {"complete": True, "metrics": {"none": {"error_count": 0}}})
    return clock


def test_smoke_is_one_cell_and_full_plan_is_replicate_major(runner):
    smoke = runner.plan_for("smoke")
    assert len(smoke) == 1 and smoke[0][3] == ("heat", "api", 1101)
    full = runner.plan_for("full")
    assert len(full) == 38
    assert [row[2] for row in full] == [1] * 12 + [2] * 12 + [3] * 12 + [0] * 2


@pytest.mark.parametrize("ending", ["completed", "failed", "timeout", "interrupted", "terminated"])
def test_every_attempt_charges_time_and_stops_its_server(runner, fake_evaluation, monkeypatch, ending):
    clock = fake_evaluation
    errors = {"failed": RuntimeError("synthetic"), "interrupted": KeyboardInterrupt(), "terminated": runner.Terminated()}

    def command(args, env, log, timeout):
        assert timeout == pytest.approx(4 * 3600 - 7)
        clock.now += timeout if ending == "timeout" else 11
        if ending == "timeout":
            raise runner.subprocess.TimeoutExpired(args, timeout)
        if ending in errors:
            raise errors[ending]

    monkeypatch.setattr(runner.engine, "command", command)
    ledger = runner.load_ledger()
    args = (ledger, 4, 0, runner.ROOT / "trial", {}, ("heat", "api", 1101))
    if ending in errors:
        with pytest.raises(type(errors[ending])):
            runner.attempt(*args)
    else:
        outcome = runner.attempt(*args)
        assert outcome == ("stopped_at_cap" if ending == "timeout" else "completed")
    assert clock.entered == clock.exited == 1
    expected = 4 * 3600 + 3 if ending == "timeout" else 21
    assert ledger["gpu_seconds"] == pytest.approx(expected)
    assert ledger["open"] is None
    saved = json.loads(runner.LEDGER.read_text())
    assert saved["gpu_seconds"] == ledger["gpu_seconds"]
    assert saved["attempts"][-1]["outcome"] == ("stopped_at_cap" if ending == "timeout" else "interrupted" if ending in ("interrupted", "terminated") else ending)


def test_insufficient_budget_starts_no_server(runner, fake_evaluation):
    ledger = runner.load_ledger()
    outcome = runner.attempt(ledger, .02, 0, runner.ROOT / "trial", {}, ("heat", "api", 1101))
    assert outcome == "stopped_at_cap"
    assert fake_evaluation.entered == 0 and ledger["gpu_seconds"] == 0


@pytest.fixture
def freeze_args(runner, tmp_path, monkeypatch):
    files = []
    for i in range(3):
        path = tmp_path / f"frozen{i}.txt"
        path.write_text(f"original {i}")
        files.append(str(path))
    monkeypatch.setattr(runner, "FROZEN_FILES", files)
    monkeypatch.setattr(runner, "historical_hash_mismatches", lambda: {})
    return SimpleNamespace(gpu=0, cap_gpu_hours=4.0, include_historical=False, attestation=None, hash_exception=[])


def test_unfrozen_run_is_refused(runner, freeze_args):
    with pytest.raises(SystemExit, match="missing"):
        runner.load_freeze(freeze_args)


def test_freeze_is_not_overwritten(runner, freeze_args):
    runner.write_freeze(freeze_args)
    before = runner.FREEZE.read_bytes()
    with pytest.raises(SystemExit, match="never overwritten"):
        runner.write_freeze(freeze_args)
    assert runner.FREEZE.read_bytes() == before


def test_freeze_detects_changed_code(runner, freeze_args):
    runner.write_freeze(freeze_args)
    Path(runner.FROZEN_FILES[0]).write_text("changed")
    with pytest.raises(SystemExit, match="Changed after freezing"):
        runner.load_freeze(freeze_args)


@pytest.mark.parametrize("field,value", [("gpu", 1), ("cap_gpu_hours", 5.0)])
def test_runtime_choices_cannot_override_freeze(runner, freeze_args, field, value):
    runner.write_freeze(freeze_args)
    setattr(freeze_args, field, value)
    with pytest.raises(SystemExit):
        runner.load_freeze(freeze_args)


def test_historical_choice_requires_attestation(runner, freeze_args):
    freeze_args.include_historical = True
    with pytest.raises(SystemExit, match="attestation"):
        runner.write_freeze(freeze_args)


def test_missing_historical_hash_list_is_not_silently_skipped(runner):
    with pytest.raises(ValueError, match="Missing historical hash list"):
        runner.historical_hash_mismatches()


def test_documented_report_exception_is_exact_and_execution_changes_are_blocked(runner, monkeypatch):
    path = "scripts/transfer_factorial_report.py"
    monkeypatch.setattr(runner, "historical_hash_mismatches", lambda: {path: ("old", "new")})
    entry = {"path": path, "recorded_sha256": "old", "current_sha256": "new", "reason": "figure audit"}
    assert runner.check_historical_hashes([entry]) == [path]
    entry["current_sha256"] = "different"
    with pytest.raises(ValueError, match="changed again"):
        runner.check_historical_hashes([entry])
    monkeypatch.setattr(runner, "historical_hash_mismatches", lambda: {"scripts/hf_batch_server.py": ("old", "new")})
    with pytest.raises(ValueError, match="execution code changed"):
        runner.check_historical_hashes(None)
