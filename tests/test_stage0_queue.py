"""Scheduling guards, with no GPU, model server, or waiting process."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def queue(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/queue_stage0_after_source_comparison.py"
    spec = importlib.util.spec_from_file_location("stage0_queue_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("SOURCE", "SOURCE_OUTPUT", "ROOT", "QUEUE"):
        target = tmp_path / name.lower()
        target.mkdir()
        monkeypatch.setattr(module, name, target)
    return module


def ready_report(queue):
    queue.save(queue.SOURCE_OUTPUT / "summary.json", dict(
        complete=True, token_and_budget_audit_passed=True, completed_evaluation_task_runs=4284))
    queue.save(queue.SOURCE_OUTPUT / "finalization.json", dict(complete=True))


def test_live_source_blocks_even_if_old_reports_look_complete(queue):
    ready_report(queue)
    queue.save(queue.SOURCE / "execution.json", dict(status="running"))
    assert queue.source_ready() == (False, "waiting_for_source_pipeline")


def test_source_failure_blocks_stage0(queue):
    ready_report(queue)
    queue.save(queue.SOURCE / "execution.json", dict(status="failed"))
    assert queue.source_ready() == (False, "source_failed")


def test_missing_report_blocks_after_source_exits(queue):
    queue.save(queue.SOURCE / "execution.json", dict(status="completed"))
    assert queue.source_ready() == (False, "waiting_for_source_report_and_audit")


def test_complete_audited_source_allows_resource_check(queue):
    ready_report(queue)
    queue.save(queue.SOURCE / "execution.json", dict(status="completed"))
    assert queue.source_ready() == (True, "source_and_report_complete")


def test_incomplete_task_accounting_blocks(queue):
    ready_report(queue)
    queue.save(queue.SOURCE / "execution.json", dict(status="completed"))
    queue.save(queue.SOURCE_OUTPUT / "summary.json", dict(
        complete=True, token_and_budget_audit_passed=True, completed_evaluation_task_runs=4283))
    assert queue.source_ready()[0] is False


def test_changed_code_is_rejected_while_waiting(queue, tmp_path):
    target = tmp_path / "changed.py"
    target.write_text("new")
    with pytest.raises(ValueError, match="Changed while queued"):
        queue.verify_pins(dict(files={str(target): "old hash"}))


def test_run_step_preserves_virtual_environment_executable(queue, monkeypatch):
    observed = []
    monkeypatch.setattr(queue.subprocess, "run", lambda args, **kwargs: observed.append(args))
    queue.run_step("test-step", ["scripts/example.py"], dict(files={}, gpu=0))
    assert observed[0][0].endswith("/.venv-model/bin/python")


def test_busy_model_port_blocks_without_querying_gpu(queue, monkeypatch):
    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(queue.socket, "create_connection", lambda *args, **kwargs: Connection())
    monkeypatch.setattr(queue.subprocess, "run", lambda *args, **kwargs: pytest.fail("GPU query should not occur"))
    assert queue.resources_ready(0) is False
