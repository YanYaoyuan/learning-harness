import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path("scripts").absolute()))
spec = importlib.util.spec_from_file_location(
    "fresh_calibration_test_module", "scripts/run_fresh_executor_calibration.py"
)
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)


class Client:
    model = "test"

    def complete_many(self, requests, *args):
        return dict(results=[dict(input_tokens=3, output_tokens=2) for _ in requests])


def test_journal_records_each_batch_request_before_terminal(tmp_path):
    path = tmp_path / "requests"
    with cal.RequestJournal(path) as journal:
        client = cal.JournalClient(Client(), journal)
        client.complete_many([{}, {}], 128, 0, "rev")
        client.complete_many([{}], 128, 0, "rev")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(r["kind"], r["turn"]) for r in rows] == [
        ("attempt", 0),
        ("attempt", 1),
        ("response", 0),
        ("response", 1),
        ("attempt", 2),
        ("response", 2),
    ]
    usage = cal.summarize_requests(path)
    assert usage["complete_usage"] and usage["known_input_tokens"] == 9


def test_failed_batch_preserves_unknown_usage(tmp_path):
    class Broken(Client):
        def complete_many(self, *args):
            raise TimeoutError("lost response")

    path = tmp_path / "requests"
    with cal.RequestJournal(path) as journal:
        with pytest.raises(TimeoutError):
            cal.JournalClient(Broken(), journal).complete_many([{}, {}], 128, 0, "rev")
    usage = cal.summarize_requests(path)
    assert not usage["complete_usage"]
    assert usage["unknown_usage_requests"] == usage["error_records"] == 2


def fixture_case(tmp_path):
    folder = tmp_path / "case"
    (folder / "evaluation").mkdir(parents=True)
    tasks, rows, events = [], [], []
    method = "json_direct"
    with cal.RequestJournal(folder / "model-requests.private") as journal:
        for i in range(32):
            task = dict(
                task_id=str(i), task_group=f"g{i}", task_type=cal.FAMILIES[i // 16]
            )
            tasks.append(task)
            rows.append(
                dict(
                    task,
                    run_id=str(i),
                    seed=11,
                    method=method,
                    success=i % 2 == 0,
                    trials=[{}],
                    calls=1,
                    input_tokens=3,
                    output_tokens=2,
                )
            )
            events.append(
                dict(
                    run_id=str(i),
                    task_id=str(i),
                    seed=11,
                    method=method,
                    trial=1,
                    phase="act",
                    call=1,
                    success=i % 2 == 0,
                    input_tokens=3,
                    output_tokens=2,
                )
            )
            journal(dict(kind="attempt", turn=i))
            journal(
                dict(
                    kind="response",
                    turn=i,
                    response=dict(input_tokens=3, output_tokens=2),
                )
            )
    for name, data in (("tasks", rows), ("events", events)):
        (folder / "evaluation" / (name + ".jsonl")).write_text(
            "".join(json.dumps(r) + "\n" for r in data)
        )
    case = dict(name=method, method=method, gpu=0)
    result = dict(
        complete=True,
        case=case,
        episodes=32,
        successes=16,
        mean_success=0.5,
        families={f: dict(tasks=16, successes=8) for f in cal.FAMILIES},
        invalid_format_decisions=0,
        thought_decisions=0,
        usage=cal.summarize_requests(folder / "model-requests.private"),
        results_sha256=cal.file_sha(folder / "evaluation/tasks.jsonl"),
        events_sha256=cal.file_sha(folder / "evaluation/events.jsonl"),
    )
    return folder, result, case, tasks


def test_report_rederives_aggregate_counts(tmp_path):
    folder, result, case, tasks = fixture_case(tmp_path)
    assert cal.audit_result(folder, result, case, tasks)["successes"] == 16
    result["successes"] += 1
    with pytest.raises(ValueError, match="Reported"):
        cal.audit_result(folder, result, case, tasks)


def test_report_rejects_event_corruption_even_if_totals_unchanged(tmp_path):
    folder, result, case, tasks = fixture_case(tmp_path)
    path = folder / "evaluation/events.jsonl"
    data = cal.read_jsonl(path)
    data[0]["task_id"] = data[1]["task_id"]
    path.write_text("".join(json.dumps(r) + "\n" for r in data))
    result["events_sha256"] = cal.file_sha(path)
    with pytest.raises(ValueError, match="Episode/event"):
        cal.audit_result(folder, result, case, tasks)
