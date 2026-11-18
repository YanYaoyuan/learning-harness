import json

import pytest

from learning_harness.webshop_agent import run_episode
from learning_harness.webshop_public import PublicView
from learning_harness.webshop_requests import RequestJournal, summarize_requests


class Episode:
    def reset(self):
        return PublicView("Public task", True, ())

    def close(self):
        self.closed = True


def test_failed_transport_is_durable_before_call_and_never_retried(tmp_path):
    path = tmp_path / "requests.jsonl"
    episode = Episode()

    class Failure:
        calls = 0

        def complete(self, *args, **kwargs):
            self.calls += 1
            # Re-read from disk while inside the transport, not callback memory.
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            assert len(rows) == 1 and rows[0]["kind"] == "attempt"
            raise TimeoutError("Private response details must not be journaled")

    client = Failure()
    with RequestJournal(path) as journal, pytest.raises(TimeoutError):
        run_episode(client, episode, "Example.", on_request=journal)
    result = summarize_requests(path)
    assert client.calls == 1 and episode.closed
    assert result["attempted_requests"] == result["unknown_usage_requests"] == 1
    assert result["error_records"] == 1 and result["pending_requests"] == 0
    assert not result["complete_usage"]
    assert "Private response" not in path.read_text()


def test_over_budget_response_retains_actual_known_usage(tmp_path):
    path = tmp_path / "requests.jsonl"

    class Overrun:
        def complete(self, *args, **kwargs):
            return dict(text="think[plan]", input_tokens=20, output_tokens=101)

    with RequestJournal(path) as journal, pytest.raises(ValueError):
        run_episode(Overrun(), Episode(), "Example.", on_request=journal)
    result = summarize_requests(path)
    assert result["known_output_tokens"] == 101
    assert result["complete_usage"] and result["attempted_requests"] == 1


def test_lost_or_incomplete_terminal_does_not_become_zero_cost(tmp_path):
    path = tmp_path / "requests.jsonl"
    with RequestJournal(path) as journal:
        journal(dict(kind="attempt", turn=0, messages=[]))
    with path.open("ab") as handle:
        handle.write(b'{"kind":"response","turn":0,')
    result = summarize_requests(path)
    assert result["truncated_tail"] and not result["complete_usage"]
    assert result["pending_requests"] == result["unknown_usage_requests"] == 1
    with pytest.raises(FileExistsError):
        RequestJournal(path)


def test_missing_usage_and_duplicate_attempt_are_rejected(tmp_path):
    path = tmp_path / "requests.jsonl"
    with RequestJournal(path) as journal:
        journal(dict(kind="attempt", turn=0, messages=[]))
        journal(dict(kind="response", turn=0, response={"text": "think[plan]"}))
        with pytest.raises(ValueError, match="retried"):
            journal(dict(kind="attempt", turn=0, messages=[]))
    result = summarize_requests(path)
    assert result["response_records"] == 1 and result["unknown_usage_requests"] == 1
    assert not result["complete_usage"]
