"""Local-only, append-only request accounting, including lost responses."""

import json
import os
from pathlib import Path
import time


class RequestJournal:
    """One exclusive journal per episode; prompts/responses never go to cloud."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("x", encoding="utf-8")
        os.chmod(self.path, 0o600)
        self.pending = set()
        self.finished = set()

    def __call__(self, event):
        kind, turn = event["kind"], event["turn"]
        if type(turn) is not int or turn < 0:
            raise ValueError("Invalid request turn")
        if kind == "attempt":
            if turn in self.pending or turn in self.finished:
                raise ValueError("A request attempt may not be retried")
        elif kind in ("response", "error"):
            if turn not in self.pending:
                raise ValueError("Response/error without a pending attempt")
        else:
            raise ValueError("Unknown request journal event")
        self.handle.write(json.dumps(dict(event, time_unix=time.time())) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        if kind == "attempt":
            self.pending.add(turn)
        else:
            self.pending.remove(turn)
            self.finished.add(turn)

    def close(self):
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def summarize_requests(path):
    """Recover durable counts; missing/truncated terminal records stay unknown.

    Token totals are explicitly *known* totals. They must not be described as
    complete whenever unknown_usage_requests is nonzero.
    """
    attempts, terminals = {}, {}
    truncated_tail = False
    path = Path(path)
    if path.exists():
        lines = path.read_bytes().splitlines(keepends=True)
        for index, raw in enumerate(lines):
            if not raw.endswith(b"\n") and index == len(lines) - 1:
                truncated_tail = True
                break
            row = json.loads(raw)
            turn = row["turn"]
            if row["kind"] == "attempt":
                if turn in attempts:
                    raise ValueError("Duplicate request attempt")
                attempts[turn] = row
            elif row["kind"] in ("response", "error"):
                if turn not in attempts or turn in terminals:
                    raise ValueError("Unpaired or duplicate terminal event")
                terminals[turn] = row
            else:
                raise ValueError("Unknown journal event")
    known_input = known_output = known = 0
    errors = 0
    for row in terminals.values():
        if row["kind"] == "error":
            errors += 1
            continue
        response = row.get("response", {})
        if isinstance(response, dict) and all(
            type(response.get(k)) is int and response[k] >= 0
            for k in ("input_tokens", "output_tokens")
        ):
            known += 1
            known_input += response["input_tokens"]
            known_output += response["output_tokens"]
    return dict(
        attempted_requests=len(attempts),
        response_records=sum(r["kind"] == "response" for r in terminals.values()),
        error_records=errors,
        pending_requests=len(attempts) - len(terminals),
        known_usage_requests=known,
        unknown_usage_requests=len(attempts) - known,
        known_input_tokens=known_input,
        known_output_tokens=known_output,
        truncated_tail=truncated_tail,
        complete_usage=known == len(attempts) and not truncated_tail,
    )
