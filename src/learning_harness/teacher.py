from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import uuid

from .api import ChatClient
from .common import digest, game_path, read_jsonl
from .environment import AlfWorld

DEFAULT_PROMPT = (
    "Write a concise reusable strategy for the household task below. "
    "Use only this public observation. Do not claim hidden object locations. "
    "Give planning/verification guidance, not a numbered oracle action trace. "
    "The same advice will be reused after several actions; avoid assuming no action has occurred. "
    'Return JSON {"hint": "at most 100 words"}.')


class BudgetLedger:
    """Reserve a conservative charge before every paid request; failed calls stay reserved."""

    def __init__(self, path, settings):
        self.path, self.settings = Path(path), settings
        for key in ("input_cny_per_million", "output_cny_per_million", "max_cost_cny"):
            value = settings.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Set a positive {key}; no assumed API prices")
        if not isinstance(settings.get("max_calls"), int) or settings["max_calls"] <= 0:
            raise ValueError("Set a positive integer max_calls")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def locked(self):
        with self.path.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0)
            value = json.loads(handle.read() or '{"calls": []}')
            try:
                yield value
            finally:
                handle.seek(0)
                handle.truncate()
                handle.write(json.dumps(value, indent=2) + "\n")
                handle.flush()
                fcntl.flock(handle, fcntl.LOCK_UN)

    def reserve(self, messages, max_output):
        # UTF-8 byte count plus message framing is deliberately conservative for
        # standard byte-based tokenizers. A failed request is not auto-refunded.
        upper_input = len(json.dumps(messages, ensure_ascii=False).encode()) + 256
        amount = (upper_input * self.settings["input_cny_per_million"] +
                  max_output * self.settings["output_cny_per_million"]) / 1_000_000
        token = uuid.uuid4().hex
        with self.locked() as value:
            allocated = sum(row.get("actual_cny", row["reserved_cny"]) for row in value["calls"])
            if len(value["calls"]) >= self.settings["max_calls"] or allocated + amount > self.settings["max_cost_cny"]:
                raise ValueError("Teacher call/cost budget exhausted before request")
            value["calls"].append(dict(id=token, reserved_cny=amount, status="reserved"))
        return token

    def settle(self, token, response):
        actual = (response["input_tokens"] * self.settings["input_cny_per_million"] +
                  response["output_tokens"] * self.settings["output_cny_per_million"]) / 1_000_000
        with self.locked() as value:
            row = next(r for r in value["calls"] if r["id"] == token)
            row.update(actual_cny=actual, status="completed")
            if actual > row["reserved_cny"]:
                raise ValueError("Actual provider accounting exceeded reservation; stop and reconcile")


def build_teacher(manifest, data_root, settings_path, output, ledger_path):
    tasks = read_jsonl(manifest)
    if any(r["purpose"] == "final_evaluation" or r["partition"] not in {"pilot", "train", "dev"} for r in tasks):
        raise ValueError("Teacher generation is restricted to pilot/train/dev, never probe/final tests")
    settings = json.loads(Path(settings_path).read_text())
    ledger, client = BudgetLedger(ledger_path, settings), ChatClient(settings)
    prompt = settings.get('system_prompt', DEFAULT_PROMPT)
    policy_fingerprint = digest(dict(model=client.model,endpoint_hash=digest(client.base_url),
        prompt=prompt,extra_body=client.extra_body,max_output_tokens=settings['max_output_tokens']))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(output) if output.exists() else []
    cached = {r["task_id"]: r for r in existing}
    if len(cached) != len(existing):
        raise ValueError("Duplicate teacher cards")
    with output.open("a") as handle:
        for task in tasks:
            if task["task_id"] in cached:
                if cached[task["task_id"]]["game_sha256"] != task["game_sha256"]:
                    raise ValueError("Teacher cache game version mismatch")
                if cached[task['task_id']].get('policy_fingerprint') != policy_fingerprint:
                    raise ValueError('Teacher cache policy/model mismatch')
                continue
            env = AlfWorld(game_path(data_root, task), seed=0)
            try:
                initial = env.reset()
            finally:
                env.close()
            messages = [{"role": "system", "content": prompt},
                        {"role": "user", "content": initial}]
            reservation = ledger.reserve(messages, settings["max_output_tokens"])
            response = client.complete(messages, settings["max_output_tokens"], 0.0)
            ledger.settle(reservation, response)
            hint = json.loads(response["text"])["hint"]
            if not isinstance(hint, str) or not hint.strip() or len(hint) > 1600:
                raise ValueError("Invalid teacher hint; charge remains in ledger")
            row = dict(task_id=task["task_id"], game_sha256=task["game_sha256"],
                       public_reset_sha256=digest(initial), hint=hint,
                       teacher_model=client.model, teacher_response_model=response["response_model"],
                       policy_fingerprint=policy_fingerprint,response_id=response.get('response_id'),
                       usage={k: response[k] for k in ("input_tokens", "output_tokens")},
                       source="public_reset_only")
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"Cached teacher hint: {task['task_id']}", flush=True)
    return dict(cards=len(read_jsonl(output)))
