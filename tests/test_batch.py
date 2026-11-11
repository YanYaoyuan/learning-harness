"""Protocol checks for the batched runner; these fixtures are not model results."""
import json
from pathlib import Path

import pytest

from learning_harness.batch_runner import rollout
from learning_harness.common import digest, file_sha, read_jsonl, write_jsonl
from learning_harness.analysis import export_sft
from learning_harness.support import SUPPORT_PREFIX


class Environment:
    def __init__(self, game, seed, max_steps):
        self.limit = 5 if Path(game).name == "long" else 2
        self.turn = 0

    def reset(self):
        return "public"

    def step(self, action):
        self.turn += 1
        return "next", self.turn == self.limit, self.turn == self.limit

    def close(self):
        pass


class Client:
    model = "test-only"
    base_url = "http://127.0.0.1:9999/v1"

    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def complete_many(self, requests, max_tokens, temperature, revision):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("fixture interruption")
        assert revision == "test-revision"
        for request in requests:
            turn = request["seed"] - 11
            if turn >= 4:
                assert not any(m["content"].startswith(SUPPORT_PREFIX) for m in request["messages"])
        return dict(padded_prompt_tokens=20 * len(requests), results=[dict(
            text='{"action":"look"}', input_tokens=20, output_tokens=5,
            response_model=self.model) for _ in requests])


class AsyncEnvironment(Environment):
    supports_async_step = True

    def step_async(self, action):
        self.pending_action = action

    def step_wait(self):
        return super().step(self.pending_action)


@pytest.mark.parametrize("env_factory", [Environment, AsyncEnvironment])
def test_batch_slots_resume_orphans_and_support_expiry(tmp_path, monkeypatch, env_factory):
    monkeypatch.setenv("LH_STUDENT_REVISION", "test-revision")
    rows = []
    for name in ("short", "long", "other"):
        (tmp_path / name).write_text("fixture")
        rows.append(dict(task_id=name, task_group=name, purpose="development", partition="train",
            task_type="pick_and_place_simple", game_relpath=name, game_sha256=file_sha(tmp_path / name)))
    manifest = tmp_path / "tasks.jsonl"
    write_jsonl(manifest, rows)
    bank = tmp_path / "support.jsonl"
    write_jsonl(bank, [dict(task_id=r["task_id"], public_reset_sha256=digest("public"), hint="strategy") for r in rows])
    config = json.loads(Path("configs/pilot.json").read_text())
    config.update(manifest=str(manifest), teacher_bank=str(bank), arms=["none", "early"],
        rollout_seeds=[11], batch_size=2, max_steps=10, max_student_output_tokens=1280)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config))
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="interruption"):
        rollout(cfg, tmp_path, output, client=Client(fail_at=4), environment_factory=env_factory)
    for _ in range(2):
        result = rollout(cfg, tmp_path, output, client=Client(), environment_factory=env_factory)
    assert result["recorded"] == result["planned"] == 6
    assert len(read_jsonl(output / "steps.jsonl")) > 18  # abandoned partial attempts remain auditable
    export_sft(output, "early", tmp_path / "sft.jsonl")
    assert len(read_jsonl(tmp_path / "sft.jsonl")) == 9
    assert len({r["attempt_id"] for r in read_jsonl(output / "episodes.jsonl")}) == 6
    config["batch_size"] = 1
    cfg.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="fingerprint"):
        rollout(cfg, tmp_path, output, client=Client(), environment_factory=env_factory)


def test_batch_evaluation_rejects_support_before_requests(tmp_path):
    manifest = tmp_path / "tasks.jsonl"
    write_jsonl(manifest, [dict(task_id="probe", purpose="development", partition="probe")])
    cfg = json.loads(Path("configs/probe.json").read_text())
    cfg.update(manifest=str(manifest), arms=["early"])
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="H0"):
        rollout(path, tmp_path, tmp_path / "run", evaluation=True, client=Client(), environment_factory=Environment)
