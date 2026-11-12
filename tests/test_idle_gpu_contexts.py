import io
import json
from types import SimpleNamespace

import pytest

from learning_harness import idle_gpu_contexts as context


@pytest.fixture
def fixture(monkeypatch):
    identity = dict(
        pid=123,
        start_ticks=456,
        uid=context.os.getuid(),
        executable="/python",
        scripts=["/fixture/ComfyUI/main.py"],
    )
    record = dict(identity, role="idle_comfyui", queue_port=8188, max_memory_mib=1024)
    monkeypatch.setattr(
        context,
        "process_identity",
        lambda pid: (identity, ["python", identity["scripts"][0], "--port", "8188"]),
    )
    state = dict(queue_running=[], queue_pending=[])
    monkeypatch.setattr(
        context.urllib.request,
        "build_opener",
        lambda *args: SimpleNamespace(
            open=lambda *args, **kwargs: io.BytesIO(json.dumps(state).encode())
        ),
    )
    return record, state


def test_recorded_idle_context_is_allowed_without_modifying_service(fixture):
    record, _ = fixture
    assert context.validate_idle_context(record, 496)["queue_running"] == 0


@pytest.mark.parametrize("change", ["queued", "running", "identity", "memory", "role"])
def test_busy_changed_or_unrecognized_background_is_rejected(fixture, change):
    record, state = fixture
    used = 496
    if change == "queued":
        state["queue_pending"] = ["private payload never returned"]
    elif change == "running":
        state["queue_running"] = ["private payload never returned"]
    elif change == "identity":
        record["start_ticks"] += 1
    elif change == "memory":
        used = 2048
    else:
        record["role"] = "other"
    with pytest.raises(ValueError):
        context.validate_idle_context(record, used)
