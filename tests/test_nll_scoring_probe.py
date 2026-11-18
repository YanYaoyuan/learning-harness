import importlib.util

import pytest

from learning_harness.nll_scoring_probe import (
    make_batch,
    prediction_positions,
    rank_correlation,
    select_stress,
    summarize,
    trials_for,
    validate_encoded,
)
from learning_harness.webshop_requests import RequestJournal, summarize_requests


def encoded():
    return dict(
        input_ids=[1, 2, 3, 4], attention_mask=[1, 1, 1, 1], labels=[-100, -100, 3, 4]
    )


def test_padding_masks_and_positions_leave_supervised_suffix_unchanged():
    row = encoded()
    assert validate_encoded(row) == 2
    batch = make_batch([row], 9, extra_padding=5, explicit_positions=True)
    assert batch["attention_mask"][0] == [0] * 5 + [1] * 4
    assert batch["position_ids"][0] == [0] * 5 + [0, 1, 2, 3]
    assert batch["labels"][0] == [-100] * 7 + [3, 4]
    assert prediction_positions(batch["labels"][0], 3) == ([0, 1], [3, 4])
    assert prediction_positions(row["labels"], 3) == ([0, 1], [3, 4])
    assert "position_ids" not in make_batch([row], 9)


def test_short_logits_window_and_invalid_labels_are_rejected():
    with pytest.raises(ValueError, match="omitted"):
        prediction_positions(encoded()["labels"], 2)
    row = encoded()
    row["labels"][1] = 6
    with pytest.raises(ValueError, match="suffix"):
        validate_encoded(row)
    with pytest.raises(ValueError, match="context limit"):
        make_batch([encoded()], 9, extra_padding=2048)


def test_sentinel_selection_is_fixed_and_requires_distinct_messages():
    entries = [
        dict(
            content_sha256=f"{i:064x}",
            observations=[dict(cached_nll=0.0), dict(cached_nll=float(i))],
        )
        for i in range(64)
    ]
    selected = select_stress(list(reversed(entries)))
    assert selected[:16] == [f"{i:064x}" for i in range(63, 47, -1)]
    assert selected[16:] == [f"{i:064x}" for i in range(16)]
    with pytest.raises(ValueError):
        select_stress(entries[:31] + [entries[0]])


def test_context_exclusions_are_declared_before_a_forward():
    row = encoded()
    row = {k: [1] * 2000 if k != "labels" else [-100] * 1999 + [1] for k in row}
    trials = trials_for(dict(encoded=row))
    assert {r["name"] for r in trials if not r["feasible"]} == {
        "pad128_default",
        "pad128_explicit",
    }


def test_alignment_and_repeat_gates_do_not_depend_on_score_direction():
    stress = [
        dict(content_sha256=str(i), condition=condition, nll=0.2)
        for i in range(32)
        for condition in ("canonical", "canonical_repeat")
    ]
    fp32 = [dict(content_sha256=str(i), nll=0.1) for i in range(32)]
    direct = [dict(content_sha256=str(i), absolute_difference=1e-7) for i in range(4)]
    result = summarize(stress, fp32, direct)
    assert result["optional_rescoring_gate_passed"]
    assert result["bf16_vs_fp32_max_absolute_difference"] == pytest.approx(0.1)
    direct[0]["absolute_difference"] = 0.001
    assert not summarize(stress, fp32, direct)["optional_rescoring_gate_passed"]
    with pytest.raises(ValueError, match="Duplicate stress"):
        summarize(stress + [stress[0]], fp32, direct)


def test_rank_correlation_handles_ties_and_constant_scores():
    assert rank_correlation([1, 1, 3], [2, 2, 4]) == pytest.approx(1.0)
    assert rank_correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert rank_correlation([1, 1], [2, 3]) is None


def runner():
    spec = importlib.util.spec_from_file_location(
        "probe_runner_test", "scripts/run_nll_scoring_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_partial_private_journal_preserves_unknown_forward_usage(tmp_path):
    module = runner()
    path = tmp_path / "forwards.private"
    with RequestJournal(path) as journal:
        journal(dict(kind="attempt", turn=0, phase="fp32"))
        journal(
            dict(
                kind="response",
                turn=0,
                response=dict(
                    phase="fp32",
                    scores=[dict(content_sha256="a", nll=0.5)],
                    input_tokens=4,
                    padded_input_positions=4,
                    output_tokens=0,
                ),
            )
        )
        journal(dict(kind="attempt", turn=1, phase="stress"))
    with path.open("ab") as handle:
        handle.write(b'{"kind":"response"')
    state = module.recover(path)
    assert state["fp32"] == [dict(content_sha256="a", nll=0.5)]
    usage = summarize_requests(path)
    assert usage["unknown_usage_requests"] == 1 and usage["pending_requests"] == 1
    assert usage["known_input_tokens"] == 4 and usage["truncated_tail"]


def test_uniform_rescoring_cannot_retry_and_double_count_a_content(tmp_path):
    module = runner()
    path = tmp_path / "forwards.private"
    with RequestJournal(path) as journal:
        for turn in range(2):
            journal(dict(kind="attempt", turn=turn))
            journal(
                dict(
                    kind="response",
                    turn=turn,
                    response=dict(
                        phase="uniform",
                        scores=[dict(content_sha256="same", nll=0.5)],
                        input_tokens=4,
                        padded_input_positions=4,
                        output_tokens=0,
                    ),
                )
            )
    with pytest.raises(ValueError, match="retried"):
        module.recover(path)


def test_parent_gate_precedes_any_queue_or_gpu_call(tmp_path, monkeypatch):
    module = runner()
    monkeypatch.setattr(module, "validate", lambda root: {})

    def parent():
        raise ValueError("Stage4 still running")

    monkeypatch.setattr(module, "require_parent_completed", parent)
    monkeypatch.setattr(
        module, "bounded_run", lambda *args: pytest.fail("GPU queue must not start")
    )
    with pytest.raises(ValueError, match="still running"):
        module.run(tmp_path)
    assert not (tmp_path / "execution.json").exists()
