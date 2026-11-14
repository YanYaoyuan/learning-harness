import pytest

from learning_harness.training_log_metrics import training_metrics


def test_carriage_returns_and_progress_bars_do_not_change_logged_steps():
    text = "\r10/192 [00:01]\r{'loss': 0.2, 'grad_norm': 1.1, 'learning_rate': 1e-4, 'epoch': 0.05}\n\r20/192\r{'loss': 0.1, 'grad_norm': 0.7, 'learning_rate': 9e-5, 'epoch': 0.1}\n{'train_runtime': 10.0, 'train_loss': 0.15}"
    rows = training_metrics(text)
    assert [r["step"] for r in rows] == [10, 20]
    assert [r["loss"] for r in rows] == [0.2, 0.1]


def test_telemetry_refuses_ambiguous_extra_training_history():
    line = "{'loss': 0.2, 'grad_norm': 1.0, 'learning_rate': 0.0001, 'epoch': 1.0}\n"
    with pytest.raises(ValueError, match="extra/resumed"):
        training_metrics(line * 20)


def test_telemetry_is_not_an_arbitrary_log_evaluator():
    with pytest.raises(ValueError):
        training_metrics(
            "{'loss': float('nan'), 'grad_norm': 1, 'learning_rate': 0, 'epoch': 1}"
        )
