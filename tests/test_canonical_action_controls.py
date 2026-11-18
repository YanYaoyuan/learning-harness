"""Scientific input guards for the new scoring-to-training handoff."""

import importlib
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
study = importlib.import_module("canonical_action_controls")


def example():
    rows = [
        dict(
            task_id="train-only-example",
            messages=[dict(role="assistant", content='{"action":"look"}')],
        )
    ]
    scores = dict(
        data_sha256="corpus",
        scores=[
            dict(
                index=0,
                task_id=rows[0]["task_id"],
                content_sha256=study.digest_messages(rows[0]["messages"]),
                mean_target_nll=0.25,
            )
        ],
    )
    return rows, scores


def test_scores_require_content_identity_not_just_row_position():
    rows, scores = example()
    assert study.aligned_scores(scores, rows, "corpus") == [0.25]
    rows[0]["messages"][0]["content"] = '{"action":"inventory"}'
    with pytest.raises(ValueError, match="alignment"):
        study.aligned_scores(scores, rows, "corpus")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1])
def test_scores_reject_invalid_losses(value):
    rows, scores = example()
    scores["scores"][0]["mean_target_nll"] = value
    with pytest.raises(ValueError, match="value"):
        study.aligned_scores(scores, rows, "corpus")


def test_scores_bind_corpus_and_all_rows():
    rows, scores = example()
    with pytest.raises(ValueError, match="corpus"):
        study.aligned_scores(scores, rows, "other-corpus")
    with pytest.raises(ValueError, match="corpus"):
        study.aligned_scores(scores, rows * 2, "corpus")
    scores["scores"][0]["task_id"] = "other-task"
    with pytest.raises(ValueError, match="alignment"):
        study.aligned_scores(scores, rows, "corpus")


def test_dependency_adapter_is_explicit_and_preserves_source():
    source = (SCRIPTS / "action_control_worker.py").read_bytes()
    original = study.worker.load_recipe
    original_report = study.report.load_study
    try:
        study.install_kernel_adapter()
        assert study.worker.load_recipe is study.load_recipe
        assert study.report.load_study is study.load_study
        assert (SCRIPTS / "action_control_worker.py").read_bytes() == source
    finally:
        study.worker.load_recipe = original
        study.report.load_study = original_report
