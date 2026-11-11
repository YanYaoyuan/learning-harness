"""Credential loading must not execute input or alter local student endpoints."""
from pathlib import Path

import pytest

from learning_harness.api import ChatClient
from learning_harness.credentials import load_teacher_env, read_literal_env


def test_literal_env_never_executes_shell_or_overrides_exported_settings(tmp_path, monkeypatch):
    marker=tmp_path/'must-not-exist'
    source=tmp_path/'local.env'
    source.write_text(f"LH_TEACHER_API_KEY='$(touch {marker})'\n"
        "LH_TEACHER_MODEL=from-file\nLH_STUDENT_BASE_URL=https://wrong.invalid\n")
    monkeypatch.delenv('LH_TEACHER_API_KEY',raising=False)
    monkeypatch.setenv('LH_TEACHER_MODEL','exported-model')
    monkeypatch.setenv('LH_STUDENT_BASE_URL','http://127.0.0.1:8000/v1')
    assert load_teacher_env(source)==['LH_TEACHER_API_KEY']
    assert not marker.exists()
    import os
    assert os.environ['LH_TEACHER_MODEL']=='exported-model'
    assert os.environ['LH_STUDENT_BASE_URL']=='http://127.0.0.1:8000/v1'


def test_env_parse_errors_do_not_reveal_values(tmp_path):
    source=tmp_path/'local.env'
    source.write_text('LH_TEACHER_API_KEY=private-value unquoted-value\n')
    with pytest.raises(ValueError) as error:read_literal_env(source)
    assert 'private-value' not in str(error.value)


def test_extra_fields_cannot_override_request_budget(monkeypatch):
    monkeypatch.setenv('TEST_URL','https://example.invalid/v1')
    monkeypatch.setenv('TEST_MODEL','test-only')
    with pytest.raises(ValueError,match='Unsupported'):
        ChatClient(dict(base_url_env='TEST_URL',model_env='TEST_MODEL',api_key_env='TEST_KEY',
            extra_body={'max_tokens':999999}))
