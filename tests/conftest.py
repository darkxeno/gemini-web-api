"""Shared test setup."""
import pytest

from gemini_openai import metrics


@pytest.fixture(autouse=True)
def _request_log_in_tmp(tmp_path, monkeypatch):
    """Per-request metrics go to a temporary file, never ~/.gemini-web-api/requests.jsonl."""
    monkeypatch.setattr(metrics, "LOG_PATH", str(tmp_path / "requests.jsonl"))
