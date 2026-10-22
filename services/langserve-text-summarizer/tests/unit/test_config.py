"""Unit tests for pydantic-settings configuration."""

import pytest

from app.core.config import Settings


def test_defaults_applied() -> None:
    s = Settings(google_api_key="k")
    assert s.gemini_model == "gemini-2.0-flash"
    assert s.llm_max_retries == 3


def test_log_level_validated() -> None:
    with pytest.raises(ValueError):
        Settings(google_api_key="k", log_level="VERBOSE")


def test_api_key_is_secret_and_not_repr() -> None:
    s = Settings(google_api_key="super-secret")
    assert "super-secret" not in repr(s)
    assert s.google_api_key.get_secret_value() == "super-secret"
