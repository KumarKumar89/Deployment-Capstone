"""Unit tests for exponential-backoff retry helpers."""

import pytest

from app.services.retry import with_retry_sync


def test_with_retry_sync_returns_first_success() -> None:
    calls: list[int] = []

    def fn() -> str:
        calls.append(1)
        return "ok"

    assert with_retry_sync(fn, max_attempts=3) == "ok"
    assert len(calls) == 1


def test_with_retry_sync_retries_then_succeeds() -> None:
    state = {"n": 0}

    def flaky() -> str:
        state["n"] += 1
        if state["n"] < 3:
            raise TimeoutError("transient")
        return "recovered"

    assert with_retry_sync(flaky, max_attempts=4) == "recovered"
    assert state["n"] == 3


def test_with_retry_sync_reraises_after_stop() -> None:
    def always_fails() -> None:
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        with_retry_sync(always_fails, max_attempts=2)


def test_non_retryable_raises_immediately() -> None:
    calls = {"n": 0}

    def boom() -> None:
        calls["n"] += 1
        raise ValueError("not transient")

    with pytest.raises(ValueError):
        with_retry_sync(boom, max_attempts=5)
    assert calls["n"] == 1
