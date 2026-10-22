"""Exponential-backoff retry helpers for LLM and external tool calls.

Wraps tenacity so call-sites stay declarative; timeouts are enforced by the
underlying clients (ChatGoogleGenerativeAI request timeout / httpx).
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, TypeVar

from tenacity import (
    AsyncRetrying,
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _is_retryable(exc: BaseException) -> bool:
    """True for transient network/timeout errors, incl. google-api-core wrappers."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    module = type(exc).__module__ or ""
    name = type(exc).__name__
    if module.startswith("google.api_core") and (
        "Timeout" in name
        or "Transient" in name
        or "ServiceUnavailable" in name
        or "TooManyRequests" in name
    ):
        return True
    if module.startswith(("httpcore", "httpx")) and (
        "Timeout" in name or "Connect" in name or "Protocol" in name
    ):
        return True
    return False


def _retryable_state(retry_state: Any) -> bool:
    """Tenacity callback: retry only transient errors."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    return exc is not None and _is_retryable(exc)


