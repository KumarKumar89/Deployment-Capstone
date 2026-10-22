"""Exponential-backoff retry helpers for LLM and external tool calls.

Shared semantics with ``langserve-text-summarizer``: only transient network /
timeout / throttling / server errors are retried; everything else propagates
immediately. Timeouts themselves are enforced by the underlying clients.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, TypeVar

from tenacity import (
    AsyncRetrying,
    Retrying,
    before_sleep_log,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

_RETRYABLE_MODULE_PREFIXES = (
    "google.api_core",
    "google.genai",
    "httpcore",
    "httpx",
)
_RETRYABLE_NAME_HINTS = (
    "Timeout",
    "Transient",
    "ServiceUnavailable",
    "TooManyRequests",
    "InternalServerError",
    "Unavailable",
    "Connect",
    "Protocol",
    "Network",
)


def _is_retryable(exc: BaseException) -> bool:
    """True for transient network/timeout/server errors incl. google wrappers."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    # google.genai raises its own typed error hierarchy (ClientError/APIError).
    from google.genai import errors as genai_errors

    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", 0) or 0
        return int(code) in (429, 500, 502, 503, 504) or int(code) >= 500
    module = type(exc).__module__ or ""
    name = type(exc).__name__
    if module.startswith(_RETRYABLE_MODULE_PREFIXES) and any(
        hint in name for hint in _RETRYABLE_NAME_HINTS
    ):
        return True
    return False


def _retryable_state(retry_state: Any) -> bool:
    """Tenacity callback: retry only transient errors."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    return exc is not None and _is_retryable(exc)


def with_retry_sync(
    fn: Callable[..., T],
    *args: Any,
    max_attempts: int = 4,
    **kwargs: Any,
) -> T:
    """Call ``fn`` synchronously with exponential backoff on transient errors.

    Args:
        fn: Callable to invoke (e.g. ``genai_client.models.generate_content``).
        *args: Positional arguments forwarded to ``fn``.
        max_attempts: Maximum number of attempts (>= 1).
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        Whatever ``fn`` returns on the first successful attempt.

    Raises:
        BaseException: The last transient failure after the budget is exhausted
            (reraised verbatim); non-retryable errors propagate immediately.
    """
    retryer = Retrying(
        stop=stop_after_attempt(max(1, max_attempts)),
        wait=wait_exponential_jitter(initial=0.5, max=10.0),
        retry=_retryable_state,
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    return retryer(fn, *args, **kwargs)


async def with_retry_async(
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    max_attempts: int = 4,
    **kwargs: Any,
) -> T:
    """Await ``fn`` with exponential backoff on transient errors.

    Same semantics as :func:`with_retry_sync` but for coroutine callables.
    """
    retryer = AsyncRetrying(
        stop=stop_after_attempt(max(1, max_attempts)),
        wait=wait_exponential_jitter(initial=0.5, max=10.0),
        retry=_retryable_state,
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    return await retryer(fn, *args, **kwargs)
