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


def with_retry_sync(
    fn: Callable[..., T],
    *args: Any,
    max_attempts: int = 3,
    **kwargs: Any,
) -> T:
    """Call ``fn`` synchronously with exponential backoff on transient errors.

    Args:
        fn: Callable to invoke (e.g. an LCEL chain's ``invoke``).
        *args: Positional arguments forwarded to ``fn``.
        max_attempts: Maximum number of attempts (>= 1).
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        Whatever ``fn`` returns on the first successful attempt.

    Raises:
        tenacity.RetryError: If all attempts fail with retryable errors.
        BaseException: Non-retryable exceptions propagate immediately.
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
    max_attempts: int = 3,
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


