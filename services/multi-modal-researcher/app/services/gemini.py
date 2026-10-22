"""Typed, retry-aware wrapper around the Google GenAI client.

Single choke point for every ``genai_client.models.generate_content`` call in
the workflow: enforces HTTP timeouts (via ``HttpOptions``), tenacity
exponential-backoff retries on transient errors, structured logging of stage
metrics, and safe extraction of text / grounding sources / inline audio.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

from google.genai import Client, types

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.middleware import AppError
from app.services.retry import with_retry_sync

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _client(settings: Settings) -> Client:
    """Build one shared GenAI client with a hard request timeout."""
    api_key = settings.api_key
    if not api_key:  # defensive; Settings validation normally prevents this.
        raise AppError("Gemini API key is not configured.", status_code=503)
    try:
        http_options: Any = types.HttpOptions(timeout=int(settings.llm_timeout_seconds))
    except Exception:  # pragma: no cover - SDK-shape fallback across versions
        http_options = {"timeout": int(settings.llm_timeout_seconds)}
    return Client(api_key=api_key, http_options=http_options)


def get_client() -> Client:
    """Return the process-wide GenAI client built from settings."""
    return _client(get_settings())


def extract_text(response: Any) -> str:
    """Extract the first non-empty text part from a GenerateContentResponse."""
    try:
        candidate = response.candidates[0]
    except (AttributeError, IndexError) as exc:
        raise AppError("The model returned an empty response.", status_code=502) from exc

    parts = getattr(candidate.content, "parts", None) or []
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            return text
    raise AppError("The model returned no text content.", status_code=502)


def extract_grounding_sources(response: Any) -> str:
    """Render grounding chunks (web citations) as a numbered plain-text list."""
    candidate = getattr(response, "candidates", [None])[0]
    metadata = getattr(candidate, "grounding_metadata", None) if candidate else None
    chunks = getattr(metadata, "grounding_chunks", None) if metadata else None
    if not chunks:
        return ""
    lines: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        web = getattr(chunk, "web", None)
        title = (getattr(web, "title", None) if web else None) or "No title"
        uri = (getattr(web, "uri", None) if web else None) or "No URI"
        lines.append(f"{i}. {title}\n   {uri}")
    logger.info(
        "grounding_sources_extracted",
        extra={"context": {"source_count": len(lines)}},
    )
    return "\n".join(lines)


def extract_inline_audio(response: Any) -> bytes:
    """Return raw PCM bytes from the first inline-data part of a TTS response."""
    try:
        parts = response.candidates[0].content.parts
    except (AttributeError, IndexError) as exc:
        raise AppError("TTS returned no audio candidates.", status_code=502) from exc
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline is not None and getattr(inline, "data", None):
            return bytes(inline.data)
    raise AppError("TTS response contained no inline audio data.", status_code=502)


def generate_content(
    stage: str,
    *,
    model: str,
    contents: Any,
    config: Any = None,
    settings: Settings | None = None,
) -> Any:
    """Call Gemini with timeout + exponential-backoff retries + stage metrics.

    Args:
        stage: Logical workflow stage name (logged, never leaked to clients).
        model: Model resource name (comes from settings, never hardcoded here).
        contents: String prompt or ``types.Content`` for multi-modal input.
        config: Optional generation config dict or ``GenerateContentConfig``.
        settings: Optional per-run settings override (timeout/retry budget);
            defaults to the process-wide cached settings.

    Returns:
        The raw ``GenerateContentResponse``.

    Raises:
        AppError: 504 on timeout after retries, 503 on upstream unavailability,
            502 when the provider returns an unusable payload.
    """
    cfg = settings or get_settings()
    client = _client(cfg)
    started = time.perf_counter()
    logger.info(
        "gemini_call_started",
        extra={"context": {"stage": stage, "model": model}},
    )
    try:
        response = with_retry_sync(
            client.models.generate_content,
            model=model,
            contents=contents,
            config=config,
            max_attempts=cfg.llm_max_attempts,
        )
    except (TimeoutError,) as exc:
        logger.error("gemini_call_timeout", extra={"context": {"stage": stage}})
        raise AppError(
            f"The '{stage}' model call timed out after retries.", status_code=504
        ) from exc
    except (ConnectionError, Exception) as exc:
        # Non-AppError transport/provider failures become typed 503s.
        if isinstance(exc, AppError):
            raise
        logger.exception(
            "gemini_call_failed",
            extra={"context": {"stage": stage, "error_type": type(exc).__name__}},
        )
        raise AppError(
            f"The '{stage}' model call failed after retries.", status_code=503
        ) from exc

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "gemini_call_completed",
        extra={"context": {"stage": stage, "model": model, "duration_ms": duration_ms}},
    )
    return response
