"""Graph runner service: async execution of the compiled LangGraph workflow."""

from __future__ import annotations

import asyncio
import time
from functools import lru_cache
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.middleware import AppError
from app.graph.graph import create_compiled_graph
from app.schemas.research import ResearchRequest, ResearchResult

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_graph() -> Any:
    """Compile the research graph once per process (cheap, no I/O)."""
    return create_compiled_graph()


async def run_research(payload: ResearchRequest) -> tuple[ResearchResult, bool, float]:
    """Execute the LangGraph state machine for one research request.

    The synchronous graph (whose nodes perform blocking Gemini HTTP calls) is
    executed on a worker thread so the event loop stays responsive; the whole
    run is bounded by ``GRAPH_TIMEOUT_SECONDS``.

    Returns:
        ``(result, video_analyzed, duration_ms)``.

    Raises:
        AppError: 504 when the overall run exceeds the configured budget;
            other typed errors from the service layer propagate unchanged.
    """
    settings = get_settings()
    graph = get_graph()
    inputs: dict[str, Any] = {
        "topic": payload.topic,
        "video_url": str(payload.video_url) if payload.video_url else None,
    }
    started = time.perf_counter()
    logger.info(
        "graph_run_started",
        extra={"context": {"topic_chars": len(payload.topic), "has_video": bool(payload.video_url)}},
    )
    try:
        state: dict[str, Any] = await asyncio.wait_for(
            asyncio.to_thread(graph.invoke, inputs),
            timeout=settings.graph_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        logger.error(
            "graph_run_timeout",
            extra={"context": {"timeout_s": settings.graph_timeout_seconds}},
        )
        raise AppError(
            "The research workflow exceeded its maximum execution time.",
            status_code=504,
        ) from exc

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    missing = [k for k in ("report", "synthesis_text", "podcast_script", "podcast_filename")
               if not state.get(k)]
    if missing:
        logger.error("graph_run_incomplete", extra={"context": {"missing": missing}})
        raise AppError(
            "The research workflow completed without producing all artifacts.",
            status_code=502,
        )

    result = ResearchResult(
        report=state["report"],
        synthesis_text=state["synthesis_text"],
        podcast_script=state["podcast_script"],
        podcast_filename=state["podcast_filename"],
    )
    logger.info(
        "graph_run_completed",
        extra={
            "context": {
                "duration_ms": duration_ms,
                "video_analyzed": bool(payload.video_url),
                "podcast_file": result.podcast_filename,
            }
        },
    )
    return result, bool(payload.video_url), duration_ms
