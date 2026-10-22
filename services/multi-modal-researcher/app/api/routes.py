"""Typed REST endpoints: health, readiness, research run, podcast download."""

from __future__ import annotations

import mimetypes

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app import __version__
from app.core.config import get_settings
from app.core.logging import get_logger, request_id_ctx
from app.core.middleware import AppError
from app.schemas.research import (
    HealthResponse,
    ReadinessResponse,
    ResearchRequest,
    ResearchResponse,
)
from app.services.audio import resolve_podcast_path
from app.services.runner import run_research

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe: process is up and serving."""
    settings = get_settings()
    return HealthResponse(service=settings.app_name, version=__version__)


@router.get("/ready", response_model=ReadinessResponse, tags=["ops"])
def ready() -> ReadinessResponse:
    """Readiness probe: verifies credentials and the media output directory."""
    settings = get_settings()
    checks: dict[str, str] = {"config": "ok"}

    try:
        checks["llm_credentials"] = "ok" if settings.api_key else "missing"
    except Exception:
        checks["llm_credentials"] = "error"

    try:
        media_dir = settings.resolved_media_output_dir
        media_dir.mkdir(parents=True, exist_ok=True)
        probe = media_dir / ".write-probe"
        probe.write_bytes(b"")
        probe.unlink()
        checks["media_directory"] = "ok"
    except Exception:
        checks["media_directory"] = "error"

    try:
        from app.services.runner import get_graph

        _ = get_graph()
        checks["graph_compiled"] = "ok"
    except Exception:
        checks["graph_compiled"] = "error"

    ok = all(v == "ok" for v in checks.values())
    return ReadinessResponse(ready=ok, checks=checks)


@router.post("/research/run", response_model=ResearchResponse, tags=["research"])
async def research_run(payload: ResearchRequest) -> ResearchResponse:
    """Run the LangGraph research workflow end-to-end for a topic.

    Performs grounded web search, optional YouTube video analysis, report
    synthesis and two-speaker podcast generation (script + TTS audio). The
    returned ``podcast_filename`` is downloadable via
    ``GET /research/podcast/{filename}``.
    """
    settings = get_settings()
    if len(payload.topic) > settings.max_topic_chars:
        raise AppError(
            f"Topic exceeds maximum of {settings.max_topic_chars} characters.",
            status_code=413,
        )
    result, video_analyzed, duration_ms = await run_research(payload)
    return ResearchResponse(
        result=result,
        video_analyzed=video_analyzed,
        duration_ms=duration_ms,
        request_id=request_id_ctx.get(),
    )


@router.get(
    "/research/podcast/{filename}",
    response_class=FileResponse,
    responses={404: {"description": "Podcast file not found"}},
    tags=["research"],
)
def download_podcast(filename: str) -> FileResponse:
    """Stream/download a generated ``.wav`` podcast from the media directory.

    Path traversal is impossible: the filename must be a plain ``.wav`` name
    and the resolved path is verified to stay inside the media dir.
    """
    path = resolve_podcast_path(filename)
    media_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
    logger.info(
        "podcast_download",
        extra={"context": {"file": path.name, "bytes": path.stat().st_size}},
    )
    return FileResponse(
        path=str(path),
        media_type=media_type,
        filename=path.name,
        headers={"X-Request-ID": request_id_ctx.get()},
    )
