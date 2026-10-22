"""FastAPI application factory for the multi-modal researcher service.

Wraps the LangGraph workflow (``app.graph.graph:create_compiled_graph``) in a
typed, observable HTTP API with correlation-ID middleware and standardized
JSON error envelopes — matching the pattern used by ``langserve-text-summarizer``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import router
from app.core.config import get_settings
from app.core.exception_handlers import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.core.middleware import RequestContextMiddleware
from app.core.tracing import configure_tracing

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown lifecycle: configure logging, tracing, compile graph."""
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_tracing(settings)
    logger.info(
        "service_startup",
        extra={"context": {"service": settings.app_name, "version": __version__}},
    )
    # Fail fast on misconfiguration / warm the compiled graph.
    from app.services.runner import get_graph

    _ = get_graph()
    yield
    logger.info("service_shutdown")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application (testable factory)."""
    settings = get_settings()
    app = FastAPI(
        title="Multi-Modal Researcher API",
        description=(
            "Production-grade multi-modal research agent: LangGraph workflow "
            "combining grounded web search, YouTube video analysis, report "
            "synthesis and two-speaker podcast generation via Google Gemini."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)

    # CORS configured via environment (CORS_ALLOW_ORIGINS etc.).
    cors_origins = [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials="*" not in cors_origins,
        allow_methods=[m.strip() for m in settings.cors_allow_methods.split(",") if m.strip()],
        allow_headers=[h.strip() for h in settings.cors_allow_headers.split(",") if h.strip()],
    )

    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def read_root() -> dict[str, str]:
        return {
            "service": settings.app_name,
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()
