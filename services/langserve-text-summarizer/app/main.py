"""FastAPI application factory with LangServe route, middleware and handlers."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from langserve import add_routes

from app import __version__
from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.exception_handlers import register_exception_handlers
from app.core.middleware import RequestContextMiddleware
from app.core.tracing import configure_tracing
from app.schemas.summarizer import SummarizeRequest
from app.services.llm import get_chain

logger = get_logger(__name__)

_PLAYGROUND_HTML = """
<html>
  <head><title>Text Summarization API</title></head>
  <body>
    <h1>LangServe Text Summarizer</h1>
    <p><a href="/summarize/playground/">Open the LangServe playground</a></p>
    <p><a href="/docs">Open the OpenAPI docs (Swagger UI)</a></p>
  </body>
</html>
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown lifecycle: configure logging, tracing, warm chain."""
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_tracing(settings)
    logger.info(
        "service_startup",
        extra={"context": {"service": settings.app_name, "version": __version__}},
    )
    # Fail fast on misconfiguration / warm up the chain.
    _ = get_chain()
    yield
    logger.info("service_shutdown")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application (testable factory)."""
    settings = get_settings()
    app = FastAPI(
        title="Text Summarization API",
        description=(
            "Production-grade text/PDF summarization powered by Google Gemini "
            "via a LangChain LCEL chain served through LangServe."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    @app.middleware("http")
    async def _attach_cors_on_errors(request, call_next):
        """Ensure CORS headers are present even on handled error responses."""
        origin = request.headers.get("origin")
        try:
            response = await call_next(request)
        except Exception:
            # Fallback envelope (handlers normally catch first).
            response = JSONResponse(
                status_code=500,
                content={"error": {
                    "code": "internal_error",
                    "message": "An unexpected internal error occurred.",
                    "status_code": 500,
                    "request_id": None,
                }},
            )
        if origin and "access-control-allow-origin" not in response.headers:
            origins = settings.cors_allow_origins_list
            allow = "*" if "*" in origins else (origin if origin in origins else "")
            if allow:
                response.headers["access-control-allow-origin"] = allow
                response.headers["vary"] = "Origin"
                response.headers["access-control-allow-headers"] = ",".join(
                    settings.cors_allow_headers_list) or "*"
                response.headers["access-control-allow-methods"] = ",".join(
                    settings.cors_allow_methods_list)
        return response

    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)

    # CORS: configured via environment (CORS_ALLOW_ORIGINS, etc.).
    # Default '*' so local file:// / static frontends can call the API.
    cors_origins = settings.cors_allow_origins_list
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials="*" not in cors_origins,
        allow_methods=settings.cors_allow_methods_list,
        allow_headers=settings.cors_allow_headers_list,
    )

    # Typed REST endpoints (/health, /ready, /summarize/text, /summarize-pdf)
    app.include_router(router)

    # LangServe route exposing the same chain with streaming/batch support.
    add_routes(
        app,
        get_chain(),
        path="/summarize",
        input_type=SummarizeRequest,
        per_req_config_modifier=None,
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def read_root() -> str:
        return _PLAYGROUND_HTML

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=get_settings().host, port=get_settings().port)
