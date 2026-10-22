"""Global exception handlers returning standardized JSON error envelopes.

Registered explicitly (not via the generic ``Exception`` handler) so responses
never leak stack traces, regardless of environment.
"""

from __future__ import annotations

import asyncio
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from app.core.logging import get_logger, request_id_ctx

logger = get_logger(__name__)


def _envelope(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "status_code": status_code,
                "request_id": request_id_ctx.get(),
            }
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach explicit handlers for all expected failure classes."""
    from app.core.middleware import AppError  # local import to avoid cycle

    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.warning("app_error", extra={"context": {"path": request.url.path, "status": exc.status_code}})
        return _envelope(exc.status_code, "app_error", exc.message)

    @app.exception_handler(HTTPException)
    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return _envelope(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _envelope(422, "validation_error", "Request validation failed.")

    @app.exception_handler(ValidationError)
    async def _pydantic_error(request: Request, exc: ValidationError) -> JSONResponse:
        return _envelope(422, "validation_error", "Payload validation failed.")

    @app.exception_handler(asyncio.TimeoutError)
    @app.exception_handler(TimeoutError)
    async def _timeout_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error("upstream_timeout", extra={"context": {"path": request.url.path}})
        return _envelope(504, "upstream_timeout", "The upstream model call timed out.")

    @app.exception_handler(ConnectionError)
    @app.exception_handler(requests.ConnectionError)
    async def _connection_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error("upstream_unavailable", extra={"context": {"path": request.url.path}})
        return _envelope(503, "upstream_unavailable", "The upstream model provider is unavailable.")

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Full detail goes to structured logs only; clients see a safe message.
        logger.exception("unhandled_exception", extra={"context": {"path": request.url.path}})
        return _envelope(500, "internal_error", "An unexpected internal error occurred.")
