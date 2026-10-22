"""Request-ID middleware and typed domain error shared across services.

Guarantees every response carries ``X-Request-ID`` and that domain failures
map onto the standardized JSON error envelope installed by
:mod:`app.core.exception_handlers`.
"""

import uuid
from typing import Awaitable, Callable

from fastapi import Request
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp

from app.core.logging import get_logger, request_id_ctx

REQUEST_ID_HEADER = "X-Request-ID"

logger = get_logger("app.http")


class RequestContextMiddleware:
    """Pure-ASGI middleware: assign/propagate a correlation id per request.

    Implemented at the ASGI level (rather than ``BaseHTTPMiddleware``) to avoid
    buffering/streaming pitfalls and to keep background tasks correlated.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers", [])}
        rid = headers.get(REQUEST_ID_HEADER.lower()) or str(uuid.uuid4())
        token = request_id_ctx.set(rid)

        async def send_with_request_id(message: dict) -> None:
            if message["type"] == "http.response.start":
                raw_headers = list(message.get("headers", []))
                raw_headers.append(
                    (REQUEST_ID_HEADER.lower().encode("latin-1"), rid.encode("latin-1"))
                )
                message = {**message, "headers": raw_headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            request_id_ctx.reset(token)


class AppError(Exception):
    """Domain-level error with an HTTP status and safe client message."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# Re-exported for typing convenience in route modules.
Handler = Callable[[Request], Awaitable[Response]]
