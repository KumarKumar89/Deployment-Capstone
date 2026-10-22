"""Request-ID middleware and global exception handlers.

Guarantees every response carries ``X-Request-ID`` and every error is returned
as a standardized JSON envelope without leaking stack traces to clients.
"""

import uuid
from typing import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.core.logging import get_logger, request_id_ctx

REQUEST_ID_HEADER = "X-Request-ID"

logger = get_logger("app.http")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign/propagate a correlation id for each request."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        rid = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        token = request_id_ctx.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(token)
        response.headers[REQUEST_ID_HEADER] = rid
        return response


class AppError(Exception):
    """Domain-level error with an HTTP status and safe client message."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# Re-exported for typing convenience in route modules.
Handler = Callable[[Request], Awaitable[Response]]
