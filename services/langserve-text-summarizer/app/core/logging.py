"""Structured JSON logging with correlation/request IDs.

Replaces all ``print()`` usage. Every log line is a single JSON object that
includes the request id propagated by :mod:`app.core.middleware`.
"""

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any

# Context variable holding the current request/correlation id.
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

_CORRELATION_ATTR = "_summarizer_correlation_handler_installed"


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON documents."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get(),
        }
        # Merge structured extras passed via ``logger.info(..., extra={"context": {...}})``
        context: Any = getattr(record, "context", None)
        if isinstance(context, dict):
            payload["context"] = context
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger (idempotent)."""
    root = logging.getLogger()
    root.setLevel(level.upper())

    handler = logging.StreamHandler(sys.stdout)
    if not hasattr(handler, _CORRELATION_ATTR):
        handler.setFormatter(JsonFormatter())
        setattr(handler, _CORRELATION_ATTR, True)
        # Avoid duplicate handlers when called more than once (e.g. reload/tests)
        for existing in list(root.handlers):
            root.removeHandler(existing)
        root.addHandler(handler)

    # Quieten noisy third-party loggers; our JSON stream is authoritative.
    for noisy in ("uvicorn.access", "httpx", "google"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger bound to the application hierarchy."""
    return logging.getLogger(name)
