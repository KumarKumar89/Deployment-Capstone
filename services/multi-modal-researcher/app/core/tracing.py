"""Optional LangSmith / OpenTelemetry tracing configured purely via env vars.

- LangSmith: set ``LANGSMITH_TRACING=true`` (+ ``LANGSMITH_API_KEY``) to export
  every graph node (they are wrapped with ``@traceable``) to a LangSmith project.
- OpenTelemetry: set ``OTEL_ENABLED=true`` (+ OTLP endpoint) to install generic
  LLM instrumentation when the optional ``opentelemetry`` extras are present.
"""

from __future__ import annotations

import os

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def configure_tracing(settings: Settings) -> None:
    """Wire observability backends from settings (best-effort, non-fatal)."""
    if settings.langsmith_tracing:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
        if settings.langsmith_api_key is not None:
            os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key.get_secret_value()
        logger.info(
            "langsmith_tracing_enabled",
            extra={"context": {"project": settings.langsmith_project}},
        )
    else:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

    if settings.otel_enabled:
        try:  # Optional dependency group: [otel]
            from opentelemetry import trace  # noqa: F401
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            provider = TracerProvider()
            exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
            logger.info(
                "otel_tracing_enabled",
                extra={"context": {"endpoint": settings.otel_exporter_otlp_endpoint}},
            )
        except ImportError:
            logger.warning(
                "otel_dependencies_missing",
                extra={"context": {"hint": "install extras: pip install .[otel]"}},
            )
