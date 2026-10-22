"""Optional LangSmith / OpenTelemetry tracing wired via environment variables.

When ``LANGSMITH_TRACING=true`` (and an API key is present) traces are exported
to LangSmith. When ``OTEL_ENABLED=true`` an OTLP tracer provider is installed.
Both are no-ops in local/test environments so the service never crashes due to
missing observability backends.
"""

from __future__ import annotations

import os

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def configure_tracing(settings: Settings) -> None:
    """Enable LangSmith and/or OpenTelemetry based on configuration."""
    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        logger.info("langsmith_tracing_enabled", extra={"context": {"project": settings.langsmith_project}})

    if settings.otel_enabled and settings.otel_exporter_otlp_endpoint:
        try:
            from opentelemetry import trace  # noqa: PLC0415
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider = TracerProvider()
            processor = BatchSpanProcessor(
                OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint)
            )
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)
            logger.info("otel_tracing_enabled", extra={"context": {"endpoint": settings.otel_exporter_otlp_endpoint}})
        except ImportError:
            logger.warning("otel_packages_missing_skipping_tracing")
