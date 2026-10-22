"""Strict input/output schemas for the summarization endpoints."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SummarizeRequest(BaseModel):
    """Payload accepted by ``POST /summarize/invoke`` and the LangServe route."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "example": {"text": "A long article about renewable energy..."}
    })

    text: str = Field(..., min_length=1, description="Plain text to summarize.")

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must contain non-whitespace content")
        return v


class SummarizeResponse(BaseModel):
    """Standardized successful summarization result."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    model: str
    input_chars: int = Field(..., ge=0)
    request_id: str | None = None


class PdfSummarizeResponse(SummarizeResponse):
    """Summarization result enriched with the processed filename."""

    filename: str


class HealthResponse(BaseModel):
    """Liveness probe response."""

    status: Literal["ok"] = "ok"
    service: str
    version: str


class ReadinessResponse(BaseModel):
    """Readiness probe response including dependency state."""

    ready: bool
    checks: dict[str, str]
