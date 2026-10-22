"""Strict Pydantic v2 input/output models for the research API."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]*\.wav$")


def is_safe_podcast_filename(name: str) -> bool:
    """True only for plain ``.wav`` filenames with no path components."""
    return bool(_SAFE_FILENAME_RE.match(name)) and "/" not in name and "\\" not in name


class ResearchRequest(BaseModel):
    """Input payload for a multi-modal research run."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    topic: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Research topic driving web search, video analysis and synthesis.",
    )
    video_url: HttpUrl | None = Field(
        default=None,
        description="Optional YouTube/video URL analyzed alongside web search.",
    )

    @field_validator("topic")
    @classmethod
    def _topic_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("topic must contain non-whitespace characters")
        return v


class ResearchResult(BaseModel):
    """Final artifacts produced by the LangGraph workflow."""

    report: str = Field(description="Markdown research report.")
    synthesis_text: str = Field(description="Executive synthesis paragraph(s).")
    podcast_script: str = Field(description="Two-speaker podcast dialogue script.")
    podcast_filename: str = Field(description="Generated .wav filename in the media dir.")


class ResearchResponse(BaseModel):
    """Envelope returned by ``POST /research/run``."""

    result: ResearchResult
    video_analyzed: bool
    duration_ms: float
    request_id: str


class HealthResponse(BaseModel):
    """Liveness payload for ``GET /health``."""

    status: str = "ok"
    service: str
    version: str


class ReadinessResponse(BaseModel):
    """Readiness payload for ``GET /ready`` including dependency checks."""

    ready: bool
    checks: dict[str, str]
