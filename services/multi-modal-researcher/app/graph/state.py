"""Typed state channels for the research and podcast generation workflow."""

from __future__ import annotations

from typing import Optional

from typing_extensions import TypedDict


class ResearchStateInput(TypedDict):
    """Inputs accepted when invoking the graph."""

    topic: str
    video_url: Optional[str]


class ResearchStateOutput(TypedDict):
    """Final outputs produced by the graph."""

    report: Optional[str]
    synthesis_text: Optional[str]
    podcast_script: Optional[str]
    podcast_filename: Optional[str]


class ResearchState(TypedDict):
    """Full state of the research and podcast generation workflow."""

    # Input fields
    topic: str
    video_url: Optional[str]

    # Intermediate results
    search_text: Optional[str]
    search_sources_text: Optional[str]
    video_text: Optional[str]

    # Final outputs
    report: Optional[str]
    synthesis_text: Optional[str]
    podcast_script: Optional[str]
    podcast_filename: Optional[str]
