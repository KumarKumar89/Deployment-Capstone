"""Research pipeline service — hardened replacement for legacy ``utils.py``.

All console formatting (``rich``) and ``print()`` statements were removed;
stage progress and metrics now flow through structured JSON logging. Every
Gemini call goes through :func:`app.services.gemini.generate_content`, which
enforces timeouts and tenacity exponential-backoff retries.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from google.genai import types

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.services.audio import save_podcast_audio
from app.services.gemini import (
    extract_grounding_sources,
    extract_inline_audio,
    extract_text,
    generate_content,
)

logger = get_logger(__name__)

_FILENAME_STRIP_RE = re.compile(r"[^A-Za-z0-9 _-]")


@dataclass(frozen=True)
class SearchOutcome:
    """Typed result of the web-research stage."""

    search_text: str
    search_sources_text: str


def slugify_topic(topic: str, max_stem_length: int = 80) -> str:
    """Derive a filesystem-safe podcast stem from an arbitrary topic string."""
    normalized = unicodedata.normalize("NFKD", topic).encode("ascii", "ignore").decode()
    cleaned = _FILENAME_STRIP_RE.sub("", normalized).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)[:max_stem_length].rstrip("_")
    return cleaned or "topic"


def run_web_research(topic: str, settings: Settings | None = None) -> SearchOutcome:
    """Grounded Google-Search research on ``topic`` via Gemini."""
    cfg = settings or get_settings()
    logger.info("stage_started", extra={"context": {"stage": "search_research"}})
    response = generate_content(
        "search_research",
        settings=cfg,
        model=cfg.search_model,
        contents=f"Research this topic and give me an overview: {topic}",
        config={
            "tools": [{"google_search": {}}],
            "temperature": cfg.search_temperature,
        },
    )
    outcome = SearchOutcome(
        search_text=extract_text(response),
        search_sources_text=extract_grounding_sources(response),
    )
    logger.info(
        "stage_completed",
        extra={
            "context": {
                "stage": "search_research",
                "chars": len(outcome.search_text),
                "sources": len(outcome.search_sources_text),
            }
        },
    )
    return outcome


def analyze_video(video_url: str, topic: str, settings: Settings | None = None) -> str:
    """Analyze a video (e.g. YouTube URL) with Gemini multi-modal input."""
    cfg = settings or get_settings()
    logger.info("stage_started", extra={"context": {"stage": "analyze_video"}})
    response = generate_content(
        "analyze_video",
        settings=cfg,
        model=cfg.video_model,
        contents=types.Content(
            parts=[
                types.Part(file_data=types.FileData(file_uri=video_url)),
                types.Part(
                    text=(
                        "Based on the video content, give me an overview "
                        f"of this topic: {topic}"
                    )
                ),
            ]
        ),
    )
    text = extract_text(response)
    logger.info(
        "stage_completed",
        extra={"context": {"stage": "analyze_video", "chars": len(text)}},
    )
    return text


def synthesize_report(
    topic: str,
    search_text: str,
    video_text: str,
    search_sources_text: str,
    video_url: str | None,
    settings: Settings | None = None,
) -> tuple[str, str]:
    """Synthesize multi-modal findings into ``(report_markdown, synthesis_text)``."""
    cfg = settings or get_settings()
    logger.info("stage_started", extra={"context": {"stage": "create_report"}})
    synthesis_prompt = f"""
    You are a research analyst. I have gathered information about "{topic}" from two sources:

    SEARCH RESULTS:
    {search_text}

    VIDEO CONTENT:
    {video_text}

    Please create a comprehensive synthesis that:
    1. Identifies key themes and insights from both sources
    2. Highlights any complementary or contrasting perspectives
    3. Provides an overall analysis of the topic based on this multi-modal research
    4. Keep it concise but thorough (3-4 paragraphs)

    Focus on creating a coherent narrative that brings together the best insights from both sources.
    """
    response = generate_content(
        "create_report",
        settings=cfg,
        model=cfg.synthesis_model,
        contents=synthesis_prompt,
        config={"temperature": cfg.synthesis_temperature},
    )
    synthesis_text = extract_text(response)
    report = f"""# Research Report: {topic}

## Executive Summary

{synthesis_text}

## Video Source
- **URL**: {video_url or "Not provided"}

## Additional Sources
{search_sources_text or "None"}

---
*Report generated using multi-modal AI research combining web search and video analysis*
"""
    logger.info(
        "stage_completed",
        extra={"context": {"stage": "create_report", "chars": len(report)}},
    )
    return report, synthesis_text


def create_podcast(
    topic: str,
    search_text: str,
    video_text: str,
    search_sources_text: str,
    video_url: str | None,
    filename: str,
    settings: Settings | None = None,
) -> tuple[str, str]:
    """Generate a 2-speaker podcast script and its TTS audio.

    Returns ``(podcast_script, filename)``; the WAV is persisted inside the
    dedicated media output directory (never the CWD root).
    """
    cfg = settings or get_settings()
    logger.info("stage_started", extra={"context": {"stage": "create_podcast"}})

    script_prompt = f"""
    Create a natural, engaging podcast conversation between Dr. Sarah (research expert) and Mike (curious interviewer) about "{topic}".

    Use this research content:

    SEARCH FINDINGS:
    {search_text}

    VIDEO INSIGHTS:
    {video_text}

    Format as a dialogue with:
    - Mike introducing the topic and asking questions
    - Dr. Sarah explaining key concepts and insights
    - Natural back-and-forth discussion (5-7 exchanges)
    - Mike asking follow-up questions
    - Dr. Sarah synthesizing the main takeaways
    - Keep it conversational and accessible (3-4 minutes when spoken)

    Format exactly like this:
    Mike: [opening question]
    Dr. Sarah: [expert response]
    Mike: [follow-up]
    Dr. Sarah: [explanation]
    [continue...]
    """
    script_response = generate_content(
        "podcast_script",
        settings=cfg,
        model=cfg.synthesis_model,
        contents=script_prompt,
        config={"temperature": cfg.podcast_script_temperature},
    )
    podcast_script = extract_text(script_response)

    tts_prompt = (
        f"TTS the following conversation between Mike and Dr. Sarah:\n{podcast_script}"
    )
    tts_response = generate_content(
        "podcast_tts",
        settings=cfg,
        model=cfg.tts_model,
        contents=tts_prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(
                            speaker="Mike",
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=cfg.mike_voice,
                                )
                            ),
                        ),
                        types.SpeakerVoiceConfig(
                            speaker="Dr. Sarah",
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=cfg.sarah_voice,
                                )
                            ),
                        ),
                    ]
                )
            ),
        ),
    )
    pcm = extract_inline_audio(tts_response)
    save_podcast_audio(filename, pcm, settings=cfg)

    logger.info(
        "stage_completed",
        extra={
            "context": {"stage": "create_podcast", "file": filename, "audio_bytes": len(pcm)}
        },
    )
    return podcast_script, filename
