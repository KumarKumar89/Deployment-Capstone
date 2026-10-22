"""LangGraph research workflow (hardened port of legacy ``agent/graph.py``).

Topology::

    START -> search_research --(video_url?)--> analyze_video -> create_report
                                   \\---------------------------> create_report
          create_report -> create_podcast -> END

Nodes delegate all provider I/O to :mod:`app.services.research` (timeout +
tenacity retry + structured JSON logging) and are traced with LangSmith when
enabled via environment variables.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.graph.configuration import Configuration
from app.graph.state import ResearchState, ResearchStateInput, ResearchStateOutput
from app.services.research import (
    analyze_video,
    create_podcast,
    run_web_research,
    slugify_topic,
    synthesize_report,
)

logger = get_logger(__name__)


def _to_settings(configuration: Configuration) -> Settings:
    """Project a RunnableConfig-resolved Configuration onto fresh Settings.

    Keeps per-run overrides (models/temperatures/voices from LangGraph
    ``configurable`` or env vars) effective inside the service layer while the
    service layer remains typed against the central pydantic Settings model.
    """
    base = get_settings().model_dump()
    base.update(
        {
            "search_model": configuration.search_model,
            "synthesis_model": configuration.synthesis_model,
            "video_model": configuration.video_model,
            "tts_model": configuration.tts_model,
            "search_temperature": configuration.search_temperature,
            "synthesis_temperature": configuration.synthesis_temperature,
            "podcast_script_temperature": configuration.podcast_script_temperature,
            "mike_voice": configuration.mike_voice,
            "sarah_voice": configuration.sarah_voice,
            "tts_channels": configuration.tts_channels,
            "tts_rate": configuration.tts_rate,
            "tts_sample_width": configuration.tts_sample_width,
        }
    )
    return Settings(**base)


@traceable(run_type="tool", name="Web Research", project_name="multi-modal-researcher")
def search_research_node(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    """Perform grounded web-search research on the topic."""
    configuration = Configuration.from_runnable_config(config)
    topic = state["topic"]
    started = time.perf_counter()
    outcome = run_web_research(topic, settings=_to_settings(configuration))
    logger.info(
        "node_completed",
        extra={
            "context": {
                "node": "search_research",
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        },
    )
    return {"search_text": outcome.search_text, "search_sources_text": outcome.search_sources_text}


@traceable(run_type="tool", name="YouTube Video Analysis", project_name="multi-modal-researcher")
def analyze_video_node(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    """Analyze video content when a URL was supplied."""
    _ = Configuration.from_runnable_config(config)
    video_url = state.get("video_url")
    topic = state["topic"]
    if not video_url:
        return {"video_text": "No video provided for analysis."}
    started = time.perf_counter()
    video_text = analyze_video(str(video_url), topic)
    logger.info(
        "node_completed",
        extra={
            "context": {
                "node": "analyze_video",
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        },
    )
    return {"video_text": video_text}


@traceable(run_type="tool", name="Create Report", project_name="multi-modal-researcher")
def create_report_node(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    """Synthesize search + video findings into a markdown report."""
    _ = Configuration.from_runnable_config(config)
    started = time.perf_counter()
    report, synthesis_text = synthesize_report(
        topic=state["topic"],
        search_text=state.get("search_text") or "",
        video_text=state.get("video_text") or "",
        search_sources_text=state.get("search_sources_text") or "",
        video_url=state.get("video_url"),
    )
    logger.info(
        "node_completed",
        extra={
            "context": {
                "node": "create_report",
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        },
    )
    return {"report": report, "synthesis_text": synthesis_text}


@traceable(run_type="tool", name="Create Podcast", project_name="multi-modal-researcher")
def create_podcast_node(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    """Generate the two-speaker script and its TTS audio artifact."""
    _ = Configuration.from_runnable_config(config)
    topic = state["topic"]
    filename = f"research_podcast_{slugify_topic(topic)}.wav"
    started = time.perf_counter()
    podcast_script, podcast_filename = create_podcast(
        topic=topic,
        search_text=state.get("search_text") or "",
        video_text=state.get("video_text") or "",
        search_sources_text=state.get("search_sources_text") or "",
        video_url=state.get("video_url"),
        filename=filename,
    )
    logger.info(
        "node_completed",
        extra={
            "context": {
                "node": "create_podcast",
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        },
    )
    return {"podcast_script": podcast_script, "podcast_filename": podcast_filename}


def should_analyze_video(state: ResearchState) -> str:
    """Conditional edge: route to video analysis only when a URL exists."""
    return "analyze_video" if state.get("video_url") else "create_report"


def create_research_graph() -> StateGraph:
    """Construct the uncompiled research workflow graph."""
    graph = StateGraph(
        ResearchState,
        input=ResearchStateInput,
        output=ResearchStateOutput,
        config_schema=Configuration,
    )

    graph.add_node("search_research", search_research_node)
    graph.add_node("analyze_video", analyze_video_node)
    graph.add_node("create_report", create_report_node)
    graph.add_node("create_podcast", create_podcast_node)

    graph.add_edge(START, "search_research")
    graph.add_conditional_edges(
        "search_research",
        should_analyze_video,
        {"analyze_video": "analyze_video", "create_report": "create_report"},
    )
    graph.add_edge("analyze_video", "create_report")
    graph.add_edge("create_report", "create_podcast")
    graph.add_edge("create_podcast", END)

    return graph


def create_compiled_graph() -> Any:
    """Compile the research graph (entrypoint used by tests and the API)."""
    return create_research_graph().compile()
