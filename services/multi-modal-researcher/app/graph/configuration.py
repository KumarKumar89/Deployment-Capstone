"""LangGraph ``config_schema`` — values sourced from env settings, never hardcoded.

Mirrors the legacy dataclass configuration but pulls defaults from the central
pydantic :class:`app.core.config.Settings`, and coerces environment overrides
to the correct types (the legacy version passed raw strings into float fields).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

from app.core.config import get_settings


@dataclass(kw_only=True)
class Configuration:
    """Runtime configuration for the multi-modal research graph."""

    # Model settings (env-overridable: SEARCH_MODEL, SYNTHESIS_MODEL, ...)
    search_model: str
    synthesis_model: str
    video_model: str
    tts_model: str

    # Generation tuning
    search_temperature: float
    synthesis_temperature: float
    podcast_script_temperature: float

    # TTS configuration
    mike_voice: str
    sarah_voice: str
    tts_channels: int
    tts_rate: int
    tts_sample_width: int

    @classmethod
    def default(cls) -> "Configuration":
        """Build a Configuration from centralized settings."""
        s = get_settings()
        return cls(
            search_model=s.search_model,
            synthesis_model=s.synthesis_model,
            video_model=s.video_model,
            tts_model=s.tts_model,
            search_temperature=s.search_temperature,
            synthesis_temperature=s.synthesis_temperature,
            podcast_script_temperature=s.podcast_script_temperature,
            mike_voice=s.mike_voice,
            sarah_voice=s.sarah_voice,
            tts_channels=s.tts_channels,
            tts_rate=s.tts_rate,
            tts_sample_width=s.tts_sample_width,
        )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration from a RunnableConfig / environment.

        Precedence: explicit ``configurable`` value > environment variable >
        centralized settings default. Values are coerced to the declared
        dataclass field type.
        """
        base = cls.default()
        configurable: dict[str, Any] = (
            config["configurable"] if config and "configurable" in config else {}
        )
        resolved: dict[str, Any] = {}
        for f in fields(cls):
            if not f.init:
                continue
            value: Any = configurable.get(f.name)
            if value is None:
                env_raw = os.environ.get(f.name.upper())
                if env_raw is not None:
                    value = _coerce(env_raw, f.type)
            resolved[f.name] = base.__dict__[f.name] if value is None else value
        return cls(**resolved)


def _coerce(raw: str, target_type: Any) -> Any:
    """Best-effort coercion of an env string to the annotated field type."""
    t = str(target_type)
    try:
        if t in ("float", "<class 'float'>"):
            return float(raw)
        if t in ("int", "<class 'int'>"):
            return int(raw)
    except (TypeError, ValueError):
        pass
    return raw
