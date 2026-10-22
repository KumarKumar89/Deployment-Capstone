"""Centralized environment-variable management using pydantic-settings.

All runtime knobs (API keys, model names, temperatures, TTS voices, timeouts,
retry budgets, output directory) are sourced from the environment / .env file.
Nothing is hardcoded in application code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Service metadata ---------------------------------------------------
    app_name: str = "multi-modal-researcher"
    environment: str = Field(default="production")
    log_level: str = Field(default="INFO")

    # --- Server ---------------------------------------------------------------
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # --- CORS -------------------------------------------------------------------
    cors_allow_origins: str = Field(default="*")
    cors_allow_methods: str = Field(default="GET,POST,OPTIONS")
    cors_allow_headers: str = Field(default="*")

    # --- Credentials (GEMINI_API_KEY takes precedence over GOOGLE_API_KEY) ------
    gemini_api_key: SecretStr | None = Field(default=None, repr=False)
    google_api_key: SecretStr | None = Field(default=None, repr=False)

    # --- Model settings -----------------------------------------------------------
    search_model: str = Field(default="gemini-2.5-flash")
    synthesis_model: str = Field(default="gemini-2.5-flash")
    video_model: str = Field(default="gemini-2.5-flash")
    tts_model: str = Field(default="gemini-2.5-flash-preview-tts")

    # --- Generation tuning ----------------------------------------------------------
    search_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    synthesis_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    podcast_script_temperature: float = Field(default=0.4, ge=0.0, le=2.0)

    # --- TTS configuration ------------------------------------------------------------
    mike_voice: str = Field(default="Kore")
    sarah_voice: str = Field(default="Puck")
    tts_channels: int = Field(default=1, ge=1, le=2)
    tts_rate: int = Field(default=24000, gt=0)
    tts_sample_width: int = Field(default=2, gt=0)

    # --- Resilience ---------------------------------------------------------------------
    llm_timeout_seconds: float = Field(default=120.0, gt=0.0)
    llm_max_attempts: int = Field(default=4, ge=1, le=10)
    graph_timeout_seconds: float = Field(default=600.0, gt=0.0)

    # --- Media / outputs --------------------------------------------------------------------
    media_output_dir: Path = Field(default=Path("media/podcasts"))
    max_topic_chars: int = Field(default=2000, gt=0)

    # --- Observability --------------------------------------------------------------------------
    langsmith_tracing: bool = Field(default=False)
    langsmith_api_key: SecretStr | None = Field(default=None, repr=False)
    langsmith_project: str = Field(default="multi-modal-researcher")
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com")
    otel_enabled: bool = Field(default=False)
    otel_exporter_otlp_endpoint: str | None = None

    @model_validator(mode="after")
    def _require_api_key(self) -> "Settings":
        if self.api_key is None:
            raise ValueError(
                "Either GEMINI_API_KEY or GOOGLE_API_KEY must be set "
                "(see .env.example)."
            )
        return self

    @property
    def api_key(self) -> str | None:
        """Resolved Gemini API key (GEMINI_API_KEY preferred)."""
        secret = self.gemini_api_key or self.google_api_key
        return secret.get_secret_value() if secret else None

    @property
    def resolved_media_output_dir(self) -> Path:
        """Absolute path of the dedicated podcast output directory."""
        return self.media_output_dir.resolve()

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return upper


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (single source of truth)."""
    return Settings()
