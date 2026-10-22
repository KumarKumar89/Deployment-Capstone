"""Centralized environment-variable management using pydantic-settings.

All runtime knobs are sourced from the environment (12-factor). No secrets,
model names or endpoints are hardcoded in application code.
"""

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Service metadata -------------------------------------------------
    app_name: str = "langserve-text-summarizer"
    environment: str = Field(default="production")
    log_level: str = Field(default="INFO")
    api_prefix: str = Field(default="")

    # --- Server -----------------------------------------------------------
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080)

    # --- CORS ---------------------------------------------------------------
    cors_allow_origins: str = Field(
        default="*",
        description=(
            "Comma-separated list of allowed origins, or '*' for any origin "
            "(convenient for local file:// / static frontend testing)."
        ),
    )
    cors_allow_methods: str = Field(default="GET,POST,OPTIONS")
    cors_allow_headers: str = Field(default="*")

    # --- LLM (Google Gemini via langchain-google-genai) -------------------
    google_api_key: SecretStr = Field(repr=False)
    gemini_model: str = Field(default="gemini-2.0-flash")
    llm_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    llm_max_retries: int = Field(default=3, ge=0, le=10)
    llm_timeout_seconds: float = Field(default=60.0, gt=0.0)
    max_input_chars: int = Field(default=100_000, gt=0)
    max_pdf_bytes: int = Field(default=25 * 1_024 * 1_024, gt=0)

    # --- Observability ------------------------------------------------------
    langsmith_tracing: bool = Field(default=False)
    langsmith_api_key: SecretStr | None = Field(default=None, repr=False)
    langsmith_project: str = Field(default="langserve-text-summarizer")
    otel_enabled: bool = Field(default=False)
    otel_exporter_otlp_endpoint: str | None = None

    @property
    def cors_allow_origins_list(self) -> list[str]:
        """Parsed list of allowed CORS origins ('*' means any origin)."""
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    @property
    def cors_allow_methods_list(self) -> list[str]:
        """Parsed list of allowed HTTP methods."""
        return [m.strip().upper() for m in self.cors_allow_methods.split(",") if m.strip()]

    @property
    def cors_allow_headers_list(self) -> list[str]:
        """Parsed list of allowed request headers."""
        return [h.strip() for h in self.cors_allow_headers.split(",") if h.strip()]

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
    return Settings()  # type: ignore[call-arg]
