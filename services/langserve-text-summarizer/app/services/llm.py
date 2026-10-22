"""LLM factory and summarization chain (typed, config-driven, retry-aware)."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.services.retry import with_retry_sync

logger = get_logger(__name__)

SUMMARIZATION_TEMPLATE = """\
You are an expert summarizer who helps readers understand complex texts.
Please summarize the following text in a clear, concise manner that captures \
the main points. Aim for a summary that is approximately 20% of the original \
length.

Guidelines:
- Use your own words and write in third person.
- Include key elements of the original and keep it brief.
- Do not include interpretation or analysis.

TEXT: {text}

SUMMARY:
"""


def create_llm(settings: Settings) -> ChatGoogleGenerativeAI:
    """Build a Gemini chat model from settings (no hardcoded keys/models)."""
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key.get_secret_value(),
        temperature=settings.llm_temperature,
        max_retries=settings.llm_max_retries,
        request_timeout=settings.llm_timeout_seconds,
    )


@lru_cache(maxsize=1)
def _chain() -> object:
    settings = get_settings()
    prompt = PromptTemplate(input_variables=["text"], template=SUMMARIZATION_TEMPLATE)
    llm: BaseLanguageModel = create_llm(settings)
    return prompt | llm | StrOutputParser()


def get_chain() -> object:
    """Return the cached LCEL summarization chain."""
    return _chain()


@traceable(run_type="chain", name="summarize_text")
def summarize_text(text: str) -> str:
    """Invoke the chain with retry + backoff; returns the summary string."""
    settings = get_settings()
    logger.info(
        "summarization_started",
        extra={"context": {"input_chars": len(text), "model": settings.gemini_model}},
    )
    result = with_retry_sync(
        get_chain().invoke,  # type: ignore[union-attr]
        {"text": text},
        max_attempts=settings.llm_max_retries,
    )
    summary: str = result if isinstance(result, str) else str(result)
    logger.info(
        "summarization_completed",
        extra={"context": {"output_chars": len(summary)}},
    )
    return summary
