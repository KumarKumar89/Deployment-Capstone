"""Typed REST endpoints: health, readiness, summarize, summarize-pdf."""

from __future__ import annotations

import time

from fastapi import APIRouter, File, UploadFile

from app import __version__
from app.core.config import get_settings
from app.core.logging import get_logger, request_id_ctx
from app.core.middleware import AppError
from app.schemas.summarizer import (
    HealthResponse,
    PdfSummarizeResponse,
    ReadinessResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from app.services.llm import summarize_text
from app.services.pdf import extract_pdf_text

logger = get_logger(__name__)

router = APIRouter()

_PDF_MAGIC = b"%PDF-"


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe: process is up and serving."""
    settings = get_settings()
    return HealthResponse(service=settings.app_name, version=__version__)


@router.get("/ready", response_model=ReadinessResponse, tags=["ops"])
def ready() -> ReadinessResponse:
    """Readiness probe: verifies configuration/dependency state."""
    settings = get_settings()
    checks: dict[str, str] = {"config": "ok"}
    try:
        _ = settings.google_api_key.get_secret_value()
        checks["llm_credentials"] = "ok" if _ else "missing"
    except Exception:
        checks["llm_credentials"] = "error"
    ok = all(v == "ok" for v in checks.values())
    return ReadinessResponse(ready=ok, checks=checks)


@router.post("/summarize/text", response_model=SummarizeResponse, tags=["summarizer"])
def summarize_text_endpoint(payload: SummarizeRequest) -> SummarizeResponse:
    """Summarize plain text with the Gemini-backed LCEL chain."""
    settings = get_settings()
    if len(payload.text) > settings.max_input_chars:
        raise AppError(
            f"Input exceeds maximum of {settings.max_input_chars} characters.",
            status_code=413,
        )
    started = time.perf_counter()
    summary = summarize_text(payload.text)
    logger.info(
        "text_summarized",
        extra={"context": {"duration_ms": round((time.perf_counter() - started) * 1000, 2)}},
    )
    return SummarizeResponse(
        summary=summary,
        model=settings.gemini_model,
        input_chars=len(payload.text),
        request_id=request_id_ctx.get(),
    )


@router.post(
    "/summarize-pdf",
    response_model=PdfSummarizeResponse,
    tags=["summarizer"],
)
async def summarize_pdf_endpoint(
    file: UploadFile = File(..., description="PDF document to summarize"),
) -> PdfSummarizeResponse:
    """Upload a PDF, extract its text and summarize it."""
    settings = get_settings()
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise AppError("Uploaded file must be a PDF.", status_code=400)

    content = await file.read()
    if not content.startswith(_PDF_MAGIC):
        raise AppError("Uploaded file does not appear to be a valid PDF.", status_code=400)

    text = extract_pdf_text(content, filename)
    if len(text) > settings.max_input_chars:
        raise AppError(
            f"Extracted text exceeds maximum of {settings.max_input_chars} characters.",
            status_code=413,
        )

    summary = summarize_text(text)
    return PdfSummarizeResponse(
        summary=summary,
        model=settings.gemini_model,
        input_chars=len(text),
        request_id=request_id_ctx.get(),
        filename=filename,
    )
