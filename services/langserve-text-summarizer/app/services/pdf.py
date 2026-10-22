"""Safe PDF text extraction with size limits and typed errors."""

from __future__ import annotations

import io
from typing import BinaryIO

import pypdf

from app.core.config import get_settings
from app.core.middleware import AppError
from app.core.logging import get_logger

logger = get_logger(__name__)


def extract_pdf_text(stream: BinaryIO | bytes, filename: str) -> str:
    """Extract all text from a PDF byte stream.

    Raises:
        AppError: on unreadable PDFs, empty extraction, or oversize input.
    """
    settings = get_settings()
    data: bytes = stream.read() if hasattr(stream, "read") else bytes(stream)  # type: ignore[arg-type]

    if len(data) > settings.max_pdf_bytes:
        raise AppError(
            f"PDF exceeds maximum size of {settings.max_pdf_bytes} bytes.",
            status_code=413,
        )

    try:
        reader = pypdf.PdfReader(io.BytesIO(data), strict=False)
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # pypdf raises many vendor-specific errors
        logger.error("pdf_parse_failed", extra={"context": {"filename": filename}})
        raise AppError("The uploaded file is not a readable PDF.", status_code=400) from exc

    text = "\n".join(pages).strip()
    if not text:
        raise AppError("No text could be extracted from the PDF.", status_code=400)

    if len(text) > settings.max_input_chars:
        text = text[: settings.max_input_chars]
        logger.warning(
            "pdf_text_truncated",
            extra={"context": {"max_input_chars": settings.max_input_chars}},
        )
    return text
