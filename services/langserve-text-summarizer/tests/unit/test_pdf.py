"""Unit tests for PDF extraction service."""

import pytest

from app.core.config import Settings, get_settings
from app.core.middleware import AppError
from app.services.pdf import extract_pdf_text


def test_rejects_oversized_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    small = Settings(google_api_key="k", max_pdf_bytes=10)
    monkeypatch.setattr("app.services.pdf.get_settings", lambda: small)
    with pytest.raises(AppError) as exc:
        extract_pdf_text(b"x" * 11, "big.pdf")
    assert exc.value.status_code == 413


def test_invalid_pdf_raises_app_error(monkeypatch: pytest.MonkeyPatch) -> None:
    default = get_settings()
    monkeypatch.setattr("app.services.pdf.get_settings", lambda: default)
    with pytest.raises(AppError) as exc:
        extract_pdf_text(b"definitely not a pdf", "bad.pdf")
    assert exc.value.status_code == 400
