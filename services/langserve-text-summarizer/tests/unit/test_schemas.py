"""Unit tests for Pydantic v2 schema validation."""

import pytest
from pydantic import ValidationError

from app.schemas.summarizer import SummarizeRequest, SummarizeResponse


def test_summarize_request_accepts_valid_text() -> None:
    req = SummarizeRequest(text="hello world")
    assert req.text == "hello world"


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_summarize_request_rejects_blank(bad: str) -> None:
    with pytest.raises(ValidationError):
        SummarizeRequest(text=bad)


def test_summarize_request_forbids_extras() -> None:
    with pytest.raises(ValidationError):
        SummarizeRequest.model_validate({"text": "ok", "unknown": 1})


def test_summarize_response_requires_fields() -> None:
    with pytest.raises(ValidationError):
        SummarizeResponse(summary="s", model="m", input_chars=-1)
