"""Shared fixtures: fake settings + FastAPI TestClient with mocked LLM."""

import os
from typing import Iterator

os.environ.setdefault("GOOGLE_API_KEY", "test-key-not-real")
os.environ.setdefault("LANGSMITH_TRACING", "false")

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """TestClient exercising the real app factory (LLM is patched per-test)."""
    from app.main import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def fake_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the LCEL chain with a deterministic stub (no network calls)."""

    class StubChain:
        def invoke(self, payload: dict) -> str:
            return f"SUMMARY({len(payload['text'])} chars)"

    monkeypatch.setattr("app.services.llm._chain", lambda: StubChain(), raising=True)
    # The lru_cache-wrapped accessor used by routes/LangServe must resolve too.
    monkeypatch.setattr("app.services.llm.get_chain", lambda: StubChain())
