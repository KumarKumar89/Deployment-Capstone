"""Integration tests for HTTP endpoints using FastAPI TestClient (mocked LLM)."""


def test_health_returns_ok(client) -> None:  # type: ignore[no-untyped-def]
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "langserve-text-summarizer"


def test_request_id_header_propagated(client) -> None:
    resp = client.get("/health", headers={"X-Request-ID": "corr-123"})
    assert resp.headers["x-request-id"] == "corr-123"


def test_ready_reports_credential_check(client) -> None:
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert "checks" in body and "llm_credentials" in body["checks"]


def test_summarize_text_validates_input(client) -> None:
    resp = client.post("/summarize/text", json={"text": "   "})
    assert resp.status_code == 422


def test_summarize_text_success(client, fake_chain) -> None:  # type: ignore[no-untyped-def]
    resp = client.post("/summarize/text", json={"text": "hello world"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"].startswith("SUMMARY(")
    assert body["input_chars"] == len("hello world")


def test_summarize_pdf_rejects_non_pdf_extension(client) -> None:
    resp = client.post(
        "/summarize-pdf",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "app_error"


def test_summarize_pdf_rejects_bad_magic(client) -> None:
    resp = client.post(
        "/summarize-pdf",
        files={"file": ("doc.pdf", b"not really a pdf", "application/pdf")},
    )
    assert resp.status_code == 400


def test_summarize_pdf_success(client, fake_chain, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "app.api.routes.extract_pdf_text",
        lambda data, name: "Extracted PDF text content for testing purposes.",
    )
    resp = client.post(
        "/summarize-pdf",
        files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "doc.pdf"
    assert body["summary"].startswith("SUMMARY(")


def test_langserve_route_registered(client) -> None:
    """The LangServe /summarize endpoint should exist on the app."""
    paths = {r.path for r in client.app.routes}  # type: ignore[attr-defined]
    assert any(p.startswith("/summarize") for p in paths)


def test_unexpected_errors_return_sanitized_envelope(client, fake_chain, monkeypatch) -> None:
    """A non-retryable crash inside summarization yields a safe 500 envelope."""

    class ExplodingChain:
        def invoke(self, payload: dict) -> str:
            raise ValueError("secret internal details at /home/appuser/x.py")

    monkeypatch.setattr("app.services.llm.get_chain", lambda: ExplodingChain())
    resp = client.post("/summarize/text", json={"text": "trigger failure"})
    assert resp.status_code == 500
    body = resp.json()
    assert "secret internal details" not in body["error"]["message"]
    assert body["error"]["code"] == "internal_error"
