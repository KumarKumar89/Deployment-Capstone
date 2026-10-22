# syntax=docker/dockerfile:1.7
########################  Stage 1: build  ########################
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install project (app package + pinned deps) into an isolated venv so the
# runtime stage can copy it wholesale.
COPY pyproject.toml README.md ./
COPY app ./app
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install .

########################  Stage 2: runtime  ######################
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="langserve-text-summarizer" \
      org.opencontainers.image.version="2.0.0"

# Non-root user with its own writable output directory.
RUN groupadd --system appuser && useradd --system --gid appuser --create-home appuser

WORKDIR /srv/app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Healthcheck hits the dedicated /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; \
r=urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4); \
sys.exit(0 if r.status==200 else 1)"

EXPOSE 8080
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
