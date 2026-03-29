# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency spec first (layer-cache friendly)
COPY pyproject.toml .

# Install all deps into an isolated prefix
RUN pip install --upgrade pip \
 && pip install --prefix=/install . --no-deps \
 && pip install --prefix=/install \
      langgraph \
      langchain \
      langchain-google-genai \
      langchain-groq \
      langgraph-checkpoint-sqlite \
      fastapi \
      "uvicorn[standard]" \
      pydantic \
      pydantic-settings \
      structlog \
      yahooquery \
      praw \
      vaderSentiment \
      requests \
      tenacity \
      python-dotenv \
      langsmith \
      aiosqlite


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: non-root user
RUN groupadd --gid 1001 appgroup \
 && useradd  --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=appuser:appgroup . .

# Checkpoint DB lives in a writable volume (not inside the image layer)
RUN mkdir -p /data && chown appuser:appgroup /data
ENV CHECKPOINT_SQLITE_PATH=/data/checkpoints.db

# Production defaults
ENV LOG_FORMAT=json
ENV LOG_LEVEL=INFO
ENV ENVIRONMENT=production
ENV PORT=8000

USER appuser
EXPOSE 8000

# Readiness / liveness probe endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready')"

CMD ["python", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "warning"]