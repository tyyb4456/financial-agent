"""
main.py — Application entrypoint
---------------------------------
Uses FastAPI's modern lifespan context manager (replaces deprecated @on_event).

Startup sequence:
  1. Configure structured logging
  2. Activate LangSmith tracing (if env vars are set)
  3. Warm the registry (eagerly import all graphs)
  4. Log startup complete

Middleware:
  - CORS
  - RequestLoggingMiddleware: logs every request/response with timing + correlation ID
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from config.logging      import configure_logging
from config.observability import setup_tracing
from config.settings      import settings
from api.routes           import router


# ── Logging must be configured first, before any other imports use structlog ──
configure_logging()
log = structlog.get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs at startup (before first request) and shutdown (after last request).
    Use this instead of deprecated @app.on_event("startup").
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    log.info("app.startup.begin", version=settings.app_port)

    tracing_on = setup_tracing()

    # Warm the registry — eagerly import and compile all graphs so the first
    # request doesn't pay the cold-start cost.
    from registry import get_registry
    registry = get_registry()
    log.info(
        "app.startup.registry_warmed",
        graphs=list(registry.keys()),
        tracing=tracing_on,
    )

    log.info("app.startup.complete", port=settings.app_port)

    yield  # ← app is running here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    log.info("app.shutdown")


# ── Request logging middleware ────────────────────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every HTTP request and response.

    Adds X-Correlation-ID response header so clients can correlate their
    request with backend log lines even for endpoints that don't return it
    in the body (like /graphs or /stream).
    """

    async def dispatch(self, request: Request, call_next):
        cid   = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        start = time.monotonic()

        structlog.contextvars.bind_contextvars(
            correlation_id=cid,
            method=request.method,
            path=request.url.path,
        )

        log.info("http.request")

        response = await call_next(request)

        elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        log.info(
            "http.response",
            status=response.status_code,
            elapsed_ms=elapsed_ms,
        )

        structlog.contextvars.clear_contextvars()

        response.headers["X-Correlation-ID"] = cid
        return response


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Workflows API",
    description=(
        "LangGraph-powered AI workflow engine. "
        "Supports financial analysis, investment advice, "
        "Reddit trends, news reporting, and medical research."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestLoggingMiddleware)

app.include_router(router, prefix="/api/v1", tags=["graphs"])


# ── Health + readiness endpoints ──────────────────────────────────────────────

@app.get("/", tags=["health"], summary="Health check")
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/ready", tags=["health"], summary="Readiness check")
async def ready():
    """
    Returns 200 only if the registry is loaded (all graphs compiled).
    Kubernetes / load-balancer readiness probe target.
    """
    from registry import get_registry
    registry = get_registry()
    return {
        "status": "ready",
        "graphs": list(registry.keys()),
    }


# # ── Dev server ────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     uvicorn.run(
#         "main:app",
#         host="0.0.0.0",
#         port=settings.app_port,
#         reload=True,
#         log_level=settings.log_level.lower(),
#     )