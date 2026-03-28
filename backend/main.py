"""
main.py — Application entrypoint
---------------------------------
Responsibilities:
  1. Configure logging
  2. Optionally enable LangSmith tracing
  3. Mount the API router
  4. Start uvicorn

Keep this file thin. Business logic lives in graphs/, tools/, chains/.
"""

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings, configure_logging
from api.routes import router


# ── Logging ───────────────────────────────────────────────────────────────────
configure_logging()

import structlog
log = structlog.get_logger(__name__)


# ── LangSmith tracing (opt-in via env) ────────────────────────────────────────
if settings.langsmith_tracing and settings.langsmith_api_key:
    os.environ["LANGSMITH_TRACING"]  = "true"
    os.environ["LANGSMITH_API_KEY"]  = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"]  = settings.langsmith_project
    log.info("langsmith.tracing.enabled", project=settings.langsmith_project)


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Workflows API",
    description="LangGraph-powered AI workflow engine.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/", tags=["health"])
async def health():
    return {"status": "ok", "version": "2.0.0"}


# ── Dev server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.app_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )