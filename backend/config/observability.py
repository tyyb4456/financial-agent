"""
config/observability.py
-----------------------
LangSmith tracing helpers.

Provides:
  - setup_tracing()       → called once at startup, reads env vars
  - trace_graph_run()     → async context manager that wraps a graph invocation
                            with metadata + tags so every run appears labelled
                            in the LangSmith UI

Design:
  - Tracing is ALWAYS opt-in: if LANGSMITH_TRACING=false (or key missing),
    all helpers are no-ops so the app works identically without any account.
  - Metadata injected per run:
      graph_name, thread_id, environment (dev/staging/prod), app_version
  - Tags injected per run:
      graph_name, environment
  - This makes it easy to filter runs in LangSmith by graph type or env.

Usage in routes.py:
    from config.observability import trace_graph_run
    async with trace_graph_run(graph_name, thread_id, inputs):
        output = await compiled.ainvoke(inputs, config=config)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import structlog

from config.settings import settings

log = structlog.get_logger(__name__)

# App version — set from env or default. Used in LangSmith metadata.
APP_VERSION  = os.getenv("APP_VERSION", "2.0.0")
ENVIRONMENT  = os.getenv("ENVIRONMENT", "development")   # development | staging | production


def setup_tracing() -> bool:
    """
    Activate LangSmith tracing if credentials are present in settings.
    Returns True if tracing was activated, False if skipped (no-op mode).

    Called once at application startup in main.py.
    """
    if not (settings.langsmith_tracing and settings.langsmith_api_key):
        log.info("langsmith.tracing.disabled")
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project

    log.info(
        "langsmith.tracing.enabled",
        project=settings.langsmith_project,
        environment=ENVIRONMENT,
        version=APP_VERSION,
    )
    return True


@asynccontextmanager
async def trace_graph_run(
    graph_name: str,
    thread_id:  str,
    inputs:     dict[str, Any],
) -> AsyncGenerator[dict, None]:
    """
    Async context manager that yields a LangGraph RunnableConfig enriched
    with LangSmith metadata and tags.

    If tracing is disabled this is a pure no-op — it just yields a bare
    config dict with the thread_id so callers don't need an if/else.

    Usage:
        async with trace_graph_run(graph_name, thread_id, inputs) as config:
            output = await compiled.ainvoke(inputs, config=config)

    The yielded `config` dict is ready to pass directly to `.ainvoke()` or
    `.astream()` — it already contains the `configurable` block with thread_id
    AND the `metadata` + `tags` blocks LangSmith reads.
    """
    # Base LangGraph config (always present — needed for checkpointing)
    lg_config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
    }

    if settings.langsmith_tracing and settings.langsmith_api_key:
        # Enrich with LangSmith metadata and tags (only when tracing is on)
        lg_config["metadata"] = {
            "graph_name":  graph_name,
            "thread_id":   thread_id,
            "environment": ENVIRONMENT,
            "app_version": APP_VERSION,
            # Truncate inputs so LangSmith UI doesn't show 6000-char data blobs
            "input_keys":  list(inputs.keys()),
        }
        lg_config["tags"] = [
            f"graph:{graph_name}",
            f"env:{ENVIRONMENT}",
        ]

    yield lg_config