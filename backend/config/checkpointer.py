"""
config/checkpointer.py
-----------------------
Async checkpointer factory.

Dev / default:  AsyncSqliteSaver  (async file-based, zero config)
Production:     swap to AsyncRedisSaver via CHECKPOINT_BACKEND=redis + REDIS_URL

Usage (in async FastAPI handlers):
    async with get_async_checkpointer() as saver:
        compiled = entry.with_checkpointer(saver)
        result   = await compiled.ainvoke(inputs, config={"configurable": {"thread_id": "abc"}})

Also provides a sync get_checkpointer() for scripts/tests.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator

import structlog

log = structlog.get_logger(__name__)

_SQLITE_PATH = os.getenv("CHECKPOINT_SQLITE_PATH", "./checkpoints.db")
_BACKEND     = os.getenv("CHECKPOINT_BACKEND", "sqlite")


# ── Async (used by FastAPI endpoints) ─────────────────────────────────────────

@asynccontextmanager
async def get_async_checkpointer() -> AsyncGenerator:
    """
    Yield an async-compatible checkpointer.

    - "sqlite"  → AsyncSqliteSaver (recommended — persists to disk)
    - "memory"  → InMemorySaver    (non-persistent, good for tests)
    """
    backend = _BACKEND.lower()

    if backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            log.info("checkpointer.async_sqlite", path=_SQLITE_PATH)
            async with AsyncSqliteSaver.from_conn_string(_SQLITE_PATH) as saver:
                yield saver
        except ImportError:
            log.warning("checkpointer.async_sqlite.unavailable", fallback="memory")
            from langgraph.checkpoint.memory import InMemorySaver
            yield InMemorySaver()

    elif backend == "memory":
        log.info("checkpointer.memory")
        from langgraph.checkpoint.memory import InMemorySaver
        yield InMemorySaver()

    else:
        log.warning("checkpointer.unknown_backend", backend=backend, fallback="memory")
        from langgraph.checkpoint.memory import InMemorySaver
        yield InMemorySaver()


# ── Sync (used by scripts, tests, CLI) ────────────────────────────────────────

@contextmanager
def get_checkpointer() -> Generator:
    """Sync checkpointer for non-async contexts (scripts, tests)."""
    backend = _BACKEND.lower()

    if backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            log.info("checkpointer.sqlite", path=_SQLITE_PATH)
            with SqliteSaver.from_conn_string(_SQLITE_PATH) as saver:
                yield saver
        except ImportError:
            log.warning("checkpointer.sqlite.unavailable", fallback="memory")
            from langgraph.checkpoint.memory import InMemorySaver
            yield InMemorySaver()
    else:
        log.info("checkpointer.memory")
        from langgraph.checkpoint.memory import InMemorySaver
        yield InMemorySaver()