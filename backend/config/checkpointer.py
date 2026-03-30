"""
config/checkpointer.py
-----------------------
Production-grade async checkpointer factory with multiple backends.

Supported backends:
  - sqlite (default):  AsyncSqliteSaver  → file-based, zero external deps
  - memory:            InMemorySaver     → non-persistent, good for tests
  - redis:             AsyncRedisSaver   → production-ready, requires Redis

Usage (in async FastAPI handlers):
    async with get_async_checkpointer() as saver:
        compiled = entry.with_checkpointer(saver)
        result   = await compiled.ainvoke(inputs, config={"configurable": {"thread_id": "abc"}})

Configuration via environment:
    CHECKPOINT_BACKEND=sqlite              # or 'memory' | 'redis'
    CHECKPOINT_SQLITE_PATH=./checkpoints.db
    REDIS_URL=redis://localhost:6379

Also provides a sync get_checkpointer() for scripts/tests.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator

import structlog

log = structlog.get_logger(__name__)

_SQLITE_PATH = os.getenv("CHECKPOINT_SQLITE_PATH", "./checkpoints.db")
_REDIS_URL   = os.getenv("REDIS_URL", "redis://localhost:6379")
_BACKEND     = os.getenv("CHECKPOINT_BACKEND", "sqlite")


# ── Async (used by FastAPI endpoints) ─────────────────────────────────────────

@asynccontextmanager
async def get_async_checkpointer() -> AsyncGenerator:
    """
    Yield an async-compatible checkpointer based on CHECKPOINT_BACKEND.
    
    Supports: sqlite (default), memory, redis
    Falls back to memory if the requested backend is unavailable.
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

    elif backend == "redis":
        try:
            from langgraph.checkpoint.redis.aio import AsyncRedisSaver
            log.info("checkpointer.async_redis", url=_REDIS_URL)
            saver = AsyncRedisSaver.from_conn_string(_REDIS_URL)
            yield saver
        except ImportError:
            log.warning("checkpointer.async_redis.unavailable_fallback_to_memory")
            from langgraph.checkpoint.memory import InMemorySaver
            yield InMemorySaver()
        except Exception as exc:
            log.error("checkpointer.async_redis.connection_failed", error=str(exc), fallback="memory")
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
    """Sync checkpointer for non-async contexts (scripts, tests).
    
    Supports: sqlite (default), memory, redis
    Falls back to memory if the requested backend is unavailable.
    """
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

    elif backend == "redis":
        try:
            from langgraph.checkpoint.redis import RedisSaver
            log.info("checkpointer.redis", url=_REDIS_URL)
            saver = RedisSaver.from_conn_string(_REDIS_URL)
            yield saver
        except ImportError:
            log.warning("checkpointer.redis.unavailable", fallback="memory")
            from langgraph.checkpoint.memory import InMemorySaver
            yield InMemorySaver()
        except Exception as exc:
            log.error("checkpointer.redis.connection_failed", error=str(exc), fallback="memory")
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