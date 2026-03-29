"""
config/logging.py
-----------------
Structured logging via structlog.

Two modes controlled by LOG_FORMAT env var:
  - "console" (default in dev)  → coloured human-readable output
  - "json"    (default in prod) → machine-readable JSON, one object per line

JSON format is what you pipe into Datadog / CloudWatch / Loki.
Console format is what you read while developing locally.

Call configure_logging() exactly once at startup (done in main.py).

Usage anywhere:
    import structlog
    log = structlog.get_logger(__name__)
    log.info("graph.run.start", graph="financial", thread_id="abc-123")
"""

import logging
import os
import structlog
from config.settings import settings


LOG_FORMAT = os.getenv("LOG_FORMAT", "console").lower()  # "console" | "json"


def configure_logging() -> None:
    """Configure structlog with the right renderer for the current environment."""

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Shared pre-chain processors (run before the renderer)
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.ExceptionRenderer(),
    ]

    if LOG_FORMAT == "json":
        # Production: one JSON object per line, machine-readable
        renderer = structlog.processors.JSONRenderer()
    else:
        # Development: coloured, human-readable
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )