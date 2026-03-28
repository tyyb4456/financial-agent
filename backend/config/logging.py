"""
Logging
-------
Structured JSON logging via structlog.
Call configure_logging() once at app startup (done in main.py).

Usage anywhere in the codebase:
    import structlog
    log = structlog.get_logger(__name__)
    log.info("graph.invoked", graph="financial", symbol="AAPL")
"""

import logging
import structlog
from config.settings import settings


def configure_logging() -> None:
    """Wire structlog → stdlib logging with JSON output."""

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer()   # swap → JSONRenderer() in prod
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )