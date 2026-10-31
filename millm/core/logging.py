"""
Structured logging configuration using structlog.

Provides consistent, machine-readable logs in JSON format for production
and human-readable colored output for development.
"""

import logging
import sys

import structlog

from millm.core.config import settings


def setup_logging() -> None:
    """
    Configure structured logging for the application.

    Uses JSON format for production and colored console output for development.
    Call this once during application startup.
    """
    # Determine renderer based on settings
    if settings.LOG_FORMAT == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    # Configure structlog
    structlog.configure(
        processors=[
            # Add log level to event dict
            structlog.stdlib.add_log_level,
            # Handle positional arguments
            structlog.stdlib.PositionalArgumentsFormatter(),
            # Add timestamp in ISO format
            structlog.processors.TimeStamper(fmt="iso"),
            # Render stack traces
            structlog.processors.StackInfoRenderer(),
            # Format exception info
            structlog.processors.format_exc_info,
            # Decode unicode
            structlog.processors.UnicodeDecoder(),
            # Final renderer (JSON or console)
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    )
    # basicConfig does nothing when the root logger already has a handler, level included; set
    # the level explicitly so the in-force LOG_LEVEL (an override loaded first, Feature 35) holds.
    apply_log_level(settings.LOG_LEVEL)

    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def apply_log_level(level: str) -> None:
    """Set the root logger's level now (Feature 35: `LOG_LEVEL` changed while the server runs).

    `setup_logging` sets the level once, through `logging.basicConfig`; without this a runtime
    change of `LOG_LEVEL` would be reported by the Settings page and ignored by the log. Every
    `millm` logger inherits the root's level (none sets its own), and structlog's stdlib loggers
    check it per call, so cached structlog loggers follow too. The quieted libraries above keep
    their own WARNING level.
    """
    logging.getLogger().setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Optional logger name. If not provided, uses the calling module's name.

    Returns:
        A bound logger instance.

    Example:
        logger = get_logger(__name__)
        logger.info("model_loaded", model_id=1, name="gpt2")
    """
    return structlog.get_logger(name)
