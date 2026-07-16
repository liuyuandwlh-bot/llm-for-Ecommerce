"""
Logging Utilities

Structured logging with structlog.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    format: str = "json",
    log_file: Optional[str] = None,
):
    """
    Setup structured logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format: Output format (json, console)
        log_file: Optional file path for logging
    """
    import structlog

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper()),
        handlers=handlers,
    )


def get_logger(name: str):
    """Get a logger instance."""
    import structlog
    return structlog.get_logger(name)


# Initialize default logging
setup_logging()
