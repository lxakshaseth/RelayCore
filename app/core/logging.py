"""Structured logging configuration for RelayCore."""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict

from app.core.context import get_current_request_id


class StructuredJsonFormatter(logging.Formatter):
    """JSON log formatter injecting request context and structured fields."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }

        request_id = get_current_request_id()
        if request_id:
            log_entry["request_id"] = request_id

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include custom extra fields if provided
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
                "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                "created", "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "message",
            }:
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


class DevelopmentFormatter(logging.Formatter):
    """Clean colorized-friendly string formatter for local development."""

    def format(self, record: logging.LogRecord) -> str:
        request_id = get_current_request_id()
        req_prefix = f"[{request_id}] " if request_id else ""
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"{time_str} | {record.levelname:<8} | {record.name}:{record.lineno} - {req_prefix}{record.getMessage()}"
        if record.exc_info:
            msg += f"\n{self.formatException(record.exc_info)}"
        return msg


def setup_logging(log_level: str = "INFO", app_env: str = "development") -> None:
    """Configure system-wide logging based on environment."""
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())

    # Clear existing handlers
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if app_env.lower() == "production":
        handler.setFormatter(StructuredJsonFormatter())
    else:
        handler.setFormatter(DevelopmentFormatter())

    root_logger.addHandler(handler)

    # Silence overly verbose third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
