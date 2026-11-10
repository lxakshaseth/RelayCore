"""Async context tracking for request correlation and telemetry."""

from contextvars import ContextVar
from typing import Optional

request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id_ctx", default=None)


def get_current_request_id() -> Optional[str]:
    """Retrieve the current request ID from the async context."""
    return request_id_ctx.get()


def set_current_request_id(request_id: str) -> None:
    """Set the current request ID in the async context."""
    request_id_ctx.set(request_id)
