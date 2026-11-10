"""Schemas and data models for LLM requests, responses, and rate limit telemetry."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat completion message."""

    role: str = Field(description="Role: 'system', 'user', or 'assistant'")
    content: str = Field(description="Textual content of the message")


class ModelRequest(BaseModel):
    """Standardized model inference request payload."""

    messages: List[Message]
    model: Optional[str] = Field(default=None, description="Model ID, defaults to settings.groq_model")
    stream: bool = Field(default=False, description="Whether to stream completion tokens")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1)
    timeout_seconds: float = Field(default=15.0, description="Per-request socket/API timeout")


class RateLimitInfo(BaseModel):
    """Extracted upstream rate limit headers from provider."""

    limit_requests: Optional[int] = Field(default=None, description="Requests per window limit")
    remaining_requests: Optional[int] = Field(default=None, description="Requests remaining in window")
    reset_requests: Optional[str] = Field(default=None, description="Time until request limit resets")
    limit_tokens: Optional[int] = Field(default=None, description="Tokens per window limit")
    remaining_tokens: Optional[int] = Field(default=None, description="Tokens remaining in window")
    reset_tokens: Optional[str] = Field(default=None, description="Time until token limit resets")
    retry_after: Optional[float] = Field(default=None, description="Retry-After header in seconds if 429")


class ModelResponse(BaseModel):
    """Standardized structured response meeting the assignment telemetry contract."""

    content: str = Field(default="", description="Generated response text or empty string on error")
    first_token_ms: Optional[int] = Field(
        default=None,
        description="Time-to-first-token in milliseconds (TTFT)",
    )
    total_ms: int = Field(description="Total round-trip latency in milliseconds")
    tokens_in: int = Field(default=0, description="Prompt input tokens consumed")
    tokens_out: int = Field(default=0, description="Completion output tokens generated")
    status: str = Field(
        default="ok",
        description="Execution status: 'ok', 'rate_limited', 'timeout', 'error'",
    )
    error_info: Optional[str] = Field(default=None, description="Sanitized error description")
    retries: int = Field(default=0, description="Number of retries triggered")
    rate_limit_info: Optional[RateLimitInfo] = Field(
        default=None,
        description="Parsed upstream provider rate limit telemetry",
    )
    model: str = Field(description="Target model identifier used for inference")
    source: str = Field(default="real", description="'real' (Groq Cloud) or 'stand-in' (Mock)")
