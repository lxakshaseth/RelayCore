"""Asynchronous Groq Cloud API model client with telemetry, streaming, and rate-limit parsing."""

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Dict, Mapping, Optional, Tuple

import groq
from groq import AsyncGroq

from app.core.config import Settings, get_settings
from app.models.base import AbstractModelClient
from app.models.sanitizer import sanitize_secrets
from app.models.schemas import ModelRequest, ModelResponse, RateLimitInfo

logger = logging.getLogger(__name__)


class GroqAsyncClient(AbstractModelClient):
    """High-performance asynchronous client for Groq Cloud API."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.api_key = self.settings.groq_api_key
        # Clean base_url for official Groq SDK
        raw_url = self.settings.groq_base_url.rstrip("/")
        if raw_url.endswith("/openai/v1"):
            raw_url = raw_url[: -len("/openai/v1")]
        self.base_url = raw_url

        self._client: Optional[AsyncGroq] = None

    def _get_client(self) -> AsyncGroq:
        """Lazy-initialize AsyncGroq client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("GROQ_API_KEY environment variable is not configured.")
            self._client = AsyncGroq(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def _parse_rate_limit_headers(self, headers: Mapping[str, Any]) -> RateLimitInfo:
        """Extract rate limit telemetry from Groq response headers."""
        def _get_int(key: str) -> Optional[int]:
            val = headers.get(key)
            if val is not None:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    return None
            return None

        def _get_float(key: str) -> Optional[float]:
            val = headers.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return None
            return None

        return RateLimitInfo(
            limit_requests=_get_int("x-ratelimit-limit-requests"),
            remaining_requests=_get_int("x-ratelimit-remaining-requests"),
            reset_requests=headers.get("x-ratelimit-reset-requests"),
            limit_tokens=_get_int("x-ratelimit-limit-tokens"),
            remaining_tokens=_get_int("x-ratelimit-remaining-tokens"),
            reset_tokens=headers.get("x-ratelimit-reset-tokens"),
            retry_after=_get_float("retry-after"),
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Execute a non-streaming chat completion with full rate-limit telemetry."""
        target_model = request.model or self.settings.groq_model
        start_time = time.perf_counter()
        retries = 0

        try:
            client = self._get_client()
            formatted_messages = [{"role": m.role, "content": m.content} for m in request.messages]

            raw_res = await asyncio.wait_for(
                client.chat.completions.with_raw_response.create(
                    messages=formatted_messages,
                    model=target_model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                ),
                timeout=request.timeout_seconds,
            )

            total_ms = int((time.perf_counter() - start_time) * 1000)
            headers = raw_res.headers
            rate_info = self._parse_rate_limit_headers(headers)
            completion = await raw_res.parse()

            content = completion.choices[0].message.content or ""
            tokens_in = completion.usage.prompt_tokens if completion.usage else 0
            tokens_out = completion.usage.completion_tokens if completion.usage else 0

            return ModelResponse(
                content=content,
                first_token_ms=total_ms,  # For non-streaming, first token arrives with response
                total_ms=total_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                status="ok",
                retries=retries,
                rate_limit_info=rate_info,
                model=target_model,
                source="real",
            )

        except groq.RateLimitError as e:
            total_ms = int((time.perf_counter() - start_time) * 1000)
            sanitized_msg = sanitize_secrets(str(e))
            logger.warning("Groq HTTP 429 RateLimitError on %s: %s", target_model, sanitized_msg)
            # Try to extract retry-after from error response
            headers = getattr(e, "response", None) and getattr(e.response, "headers", {}) or {}
            rate_info = self._parse_rate_limit_headers(headers)

            return ModelResponse(
                content="",
                first_token_ms=None,
                total_ms=total_ms,
                tokens_in=0,
                tokens_out=0,
                status="rate_limited",
                error_info=sanitized_msg,
                retries=retries,
                rate_limit_info=rate_info,
                model=target_model,
                source="real",
            )

        except (asyncio.TimeoutError, groq.APITimeoutError) as e:
            total_ms = int((time.perf_counter() - start_time) * 1000)
            logger.warning("Groq request timed out on %s after %d ms", target_model, total_ms)
            return ModelResponse(
                content="",
                first_token_ms=None,
                total_ms=total_ms,
                tokens_in=0,
                tokens_out=0,
                status="timeout",
                error_info=f"Request timed out after {request.timeout_seconds}s",
                retries=retries,
                model=target_model,
                source="real",
            )

        except Exception as e:
            total_ms = int((time.perf_counter() - start_time) * 1000)
            sanitized_msg = sanitize_secrets(f"{type(e).__name__}: {str(e)}")
            logger.error("Groq inference failed on %s: %s", target_model, sanitized_msg)
            return ModelResponse(
                content="",
                first_token_ms=None,
                total_ms=total_ms,
                tokens_in=0,
                tokens_out=0,
                status="error",
                error_info=sanitized_msg,
                retries=retries,
                model=target_model,
                source="real",
            )

    async def stream_complete(
        self, request: ModelRequest
    ) -> AsyncIterator[Tuple[str, Optional[ModelResponse]]]:
        """Stream token chunks and yield final ModelResponse with measured TTFT."""
        target_model = request.model or self.settings.groq_model
        start_time = time.perf_counter()
        first_token_ms: Optional[int] = None
        collected_chunks = []
        retries = 0

        try:
            client = self._get_client()
            formatted_messages = [{"role": m.role, "content": m.content} for m in request.messages]

            stream = await asyncio.wait_for(
                client.chat.completions.create(
                    messages=formatted_messages,
                    model=target_model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    stream=True,
                ),
                timeout=request.timeout_seconds,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    if first_token_ms is None:
                        first_token_ms = int((time.perf_counter() - start_time) * 1000)
                    collected_chunks.append(delta)
                    yield (delta, None)

            total_ms = int((time.perf_counter() - start_time) * 1000)
            full_content = "".join(collected_chunks)

            # Approximation of token counts for streaming (standard ~4 chars per token)
            tokens_in_approx = sum(len(m.content) for m in request.messages) // 4
            tokens_out_approx = len(full_content) // 4

            final_response = ModelResponse(
                content=full_content,
                first_token_ms=first_token_ms or total_ms,
                total_ms=total_ms,
                tokens_in=tokens_in_approx,
                tokens_out=tokens_out_approx,
                status="ok",
                retries=retries,
                model=target_model,
                source="real",
            )
            yield ("", final_response)

        except groq.RateLimitError as e:
            total_ms = int((time.perf_counter() - start_time) * 1000)
            sanitized_msg = sanitize_secrets(str(e))
            headers = getattr(e, "response", None) and getattr(e.response, "headers", {}) or {}
            rate_info = self._parse_rate_limit_headers(headers)
            yield (
                "",
                ModelResponse(
                    content="",
                    first_token_ms=None,
                    total_ms=total_ms,
                    tokens_in=0,
                    tokens_out=0,
                    status="rate_limited",
                    error_info=sanitized_msg,
                    retries=retries,
                    rate_limit_info=rate_info,
                    model=target_model,
                    source="real",
                ),
            )

        except (asyncio.TimeoutError, groq.APITimeoutError):
            total_ms = int((time.perf_counter() - start_time) * 1000)
            yield (
                "",
                ModelResponse(
                    content="",
                    first_token_ms=None,
                    total_ms=total_ms,
                    tokens_in=0,
                    tokens_out=0,
                    status="timeout",
                    error_info=f"Stream timed out after {request.timeout_seconds}s",
                    retries=retries,
                    model=target_model,
                    source="real",
                ),
            )

        except Exception as e:
            total_ms = int((time.perf_counter() - start_time) * 1000)
            sanitized_msg = sanitize_secrets(f"{type(e).__name__}: {str(e)}")
            yield (
                "",
                ModelResponse(
                    content="",
                    first_token_ms=None,
                    total_ms=total_ms,
                    tokens_in=0,
                    tokens_out=0,
                    status="error",
                    error_info=sanitized_msg,
                    retries=retries,
                    model=target_model,
                    source="real",
                ),
            )
