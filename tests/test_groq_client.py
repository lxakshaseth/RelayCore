"""Unit and integration tests for GroqAsyncClient, secret scrubbing, and rate limit handling."""

import os
import pytest
from app.models.groq_client import GroqAsyncClient
from app.models.sanitizer import sanitize_secrets
from app.models.schemas import Message, ModelRequest, RateLimitInfo


def test_secret_scrubber_redaction():
    """Verify that sensitive API keys are thoroughly redacted from logs and errors."""
    raw_error = "Failed to authenticate with key: gsk_1234567890abcdef1234567890 and Bearer token_secret_9876543210123456"
    sanitized = sanitize_secrets(raw_error)

    assert "gsk_" not in sanitized
    assert "token_secret" not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized


def test_rate_limit_header_parser():
    """Verify rate limit headers are cleanly converted into typed RateLimitInfo."""
    client = GroqAsyncClient()
    mock_headers = {
        "x-ratelimit-limit-requests": "1000",
        "x-ratelimit-remaining-requests": "995",
        "x-ratelimit-reset-requests": "4m12s",
        "x-ratelimit-limit-tokens": "8000",
        "x-ratelimit-remaining-tokens": "7850",
        "x-ratelimit-reset-tokens": "210ms",
        "retry-after": "5.5",
    }

    info: RateLimitInfo = client._parse_rate_limit_headers(mock_headers)
    assert info.limit_requests == 1000
    assert info.remaining_requests == 995
    assert info.reset_requests == "4m12s"
    assert info.limit_tokens == 8000
    assert info.remaining_tokens == 7850
    assert info.reset_tokens == "210ms"
    assert info.retry_after == 5.5


@pytest.mark.asyncio
async def test_groq_timeout_handling():
    """Verify that client timeout returns status='timeout' rather than crashing."""
    client = GroqAsyncClient()
    req = ModelRequest(
        messages=[Message(role="user", content="Ping test.")],
        timeout_seconds=0.0001,  # Guaranteed timeout
    )
    res = await client.complete(req)

    assert res.status == "timeout"
    assert res.content == ""
    assert res.first_token_ms is None
    assert "timed out" in (res.error_info or "").lower()


@pytest.mark.asyncio
async def test_groq_real_completion_if_configured():
    """Verify real hosted model completion when GROQ_API_KEY is available."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not configured in environment.")

    client = GroqAsyncClient()
    req = ModelRequest(
        messages=[Message(role="user", content="Respond with exactly one word: 'CONFIRMED'.")],
        max_tokens=10,
    )
    res = await client.complete(req)

    assert res.status == "ok"
    assert res.tokens_in > 0
    assert res.tokens_out > 0
    assert res.total_ms > 0
    assert "CONFIRMED" in res.content.upper()
    assert res.rate_limit_info is not None
    assert res.rate_limit_info.remaining_requests is not None


@pytest.mark.asyncio
async def test_groq_real_streaming_ttft_if_configured():
    """Verify real streaming measures TTFT accurately."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not configured in environment.")

    client = GroqAsyncClient()
    req = ModelRequest(
        messages=[Message(role="user", content="Count from 1 to 3.")],
        max_tokens=20,
        stream=True,
    )

    chunks = []
    final_res = None
    async for chunk, response in client.stream_complete(req):
        if chunk:
            chunks.append(chunk)
        if response:
            final_res = response

    assert final_res is not None
    assert final_res.status == "ok"
    assert final_res.first_token_ms is not None
    assert final_res.first_token_ms > 0
    assert final_res.total_ms >= final_res.first_token_ms
    assert len(chunks) > 0
