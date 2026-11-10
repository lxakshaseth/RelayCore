"""Unit tests for TokenBucket, RelayRateLimiter, Concurrency Limiting, and Retry Policy."""

import asyncio
import time
import pytest

from app.core.config import Settings
from app.core.queue_manager import BoundedPriorityQueueManager
from app.core.rate_limiter import RelayRateLimiter, TokenBucket
from app.core.retry import RetryPolicy
from app.models.schemas import ModelResponse, RateLimitInfo


def test_token_bucket_refill_math():
    """Verify continuous fractional refill and consumption mechanics."""
    # Capacity 10, refills at 10 tokens/sec
    bucket = TokenBucket(capacity=10.0, refill_rate_per_sec=10.0)

    # Consume all 10 tokens
    assert bucket.consume(10.0) is True
    assert bucket.can_consume(1.0) is False

    # Simulate 0.1s time passing (refills ~1.0 token)
    bucket.last_update -= 0.1
    assert bucket.can_consume(1.0) is True
    assert bucket.consume(1.0) is True


def test_relay_rate_limiter_rpm_and_tpm():
    """Verify RelayRateLimiter acquires slots and halts on bucket exhaustion."""
    async def _test():
        settings = Settings(
            GROQ_RPM_LIMIT=2,
            GROQ_TPM_LIMIT=200,
            GROQ_RPD_LIMIT=10,
        )
        limiter = RelayRateLimiter(settings=settings)

        # First 2 requests should be permitted immediately
        ok1, reason1 = await limiter.acquire_slot(estimated_tokens=50, max_wait_seconds=0.1)
        assert ok1 is True
        assert reason1 is None

        ok2, reason2 = await limiter.acquire_slot(estimated_tokens=50, max_wait_seconds=0.1)
        assert ok2 is True

        # 3rd request exceeds RPM capacity and should timeout waiting
        ok3, reason3 = await limiter.acquire_slot(estimated_tokens=50, max_wait_seconds=0.05)
        assert ok3 is False
        assert "timed out" in (reason3 or "").lower()

    asyncio.run(_test())


def test_daily_budget_exhaustion_fast_fail():
    """Verify daily budget exhaustion stops further calls without waiting or retrying."""
    async def _test():
        settings = Settings(
            GROQ_RPM_LIMIT=10,
            GROQ_TPM_LIMIT=1000,
            GROQ_RPD_LIMIT=2,
        )
        limiter = RelayRateLimiter(settings=settings)

        # Consume 2 requests
        await limiter.acquire_slot(estimated_tokens=10)
        await limiter.acquire_slot(estimated_tokens=10)

        assert limiter.is_daily_budget_exhausted() is True

        # 3rd request must fail immediately with daily budget reason
        ok, reason = await limiter.acquire_slot(estimated_tokens=10, max_wait_seconds=1.0)
        assert ok is False
        assert "daily request budget exhausted" in (reason or "").lower()

    asyncio.run(_test())


def test_concurrency_limiter_semaphore():
    """Verify concurrency limiter throttles in-flight executions."""
    async def _test():
        settings = Settings(WORKER_CONCURRENCY=2)
        limiter = RelayRateLimiter(settings=settings)

        in_flight = 0
        max_observed = 0

        async def worker():
            nonlocal in_flight, max_observed
            async with limiter.concurrency_semaphore:
                in_flight += 1
                max_observed = max(max_observed, in_flight)
                await asyncio.sleep(0.01)
                in_flight -= 1

        await asyncio.gather(*(worker() for _ in range(6)))
        assert max_observed <= 2

    asyncio.run(_test())


def test_bounded_queue_priority_ordering():
    """Verify high priority requests are dequeued before normal requests."""
    async def _test():
        manager = BoundedPriorityQueueManager(max_depth=10, max_concurrency=2)

        item_low = await manager.enqueue("low_1", "payload_low", deadline_seconds=10.0, priority=1)
        item_high = await manager.enqueue("high_1", "payload_high", deadline_seconds=10.0, priority=5)

        dequeued_1 = await manager.dequeue_next()
        dequeued_2 = await manager.dequeue_next()

        assert dequeued_1.item_id == "high_1"
        assert dequeued_2.item_id == "low_1"

    asyncio.run(_test())


def test_bounded_queue_capacity_rejection():
    """Verify bounded queue rejects when capacity ceiling is breached."""
    async def _test():
        manager = BoundedPriorityQueueManager(max_depth=2, max_concurrency=1)

        await manager.enqueue("req_1", "p1", deadline_seconds=10.0)
        await manager.enqueue("req_2", "p2", deadline_seconds=10.0)

        with pytest.raises(RuntimeError) as exc_info:
            await manager.enqueue("req_3", "p3", deadline_seconds=10.0)

        assert "capacity saturated" in str(exc_info.value)
        assert manager.total_rejected_capacity == 1

    asyncio.run(_test())


def test_bounded_queue_deadline_drop():
    """Verify expired requests are discarded during dequeue."""
    async def _test():
        manager = BoundedPriorityQueueManager(max_depth=5, max_concurrency=1)

        item = await manager.enqueue("expired_req", "payload", deadline_seconds=0.01)
        await asyncio.sleep(0.02)

        dequeued = await manager.dequeue_next()
        assert dequeued is None
        assert manager.total_dropped_deadline == 1
        assert item.future.done()

    asyncio.run(_test())


def test_retry_policy_backoff_and_jitter():
    """Verify exponential backoff calculation and jitter bounds."""
    policy = RetryPolicy(max_retries=3, base_delay_seconds=0.5, max_delay_seconds=4.0)

    # Attempt 0: ~0.5s * Uniform(0.5, 1.0) -> between 0.25 and 0.5
    delay_0 = policy.compute_backoff(attempt=0)
    assert 0.20 <= delay_0 <= 0.55

    # Attempt 2: min(4.0, 0.5 * 4) = 2.0s -> between 1.0 and 2.0
    delay_2 = policy.compute_backoff(attempt=2)
    assert 0.90 <= delay_2 <= 2.05


def test_retry_policy_retry_after_adoption():
    """Verify upstream Retry-After header is adopted when present."""
    policy = RetryPolicy(max_retries=2, base_delay_seconds=0.5, max_delay_seconds=4.0)

    delay = policy.compute_backoff(attempt=0, retry_after=2.5)
    assert delay == 2.5


def test_retry_policy_daily_exhaustion_stops_retries():
    """Verify that when daily budget is exhausted, retrying is strictly prohibited."""
    policy = RetryPolicy(max_retries=2)
    err_res = ModelResponse(status="rate_limited", total_ms=50, model="test", source="stand-in")

    # When daily budget is exhausted, is_retriable must return False
    can_retry = policy.is_retriable(
        response=err_res,
        attempt=0,
        deadline_time=time.time() + 10.0,
        is_daily_exhausted=True,
    )
    assert can_retry is False


def test_retry_policy_context_shedding_trigger():
    """Verify that a 429 TPM error signals context reduction."""
    policy = RetryPolicy()

    tpm_err = ModelResponse(
        status="rate_limited",
        total_ms=40,
        error_info="Rate limit reached for model on tokens per minute (TPM). Limit 8000.",
        model="test",
        source="stand-in",
    )
    assert policy.should_shed_context_on_retry(tpm_err) is True

    generic_err = ModelResponse(
        status="error",
        total_ms=40,
        error_info="Internal server error",
        model="test",
        source="stand-in",
    )
    assert policy.should_shed_context_on_retry(generic_err) is False
