"""Comprehensive Rate Limiter supporting Token Bucket, Concurrency Limiting, and Daily Quotas."""

import asyncio
import time
from typing import Optional, Tuple
from app.core.config import Settings, get_settings


class TokenBucket:
    """Continuous Token Bucket implementation with deterministic fractional refill."""

    def __init__(self, capacity: float, refill_rate_per_sec: float):
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate_per_sec)
        self.tokens = float(capacity)
        self.last_update = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_update = now

    def can_consume(self, amount: float) -> bool:
        self._refill()
        return self.tokens >= amount

    def consume(self, amount: float) -> bool:
        self._refill()
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False

    def time_until_available(self, amount: float) -> float:
        self._refill()
        if self.tokens >= amount:
            return 0.0
        missing = amount - self.tokens
        return missing / self.refill_rate if self.refill_rate > 0 else float("inf")

    def reset(self) -> None:
        self.tokens = self.capacity
        self.last_update = time.monotonic()


class RelayRateLimiter:
    """Central Traffic Controller guarding RPM, TPM, RPD, TPD, and Concurrency."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

        # Limits from config
        self.rpm_limit = self.settings.groq_rpm_limit
        self.tpm_limit = self.settings.groq_tpm_limit
        self.rpd_limit = self.settings.groq_rpd_limit
        self.tpd_limit = 500000  # Daily token limit
        self.max_concurrency = self.settings.worker_concurrency

        # Continuous Token Buckets (refilled every second)
        # 30 RPM = 0.5 requests/sec; 8,000 TPM = 133.33 tokens/sec
        self.request_bucket = TokenBucket(
            capacity=self.rpm_limit,
            refill_rate_per_sec=self.rpm_limit / 60.0,
        )
        self.token_bucket = TokenBucket(
            capacity=self.tpm_limit,
            refill_rate_per_sec=self.tpm_limit / 60.0,
        )

        # Concurrency Limiter Semaphore
        self._concurrency_semaphore: Optional[asyncio.Semaphore] = None
        self.active_in_flight: int = 0

        # Daily Quota Counters
        self.daily_requests_count: int = 0
        self.daily_tokens_count: int = 0
        self._daily_start_time: float = time.time()

        self._lock: Optional[asyncio.Lock] = None

    @property
    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @property
    def concurrency_semaphore(self) -> asyncio.Semaphore:
        if self._concurrency_semaphore is None:
            self._concurrency_semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._concurrency_semaphore

    def _check_daily_reset(self) -> None:
        """Reset 24-hour counters if 86,400 seconds elapsed."""
        now = time.time()
        if now - self._daily_start_time > 86400.0:
            self.daily_requests_count = 0
            self.daily_tokens_count = 0
            self._daily_start_time = now

    def is_daily_budget_exhausted(self) -> bool:
        """Deterministically check if daily request or token quota has been depleted."""
        self._check_daily_reset()
        return (
            self.daily_requests_count >= self.rpd_limit
            or self.daily_tokens_count >= self.tpd_limit
        )

    async def acquire_slot(
        self, estimated_tokens: int = 500, max_wait_seconds: float = 10.0
    ) -> Tuple[bool, Optional[str]]:
        """Attempt to acquire permission to proceed through rate limits.

        Returns (True, None) if permitted, or (False, reason) if rejected/timed out.
        """
        start_time = time.monotonic()

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed >= max_wait_seconds:
                return False, f"Rate limiter wait timed out after {max_wait_seconds:.1f}s"

            async with self.lock:
                self._check_daily_reset()

                # 1. Daily Budget Check - Never wait or retry if daily budget is gone
                if self.daily_requests_count >= self.rpd_limit:
                    return False, f"Daily request budget exhausted ({self.rpd_limit} RPD reached)"
                if self.daily_tokens_count + estimated_tokens > self.tpd_limit:
                    return False, f"Daily token budget exhausted ({self.tpd_limit} TPD reached)"

                # 2. Token Bucket Checks
                can_req = self.request_bucket.can_consume(1.0)
                can_tok = self.token_bucket.can_consume(float(estimated_tokens))

                if can_req and can_tok:
                    self.request_bucket.consume(1.0)
                    self.token_bucket.consume(float(estimated_tokens))
                    self.daily_requests_count += 1
                    self.daily_tokens_count += estimated_tokens
                    return True, None

                # Calculate minimum wait time for buckets to refill
                wait_req = self.request_bucket.time_until_available(1.0)
                wait_tok = self.token_bucket.time_until_available(float(estimated_tokens))
                needed_wait = max(wait_req, wait_tok)

            remaining = max_wait_seconds - (time.monotonic() - start_time)
            if remaining <= 0:
                return False, f"Rate limiter wait timed out after {max_wait_seconds:.1f}s"

            wait_step = min(max(0.005, needed_wait), min(0.5, remaining))
            await asyncio.sleep(wait_step)

    async def record_actual_tokens(self, estimated_tokens: int, actual_tokens: int) -> None:
        """Reconcile difference between estimated tokens and actual consumed tokens."""
        diff = actual_tokens - estimated_tokens
        if diff > 0:
            async with self.lock:
                self.daily_tokens_count += diff
                # If consumed more than estimated, drain extra tokens from bucket
                self.token_bucket.consume(float(diff))

    def reset_all(self) -> None:
        """Reset all rate limiter buckets, counters, and in-flight tracking."""
        self.request_bucket.reset()
        self.token_bucket.reset()
        self.daily_requests_count = 0
        self.daily_tokens_count = 0
        self._daily_start_time = time.time()
        self.active_in_flight = 0


# Backwards compatibility alias
ClientRateLimiter = RelayRateLimiter
