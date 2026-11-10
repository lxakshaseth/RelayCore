"""Proactive client-side Rate Limiter (Token Bucket & Sliding Window) for RelayCore."""

import asyncio
import time
from typing import Optional, Tuple
from app.core.config import Settings, get_settings


class ClientRateLimiter:
    """Client-side Token Bucket protecting upstream Groq RPM, TPM, and RPD limits."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

        # Limits from config
        self.rpm_limit = self.settings.groq_rpm_limit
        self.tpm_limit = self.settings.groq_tpm_limit
        self.rpd_limit = self.settings.groq_rpd_limit

        # Sliding window history: (timestamp, tokens)
        self._requests_history = []
        self._tokens_history = []

        # Daily tracking
        self._daily_requests_count = 0
        self._daily_tokens_count = 0
        self._daily_start_time = time.time()

        self._lock = asyncio.Lock()

    def _purge_expired(self, now: float) -> None:
        """Evict records older than 60 seconds and reset daily counters if 24 hours elapsed."""
        cutoff = now - 60.0
        self._requests_history = [t for t in self._requests_history if t > cutoff]
        self._tokens_history = [item for item in self._tokens_history if item[0] > cutoff]

        if now - self._daily_start_time > 86400.0:
            self._daily_requests_count = 0
            self._daily_tokens_count = 0
            self._daily_start_time = now

    def is_daily_budget_exhausted(self) -> bool:
        """Check if daily requests or tokens quota is completely spent."""
        now = time.time()
        self._purge_expired(now)
        return self._daily_requests_count >= self.rpd_limit

    async def acquire_or_wait(
        self, estimated_tokens: int = 500, max_wait_seconds: float = 10.0
    ) -> Tuple[bool, Optional[str]]:
        """Attempt to acquire permission to execute an upstream call.

        Waits if transient per-minute limit is near capacity.
        Returns (True, None) if permitted, or (False, reason) if blocked/exhausted.
        """
        start_wait = time.time()

        while True:
            now = time.time()
            if (now - start_wait) > max_wait_seconds:
                return False, f"Client rate limiter wait timeout ({max_wait_seconds}s exceeded)"

            async with self._lock:
                self._purge_expired(now)

                # 1. Daily budget exhaustion check
                if self._daily_requests_count >= self.rpd_limit:
                    return False, f"Daily request budget exhausted ({self.rpd_limit} RPD reached)"

                # 2. RPM check
                rpm_used = len(self._requests_history)
                # 3. TPM check
                tpm_used = sum(t[1] for t in self._tokens_history)

                if rpm_used < self.rpm_limit and (tpm_used + estimated_tokens) <= self.tpm_limit:
                    # Success: allocate slot
                    self._requests_history.append(now)
                    self._tokens_history.append((now, estimated_tokens))
                    self._daily_requests_count += 1
                    self._daily_tokens_count += estimated_tokens
                    return True, None

                # Compute wait duration until oldest entry leaves 60s window
                wait_time = 0.5
                if self._requests_history:
                    oldest_req = self._requests_history[0]
                    wait_time = max(0.1, (oldest_req + 60.0) - now)
                if self._tokens_history:
                    oldest_tok = self._tokens_history[0][0]
                    tok_wait = max(0.1, (oldest_tok + 60.0) - now)
                    wait_time = min(wait_time, tok_wait)

            # Wait outside lock
            wait_step = min(wait_time, 0.5)
            await asyncio.sleep(wait_step)

    async def record_actual_tokens(self, estimated_tokens: int, actual_tokens: int) -> None:
        """Reconcile difference between estimated and actual consumed tokens."""
        diff = actual_tokens - estimated_tokens
        if diff != 0:
            async with self._lock:
                now = time.time()
                self._tokens_history.append((now, diff))
                self._daily_tokens_count += diff

    def reset(self) -> None:
        """Reset limiter state."""
        self._requests_history.clear()
        self._tokens_history.clear()
        self._daily_requests_count = 0
        self._daily_tokens_count = 0
        self._daily_start_time = time.time()
