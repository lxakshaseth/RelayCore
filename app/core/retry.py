"""Intelligent Retry Policy with Exponential Backoff, Full Jitter, and Overload Protection."""

import logging
import random
import time
from typing import Optional

from app.models.schemas import ModelResponse

logger = logging.getLogger(__name__)


class RetryPolicy:
    """Manages retry evaluations, jittered exponential backoffs, and overload context shedding."""

    def __init__(
        self,
        max_retries: int = 2,
        base_delay_seconds: float = 0.5,
        max_delay_seconds: float = 3.0,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay_seconds
        self.max_delay = max_delay_seconds

    def compute_backoff(
        self, attempt: int, retry_after: Optional[float] = None
    ) -> float:
        """Compute wait duration using upstream Retry-After header or exponential backoff with full jitter."""
        if retry_after is not None and retry_after > 0:
            # Respect upstream header, clamped to safe maximum
            return min(self.max_delay * 1.5, retry_after)

        # Exponential backoff: base * 2^attempt
        exp_backoff = min(self.max_delay, self.base_delay * (2**attempt))
        # Full jitter: Uniform(0.5, 1.0) * exp_backoff
        jitter = random.uniform(0.5, 1.0)
        return round(exp_backoff * jitter, 3)

    def is_retriable(
        self,
        response: Optional[ModelResponse],
        attempt: int,
        deadline_time: float,
        is_daily_exhausted: bool = False,
    ) -> bool:
        """Determine if a request should be retried or fail fast.

        Returns False if:
        - Max retries exceeded.
        - Daily budget is exhausted (avoids repeatedly retrying doomed requests).
        - Error is permanent (e.g. 400 Bad Request).
        - Remaining time until client deadline cannot accommodate backoff delay.
        """
        if attempt >= self.max_retries:
            return False

        if is_daily_exhausted:
            logger.info("Daily budget exhausted: aborting retries immediately.")
            return False

        if response is None:
            return True

        # Non-retriable status
        if response.status == "ok":
            return False

        # If client deadline is close, retrying will worsen queue congestion
        remaining_time = deadline_time - time.time()
        estimated_backoff = self.compute_backoff(
            attempt,
            response.rate_limit_info.retry_after if response.rate_limit_info else None,
        )

        if remaining_time <= (estimated_backoff + 0.1):
            logger.warning(
                "Client deadline (%.2fs remaining) cannot fit retry delay (%.2fs). Dropping retry.",
                remaining_time,
                estimated_backoff,
            )
            return False

        # Retriable errors: 429 rate limit or 5xx server failures
        if response.status in ("rate_limited", "error"):
            return True

        return False

    def should_shed_context_on_retry(self, response: Optional[ModelResponse]) -> bool:
        """Identify if a 429 was caused by Tokens Per Minute (TPM), requiring context shedding."""
        if not response or response.status != "rate_limited":
            return False

        err_text = (response.error_info or "").lower()
        # If error explicitly mentions TPM or tokens
        return "tokens" in err_text or "tpm" in err_text or "token" in err_text
