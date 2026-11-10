"""Controllable Stand-in Model Service simulating Groq API behavior and constraints."""

import asyncio
import logging
import random
import time
from typing import AsyncIterator, List, Optional, Tuple

from app.models.base import AbstractModelClient
from app.models.schemas import ModelRequest, ModelResponse, RateLimitInfo
from app.simulator.audit import SimulatorAuditRecorder
from app.simulator.config import StandinConfig

logger = logging.getLogger(__name__)


# Default synthetic grounding text when prompt is technical
SYNTHETIC_RESPONSES = [
    "According to the provided research papers [1706.03762v7, Section 3.1], the Transformer relies entirely on self-attention mechanisms to compute representations without sequence-aligned RNNs.",
    "The Mimi audio codec [2410.00037v1, Section 3.3] compresses 24 kHz audio into a 12.5 Hz frame rate representation using Residual Vector Quantization (RVQ).",
    "Dynamic Chunking [2507.07955v2, Section 2] introduces H-Net U-Net-like hierarchical subsampling learned end-to-end to eliminate external heuristics.",
    "Based on the analysis, the model produces representations of dimension dmodel = 512 across multi-head projections.",
]


class StandinModelService(AbstractModelClient):
    """Calibrated simulator for Groq LLM API with deterministic control and fault injection."""

    def __init__(self, config: Optional[StandinConfig] = None):
        self.config = config or StandinConfig()
        self.audit = SimulatorAuditRecorder()
        self._lock = asyncio.Lock()

        # Deterministic random generator
        self.rng = random.Random(self.config.seed) if self.config.seed is not None else random.Random()

        # Concurrency tracker
        self.active_concurrency: int = 0

        # Rate limit sliding window tracking: list of (timestamp, token_count)
        self._sliding_requests: List[float] = []
        self._sliding_tokens: List[Tuple[float, int]] = []

        # Daily cumulative counters
        self._daily_requests_count: int = 0
        self._daily_tokens_count: int = 0
        self._daily_start_time: float = time.time()

    def set_config(self, config: StandinConfig) -> None:
        """Update simulator configuration dynamically during experiments."""
        self.config = config
        if config.seed is not None:
            self.rng = random.Random(config.seed)

    def reset_quotas(self) -> None:
        """Reset all in-flight sliding window and daily counters."""
        self._sliding_requests.clear()
        self._sliding_tokens.clear()
        self._daily_requests_count = 0
        self._daily_tokens_count = 0
        self._daily_start_time = time.time()
        self.active_concurrency = 0

    def _purge_expired_windows(self, now: float) -> None:
        """Evict records older than their respective sliding windows."""
        # Purge RPM window
        rpm_cutoff = now - self.config.rpm_window_seconds
        self._sliding_requests = [t for t in self._sliding_requests if t > rpm_cutoff]

        # Purge TPM window
        tpm_cutoff = now - self.config.tpm_window_seconds
        self._sliding_tokens = [item for item in self._sliding_tokens if item[0] > tpm_cutoff]

        # Purge daily counters if 24 hours elapsed
        if now - self._daily_start_time > 86400.0:
            self._daily_requests_count = 0
            self._daily_tokens_count = 0
            self._daily_start_time = now

    def _calculate_latencies(self, tokens_out: int) -> Tuple[int, int]:
        """Compute TTFT and total latency factoring in concurrency degradation."""
        # Sample TTFT from Gaussian/Log-normal distribution
        base_ttft = max(25.0, self.rng.gauss(self.config.base_ttft_ms, self.config.ttft_std_ms))

        # Concurrency penalty: latency increases when active requests exceed single slot
        load_multiplier = 1.0
        if self.active_concurrency > 1:
            load_factor = (self.active_concurrency - 1) / max(1, self.config.max_concurrency)
            load_multiplier += self.config.concurrency_penalty_factor * load_factor

        effective_ttft_ms = int(base_ttft * load_multiplier)

        # Generation time based on tokens/second throughput
        decode_ms = int((tokens_out / max(1.0, self.config.tokens_per_second)) * 1000 * load_multiplier)
        total_ms = effective_ttft_ms + decode_ms

        return effective_ttft_ms, total_ms

    async def _check_rate_limits(self, prompt_tokens: int, now: float) -> Optional[ModelResponse]:
        """Verify request against RPM, TPM, RPD, and daily token limits."""
        self._purge_expired_windows(now)

        # Check Requests Per Minute (RPM)
        if len(self._sliding_requests) >= self.config.rpm_limit:
            oldest = self._sliding_requests[0]
            retry_after = round(max(0.1, (oldest + self.config.rpm_window_seconds) - now), 2)
            return self._build_429_response(
                reason=f"Rate limit reached for model on requests per minute (RPM). Limit {self.config.rpm_limit}.",
                retry_after=retry_after,
                model=self.config.model_name,
            )

        # Check Tokens Per Minute (TPM)
        current_window_tokens = sum(t[1] for t in self._sliding_tokens)
        if current_window_tokens + prompt_tokens > self.config.tpm_limit:
            oldest = self._sliding_tokens[0][0] if self._sliding_tokens else now
            retry_after = round(max(0.1, (oldest + self.config.tpm_window_seconds) - now), 2)
            return self._build_429_response(
                reason=f"Rate limit reached for model on tokens per minute (TPM). Limit {self.config.tpm_limit}, requested {prompt_tokens}.",
                retry_after=retry_after,
                model=self.config.model_name,
            )

        # Check Requests Per Day (RPD)
        if self._daily_requests_count >= self.config.rpd_limit:
            seconds_to_midnight = max(1.0, 86400.0 - (now - self._daily_start_time))
            return self._build_429_response(
                reason=f"Daily request budget exhausted (RPD limit {self.config.rpd_limit} reached).",
                retry_after=round(seconds_to_midnight, 1),
                model=self.config.model_name,
            )

        # Check Daily Tokens
        if self._daily_tokens_count + prompt_tokens > self.config.daily_token_limit:
            seconds_to_midnight = max(1.0, 86400.0 - (now - self._daily_start_time))
            return self._build_429_response(
                reason=f"Daily token budget exhausted (TPD limit {self.config.daily_token_limit} reached).",
                retry_after=round(seconds_to_midnight, 1),
                model=self.config.model_name,
            )

        return None

    def _build_429_response(self, reason: str, retry_after: float, model: str) -> ModelResponse:
        """Construct synthetic HTTP 429 ModelResponse identical to Groq provider payload."""
        rate_info = RateLimitInfo(
            limit_requests=self.config.rpm_limit,
            remaining_requests=max(0, self.config.rpm_limit - len(self._sliding_requests)),
            reset_requests=f"{retry_after:.1f}s",
            limit_tokens=self.config.tpm_limit,
            remaining_tokens=max(0, self.config.tpm_limit - sum(t[1] for t in self._sliding_tokens)),
            reset_tokens=f"{retry_after:.1f}s",
            retry_after=retry_after,
        )
        return ModelResponse(
            content="",
            first_token_ms=None,
            total_ms=45,  # 429 rejection is returned quickly
            tokens_in=0,
            tokens_out=0,
            status="rate_limited",
            error_info=reason,
            retries=0,
            rate_limit_info=rate_info,
            model=model,
            source="stand-in",
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Execute a simulated non-streaming request."""
        target_model = request.model or self.config.model_name
        prompt_len = sum(len(m.content) for m in request.messages)
        prompt_tokens = max(1, prompt_len // 4)
        now = time.time()

        async with self._lock:
            # 1. Rate Limit Checks
            rate_limit_err = await self._check_rate_limits(prompt_tokens, now)
            if rate_limit_err:
                self.audit.record(
                    model=target_model,
                    stream=False,
                    tokens_in=prompt_tokens,
                    tokens_out=0,
                    status="rate_limited",
                    error_type="HTTP_429",
                    retry_after=rate_limit_err.rate_limit_info.retry_after if rate_limit_err.rate_limit_info else None,
                    refusal_reason=rate_limit_err.error_info,
                )
                return rate_limit_err

            # 2. Check Fault Injection (500 / 503 / Timeout)
            roll = self.rng.random()
            if roll < self.config.failure_rate_500:
                self.audit.record(model=target_model, status="error", error_type="HTTP_500")
                return ModelResponse(
                    content="",
                    total_ms=80,
                    status="error",
                    error_info="HTTP 500: Internal Server Error (Simulated)",
                    model=target_model,
                    source="stand-in",
                )
            elif roll < self.config.failure_rate_500 + self.config.failure_rate_503:
                self.audit.record(model=target_model, status="error", error_type="HTTP_503")
                return ModelResponse(
                    content="",
                    total_ms=120,
                    status="error",
                    error_info="HTTP 503: Service Unavailable - Capacity Exhausted (Simulated)",
                    model=target_model,
                    source="stand-in",
                )
            elif roll < (self.config.failure_rate_500 + self.config.failure_rate_503 + self.config.timeout_probability):
                delay = self.config.simulated_timeout_delay_ms
                await asyncio.sleep(delay / 1000.0)
                self.audit.record(model=target_model, status="timeout", total_ms=int(delay))
                return ModelResponse(
                    content="",
                    total_ms=int(delay),
                    status="timeout",
                    error_info=f"Request timed out after {delay}ms (Simulated)",
                    model=target_model,
                    source="stand-in",
                )

            # 3. Register In-Flight & Rate Consumption
            self.active_concurrency += 1
            self._sliding_requests.append(now)
            self._daily_requests_count += 1

        try:
            # Generate simulated response content
            candidate_text = self.rng.choice(SYNTHETIC_RESPONSES)
            completion_tokens = max(5, len(candidate_text) // 4)

            # Compute calibrated latencies with concurrency load penalty
            effective_ttft, total_ms = self._calculate_latencies(completion_tokens)

            # Simulate execution time
            await asyncio.sleep(total_ms / 1000.0)

            async with self._lock:
                self._sliding_tokens.append((time.time(), prompt_tokens + completion_tokens))
                self._daily_tokens_count += prompt_tokens + completion_tokens

            rate_info = RateLimitInfo(
                limit_requests=self.config.rpm_limit,
                remaining_requests=max(0, self.config.rpm_limit - len(self._sliding_requests)),
                limit_tokens=self.config.tpm_limit,
                remaining_tokens=max(0, self.config.tpm_limit - sum(t[1] for t in self._sliding_tokens)),
            )

            res = ModelResponse(
                content=candidate_text,
                first_token_ms=effective_ttft,
                total_ms=total_ms,
                tokens_in=prompt_tokens,
                tokens_out=completion_tokens,
                status="ok",
                rate_limit_info=rate_info,
                model=target_model,
                source="stand-in",
            )

            self.audit.record(
                model=target_model,
                stream=False,
                tokens_in=prompt_tokens,
                tokens_out=completion_tokens,
                first_token_ms=effective_ttft,
                total_ms=total_ms,
                active_concurrency=self.active_concurrency,
                status="ok",
            )
            return res

        finally:
            async with self._lock:
                self.active_concurrency = max(0, self.active_concurrency - 1)

    async def stream_complete(
        self, request: ModelRequest
    ) -> AsyncIterator[Tuple[str, Optional[ModelResponse]]]:
        """Stream simulated token chunks with gradual token delivery."""
        target_model = request.model or self.config.model_name
        prompt_len = sum(len(m.content) for m in request.messages)
        prompt_tokens = max(1, prompt_len // 4)
        now = time.time()

        async with self._lock:
            rate_limit_err = await self._check_rate_limits(prompt_tokens, now)
            if rate_limit_err:
                self.audit.record(
                    model=target_model,
                    stream=True,
                    tokens_in=prompt_tokens,
                    status="rate_limited",
                    error_type="HTTP_429",
                )
                yield ("", rate_limit_err)
                return

            self.active_concurrency += 1
            self._sliding_requests.append(now)
            self._daily_requests_count += 1

        try:
            candidate_text = self.rng.choice(SYNTHETIC_RESPONSES)
            words = candidate_text.split()
            completion_tokens = max(len(words), len(candidate_text) // 4)

            effective_ttft, total_ms = self._calculate_latencies(completion_tokens)

            # Wait for TTFT
            await asyncio.sleep(effective_ttft / 1000.0)

            # Inter-token delay for remaining tokens
            remaining_ms = max(0, total_ms - effective_ttft)
            inter_token_delay = (remaining_ms / max(1, len(words))) / 1000.0

            for idx, word in enumerate(words):
                chunk = word + (" " if idx < len(words) - 1 else "")
                yield (chunk, None)
                if inter_token_delay > 0:
                    await asyncio.sleep(inter_token_delay)

            async with self._lock:
                self._sliding_tokens.append((time.time(), prompt_tokens + completion_tokens))
                self._daily_tokens_count += prompt_tokens + completion_tokens

            final_res = ModelResponse(
                content=candidate_text,
                first_token_ms=effective_ttft,
                total_ms=total_ms,
                tokens_in=prompt_tokens,
                tokens_out=completion_tokens,
                status="ok",
                model=target_model,
                source="stand-in",
            )

            self.audit.record(
                model=target_model,
                stream=True,
                tokens_in=prompt_tokens,
                tokens_out=completion_tokens,
                first_token_ms=effective_ttft,
                total_ms=total_ms,
                active_concurrency=self.active_concurrency,
                status="ok",
            )
            yield ("", final_res)

        finally:
            async with self._lock:
                self.active_concurrency = max(0, self.active_concurrency - 1)
