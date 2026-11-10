"""Configuration models for the calibrated Stand-in Model Service."""

from typing import Optional
from pydantic import BaseModel, Field


class StandinConfig(BaseModel):
    """Complete parameter specification for the Stand-in Model Simulator.

    Distinguishes parameters calibrated directly from Groq empirical measurements
    from synthetic architectural assumptions.
    """

    # --- Determinism & Model Identity ---
    seed: Optional[int] = Field(default=None, description="Random seed for deterministic experiments")
    model_name: str = Field(default="qwen/qwen3.8-27b", description="Model identifier being simulated")

    # --- Empirical Calibrated Latency Parameters (From Real Groq Telemetry) ---
    # Measured on Groq free tier: TTFT median ~180ms, decode speed ~250 tokens/sec
    base_ttft_ms: float = Field(default=180.0, ge=1.0, description="Base median Time To First Token in ms (Measured)")
    ttft_std_ms: float = Field(default=35.0, ge=0.0, description="Log-normal/Gaussian stddev of TTFT in ms (Measured)")
    tokens_per_second: float = Field(default=250.0, ge=1.0, description="Token generation speed in tokens/s (Measured)")
    inter_token_jitter_ms: float = Field(default=0.5, ge=0.0, description="Jitter between streamed tokens in ms (Assumption)")

    # --- Empirical Rate Limits (From Groq x-ratelimit-* headers) ---
    rpm_limit: int = Field(default=30, ge=1, description="Requests per minute limit (Measured from Groq RPM)")
    tpm_limit: int = Field(default=8000, ge=1, description="Tokens per minute limit (Measured from x-ratelimit-limit-tokens)")
    rpd_limit: int = Field(default=1000, ge=1, description="Requests per day limit (Measured from x-ratelimit-limit-requests)")
    daily_token_limit: int = Field(default=500000, ge=1, description="Daily tokens ceiling (Documented Free Tier)")

    # --- Concurrency & Degradation Assumptions ---
    # Groq LPUs handle concurrent batches efficiently until capacity saturation
    max_concurrency: int = Field(default=3, ge=1, description="Concurrency threshold before slowdown occurs (Assumption)")
    concurrency_penalty_factor: float = Field(
        default=0.4,
        ge=0.0,
        description="Latency multiplier added per active request above 1: Latency *= (1 + penalty * active / max_concurrency) (Assumption)",
    )

    # --- Synthetic Fault & Timeout Injection ---
    failure_rate_500: float = Field(default=0.0, ge=0.0, le=1.0, description="Probability of HTTP 500 Internal Error")
    failure_rate_503: float = Field(default=0.0, ge=0.0, le=1.0, description="Probability of HTTP 503 Service Unavailable")
    timeout_probability: float = Field(default=0.0, ge=0.0, le=1.0, description="Probability of simulated network timeout")
    simulated_timeout_delay_ms: float = Field(default=5000.0, description="Delay applied when timeout is triggered")

    # --- Rate Limit Recovery Windows ---
    tpm_window_seconds: float = Field(default=60.0, description="Sliding window duration for token limits in seconds")
    rpm_window_seconds: float = Field(default=60.0, description="Sliding window duration for request limits in seconds")
