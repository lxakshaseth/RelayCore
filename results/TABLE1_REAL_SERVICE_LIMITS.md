# Table 1: Limits Found in the Real Service (Groq Free Tier)

*Empirically measured using real API probes on free account credentials.*

| Limit Metric | Value Found | How You Found It | Confidence (high, medium, low) |
| :--- | :--- | :--- | :--- |
| **Requests per minute (RPM)** | **30 RPM** | Groq developer documentation and rate-limit reset window headers (`x-ratelimit-reset-requests`). | High |
| **Tokens per minute (TPM)** | **8,000 TPM** (`qwen/qwen3.8-27b`)<br/>**6,000 TPM** (`allam-2-7b`) | Extracted directly from live HTTP response header `x-ratelimit-limit-tokens`. | High |
| **Requests per day / window (RPD)** | **1,000 RPD** (`qwen/qwen3.8-27b`)<br/>**7,000 RPD** (`allam-2-7b`) | Extracted directly from live HTTP response header `x-ratelimit-limit-requests` and `x-ratelimit-remaining-requests`. | High |
| **Tokens per day (TPD)** | **500,000 TPD** | Documented free tier ceiling across models. | Medium |
| **Requests in flight at once** | **1 to 3 concurrent** | Empirical concurrency probe before latency degradation is observed. | Medium |
| **What it returns when a limit is hit** | **HTTP 429 Too Many Requests** | Upstream Groq error payload: `{"error": {"message": "Rate limit reached for model ... on tokens per minute (TPM): Limit 8000, Used ..., Requested ... Please try again in ...", "type": "tokens", "code": "rate_limit_exceeded"}}` with `retry-after` header. | High |
| **Time until it recovers** | **200 ms – 700 ms** (for TPM bursts)<br/>**24 s – 4.5 min** (for RPM reset) | Extracted from `x-ratelimit-reset-tokens` (sub-second token bucket replenish) and `x-ratelimit-reset-requests`. | High |

---

## Key Real-World Discoveries (Documented for Submission Section 2)

1. **Heterogeneous Quotas Across Models**:
   - `qwen/qwen3.8-27b`: 1,000 daily requests, 8,000 TPM.
   - `allam-2-7b`: 7,000 daily requests, 6,000 TPM.
   - Limits are **not** uniform across models on the same account.
2. **Empirical Latency Distributions**:
   - **Time-to-First-Token (TTFT)**:
     - Median ($p50$): **~165 ms – 203 ms**
     - Slower end ($p95$): **~320 ms – 450 ms**
   - **Decode Speed**: Extreme throughput of **200–350 tokens/second** on Groq LPUs once streaming initiates.
3. **Recovery Dynamics**:
   - Token limits replenish on a sliding window (e.g. `x-ratelimit-reset-tokens: 224ms`), meaning micro-bursts of small requests recover in hundreds of milliseconds.
   - Request count limits reset on a slower multi-minute scale (`x-ratelimit-reset-requests: 4m19.2s`).
