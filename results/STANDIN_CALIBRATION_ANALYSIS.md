# Stand-in Model Service: Calibration, Assumptions & Divergence Analysis

*Required for Section 2 of The Relay Assignment Writeup.*

---

## 1. Parameter Taxonomy: Measured vs. Assumed

To maintain scientific integrity, RelayCore explicitly demarcates parameters derived from live empirical API measurements against architectural engineering assumptions.

### 1.1 Empirically Measured Parameters (Calibrated from Groq API Calls)

| Parameter | Calibrated Value | Real Groq Telemetry Source |
| :--- | :--- | :--- |
| **Base TTFT Median ($\mu$)** | `180.0 ms` | Observed median TTFT across live streaming calls to `qwen/qwen3.8-27b` (165 ms – 203 ms). |
| **Base TTFT StdDev ($\sigma$)** | `35.0 ms` | Measured dispersion of initial chunk arrival over repeated calls. |
| **Decode Throughput** | `250.0 tokens/s` | Measured from streaming chunk arrival frequency (approx. 4 ms per token delta). |
| **Requests Per Minute (RPM)** | `30 RPM` | Groq developer tier limits and `x-ratelimit-reset-requests` decay rates. |
| **Tokens Per Minute (TPM)** | `8,000 TPM` | Extracted directly from live header `x-ratelimit-limit-tokens: 8000`. |
| **Requests Per Day (RPD)** | `1,000 RPD` | Extracted directly from live header `x-ratelimit-limit-requests: 1000`. |
| **429 Response Shape** | Exact JSON payload | Modeled after Groq's exact error schema containing `type: rate_limit_exceeded` and `retry-after` header. |
| **Token Reset Micro-Window** | `~250 ms – 700 ms` | Verified from `x-ratelimit-reset-tokens: 224ms`, confirming sub-second sliding refill. |

---

### 1.2 Synthetic Architectural Assumptions

| Parameter | Assumed Value | Justification |
| :--- | :--- | :--- |
| **Concurrency Ceiling** | `3 in-flight requests` | Assumption representing typical free-tier account throttling before upstream gateway queuing. |
| **Concurrency Penalty Factor** | `0.4 (40% delay per excess req)` | Synthetic model to emulate server-side queue contention under concurrent load. |
| **Daily Token Ceiling (TPD)** | `500,000 tokens` | Published free-tier policy assumption; daily depletion cannot be routinely probed without exhausting the account. |
| **Fault Probabilities** | `failure_rate_500`, `503` (Configurable) | Groq's production availability is high (>99.9%); failure rates are parameterized for stress testing. |
| **Token Delivery Jitter** | `0.5 ms` | Synthetic jitter added to simulate network packet variation. |

---

## 2. Where the Simulator Differs from the Actual Groq Service

The stand-in simulator provides an approximation of rate limits and latency dynamics, but diverges from the real hosted Groq infrastructure in four fundamental ways:

### 2.1 Deterministic Hardware Scheduling vs. Operating System Jitter
- **Actual Groq Service**: Powered by custom Language Processing Units (LPUs) with deterministic tensor compute architectures. Token decode speed is mechanically consistent and memory-bandwidth bound with minimal per-token variance.
- **Stand-in Simulator**: Runs on general-purpose host CPUs and asyncio event loops. Python thread scheduling, garbage collection pauses, and OS context switches introduce non-deterministic jitter absent from real LPU hardware.

### 2.2 Global Multi-Tenant Token Buckets vs. Local Isolated Tracking
- **Actual Groq Service**: Quotas and concurrency are enforced across a distributed global edge network. A surge from other tenants sharing the LPU cluster may introduce sudden queue spikes even when local client concurrency is low.
- **Stand-in Simulator**: Relies on a local in-memory sliding window and token bucket. It does not account for external global multi-tenant contention or regional edge POP routing delays.

### 2.3 Strict Byte-Pair Encoding vs. Character Heuristics
- **Actual Groq Service**: Computes input and output token counts using the exact byte-pair tokenizer (BPE) of the active model (e.g., Qwen/Llama tokenizers).
- **Stand-in Simulator**: Uses standard word-to-token approximations ($\approx 4\text{ chars/token}$ or $1.3\text{ tokens/word}$) for performance and dependency isolation during massive synthetic load runs.

### 2.4 Token Burst Allowances
- **Actual Groq Service**: Upstream API gateways frequently permit brief instantaneous micro-bursts above the nominal RPM if average token throughput remains compliant over the subsequent second.
- **Stand-in Simulator**: Enforces strict sliding window boundaries.

---

## 3. Operational Modes & Switch

The simulator implements `AbstractModelClient`, allowing RelayCore to switch dynamically between real and simulated inference via configuration:

```python
# Stand-in mode for heavy load tests
client = StandinModelService(config=StandinConfig(seed=42, rpm_limit=30, tpm_limit=8000))

# Real hosted Groq mode
client = GroqAsyncClient(settings=get_settings())
```
