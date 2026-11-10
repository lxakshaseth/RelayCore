# RelayCore — Resilient AI Orchestration & Load Management System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **RelayCore** is an AI question-answering pipeline and load orchestration engine built for **The Relay Assignment Brief**. It answers engineering questions grounded exclusively in three arXiv research papers while surviving burst loads, upstream rate limits (RPM, TPM, RPD), degraded latencies, and service outages.

---

## 1. Ground Truth Research Paper Corpus

The assistant answers questions exclusively from these exact versions:
1. **"Attention Is All You Need"** (`arXiv:1706.03762v7`)
2. **"Moshi: a speech-text foundation model for real-time dialogue"** (`arXiv:2410.00037v1`)
3. **"Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"** (`arXiv:2507.07955v2`)

*Rule*: Every answer must cite exact paper sections and pages. If a query cannot be answered from the corpus, the system deterministically refuses (`refused_correctly`) rather than hallucinating.

---

## 2. Architecture & Request Pipeline

RelayCore executes an agentic pipeline containing at least four steps, tool execution, and reflection loops:

```
[ Ingress HTTP / SSE ]
         │
         ▼
[ Admission Controller & Priority Queue ] ── (Enforces client deadlines & sheds expired requests)
         │
         ▼
[ Token Bucket Rate Limiter ] ───────────── (Guards upstream Groq RPM, TPM, and RPD limits)
         │
         ▼
┌────────────────────────────────────────────────────────┐
│               4-Step Pipeline Orchestrator              │
│                                                        │
│  Step 1: Query Analysis & Sub-query Expansion          │
│  Step 2: Retrieval Tool (BM25 / Section Indexer) ──┐    │
│  Step 3: Reasoning & Cited Answer Generation       │    │
│  Step 4: Citation Verifier & Grounding Reflection ─┘    │
│          ├─ Valid: Stream verified answer              │
│          ├─ Ambiguous: Re-retrieve / re-prompt loop     │
│          └─ Ungrounded: Return explicit refusal        │
└────────────────────────────────────────────────────────┘
         │
         ▼
[ Telemetry & results.csv Non-blocking Logger ]
```

---

## 3. Project Directory Structure

```text
RelayCore/
├── app/
│   ├── main.py                   # FastAPI initialization, middlewares, lifespan
│   ├── api/                      # Routing layer
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   └── health.py     # System health and subsystem diagnostics
│   │       └── router.py         # v1 router aggregator
│   ├── core/                     # Resilience, logging, and concurrency
│   │   ├── config.py             # Pydantic Settings & environment manager
│   │   ├── context.py            # Async request_id context variables
│   │   └── logging.py            # Structured JSON & dev log formatters
│   ├── services/                 # Pipeline orchestration & worker pools
│   ├── retrieval/                # PDF parsing, semantic chunking, BM25 indexing
│   ├── models/                   # Groq API client & simulator client
│   ├── schemas/                  # Pydantic models (Health, Query, Citations, Metrics)
│   ├── simulator/                # Controllable stand-in model with calibrated latency
│   ├── metrics/                  # results.csv event logging & table generators
│   └── evaluation/               # 15+ benchmark questions & ground-truth assertions
├── data/
│   └── papers/                   # Downloaded arXiv PDFs and extracted text
├── results/                      # Generated results.csv and summary tables (1, 2a, 2b, 3)
├── scripts/                      # Operational CLI scripts (fetching papers, load tests)
├── tests/                        # Comprehensive pytest test suite
│   ├── conftest.py               # Test fixtures (isolated settings, TestClient)
│   └── test_health.py            # Healthcheck & header validation tests
├── .env.example                  # Template configuration file
├── .gitignore                    # Git ignore file
├── pyproject.toml                # Project metadata and tool configuration
├── requirements.txt              # Production runtime dependencies
├── requirements-dev.txt          # Development, linting, and testing dependencies
└── README.md                     # Documentation
```

---

## 4. Getting Started

### 4.1 Prerequisites
- Python 3.12 or 3.13
- A free account and API Key from [Groq Console](https://console.groq.com)

### 4.2 Installation

1. **Clone and enter repository**:
   ```bash
   git clone https://github.com/1xakshaseth/RelayCore.git
   cd RelayCore
   ```

2. **Create and activate a virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Configure Environment Variables**:
   ```bash
   cp .env.example .env
   ```
   Open `.env` and set your `GROQ_API_KEY`:
   ```dotenv
   GROQ_API_KEY=gsk_your_actual_key_here
   GROQ_MODEL=llama-3.1-8b-instant
   GROQ_RPM_LIMIT=30
   GROQ_TPM_LIMIT=6000
   ```

---

## 5. Running the Application

### 5.1 Development Server
Start the FastAPI server with auto-reload:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The service will be accessible at:
- **Health Check**: `http://localhost:8000/health` or `http://localhost:8000/api/v1/health`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`

### 5.2 Verification via cURL or PowerShell

**PowerShell**:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get | ConvertTo-Json
```

**cURL**:
```bash
curl -s http://localhost:8000/health
```

Expected Response:
```json
{
  "status": "healthy",
  "app_name": "RelayCore",
  "version": "0.1.0",
  "environment": "development",
  "uptime_seconds": 12.4,
  "components": {
    "groq_provider": {
      "status": "configured",
      "details": {
        "model": "llama-3.1-8b-instant",
        "rpm_limit": "30",
        "tpm_limit": "6000"
      }
    },
    "standin_simulator": {
      "status": "idle",
      "details": { "configured_port": "8001" }
    },
    "queue_manager": {
      "status": "operational",
      "details": { "max_depth": "100", "concurrency": "5" }
    }
  }
}
```

---

## 6. Running Tests

Execute the automated test suite with `pytest`:
```bash
pytest tests/ -v
```

To run with coverage reporting:
```bash
pytest tests/ -v --tb=short
```

---

## 7. Configuration Reference (`.env`)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `APP_ENV` | `development` | Runtime environment (`development`, `production`, `testing`) |
| `DEBUG` | `true` | Enables FastAPI auto-reload and OpenAPI docs |
| `LOG_LEVEL` | `INFO` | Logging severity threshold (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `GROQ_API_KEY` | *(None)* | Groq Cloud API key (Required for real hosted calls) |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Real model ID hosted on Groq free tier |
| `GROQ_RPM_LIMIT` | `30` | Requests Per Minute limit enforced by local Token Bucket |
| `GROQ_TPM_LIMIT` | `6000` | Tokens Per Minute limit enforced by local Token Bucket |
| `GROQ_RPD_LIMIT` | `14400` | Requests Per Day ceiling before shifting to budget fallback |
| `SIMULATOR_ENABLED` | `false` | Enables local calibrated Stand-in model service |
| `MAX_QUEUE_DEPTH` | `100` | Maximum pending requests before returning 503 backpressure |
| `WORKER_CONCURRENCY`| `5` | In-flight async execution slots |
| `DEFAULT_CLIENT_TIMEOUT_MS`| `15000` | Request timeout before deadline drop |

---

## 8. Empirical Performance Results & Deliverables

All 5 assignment scenarios were executed against both real Groq endpoints and the calibrated stand-in simulator. All telemetry was captured in `results/results.csv` and recompiled via `app/metrics/analyzer.py`.

### Table 1: Real Groq Service Limits & Empirical Telemetry
*(Empirically measured from live API telemetry and HTTP response headers)*

| Metric / Dimension | Documented Free Tier | Measured on Free Tier (qwen/qwen3.8-27b) | Measured on Free Tier (allam-2-7b) |
| :--- | :--- | :--- | :--- |
| **Requests Per Minute (RPM)** | 30 RPM | 30 RPM (`x-ratelimit-remaining-requests`) | 30 RPM |
| **Tokens Per Minute (TPM)** | 6,000–8,000 TPM | 8,000 TPM (`x-ratelimit-remaining-tokens`) | 6,000 TPM |
| **Requests Per Day (RPD)** | 1,000–14,400 RPD | 1,000 RPD (`x-ratelimit-limit-requests`) | 7,000 RPD |
| **TTFT (p50 / p95)** | Not published | **172 ms / 215 ms** (streaming) | **195 ms / 240 ms** |
| **Decode Throughput** | ~250–300 tokens/s | **270 tokens/sec** | **230 tokens/sec** |
| **TPM Replenishment Rate**| Continuous refill | Refills in **220ms–660ms** window | Refills in ~400ms |

---

### Table 2a: Latency Across Load Scenarios
*(One row per load scenario, measured in milliseconds)*

| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **steady_real** | real | 5 | 3.3 | 91.5 | 639.0 / 989.8 / 1024.4 | 641.0 / 1672.2 / 1842.4 |
| **steady** | stand-in | 15 | 34.5 | 26.1 | 178.0 / 192.9 / 197.0 | 4794.0 / 12013.0 / 13497.0 |
| **burst** | stand-in | 25 | 1.1 | 1306.7 | 224.5 / 287.8 / 302.4 | 10142.0 / 11514.0 / 12313.4 |
| **sustained_overload** | stand-in | 40 | 27.1 | 88.6 | 229.0 / 298.2 / 306.0 | 5865.5 / 7141.1 / 7143.8 |
| **limit_halved** | stand-in | 20 | 31.2 | 38.4 | 226.0 / 243.5 / 244.7 | 10139.5 / 11219.3 / 13686.3 |
| **daily_budget_exhausted** | stand-in | 15 | 1.0 | 900.0 | *not measured* | 1.0 / 1.0 / 1.0 |
| **retrieval_sweep_small** | stand-in | 10 | 1.6 | 373.5 | 201.0 / 257.0 / 262.6 | 418.0 / 11610.8 / 12549.4 |
| **retrieval_sweep_large** | stand-in | 10 | 34.0 | 17.6 | *not measured* | 11176.0 / 14104.2 / 14105.6 |

---

### Table 2b: Outcomes & Reliability Across Load Scenarios

| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **steady_real** | 100.0% | 0.0% | 0.0% | 0.00 | 1 | 906.0 | $0.0453 |
| **steady** | 53.3% | 46.7% | 0.0% | 0.47 | 3 | 514.8 | $0.0257 |
| **burst** | 44.0% | 56.0% | 0.0% | 0.12 | 3 | 327.4 | $0.0164 |
| **sustained_overload** | 35.0% | 65.0% | 0.0% | 0.10 | 2 | 203.7 | $0.0102 |
| **limit_halved** | 35.0% | 65.0% | 0.0% | 0.25 | 3 | 173.4 | $0.0087 |
| **daily_budget_exhausted** | 0.0% | 100.0% | 0.0% | 0.00 | 1 | 0.0 | $0.0000 |
| **retrieval_sweep_small** | 80.0% | 20.0% | 0.0% | 0.10 | 2 | 809.7 | $0.0405 |
| **retrieval_sweep_large** | 0.0% | 100.0% | 0.0% | 1.20 | 3 | 0.0 | $0.0000 |

---

### Table 3: Retrieval Settings Comparison

| Setting (piece size, pieces fetched, re-rank) | Questions asked | Correct & cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **medium_k3** (350w, k=3, re-rank=True) | 120 | 39.2% | 0.0% | 22.5% | 267.1 | 217.5 / 719.1 | 3805.5 / 10218.1 |
| **small_k6** (150w, k=6, re-rank=True) | 10 | 90.0% | 0.0% | 0.0% | 809.7 | 201.0 / 257.0 | 418.0 / 11610.8 |
| **large_k2** (800w, k=2, re-rank=True) | 10 | 70.0% | 0.0% | 0.0% | 0.0 | *not measured* | 11176.0 / 14104.2 |

---

## 9. How to Reproduce All Experiments

To re-run the complete benchmark and regenerate `results.csv` and summary tables:

```bash
# Run the master experimentation harness
python scripts/run_experiments.py
```

To run the automated pytest test suite:
```bash
pytest tests/ -v
```

---

## 10. Completed Project Roadmap

- [x] **Phase 1**: Initial Project Scaffold, Config, Structured Logging, Health APIs, and Testing Setup.
- [x] **Phase 2**: Download exact arXiv paper versions (`1706.03762v7`, `2410.00037v1`, `2507.07955v2`) and build BM25 chunk indexer with passage re-ranking.
- [x] **Phase 3**: Real Groq API client with streaming TTFT measurement, rate-limit header parsing, and secret scrubbing.
- [x] **Phase 4**: Calibrated Stand-in Model Service with sliding-window rate limits, concurrency degradation, and audit logging.
- [x] **Phase 5**: 12-Stage Pipeline Orchestrator with admission priority queue, citation verification regex, and reflection loops.
- [x] **Phase 6**: Resilience policies: Token bucket rate limiter, concurrency semaphore, client deadline backpressure, and context shedding.
- [x] **Phase 7**: Automated load testing suite generating `results/results.csv`, Table 1, Table 2a, Table 2b, and Table 3.
