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

## 8. Next Milestones (Implementation Roadmap)

- [x] Initial Project Scaffold, Config, Structured Logging, Health APIs, and Testing Setup.
- [ ] **Phase 1**: Fetch exact arXiv paper versions (`1706.03762v7`, `2410.00037v1`, `2507.07955v2`) and build BM25 chunk indexer.
- [ ] **Phase 2**: Real Groq API baseline profiling and Stand-in Model Service calibration.
- [ ] **Phase 3**: 4-Step Pipeline Orchestrator with Reflection & Citation Verifier.
- [ ] **Phase 4**: Priority Queue, Token Bucket, and Deadline Backpressure Engine.
- [ ] **Phase 5**: Automated load testing suite generating `results.csv` and Tables 1, 2a, 2b, and 3.
