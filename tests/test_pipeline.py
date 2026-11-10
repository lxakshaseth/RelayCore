"""Integration and unit tests for the complete 12-stage RelayCore pipeline."""

import asyncio
import os
import pytest

from app.core.config import Settings
from app.core.rate_limiter import ClientRateLimiter
from app.retrieval.service import get_retrieval_service
from app.schemas.query import QueryRequest
from app.services.orchestrator import OrchestrationPipeline
from app.simulator.config import StandinConfig
from app.simulator.service import StandinModelService


@pytest.fixture
def mock_pipeline() -> OrchestrationPipeline:
    """Fixture providing a fast-running pipeline backed by StandinModelService."""
    settings = Settings(
        APP_ENV="testing",
        DEBUG=True,
        DATA_DIR="data/papers",
        GROQ_RPM_LIMIT=10,
        GROQ_TPM_LIMIT=5000,
        GROQ_RPD_LIMIT=50,
        DEFAULT_CLIENT_TIMEOUT_MS=10000,
    )
    retrieval = get_retrieval_service(data_dir="data/papers")
    rate_limiter = ClientRateLimiter(settings=settings)

    # Fast stand-in client
    standin_client = StandinModelService(
        config=StandinConfig(
            seed=42,
            base_ttft_ms=10.0,
            ttft_std_ms=2.0,
            tokens_per_second=1000.0,
            rpm_limit=20,
            tpm_limit=5000,
        )
    )

    pipeline = OrchestrationPipeline(
        settings=settings,
        retrieval_service=retrieval,
        model_client=standin_client,
        rate_limiter=rate_limiter,
    )
    return pipeline


@pytest.mark.asyncio
async def test_pipeline_e2e_successful_query(mock_pipeline: OrchestrationPipeline):
    """Verify standard grounded technical query completes through all 12 stages."""
    req = QueryRequest(
        question="What is the dimension of the embedding layers dmodel in the Transformer?",
        retrieval_setting="medium_k3",
        timeout_ms=10000,
    )
    res = await mock_pipeline.process_query(req, request_id="test_req_001", scenario="unit_test")

    assert res.request_id == "test_req_001"
    assert res.metrics.status == "ok"
    assert res.metrics.total_ms > 0
    assert res.metrics.first_token_ms is not None
    assert len(res.answer) > 0
    assert res.metrics.tokens_in > 0
    assert res.metrics.tokens_out > 0


@pytest.mark.asyncio
async def test_pipeline_out_of_corpus_refusal(mock_pipeline: OrchestrationPipeline):
    """Verify query with no evidence triggers deterministic refusal without guessing."""
    req = QueryRequest(
        question="What is the best secret recipe for chocolate strawberry cheesecake?",
        retrieval_setting="medium_k3",
    )
    res = await mock_pipeline.process_query(req, request_id="test_req_refusal", scenario="unit_test")

    assert res.metrics.status == "ok"
    assert res.metrics.answer_grade == "refused_correctly"
    assert "not contain sufficient evidence" in res.answer.lower() or "do not contain references" in res.answer.lower()
    assert len(res.citations) == 0


@pytest.mark.asyncio
async def test_pipeline_daily_budget_exhaustion(mock_pipeline: OrchestrationPipeline):
    """Verify that when daily budget is exhausted, system immediately emits 3pm refusal."""
    # Force daily limit to 0
    mock_pipeline.rate_limiter.rpd_limit = 0

    req = QueryRequest(question="How does Mimi codec discretize audio?")
    res = await mock_pipeline.process_query(req, request_id="test_req_daily_spent", scenario="daily_budget_test")

    assert res.metrics.status == "rate_limited"
    assert res.metrics.answer_grade == "refused_correctly"
    assert "daily research compute budget has been exhausted" in res.answer.lower()
    assert res.metrics.tokens_in == 0
    assert res.metrics.tokens_out == 0


@pytest.mark.asyncio
async def test_pipeline_client_deadline_timeout(mock_pipeline: OrchestrationPipeline):
    """Verify request where model latency exceeds client deadline drops with status='timeout'."""
    # Temporarily set model latency to 80ms while client deadline is 15ms
    mock_pipeline.model_client.config.base_ttft_ms = 80.0
    req = QueryRequest(
        question="Explain Transformer encoder stack",
        timeout_ms=15,  # 15ms deadline < 80ms TTFT
    )
    res = await mock_pipeline.process_query(req, request_id="test_req_timeout", scenario="timeout_test")

    assert res.metrics.status == "timeout"
    assert "timed out" in res.answer.lower()
    # Reset base_ttft_ms
    mock_pipeline.model_client.config.base_ttft_ms = 10.0


@pytest.mark.asyncio
async def test_pipeline_concurrent_load(mock_pipeline: OrchestrationPipeline):
    """Verify multiple concurrent requests complete successfully under load."""
    questions = [
        "What is self-attention?",
        "What is Mimi codec frame rate?",
        "What is H-Net hierarchical sequence modeling?",
        "What is the multi-head attention mechanism?",
    ]
    tasks = [
        mock_pipeline.process_query(
            QueryRequest(question=q, timeout_ms=15000),
            request_id=f"concurrent_{idx}",
            scenario="concurrency_test",
        )
        for idx, q in enumerate(questions)
    ]

    results = await asyncio.gather(*tasks)
    assert len(results) == 4
    for r in results:
        assert r.metrics.status in ("ok", "rate_limited")
        assert r.metrics.total_ms > 0


@pytest.mark.asyncio
async def test_pipeline_results_csv_logging(mock_pipeline: OrchestrationPipeline):
    """Verify telemetry record is logged conforming to results.csv schema."""
    req = QueryRequest(question="What is the learning rate warmup for Transformer?")
    res = await mock_pipeline.process_query(req, request_id="test_csv_001", scenario="audit_test")

    csv_records = mock_pipeline.results_logger.get_records()
    matched = [r for r in csv_records if r.request_id == "test_csv_001"]
    assert len(matched) == 1

    rec = matched[0]
    assert rec.scenario == "audit_test"
    assert rec.status == "ok"
    assert rec.retrieval_setting == "medium_k3"
    assert rec.total_ms > 0
