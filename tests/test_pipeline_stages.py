"""Comprehensive 15-scenario regression and unit test suite for RelayCore 12-stage pipeline.

Validates:
1. Supported question with valid citations.
2. Unsupported question that must be refused.
3. Relevant question not incorrectly refused.
4. Citation referring to a nonexistent paper (9999.99999).
5. Citation with invalid page or section metadata (Page 999).
6. Citation verification failure (hallucination rejection).
7. Real-model request metrics (source='real', tokens > 0).
8. Pre-generation refusal metrics (source='not_called', tokens=0, first_token_ms=null).
9. Stand-in request metrics (source='stand-in').
10. RPM limit exhaustion.
11. TPM limit exhaustion.
12. Daily quota exhaustion.
13. Concurrent requests under semaphore.
14. Retry after rate limiting.
15. Timeout handling.
"""

import asyncio
import time
from typing import AsyncIterator, Optional, Tuple
import pytest

from app.core.config import Settings
from app.core.rate_limiter import RelayRateLimiter
from app.core.retry import RetryPolicy
from app.models.base import AbstractModelClient
from app.models.schemas import RateLimitInfo, ModelRequest, ModelResponse
from app.retrieval.models import DocumentChunk, RetrievalResult
from app.retrieval.service import get_retrieval_service
from app.schemas.query import QueryRequest
from app.services.orchestrator import OrchestrationPipeline
from app.simulator.config import StandinConfig
from app.simulator.service import StandinModelService


class DummyGroqClient(AbstractModelClient):
    """Simulates a real Groq client for metrics verification."""

    def __init__(self, canned_response: Optional[ModelResponse] = None):
        self.canned_response = canned_response

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if self.canned_response:
            return self.canned_response
        return ModelResponse(
            content="According to [Attention Is All You Need, Section 3.2, Page 4], multi-head attention allows the model to jointly attend to information.",
            tokens_in=512,
            tokens_out=64,
            first_token_ms=180,
            total_ms=200,
            status="ok",
            model="llama-3.3-70b-versatile",
        )

    async def stream_complete(
        self, request: ModelRequest
    ) -> AsyncIterator[Tuple[str, Optional[ModelResponse]]]:
        res = await self.complete(request)
        yield res.content, res


class FlakyRateLimitClient(AbstractModelClient):
    """Simulates a model that hits 429 RateLimit on attempt 0, then succeeds on attempt 1."""

    def __init__(self):
        self.attempts = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.attempts += 1
        if self.attempts == 1:
            return ModelResponse(
                content="Rate limit exceeded",
                tokens_in=10,
                tokens_out=0,
                total_ms=50,
                status="rate_limited",
                model="llama-3.3-70b-versatile",
                rate_limit_info=RateLimitInfo(retry_after=0.05),
            )
        return ModelResponse(
            content="The Transformer uses multi-head attention to attend to information from different representation subspaces [Attention Is All You Need, Section 3.2, Page 4].",
            tokens_in=400,
            tokens_out=50,
            first_token_ms=120,
            total_ms=150,
            status="ok",
            model="llama-3.3-70b-versatile",
        )

    async def stream_complete(
        self, request: ModelRequest
    ) -> AsyncIterator[Tuple[str, Optional[ModelResponse]]]:
        res = await self.complete(request)
        yield res.content, res


class SlowSleepingClient(AbstractModelClient):
    """Simulates an upstream client that takes longer than the client timeout."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        await asyncio.sleep(0.5)
        return ModelResponse(
            content="Slow answer",
            tokens_in=100,
            tokens_out=20,
            total_ms=500,
            status="ok",
            model="llama-3.3-70b-versatile",
        )

    async def stream_complete(
        self, request: ModelRequest
    ) -> AsyncIterator[Tuple[str, Optional[ModelResponse]]]:
        res = await self.complete(request)
        yield res.content, res


@pytest.fixture
def base_pipeline() -> OrchestrationPipeline:
    """Configured pipeline with local indexed papers and Standin model."""
    settings = Settings(
        APP_ENV="testing",
        DEBUG=True,
        DATA_DIR="data/papers",
        GROQ_RPM_LIMIT=30,
        GROQ_TPM_LIMIT=10000,
        GROQ_RPD_LIMIT=100,
        DEFAULT_CLIENT_TIMEOUT_MS=10000,
        MAX_RETRIES=2,
    )
    retrieval = get_retrieval_service(data_dir="data/papers")
    rate_limiter = RelayRateLimiter(settings=settings)
    standin = StandinModelService(
        config=StandinConfig(
            seed=42,
            base_ttft_ms=5.0,
            ttft_std_ms=1.0,
            tokens_per_second=2000.0,
        )
    )
    return OrchestrationPipeline(
        settings=settings,
        retrieval_service=retrieval,
        model_client=standin,
        rate_limiter=rate_limiter,
    )


# -----------------------------------------------------------------------------
# 1. Supported question with valid citations
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_01_supported_question_valid_citations(base_pipeline: OrchestrationPipeline):
    client = DummyGroqClient(
        canned_response=ModelResponse(
            content="The Transformer uses multi-head attention to attend to different representation subspaces [Attention Is All You Need, Section 3.2, Page 4].",
            tokens_in=600,
            tokens_out=80,
            first_token_ms=210,
            total_ms=250,
            status="ok",
            model="llama-3.3-70b-versatile",
        )
    )
    base_pipeline.set_model_client(client)
    req = QueryRequest(
        question="Why does the Transformer use multi-head attention?",
        target_papers=["1706.03762"],
        retrieval_setting="medium_k3",
    )
    res = await base_pipeline.process_query(req, request_id="req_01")
    assert res.metrics.status == "ok"
    assert res.metrics.answer_grade == "correct"
    assert len(res.citations) > 0
    assert any("1706.03762" in c.arxiv_id for c in res.citations)


# -----------------------------------------------------------------------------
# 2. Unsupported question that must be refused
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_02_unsupported_question_refusal(base_pipeline: OrchestrationPipeline):
    req = QueryRequest(
        question="What is the main purpose of RelayCore in distributed machine learning?",
        target_papers=["1706.03762"],
        retrieval_setting="medium_k3",
    )
    res = await base_pipeline.process_query(req, request_id="req_02")
    assert res.metrics.status == "ok"
    assert res.metrics.answer_grade == "refused_correctly"
    assert "not contain sufficient evidence" in res.answer.lower()
    assert len(res.citations) == 0
    assert res.metrics.source == "not_called"
    assert res.metrics.tokens_in == 0
    assert res.metrics.tokens_out == 0
    assert res.metrics.first_token_ms is None


# -----------------------------------------------------------------------------
# 3. Relevant question not incorrectly refused
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_03_relevant_question_not_incorrectly_refused(base_pipeline: OrchestrationPipeline):
    req = QueryRequest(
        question="What is the dimensionality of input and output layers dmodel in Transformer?",
        target_papers=["1706.03762"],
        retrieval_setting="medium_k3",
    )
    res = await base_pipeline.process_query(req, request_id="req_03")
    assert res.metrics.status == "ok"
    assert res.metrics.answer_grade != "refused_wrongly"
    assert len(res.answer) > 0


# -----------------------------------------------------------------------------
# 4. Citation referring to a nonexistent paper (9999.99999)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_04_citation_nonexistent_paper(base_pipeline: OrchestrationPipeline):
    client = DummyGroqClient(
        canned_response=ModelResponse(
            content="The Transformer was invented by [Nonexistent Paper 9999.99999, Section 1, Page 4].",
            tokens_in=500,
            tokens_out=40,
            first_token_ms=150,
            total_ms=180,
            status="ok",
            model="llama-3.3-70b-versatile",
        )
    )
    base_pipeline.set_model_client(client)
    req = QueryRequest(
        question="Who invented the Transformer architecture?",
        target_papers=["1706.03762"],
    )
    res = await base_pipeline.process_query(req, request_id="req_04")
    assert res.metrics.answer_grade == "wrong"
    assert len(res.citations) == 0
    assert "Answer rejected: generated response contained ungrounded claims" in res.answer


# -----------------------------------------------------------------------------
# 5. Citation with invalid page or section metadata (Page 999)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_05_citation_invalid_page_metadata(base_pipeline: OrchestrationPipeline):
    client = DummyGroqClient(
        canned_response=ModelResponse(
            content="Multi-head attention projects queries [Attention Is All You Need, Section 3.2, Page 999].",
            tokens_in=500,
            tokens_out=40,
            first_token_ms=150,
            total_ms=180,
            status="ok",
            model="llama-3.3-70b-versatile",
        )
    )
    base_pipeline.set_model_client(client)
    req = QueryRequest(
        question="Explain multi-head attention projection",
        target_papers=["1706.03762"],
    )
    res = await base_pipeline.process_query(req, request_id="req_05")
    assert res.metrics.answer_grade == "wrong"
    assert len(res.citations) == 0
    assert "Answer rejected" in res.answer


# -----------------------------------------------------------------------------
# 6. Citation verification failure cannot bypass
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_06_citation_verification_cannot_bypass(base_pipeline: OrchestrationPipeline):
    # LLM returns status='ok' (HTTP 200 equivalent) with unsupported hallucinated claims and no citations
    client = DummyGroqClient(
        canned_response=ModelResponse(
            content="Transformers utilize quantum fluctuation entanglement matrices to predict future tokens across dimensional wormholes without any mathematical equations.",
            tokens_in=450,
            tokens_out=50,
            first_token_ms=140,
            total_ms=170,
            status="ok",
            model="llama-3.3-70b-versatile",
        )
    )
    base_pipeline.set_model_client(client)
    req = QueryRequest(
        question="How does self-attention work in the Transformer?",
        target_papers=["1706.03762"],
    )
    res = await base_pipeline.process_query(req, request_id="req_06")
    assert res.metrics.answer_grade == "wrong"
    assert len(res.citations) == 0
    assert "quantum fluctuation" not in res.answer
    assert "Answer rejected: generated response contained ungrounded claims" in res.answer


# -----------------------------------------------------------------------------
# 7. Real-model request metrics (source='real', tokens > 0)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_07_real_model_request_metrics(base_pipeline: OrchestrationPipeline):
    class GroqAsyncClient(AbstractModelClient):
        async def complete(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(
                content="Transformer attention is computed as softmax(QK^T / sqrt(dk))V [Attention Is All You Need, Section 3.2, Page 4].",
                tokens_in=720,
                tokens_out=95,
                first_token_ms=230,
                total_ms=260,
                status="ok",
                model="llama-3.3-70b-versatile",
            )

        async def stream_complete(
            self, request: ModelRequest
        ) -> AsyncIterator[Tuple[str, Optional[ModelResponse]]]:
            res = await self.complete(request)
            yield res.content, res

    base_pipeline.set_model_client(GroqAsyncClient())
    req = QueryRequest(
        question="How is scaled dot product attention defined?",
        target_papers=["1706.03762"],
    )
    res = await base_pipeline.process_query(req, request_id="req_07")
    assert res.metrics.source == "real"
    assert res.metrics.tokens_in == 720
    assert res.metrics.tokens_out == 95
    assert res.metrics.first_token_ms == 230


# -----------------------------------------------------------------------------
# 8. Pre-generation refusal metrics (source='not_called', tokens=0, first_token_ms=null)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_08_pre_generation_refusal_metrics(base_pipeline: OrchestrationPipeline):
    req = QueryRequest(
        question="Can you give me a recipe for baked chocolate brownies?",
        retrieval_setting="medium_k3",
    )
    res = await base_pipeline.process_query(req, request_id="req_08")
    assert res.metrics.source == "not_called"
    assert res.metrics.tokens_in == 0
    assert res.metrics.tokens_out == 0
    assert res.metrics.first_token_ms is None
    assert res.metrics.answer_grade == "refused_correctly"


# -----------------------------------------------------------------------------
# 9. Stand-in request metrics (source='stand-in')
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_09_standin_request_metrics(base_pipeline: OrchestrationPipeline):
    req = QueryRequest(
        question="What is the encoder architecture of Transformer?",
        target_papers=["1706.03762"],
    )
    res = await base_pipeline.process_query(req, request_id="req_09")
    assert res.metrics.source == "stand-in"
    assert res.metrics.tokens_in > 0
    assert res.metrics.tokens_out > 0
    assert res.metrics.first_token_ms is not None


# -----------------------------------------------------------------------------
# 10. RPM limit exhaustion
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_10_rpm_limit_exhaustion(base_pipeline: OrchestrationPipeline):
    base_pipeline.rate_limiter.request_bucket.tokens = 0.0
    base_pipeline.rate_limiter.request_bucket.refill_rate = 0.0001

    req = QueryRequest(
        question="Explain scaled dot product attention",
        timeout_ms=50,
    )
    res = await base_pipeline.process_query(req, request_id="req_10")
    assert res.metrics.status == "rate_limited"
    assert res.metrics.source == "not_called"
    assert res.metrics.tokens_in == 0
    assert res.metrics.tokens_out == 0


# -----------------------------------------------------------------------------
# 11. TPM limit exhaustion
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_11_tpm_limit_exhaustion(base_pipeline: OrchestrationPipeline):
    base_pipeline.rate_limiter.token_bucket.tokens = 0.0
    base_pipeline.rate_limiter.token_bucket.refill_rate = 0.0001

    req = QueryRequest(
        question="Explain the encoder self-attention mechanism and feedforward layers",
        timeout_ms=50,
    )
    res = await base_pipeline.process_query(req, request_id="req_11")
    assert res.metrics.status == "rate_limited"
    assert res.metrics.source == "not_called"


# -----------------------------------------------------------------------------
# 12. Daily quota exhaustion
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_12_daily_quota_exhaustion(base_pipeline: OrchestrationPipeline):
    base_pipeline.rate_limiter.rpd_limit = 0
    assert base_pipeline.rate_limiter.is_daily_budget_exhausted() is True

    req = QueryRequest(question="What is Mimi codec?")
    res = await base_pipeline.process_query(req, request_id="req_12")
    assert res.metrics.status == "rate_limited"
    assert res.metrics.answer_grade == "refused_correctly"
    assert res.metrics.source == "not_called"
    assert res.metrics.tokens_in == 0
    assert res.metrics.tokens_out == 0
    assert "daily research compute budget has been exhausted" in res.answer.lower()


# -----------------------------------------------------------------------------
# 13. Concurrent requests under semaphore
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_13_concurrent_requests_under_semaphore(base_pipeline: OrchestrationPipeline):
    reqs = [
        QueryRequest(question=f"What is self attention question {i}?", timeout_ms=8000)
        for i in range(5)
    ]
    tasks = [
        base_pipeline.process_query(req, request_id=f"concurrent_req_{i}")
        for i, req in enumerate(reqs)
    ]
    results = await asyncio.gather(*tasks)
    assert len(results) == 5
    for r in results:
        assert r.metrics.status in ("ok", "rate_limited")
        assert r.metrics.total_ms > 0


# -----------------------------------------------------------------------------
# 14. Retry after rate limiting
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_14_retry_after_rate_limiting(base_pipeline: OrchestrationPipeline):
    flaky_client = FlakyRateLimitClient()
    base_pipeline.set_model_client(flaky_client)
    req = QueryRequest(
        question="Why does the Transformer use multi-head attention?",
        target_papers=["1706.03762"],
        timeout_ms=10000,
    )
    res = await base_pipeline.process_query(req, request_id="req_14")
    assert res.metrics.status == "ok"
    assert res.metrics.retries == 1
    assert flaky_client.attempts == 2
    assert res.metrics.answer_grade == "correct"


# -----------------------------------------------------------------------------
# 15. Timeout handling
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_15_timeout_handling(base_pipeline: OrchestrationPipeline):
    slow_client = SlowSleepingClient()
    base_pipeline.set_model_client(slow_client)
    req = QueryRequest(
        question="Explain Transformer multi-head attention",
        target_papers=["1706.03762"],
        timeout_ms=50,  # 50ms deadline << 500ms sleep
    )
    res = await base_pipeline.process_query(req, request_id="req_15")
    assert res.metrics.status == "timeout"
    assert "timed out" in res.answer.lower()
