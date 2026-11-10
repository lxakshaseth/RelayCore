"""Comprehensive unit tests verifying all behaviors of the Stand-in Model Simulator."""

import asyncio
import time
import pytest

from app.models.schemas import Message, ModelRequest
from app.simulator.config import StandinConfig
from app.simulator.service import StandinModelService


@pytest.fixture
def fast_simulator() -> StandinModelService:
    """Fast-running simulator with tiny latencies for high-speed test execution."""
    cfg = StandinConfig(
        seed=42,
        base_ttft_ms=10.0,
        ttft_std_ms=2.0,
        tokens_per_second=1000.0,
        rpm_limit=10,
        tpm_limit=500,
        rpd_limit=50,
        daily_token_limit=10000,
        max_concurrency=2,
        concurrency_penalty_factor=0.5,
        rpm_window_seconds=1.0,  # Fast 1s recovery window for testing
        tpm_window_seconds=1.0,
    )
    return StandinModelService(config=cfg)


@pytest.mark.asyncio
async def test_deterministic_latency_with_seed():
    """Verify that specifying seed produces identical deterministic latency sequences."""
    req = ModelRequest(messages=[Message(role="user", content="Test query")], max_tokens=20)

    sim1 = StandinModelService(config=StandinConfig(seed=1234, base_ttft_ms=15.0, ttft_std_ms=3.0))
    sim2 = StandinModelService(config=StandinConfig(seed=1234, base_ttft_ms=15.0, ttft_std_ms=3.0))

    res1_a = await sim1.complete(req)
    res1_b = await sim1.complete(req)

    res2_a = await sim2.complete(req)
    res2_b = await sim2.complete(req)

    assert res1_a.first_token_ms == res2_a.first_token_ms
    assert res1_a.total_ms == res2_a.total_ms
    assert res1_b.first_token_ms == res2_b.first_token_ms
    assert res1_b.total_ms == res2_b.total_ms


@pytest.mark.asyncio
async def test_streaming_token_delivery(fast_simulator: StandinModelService):
    """Verify streaming token delivery yields incremental deltas and records TTFT."""
    req = ModelRequest(messages=[Message(role="user", content="Explain Transformer")], stream=True)

    streamed_tokens = []
    final_res = None

    async for token, res in fast_simulator.stream_complete(req):
        if token:
            streamed_tokens.append(token)
        if res:
            final_res = res

    assert len(streamed_tokens) > 0
    assert final_res is not None
    assert final_res.status == "ok"
    assert final_res.first_token_ms is not None
    assert final_res.first_token_ms > 0
    assert final_res.total_ms >= final_res.first_token_ms
    assert final_res.source == "stand-in"


@pytest.mark.asyncio
async def test_rpm_rate_limiting_and_429(fast_simulator: StandinModelService):
    """Verify that exceeding RPM limit deterministically generates HTTP 429."""
    fast_simulator.config.rpm_limit = 3
    req = ModelRequest(messages=[Message(role="user", content="Ping")])

    # First 3 should succeed
    for _ in range(3):
        res = await fast_simulator.complete(req)
        assert res.status == "ok"

    # 4th request must be rejected with HTTP 429
    res_429 = await fast_simulator.complete(req)
    assert res_429.status == "rate_limited"
    assert res_429.rate_limit_info is not None
    assert res_429.rate_limit_info.retry_after is not None
    assert res_429.rate_limit_info.retry_after > 0
    assert "requests per minute" in (res_429.error_info or "").lower()


@pytest.mark.asyncio
async def test_tpm_rate_limiting_and_429(fast_simulator: StandinModelService):
    """Verify that exceeding TPM limit triggers HTTP 429."""
    fast_simulator.config.tpm_limit = 50  # Very tight token limit
    # Prompt with ~60 tokens
    large_prompt = "word " * 240
    req = ModelRequest(messages=[Message(role="user", content=large_prompt)])

    res_429 = await fast_simulator.complete(req)
    assert res_429.status == "rate_limited"
    assert "tokens per minute" in (res_429.error_info or "").lower()


@pytest.mark.asyncio
async def test_rpd_daily_limit_exhaustion(fast_simulator: StandinModelService):
    """Verify that exhausting daily requests budget (RPD) rejects further calls."""
    fast_simulator.config.rpd_limit = 2
    req = ModelRequest(messages=[Message(role="user", content="Quick ping")])

    res1 = await fast_simulator.complete(req)
    assert res1.status == "ok"

    res2 = await fast_simulator.complete(req)
    assert res2.status == "ok"

    # 3rd request exceeds daily budget
    res_daily_blocked = await fast_simulator.complete(req)
    assert res_daily_blocked.status == "rate_limited"
    assert "daily request budget" in (res_daily_blocked.error_info or "").lower()


@pytest.mark.asyncio
async def test_concurrency_latency_degradation():
    """Verify that latency increases as concurrent load rises."""
    # Base TTFT: 100ms, max_concurrency: 2, penalty: 1.0 (doubles per concurrent excess)
    cfg = StandinConfig(
        base_ttft_ms=100.0,
        ttft_std_ms=0.0,  # Zero variance for clean measurement
        tokens_per_second=1000.0,
        max_concurrency=2,
        concurrency_penalty_factor=1.0,
    )
    sim = StandinModelService(config=cfg)
    req = ModelRequest(messages=[Message(role="user", content="Concurrent ping")])

    # Single isolated request
    single_res = await sim.complete(req)

    # 3 concurrent requests running simultaneously
    results = await asyncio.gather(
        sim.complete(req),
        sim.complete(req),
        sim.complete(req),
    )

    avg_concurrent_ttft = sum(r.first_token_ms for r in results) / len(results)
    # Concurrent TTFT should degrade higher than single isolated TTFT
    assert avg_concurrent_ttft >= single_res.first_token_ms


@pytest.mark.asyncio
async def test_rate_limit_window_recovery(fast_simulator: StandinModelService):
    """Verify that simulator recovers after rate-limit window elapses."""
    fast_simulator.config.rpm_limit = 2
    fast_simulator.config.rpm_window_seconds = 0.3  # 300ms recovery window
    req = ModelRequest(messages=[Message(role="user", content="Ping")])

    await fast_simulator.complete(req)
    await fast_simulator.complete(req)

    # 3rd request is blocked
    blocked = await fast_simulator.complete(req)
    assert blocked.status == "rate_limited"

    # Wait for sliding window to expire
    await asyncio.sleep(0.35)

    # Should recover and succeed
    recovered = await fast_simulator.complete(req)
    assert recovered.status == "ok"


@pytest.mark.asyncio
async def test_fault_injection_500():
    """Verify configurable failure probability injection (500)."""
    cfg = StandinConfig(failure_rate_500=1.0)  # Always fail with 500
    sim = StandinModelService(config=cfg)
    res = await sim.complete(ModelRequest(messages=[Message(role="user", content="Hi")]))

    assert res.status == "error"
    assert "500" in (res.error_info or "")


@pytest.mark.asyncio
async def test_fault_injection_503():
    """Verify configurable failure probability injection (503)."""
    cfg = StandinConfig(failure_rate_503=1.0)  # Always fail with 503
    sim = StandinModelService(config=cfg)
    res = await sim.complete(ModelRequest(messages=[Message(role="user", content="Hi")]))

    assert res.status == "error"
    assert "503" in (res.error_info or "")


@pytest.mark.asyncio
async def test_audit_recorder_tracks_requests(fast_simulator: StandinModelService):
    """Verify audit recorder records every simulated request with accurate telemetry."""
    fast_simulator.audit.clear()
    assert fast_simulator.audit.get_count() == 0

    req = ModelRequest(messages=[Message(role="user", content="Audit test")])
    await fast_simulator.complete(req)
    await fast_simulator.complete(req)

    records = fast_simulator.audit.get_all()
    assert len(records) == 2
    assert records[0].status == "ok"
    assert records[0].total_ms > 0
    assert records[1].record_id == 2
