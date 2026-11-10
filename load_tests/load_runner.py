"""Load-testing harness executing the 5 mandatory scenarios for RelayCore."""

import asyncio
import logging
import time
from typing import List, Optional

from app.core.config import Settings
from app.core.rate_limiter import RelayRateLimiter
from app.evaluation.benchmarks import BenchmarkQuestion, load_benchmark_questions
from app.models.base import AbstractModelClient
from app.models.groq_client import GroqAsyncClient
from app.retrieval.service import get_retrieval_service
from app.schemas.query import QueryRequest, QueryResponse
from app.services.orchestrator import OrchestrationPipeline
from app.simulator.config import StandinConfig
from app.simulator.service import StandinModelService

logger = logging.getLogger(__name__)


class LoadTestRunner:
    """Orchestrates multi-user concurrent synthetic and real load scenarios."""

    def __init__(self, data_questions_path: str = "data/test_questions.json"):
        self.questions = load_benchmark_questions(data_questions_path)
        self.retrieval_service = get_retrieval_service()

    def _create_pipeline(
        self,
        model_client: AbstractModelClient,
        rpm: int = 30,
        tpm: int = 8000,
        rpd: int = 1000,
        concurrency: int = 5,
        max_queue_depth: int = 100,
    ) -> OrchestrationPipeline:
        settings = Settings(
            GROQ_RPM_LIMIT=rpm,
            GROQ_TPM_LIMIT=tpm,
            GROQ_RPD_LIMIT=rpd,
            WORKER_CONCURRENCY=concurrency,
            MAX_QUEUE_DEPTH=max_queue_depth,
        )
        rate_limiter = RelayRateLimiter(settings=settings)
        return OrchestrationPipeline(
            settings=settings,
            retrieval_service=self.retrieval_service,
            model_client=model_client,
            rate_limiter=rate_limiter,
        )

    async def run_scenario(
        self,
        scenario_name: str,
        pipeline: OrchestrationPipeline,
        num_requests: int = 15,
        concurrency: int = 3,
        inter_request_delay_s: float = 0.0,
        timeout_ms: int = 15000,
        retrieval_setting: str = "medium_k3",
    ) -> List[QueryResponse]:
        """Execute a batch of queries with controlled concurrency and spacing."""
        logger.info("Starting scenario '%s': %d requests, concurrency=%d", scenario_name, num_requests, concurrency)
        semaphore = asyncio.Semaphore(concurrency)
        responses: List[QueryResponse] = []

        async def worker(idx: int, q_item: BenchmarkQuestion):
            async with semaphore:
                req_id = f"{scenario_name}_{idx:03d}"
                req = QueryRequest(
                    question=q_item.question,
                    target_papers=[q_item.target_paper] if q_item.target_paper and q_item.target_paper != "none" else None,
                    retrieval_setting=retrieval_setting,
                    timeout_ms=timeout_ms,
                )
                res = await pipeline.process_query(req, request_id=req_id, scenario=scenario_name)
                responses.append(res)
                if inter_request_delay_s > 0:
                    await asyncio.sleep(inter_request_delay_s)

        tasks = []
        for i in range(num_requests):
            q = self.questions[i % len(self.questions)]
            tasks.append(worker(i + 1, q))

        await asyncio.gather(*tasks)
        logger.info("Scenario '%s' completed. Collected %d responses.", scenario_name, len(responses))
        return responses

    async def run_steady_load(
        self, num_requests: int = 10, is_real: bool = False
    ) -> List[QueryResponse]:
        """Scenario 1: Steady load within normal system capacity."""
        if is_real:
            model = GroqAsyncClient()
        else:
            model = StandinModelService(StandinConfig(seed=42, rpm_limit=30, tpm_limit=8000))

        pipeline = self._create_pipeline(model, rpm=30, tpm=8000, rpd=1000, concurrency=2)
        # Moderate uniform spacing (1.5 seconds between requests)
        return await self.run_scenario(
            scenario_name="steady_real" if is_real else "steady",
            pipeline=pipeline,
            num_requests=num_requests,
            concurrency=2,
            inter_request_delay_s=0.5,
        )

    async def run_sudden_burst(self, num_requests: int = 25) -> List[QueryResponse]:
        """Scenario 2: Sudden surge of requests hitting admission queue all at once."""
        model = StandinModelService(
            StandinConfig(seed=101, rpm_limit=30, tpm_limit=8000, max_concurrency=3)
        )
        pipeline = self._create_pipeline(model, rpm=30, tpm=8000, rpd=1000, concurrency=3, max_queue_depth=30)
        # 25 requests with concurrency=15, 0 delay (flash burst)
        return await self.run_scenario(
            scenario_name="burst",
            pipeline=pipeline,
            num_requests=num_requests,
            concurrency=15,
            inter_request_delay_s=0.0,
        )

    async def run_sustained_overload(self, num_requests: int = 40) -> List[QueryResponse]:
        """Scenario 3: Prolonged high-traffic arrival rate exceeding capacity."""
        model = StandinModelService(
            StandinConfig(seed=202, rpm_limit=30, tpm_limit=8000, max_concurrency=3)
        )
        pipeline = self._create_pipeline(model, rpm=30, tpm=8000, rpd=1000, concurrency=5, max_queue_depth=50)
        # 40 requests sent rapidly with tight deadline (7s) to measure deadline shedding
        return await self.run_scenario(
            scenario_name="sustained_overload",
            pipeline=pipeline,
            num_requests=num_requests,
            concurrency=6,
            inter_request_delay_s=0.05,
            timeout_ms=7000,
        )

    async def run_limit_halved(self, num_requests: int = 20) -> List[QueryResponse]:
        """Scenario 4: Upstream per-minute limits halved (RPM=15, TPM=4000)."""
        model = StandinModelService(
            StandinConfig(seed=303, rpm_limit=15, tpm_limit=4000, max_concurrency=2)
        )
        pipeline = self._create_pipeline(model, rpm=15, tpm=4000, rpd=1000, concurrency=2)
        return await self.run_scenario(
            scenario_name="limit_halved",
            pipeline=pipeline,
            num_requests=num_requests,
            concurrency=4,
            inter_request_delay_s=0.1,
        )

    async def run_daily_budget_exhausted(self, num_requests: int = 15) -> List[QueryResponse]:
        """Scenario 5: 3 PM daily budget exhaustion scenario (RPD=0)."""
        model = StandinModelService(StandinConfig(seed=404, rpd_limit=0))
        pipeline = self._create_pipeline(model, rpm=30, tpm=8000, rpd=0)
        return await self.run_scenario(
            scenario_name="daily_budget_exhausted",
            pipeline=pipeline,
            num_requests=num_requests,
            concurrency=5,
        )
