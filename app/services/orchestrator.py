"""Complete 12-stage RelayCore Orchestration Pipeline."""

import asyncio
import logging
import random
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import Settings, get_settings
from app.core.queue_manager import BoundedPriorityQueueManager
from app.core.rate_limiter import RelayRateLimiter
from app.core.retry import RetryPolicy
from app.metrics.results_logger import RequestMetricRecord, get_results_logger
from app.models.base import AbstractModelClient
from app.models.groq_client import GroqAsyncClient
from app.models.schemas import Message, ModelRequest, ModelResponse
from app.retrieval.models import DocumentChunk, RetrievalResponse, RetrievalResult
from app.retrieval.parser import CANONICAL_PAPERS
from app.retrieval.service import RetrievalService, get_retrieval_service
from app.schemas.query import CitationItem, ExecutionMetrics, QueryRequest, QueryResponse
from app.simulator.service import StandinModelService

logger = logging.getLogger(__name__)

# System instructions enforcing strict paper grounding and verified citations
SYSTEM_PROMPT = """You are RelayCore, a senior AI engineering research assistant.
Answer the user's technical question using ONLY the provided research paper passages.

Rules:
1. Every claim MUST be supported by the text and explicitly cited using bracketed citations: [Paper Title, Section X.Y, Page Z].
2. If the provided passages do not contain sufficient evidence to answer the question, you MUST say:
   "The provided research papers do not contain sufficient evidence to answer this question."
3. Do NOT extrapolate, hallucinate, or guess facts not present in the excerpts.
"""


class OrchestrationPipeline:
    """Manages the 12-stage end-to-end request lifecycle with resilience, queues, and quality verification."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        retrieval_service: Optional[RetrievalService] = None,
        model_client: Optional[AbstractModelClient] = None,
        rate_limiter: Optional[Any] = None,
        queue_manager: Optional[BoundedPriorityQueueManager] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ):
        self.settings = settings or get_settings()
        self.retrieval_service = retrieval_service or get_retrieval_service(data_dir=self.settings.data_dir)
        self.rate_limiter = rate_limiter or RelayRateLimiter(settings=self.settings)
        self.queue_manager = queue_manager or BoundedPriorityQueueManager(
            max_depth=self.settings.max_queue_depth,
            max_concurrency=self.settings.worker_concurrency,
        )
        self.retry_policy = retry_policy or RetryPolicy(max_retries=self.settings.max_retries)
        self.results_logger = get_results_logger()

        # Model backend: defaults to GroqAsyncClient or StandinModelService based on SIMULATOR_ENABLED
        if model_client:
            self.model_client = model_client
        elif self.settings.simulator_enabled:
            self.model_client = StandinModelService()
        else:
            self.model_client = GroqAsyncClient(settings=self.settings)

    def set_model_client(self, client: AbstractModelClient) -> None:
        """Dynamically swap model client (e.g. between real Groq and Standin)."""
        self.model_client = client

    # -------------------------------------------------------------------------
    # STAGE 4: Question Classification & Target Paper Routing
    # -------------------------------------------------------------------------
    def classify_question(self, question: str) -> Optional[List[str]]:
        """Identify target paper scope and resolve cross-paper terminology collisions."""
        q_lower = question.lower()

        # Disambiguate "chunking" in audio (Moshi) vs sequence modeling (Dynamic Chunking)
        if "dynamic chunking" in q_lower or "h-net" in q_lower or "hierarchical sequence" in q_lower:
            return ["2507.07955v2"]
        if "moshi" in q_lower or "mimi" in q_lower or "speech" in q_lower or "codec" in q_lower or "rvq" in q_lower:
            return ["2410.00037v1"]
        if "transformer" in q_lower or "attention is all you need" in q_lower or "vaswani" in q_lower or "multi-head" in q_lower or "dmodel" in q_lower:
            return ["1706.03762v7"]

        return None  # Broad search across all 3 papers

    # -------------------------------------------------------------------------
    # STAGE 6: Context Selection & Prompt Construction
    # -------------------------------------------------------------------------
    def format_passages_context(self, passages: List[RetrievalResult], max_passages: int = 4) -> str:
        """Construct structured, grounded context block for LLM prompt."""
        formatted_blocks = []
        for idx, p in enumerate(passages[:max_passages], start=1):
            chunk = p.chunk
            header = f"[PASSAGE {idx}] (Paper: '{chunk.paper_title}', arXiv: {chunk.paper_id}, Section: '{chunk.section}', Page: {chunk.page_number})"
            formatted_blocks.append(f"{header}\n{chunk.content}")
        return "\n\n".join(formatted_blocks)

    # -------------------------------------------------------------------------
    # STAGE 8 & 9: Citation Validation & Answer Quality Checking
    # -------------------------------------------------------------------------
    def validate_citations_and_quality(
        self, answer: str, passages: List[RetrievalResult], has_evidence: bool
    ) -> Tuple[str, List[CitationItem], bool]:
        """Verify grounding, parse bracketed citations, and assign answer_grade.

        Returns (answer_grade, parsed_citations, is_valid).
        """
        answer_lower = answer.lower()

        # Check for explicit refusal in answer text
        refusal_phrases = [
            "do not contain sufficient evidence",
            "does not contain sufficient evidence",
            "papers do not state",
            "papers do not provide",
            "cannot be answered from the provided",
            "not mentioned in the",
            "insufficient evidence to answer",
        ]
        is_refusal = any(p in answer_lower for p in refusal_phrases)

        if not has_evidence:
            if is_refusal:
                return "refused_correctly", [], True
            else:
                return "wrong", [], False

        if is_refusal:
            # Model refused despite passages having evidence
            return "refused_wrongly", [], False

        # Canonical papers validation map
        canonical_arxiv_bases = {entry.arxiv_id for entry in CANONICAL_PAPERS.values()}

        # 1. Detect hallucinated or nonexistent arXiv IDs in text or bracketed citations
        # e.g., '9999.99999' or 'arxiv:9999.99999'
        arxiv_matches = re.findall(r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b", answer)
        for matched_id in arxiv_matches:
            base_id = matched_id.split("v")[0]
            if base_id not in canonical_arxiv_bases:
                logger.warning("Citation references nonexistent paper arXiv ID: %s", matched_id)
                return "wrong", [], False

        # 2. Detect out-of-bounds page numbers (all 3 papers have <= 60 pages)
        page_matches = re.findall(r"(?:page|p\.)\s*(\d+)", answer, re.IGNORECASE)
        for p_str in page_matches:
            p_num = int(p_str)
            if p_num <= 0 or p_num > 60:
                logger.warning("Citation contains out-of-bounds page number: %d", p_num)
                return "wrong", [], False

        # 3. Check bracketed citations for section/page/paper consistency against retrieved chunks
        retrieved_paper_bases = {p.chunk.paper_id.split("v")[0] for p in passages}
        retrieved_pages = {p.chunk.page_number for p in passages}
        bracket_citations = re.findall(r"\[([^\]]+)\]", answer)
        for b_cit in bracket_citations:
            # Check arXiv IDs inside brackets
            b_arxiv = re.findall(r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b", b_cit)
            for m_id in b_arxiv:
                b_base = m_id.split("v")[0]
                if b_base not in canonical_arxiv_bases:
                    logger.warning("Bracket citation references nonexistent paper: %s", m_id)
                    return "wrong", [], False
                if b_base not in retrieved_paper_bases:
                    logger.warning("Bracket citation references paper not in retrieved passages: %s", m_id)
                    return "wrong", [], False

            # Check pages inside brackets
            b_pages = re.findall(r"(?:page|p\.)\s*(\d+)", b_cit, re.IGNORECASE)
            for p_str in b_pages:
                p_num = int(p_str)
                if p_num <= 0 or p_num > 60:
                    logger.warning("Bracket citation page out of bounds: %d", p_num)
                    return "wrong", [], False
                if p_num not in retrieved_pages:
                    logger.warning("Bracket citation page %d not in retrieved chunks (%s)", p_num, retrieved_pages)
                    return "wrong", [], False

        # 4. Check claim grounding lexical overlap against retrieved passages
        combined_passages_text = " ".join(p.chunk.content for p in passages)
        stopwords = {
            "the", "is", "at", "which", "on", "and", "a", "an", "in", "to", "of",
            "for", "with", "as", "by", "that", "this", "it", "from", "are", "be",
            "was", "were", "or", "an", "will", "can", "has", "have", "had", "not",
            "but", "what", "how", "why", "when", "where", "who", "whom", "its"
        }
        answer_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", answer_lower)) - stopwords
        passages_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", combined_passages_text.lower())) - stopwords

        if answer_words:
            overlap = len(answer_words.intersection(passages_words)) / len(answer_words)
            if overlap < 0.15:
                logger.warning(
                    "Answer grounding lexical overlap too low: %.1f%% (minimum 15%% required)",
                    overlap * 100,
                )
                return "wrong", [], False

        # 5. Extract structured citations from model output and match against retrieved chunks
        extracted_citations: List[CitationItem] = []
        matched_chunks = set()

        for p in passages:
            chunk = p.chunk
            # Check if paper title, arXiv ID, or section is mentioned in the text
            paper_ref = (
                chunk.paper_id in answer
                or chunk.paper_id.split("v")[0] in answer
                or chunk.paper_title[:20].lower() in answer_lower
            )
            section_ref = (
                chunk.section.lower() in answer_lower
                or f"page {chunk.page_number}" in answer_lower
                or f"p. {chunk.page_number}" in answer_lower
            )

            if paper_ref or section_ref:
                matched_chunks.add(chunk.chunk_id)
                extracted_citations.append(
                    CitationItem(
                        paper_title=chunk.paper_title,
                        arxiv_id=chunk.paper_id,
                        section=chunk.section,
                        page=chunk.page_number,
                        passage_snippet=chunk.content[:200] + "...",
                    )
                )

        if extracted_citations:
            return "correct", extracted_citations, True

        # If answer makes claims but contains zero citations matching retrieved text
        return "wrong", [], False

    # -------------------------------------------------------------------------
    # MAIN 12-STAGE PIPELINE EXECUTION
    # -------------------------------------------------------------------------
    async def process_query(
        self, request: QueryRequest, request_id: str, scenario: str = "steady"
    ) -> QueryResponse:
        """Execute full 12-stage pipeline with backpressure, rate-limiting, and retries."""
        start_time_iso = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        retries = 0
        total_tokens_in = 0
        total_tokens_out = 0
        first_token_ms: Optional[int] = None
        status = "ok"
        answer_grade = "correct"
        final_answer = ""
        final_citations: List[CitationItem] = []
        setting_name = request.retrieval_setting or "medium_k3"

        timeout_seconds = (request.timeout_ms or self.settings.default_client_timeout_ms) / 1000.0
        deadline_time = time.time() + timeout_seconds

        try:
            # -----------------------------------------------------------------
            # STAGE 1: Request Validation
            # -----------------------------------------------------------------
            clean_question = request.question.strip()
            if len(clean_question) < 3:
                status = "error"
                answer_grade = "refused_correctly"
                final_answer = "Invalid query: question must be at least 3 characters long."
                total_ms = max(1, int((time.perf_counter() - t0) * 1000))
                await self._log_metrics(
                    request_id=request_id,
                    scenario=scenario,
                    start_time=start_time_iso,
                    first_token_ms=None,
                    total_ms=total_ms,
                    status=status,
                    retries=0,
                    tokens_in=0,
                    tokens_out=0,
                    retrieval_setting=setting_name,
                    answer_grade=answer_grade,
                    source="not_called",
                )
                return self._build_response(
                    request_id=request_id,
                    question=clean_question,
                    answer=final_answer,
                    citations=[],
                    first_token_ms=None,
                    total_ms=total_ms,
                    status=status,
                    tokens_in=0,
                    tokens_out=0,
                    retries=0,
                    setting_name=setting_name,
                    answer_grade=answer_grade,
                    source="not_called",
                )

            # -----------------------------------------------------------------
            # STAGE 2: Request Queueing & Admission Control
            # -----------------------------------------------------------------
            if time.time() > deadline_time:
                raise TimeoutError("Client timeout expired before admission to pipeline.")

            # -----------------------------------------------------------------
            # STAGE 3: Rate-Limit Checking & Daily Budget Exhaustion
            # -----------------------------------------------------------------
            if self.rate_limiter.is_daily_budget_exhausted():
                status = "rate_limited"
                answer_grade = "refused_correctly"
                final_answer = (
                    "The daily research compute budget has been exhausted for today (resets at 00:00 UTC). "
                    "New generative inferences cannot be processed, but indexed research passages remain accessible."
                )
                total_ms = max(1, int((time.perf_counter() - t0) * 1000))
                await self._log_metrics(
                    request_id=request_id,
                    scenario=scenario,
                    start_time=start_time_iso,
                    first_token_ms=None,
                    total_ms=total_ms,
                    status=status,
                    retries=0,
                    tokens_in=0,
                    tokens_out=0,
                    retrieval_setting=setting_name,
                    answer_grade=answer_grade,
                    source="not_called",
                )
                return self._build_response(
                    request_id=request_id,
                    question=clean_question,
                    answer=final_answer,
                    citations=[],
                    first_token_ms=None,
                    total_ms=total_ms,
                    status=status,
                    tokens_in=0,
                    tokens_out=0,
                    retries=0,
                    setting_name=setting_name,
                    answer_grade=answer_grade,
                    source="not_called",
                )

            # -----------------------------------------------------------------
            # STAGE 4: Question Classification
            # -----------------------------------------------------------------
            target_papers = request.target_papers or self.classify_question(clean_question)

            # -----------------------------------------------------------------
            # STAGE 5: Research-Paper Retrieval (BM25 Tool Call)
            # -----------------------------------------------------------------
            if time.time() > deadline_time:
                raise TimeoutError("Client timeout expired before retrieval tool execution.")

            retrieval_res: RetrievalResponse = self.retrieval_service.retrieve(
                query=clean_question,
                setting_name=setting_name,
                target_papers=target_papers,
            )

            # If retrieval found no evidence or question is out of domain
            if not retrieval_res.has_evidence:
                status = "ok"
                answer_grade = "refused_correctly"
                target_desc = f" (targeted papers: {target_papers})" if target_papers else " (across all indexed papers)"
                final_answer = (
                    f"The provided research papers{target_desc} do not contain sufficient evidence to answer: '{clean_question}'. "
                    f"Reason: {retrieval_res.refusal_reason}"
                )
                total_ms = max(1, int((time.perf_counter() - t0) * 1000))
                await self._log_metrics(
                    request_id=request_id,
                    scenario=scenario,
                    start_time=start_time_iso,
                    first_token_ms=None,
                    total_ms=total_ms,
                    status=status,
                    retries=retries,
                    tokens_in=0,
                    tokens_out=0,
                    retrieval_setting=setting_name,
                    answer_grade=answer_grade,
                    source="not_called",
                )
                return self._build_response(
                    request_id=request_id,
                    question=clean_question,
                    answer=final_answer,
                    citations=[],
                    first_token_ms=None,
                    total_ms=total_ms,
                    status=status,
                    tokens_in=0,
                    tokens_out=0,
                    retries=retries,
                    setting_name=setting_name,
                    answer_grade=answer_grade,
                    source="not_called",
                )

            # -----------------------------------------------------------------
            # STAGE 6: Context Selection & Token Budgeting
            # -----------------------------------------------------------------
            passages_context = self.format_passages_context(retrieval_res.passages)
            estimated_prompt_tokens = len(SYSTEM_PROMPT + passages_context + clean_question) // 4

            # Acquire rate limit slot from client token bucket
            remaining_time = max(0.5, deadline_time - time.time())
            if hasattr(self.rate_limiter, "acquire_slot"):
                acquired, limit_reason = await self.rate_limiter.acquire_slot(
                    estimated_tokens=estimated_prompt_tokens,
                    max_wait_seconds=min(remaining_time, 10.0),
                )
            else:
                acquired, limit_reason = await self.rate_limiter.acquire_or_wait(
                    estimated_tokens=estimated_prompt_tokens,
                    max_wait_seconds=min(remaining_time, 10.0),
                )
            if not acquired:
                status = "rate_limited"
                answer_grade = "refused_wrongly"
                final_answer = f"Request delayed and shed due to provider rate limits: {limit_reason}"
                total_ms = max(1, int((time.perf_counter() - t0) * 1000))
                await self._log_metrics(
                    request_id=request_id,
                    scenario=scenario,
                    start_time=start_time_iso,
                    first_token_ms=None,
                    total_ms=total_ms,
                    status=status,
                    retries=retries,
                    tokens_in=0,
                    tokens_out=0,
                    retrieval_setting=setting_name,
                    answer_grade=answer_grade,
                    source="not_called",
                )
                return self._build_response(
                    request_id=request_id,
                    question=clean_question,
                    answer=final_answer,
                    citations=[],
                    first_token_ms=None,
                    total_ms=total_ms,
                    status=status,
                    tokens_in=0,
                    tokens_out=0,
                    retries=retries,
                    setting_name=setting_name,
                    answer_grade=answer_grade,
                    source="not_called",
                )

            # -----------------------------------------------------------------
            # STAGE 7: Model Generation with Retry & Reflection Loop (Stages 7-10)
            # -----------------------------------------------------------------
            max_attempts = self.settings.max_retries + 1
            model_response: Optional[ModelResponse] = None

            for attempt in range(max_attempts):
                remaining_deadline = deadline_time - time.time()
                if remaining_deadline <= 0:
                    status = "timeout"
                    answer_grade = "refused_wrongly"
                    final_answer = "Request timed out: client deadline expired before model invocation."
                    break

                prompt_messages = [
                    Message(role="system", content=SYSTEM_PROMPT),
                    Message(
                        role="user",
                        content=f"Research Excerpts:\n{passages_context}\n\nQuestion: {clean_question}",
                    ),
                ]

                # Invoke Model Backend with Concurrency Semaphore and strict deadline timeout
                call_timeout = min(15.0, max(0.01, remaining_deadline))
                model_req = ModelRequest(
                    messages=prompt_messages,
                    timeout_seconds=call_timeout,
                    temperature=0.1,
                    max_tokens=256,
                )
                if hasattr(self.rate_limiter, "concurrency_semaphore"):
                    async with self.rate_limiter.concurrency_semaphore:
                        model_response = await asyncio.wait_for(
                            self.model_client.complete(model_req),
                            timeout=call_timeout,
                        )
                else:
                    model_response = await asyncio.wait_for(
                        self.model_client.complete(model_req),
                        timeout=call_timeout,
                    )

                if time.time() > deadline_time:
                    status = "timeout"
                    answer_grade = "refused_wrongly"
                    final_answer = "Request timed out: client deadline was exceeded during inference."
                    break

                if attempt == 0:
                    first_token_ms = model_response.first_token_ms

                total_tokens_in += model_response.tokens_in
                total_tokens_out += model_response.tokens_out

                # Reconcile actual tokens in client rate limiter
                await self.rate_limiter.record_actual_tokens(
                    estimated_tokens=estimated_prompt_tokens,
                    actual_tokens=model_response.tokens_in + model_response.tokens_out,
                )

                # STAGE 8 & 9: Citation Validation & Answer Quality
                if model_response.status == "ok":
                    grade, citations, is_valid = self.validate_citations_and_quality(
                        model_response.content, retrieval_res.passages, retrieval_res.has_evidence
                    )
                    if is_valid or grade == "refused_correctly":
                        answer_grade = grade
                        final_answer = model_response.content
                        final_citations = citations
                        status = "ok"
                        break
                    else:
                        logger.warning(
                            "Attempt %d: Output lacked valid citations or contained hallucinations (grade=%s).",
                            attempt + 1,
                            grade,
                        )
                        answer_grade = grade
                        final_citations = []
                        final_answer = (
                            "Answer rejected: generated response contained ungrounded claims "
                            "or invalid citations that could not be verified against the source text."
                        )
                        # Check if retry attempt is permitted and fits client deadline
                        if attempt < self.settings.max_retries and (deadline_time - time.time()) > 1.0:
                            retries += 1
                            backoff = self.retry_policy.compute_backoff(attempt=attempt)
                            await asyncio.sleep(backoff)
                            continue
                        else:
                            # Retries exhausted: ensure ungrounded claim is NEVER returned
                            status = "ok"
                            break
                else:
                    # Upstream model returned non-OK status (e.g. rate_limited, error)
                    daily_exhausted = (
                        self.rate_limiter.is_daily_budget_exhausted()
                        if hasattr(self.rate_limiter, "is_daily_budget_exhausted")
                        else False
                    )
                    if self.retry_policy.is_retriable(
                        response=model_response,
                        attempt=attempt,
                        deadline_time=deadline_time,
                        is_daily_exhausted=daily_exhausted,
                    ):
                        retries += 1
                        if self.retry_policy.should_shed_context_on_retry(model_response):
                            logger.warning("Upstream rate limited on TPM. Shedding context passages for retry...")
                            passages_context = self.format_passages_context(retrieval_res.passages, max_passages=2)

                        retry_after = (
                            model_response.rate_limit_info.retry_after
                            if model_response and model_response.rate_limit_info
                            else None
                        )
                        backoff = self.retry_policy.compute_backoff(attempt=attempt, retry_after=retry_after)
                        await asyncio.sleep(backoff)
                    else:
                        status = model_response.status if model_response else "error"
                        final_answer = model_response.content if model_response else "Failed after maximum retries."
                        answer_grade = "refused_wrongly" if status == "rate_limited" else "wrong"
                        break

            if status == "timeout" and not final_answer:
                final_answer = "Request timed out: client deadline was exceeded."

            if status == "ok" and not final_answer:
                final_answer = model_response.content if model_response else "Unable to synthesize answer."

        except (TimeoutError, asyncio.TimeoutError) as e:
            status = "timeout"
            answer_grade = "refused_wrongly"
            final_answer = f"Request timed out: {str(e) if str(e) else 'client deadline exceeded.'}"
        except Exception as e:
            status = "error"
            answer_grade = "wrong"
            final_answer = f"Internal pipeline error: {type(e).__name__}: {str(e)}"
            logger.exception("Orchestration pipeline exception for req %s", request_id)

        # ---------------------------------------------------------------------
        # STAGE 11: Final Response Generation
        # ---------------------------------------------------------------------
        total_ms = max(1, int((time.perf_counter() - t0) * 1000))
        if total_tokens_in > 0 or total_tokens_out > 0 or first_token_ms is not None:
            source_name = getattr(self.model_client, "__class__", None).__name__
            source_tag = "stand-in" if "Standin" in source_name else "real"
        else:
            source_tag = "not_called"

        # ---------------------------------------------------------------------
        # STAGE 12: Metrics Logging
        # ---------------------------------------------------------------------
        await self._log_metrics(
            request_id=request_id,
            scenario=scenario,
            start_time=start_time_iso,
            first_token_ms=first_token_ms,
            total_ms=total_ms,
            status=status,
            retries=retries,
            tokens_in=total_tokens_in,
            tokens_out=total_tokens_out,
            retrieval_setting=setting_name,
            answer_grade=answer_grade,
            source=source_tag,
        )

        return self._build_response(
            request_id=request_id,
            question=clean_question if 'clean_question' in locals() else request.question,
            answer=final_answer,
            citations=final_citations,
            first_token_ms=first_token_ms,
            total_ms=total_ms,
            status=status,
            tokens_in=total_tokens_in,
            tokens_out=total_tokens_out,
            retries=retries,
            setting_name=setting_name,
            answer_grade=answer_grade,
            source=source_tag,
        )

    async def _log_metrics(
        self,
        request_id: str,
        scenario: str,
        start_time: str,
        first_token_ms: Optional[int],
        total_ms: int,
        status: str,
        retries: int,
        tokens_in: int,
        tokens_out: int,
        retrieval_setting: str,
        answer_grade: str,
        source: Optional[str] = None,
    ) -> None:
        """Write record to results.csv."""
        if source is None:
            if tokens_in == 0 and tokens_out == 0 and first_token_ms is None:
                source = "not_called"
            else:
                source_name = getattr(self.model_client, "__class__", None).__name__
                source = "stand-in" if "Standin" in source_name else "real"
        record = RequestMetricRecord(
            request_id=request_id,
            scenario=scenario,
            source=source,
            start_time=start_time,
            first_token_ms=first_token_ms,
            total_ms=total_ms,
            status=status,
            retries=retries,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            retrieval_setting=retrieval_setting,
            answer_grade=answer_grade,
        )
        await self.results_logger.log_request(record)

    def _build_response(
        self,
        request_id: str,
        question: str,
        answer: str,
        citations: List[CitationItem],
        first_token_ms: Optional[int],
        total_ms: int,
        status: str,
        tokens_in: int,
        tokens_out: int,
        retries: int,
        setting_name: str,
        answer_grade: str,
        source: str = "not_called",
    ) -> QueryResponse:
        """Assemble standardized QueryResponse."""
        return QueryResponse(
            request_id=request_id,
            question=question,
            answer=answer,
            citations=citations,
            metrics=ExecutionMetrics(
                first_token_ms=first_token_ms,
                total_ms=total_ms,
                status=status,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                retries=retries,
                source=source,
                retrieval_setting=setting_name,
                answer_grade=answer_grade,
            ),
        )


_pipeline_instance: Optional[OrchestrationPipeline] = None


def get_pipeline() -> OrchestrationPipeline:
    """Retrieve singleton OrchestrationPipeline."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = OrchestrationPipeline()
    return _pipeline_instance
