"""Comprehensive Evaluation Framework for RelayCore Question-Answering and Retrieval."""

import asyncio
import csv
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import Settings
from app.core.rate_limiter import RelayRateLimiter
from app.evaluation.benchmarks import EvaluationQuestion, load_evaluation_questions
from app.metrics.analyzer import calculate_percentile
from app.metrics.results_logger import get_results_logger
from app.models.base import AbstractModelClient
from app.models.groq_client import GroqAsyncClient
from app.retrieval.service import RETRIEVAL_PRESETS, get_retrieval_service
from app.schemas.query import QueryRequest, QueryResponse
from app.services.orchestrator import OrchestrationPipeline
from app.simulator.config import StandinConfig
from app.simulator.service import StandinModelService

logger = logging.getLogger(__name__)


@dataclass
class QuestionEvaluationRecord:
    """Individual question evaluation record under a specific retrieval setting."""
    question_id: str
    category: str
    setting_name: str
    question: str
    expected_answer: str
    canonical_citations: str
    actual_answer: str
    citations_returned: List[str]
    is_answerable: bool
    status: str
    answer_grade: str  # correct, wrong, refused_correctly, refused_wrongly
    citation_correct: bool
    tokens_in: int
    tokens_out: int
    first_token_ms: Optional[int]
    total_ms: int


@dataclass
class SettingEvaluationSummary:
    """Aggregated evaluation metrics for a specific retrieval configuration."""
    setting_name: str
    setting_description: str
    questions_asked: int
    correct_count: int
    wrong_count: int
    refused_correctly_count: int
    refused_wrongly_count: int
    correct_and_cited_pct: float
    wrong_pct: float
    correctly_refused_pct: float
    incorrectly_refused_pct: float
    citation_correct_pct: float
    tokens_per_answer: float
    cost_per_1000_answers: str
    first_token_p50_ms: str
    first_token_p95_ms: str
    total_time_p50_ms: str
    total_time_p95_ms: str


class EvaluationFramework:
    """End-to-end evaluation harness benchmarking retrieval quality, cost, and latency."""

    def __init__(
        self,
        questions_path: str = "data/evaluation_questions.json",
        results_csv_path: str = "results/results.csv",
    ):
        self.questions_path = questions_path
        self.results_csv_path = results_csv_path
        self.questions: List[EvaluationQuestion] = load_evaluation_questions(questions_path)
        self.retrieval_service = get_retrieval_service()
        self.results_logger = get_results_logger(results_csv_path)

    def _create_pipeline(
        self, model_client: AbstractModelClient, rpm: int = 30, tpm: int = 8000
    ) -> OrchestrationPipeline:
        """Create a dedicated orchestration pipeline for evaluation."""
        settings = Settings(
            GROQ_RPM_LIMIT=rpm,
            GROQ_TPM_LIMIT=tpm,
            WORKER_CONCURRENCY=5,
            MAX_QUEUE_DEPTH=100,
        )
        rate_limiter = RelayRateLimiter(settings=settings)
        return OrchestrationPipeline(
            settings=settings,
            retrieval_service=self.retrieval_service,
            model_client=model_client,
            rate_limiter=rate_limiter,
        )

    def grade_response(
        self, q: EvaluationQuestion, res: QueryResponse
    ) -> Tuple[str, bool]:
        """Strictly evaluate answer accuracy, refusal validity, and citation correctness.

        Returns (answer_grade, citation_correct).
        Possible grades:
        - 'correct': Grounded with valid citations and factual content.
        - 'wrong': Hallucination, unsupported claims, or ungrounded answer to unanswerable query.
        - 'refused_correctly': Refused an unanswerable or ungrounded query.
        - 'refused_wrongly': Refused an answerable query that papers contain.
        """
        answer_text = res.answer or ""
        answer_lower = answer_text.lower()
        has_citations = len(res.citations) > 0

        # Check refusal indicators
        refusal_indicators = [
            "do not contain sufficient evidence",
            "not mentioned in the",
            "unanswerable",
            "cannot answer",
            "exhausted",
            "unable to find",
        ]
        is_refusal = (
            res.metrics.answer_grade in ("refused_correctly", "refused_wrongly")
            or any(phrase in answer_lower for phrase in refusal_indicators)
        )

        # 1. Unanswerable Question Assessment
        if not q.is_answerable:
            if is_refusal:
                # Correctly refused without hallucinating
                return "refused_correctly", True
            else:
                # Answered an unanswerable question -> Hallucination!
                return "wrong", False

        # 2. Answerable Question Assessment
        if is_refusal:
            # Answerable in corpus, but system refused
            return "refused_wrongly", False

        # Check expected factual keyword coverage
        if q.expected_answer_keywords:
            matched_keywords = [
                kw for kw in q.expected_answer_keywords if kw.lower() in answer_lower
            ]
            # Need at least 50% keyword match or at least 1 keyword
            required_matches = max(1, len(q.expected_answer_keywords) // 2)
            has_factual_content = len(matched_keywords) >= required_matches
        else:
            has_factual_content = len(answer_text.strip()) > 20

        # Verify citations: must have at least one structured citation
        if has_factual_content and has_citations:
            return "correct", True
        elif has_factual_content and not has_citations:
            # Correct facts but missing required grounding citations
            return "wrong", False
        else:
            # Did not contain required factual keywords
            return "wrong", has_citations

    async def evaluate_setting(
        self,
        setting_name: str,
        pipeline: OrchestrationPipeline,
        inter_request_delay_s: float = 0.0,
    ) -> List[QuestionEvaluationRecord]:
        """Run all evaluation questions through the pipeline for one retrieval setting."""
        records: List[QuestionEvaluationRecord] = []
        logger.info("Evaluating Setting '%s' across %d benchmark questions...", setting_name, len(self.questions))

        print(f"\n>>> Evaluating Setting: {setting_name} ({len(self.questions)} questions)...", flush=True)
        for idx, q in enumerate(self.questions):
            print(f"  [{setting_name}] ({idx+1}/{len(self.questions)}) Question {q.id} ({q.category}): '{q.question[:45]}...'", flush=True)
            req_id = f"eval_{setting_name}_{idx+1:02d}"
            req = QueryRequest(
                question=q.question,
                target_papers=q.target_papers if q.target_papers else None,
                retrieval_setting=setting_name,
                timeout_ms=20000,
            )

            try:
                res: QueryResponse = await pipeline.process_query(
                    request=req,
                    request_id=req_id,
                    scenario=f"eval_{setting_name}",
                )
            except Exception as e:
                logger.error("Error evaluating question %s on setting %s: %s", q.id, setting_name, e)
                continue

            grade, citation_correct = self.grade_response(q, res)
            citations_str = [
                f"[{c.paper_title} ({c.arxiv_id}), Sec: '{c.section}', p.{c.page}]"
                for c in res.citations
            ]

            rec = QuestionEvaluationRecord(
                question_id=q.id,
                category=q.category,
                setting_name=setting_name,
                question=q.question,
                expected_answer=q.expected_answer,
                canonical_citations=q.canonical_citations,
                actual_answer=res.answer,
                citations_returned=citations_str,
                is_answerable=q.is_answerable,
                status=res.metrics.status,
                answer_grade=grade,
                citation_correct=citation_correct,
                tokens_in=res.metrics.tokens_in,
                tokens_out=res.metrics.tokens_out,
                first_token_ms=res.metrics.first_token_ms,
                total_ms=res.metrics.total_ms,
            )
            records.append(rec)

            if inter_request_delay_s > 0:
                await asyncio.sleep(inter_request_delay_s)

        return records

    def compute_summary(
        self, setting_name: str, description: str, records: List[QuestionEvaluationRecord]
    ) -> SettingEvaluationSummary:
        """Compute statistical percentiles, accuracy rates, and costs for a setting."""
        total_q = len(records)
        if total_q == 0:
            raise ValueError(f"No records found for setting {setting_name}")

        correct_count = sum(1 for r in records if r.answer_grade == "correct")
        wrong_count = sum(1 for r in records if r.answer_grade == "wrong")
        refused_corr_count = sum(1 for r in records if r.answer_grade == "refused_correctly")
        refused_wrong_count = sum(1 for r in records if r.answer_grade == "refused_wrongly")
        citation_corr_count = sum(1 for r in records if r.citation_correct)

        corr_pct = round((correct_count / total_q) * 100.0, 1)
        wrong_pct = round((wrong_count / total_q) * 100.0, 1)
        ref_corr_pct = round((refused_corr_count / total_q) * 100.0, 1)
        ref_wrong_pct = round((refused_wrong_count / total_q) * 100.0, 1)
        cit_corr_pct = round((citation_corr_count / total_q) * 100.0, 1)

        total_tokens = [r.tokens_in + r.tokens_out for r in records]
        tokens_per_ans = round(sum(total_tokens) / max(1, total_q), 1)

        # Commercial rate: $0.05 / 1,000,000 tokens
        cost_val = (tokens_per_ans * 1000.0 / 1_000_000.0) * 0.05
        cost_str = f"${cost_val:.4f}"

        # First token latency
        ft_clean = [r.first_token_ms for r in records if r.first_token_ms is not None]
        if ft_clean:
            ft_p50 = f"{calculate_percentile(ft_clean, 50):.1f}"
            ft_p95 = f"{calculate_percentile(ft_clean, 95):.1f}"
        else:
            ft_p50, ft_p95 = "not measured", "not measured"

        # Total latency
        tot_clean = [r.total_ms for r in records]
        tot_p50 = f"{calculate_percentile(tot_clean, 50):.1f}"
        tot_p95 = f"{calculate_percentile(tot_clean, 95):.1f}"

        return SettingEvaluationSummary(
            setting_name=setting_name,
            setting_description=description,
            questions_asked=total_q,
            correct_count=correct_count,
            wrong_count=wrong_count,
            refused_correctly_count=refused_corr_count,
            refused_wrongly_count=refused_wrong_count,
            correct_and_cited_pct=corr_pct,
            wrong_pct=wrong_pct,
            correctly_refused_pct=ref_corr_pct,
            incorrectly_refused_pct=ref_wrong_pct,
            citation_correct_pct=cit_corr_pct,
            tokens_per_answer=tokens_per_ans,
            cost_per_1000_answers=cost_str,
            first_token_p50_ms=ft_p50,
            first_token_p95_ms=ft_p95,
            total_time_p50_ms=tot_p50,
            total_time_p95_ms=tot_p95,
        )

    async def run_full_evaluation(
        self, is_real: bool = False, inter_request_delay_s: float = 0.0
    ) -> Tuple[List[SettingEvaluationSummary], List[QuestionEvaluationRecord]]:
        """Run all 18 benchmark questions across Settings A, B, and C."""
        settings_to_run = [
            (
                "setting_a",
                "Setting A: Small chunks (150w), low top-k (k=2), no re-ranking",
            ),
            (
                "setting_b",
                "Setting B: Medium chunks (350w), moderate top-k (k=4), no re-ranking",
            ),
            (
                "setting_c",
                "Setting C: Larger chunks (800w), moderate top-k (k=3), with re-ranking",
            ),
        ]

        all_summaries: List[SettingEvaluationSummary] = []
        all_records: List[QuestionEvaluationRecord] = []

        for setting_key, desc in settings_to_run:
            if is_real:
                model = GroqAsyncClient()
                # Pacing for real Groq free-tier rate limits
                delay = max(2.1, inter_request_delay_s)
            else:
                model = StandinModelService(
                    StandinConfig(seed=42, rpm_limit=60, tpm_limit=20000)
                )
                delay = inter_request_delay_s

            pipeline = self._create_pipeline(model)
            records = await self.evaluate_setting(
                setting_name=setting_key,
                pipeline=pipeline,
                inter_request_delay_s=delay,
            )
            summary = self.compute_summary(setting_key, desc, records)
            all_summaries.append(summary)
            all_records.extend(records)

        return all_summaries, all_records


def format_evaluation_table_markdown(summaries: List[SettingEvaluationSummary]) -> str:
    """Format SettingEvaluationSummary list into the exact Table 3 format."""
    headers = [
        "Setting (piece size, pieces fetched, re-rank yes or no)",
        "Questions asked",
        "Correct & cited %",
        "Wrong %",
        "Correctly refused %",
        "Incorrectly refused %",
        "Citation correctness %",
        "Tokens per answer",
        "Cost per 1,000",
        "First token p50 / p95 (ms)",
        "Total time p50 / p95 (ms)",
    ]
    rows = [
        [
            s.setting_description,
            str(s.questions_asked),
            f"{s.correct_and_cited_pct}%",
            f"{s.wrong_pct}%",
            f"{s.correctly_refused_pct}%",
            f"{s.incorrectly_refused_pct}%",
            f"{s.citation_correct_pct}%",
            str(s.tokens_per_answer),
            s.cost_per_1000_answers,
            f"{s.first_token_p50_ms} / {s.first_token_p95_ms}",
            f"{s.total_time_p50_ms} / {s.total_time_p95_ms}",
        ]
        for s in summaries
    ]

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([header_line, separator_line] + body_lines)
