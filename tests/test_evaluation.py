"""Unit tests for the RelayCore Evaluation Framework."""

import pytest
from app.evaluation.benchmarks import EvaluationQuestion, load_evaluation_questions
from app.evaluation.evaluator import (
    EvaluationFramework,
    QuestionEvaluationRecord,
    SettingEvaluationSummary,
    format_evaluation_table_markdown,
)
from app.schemas.query import CitationItem, ExecutionMetrics, QueryResponse


def test_load_evaluation_questions():
    """Verify that all 18 questions load cleanly and all 5 required categories are covered."""
    questions = load_evaluation_questions("data/evaluation_questions.json")
    assert len(questions) >= 15
    assert len(questions) == 18

    categories = {q.category for q in questions}
    assert "direct_answer" in categories
    assert "multiple_passages" in categories
    assert "technical_details" in categories
    assert "unanswerable_out_of_corpus" in categories
    assert "retrieval_failure_trap" in categories

    # Check that unanswerable questions have is_answerable=False
    unanswerable = [q for q in questions if not q.is_answerable]
    assert len(unanswerable) >= 4
    for q in unanswerable:
        assert q.category == "unanswerable_out_of_corpus"


def test_evaluator_grading_logic():
    """Verify that the evaluator correctly scores answers, refusals, and hallucinations."""
    framework = EvaluationFramework()

    # 1. Answerable question with valid citations and keywords
    q_ans = EvaluationQuestion(
        id="test_01",
        category="direct_answer",
        question="What is dmodel?",
        target_papers=["1706.03762v7"],
        expected_answer="dmodel is 512",
        canonical_citations="Section 3.1",
        expected_answer_keywords=["512", "dmodel"],
        is_answerable=True,
    )
    res_correct = QueryResponse(
        request_id="req1",
        question="What is dmodel?",
        answer="In the Transformer, the embedding dimension dmodel is 512 according to Section 3.1.",
        citations=[
            CitationItem(
                paper_title="Attention Is All You Need",
                arxiv_id="1706.03762v7",
                section="3.1",
                page=3,
                passage_snippet="...",
            )
        ],
        metrics=ExecutionMetrics(
            total_ms=100, status="ok", tokens_in=50, tokens_out=20, retries=0
        ),
    )
    grade, cit_correct = framework.grade_response(q_ans, res_correct)
    assert grade == "correct"
    assert cit_correct is True

    # 2. Answerable question that was incorrectly refused
    res_wrong_refusal = QueryResponse(
        request_id="req2",
        question="What is dmodel?",
        answer="The provided research papers do not contain sufficient evidence to answer: 'What is dmodel?'.",
        citations=[],
        metrics=ExecutionMetrics(
            total_ms=100,
            status="ok",
            tokens_in=0,
            tokens_out=0,
            retries=0,
            answer_grade="refused_correctly",
        ),
    )
    grade, cit_correct = framework.grade_response(q_ans, res_wrong_refusal)
    assert grade == "refused_wrongly"
    assert cit_correct is False

    # 3. Unanswerable question correctly refused
    q_unans = EvaluationQuestion(
        id="test_02",
        category="unanswerable_out_of_corpus",
        question="How to bake cookies?",
        target_papers=[],
        expected_answer="Refuse",
        canonical_citations="None",
        is_answerable=False,
    )
    res_good_refusal = QueryResponse(
        request_id="req3",
        question="How to bake cookies?",
        answer="The provided research papers do not contain sufficient evidence to answer: 'How to bake cookies?'.",
        citations=[],
        metrics=ExecutionMetrics(
            total_ms=100,
            status="ok",
            tokens_in=0,
            tokens_out=0,
            retries=0,
            answer_grade="refused_correctly",
        ),
    )
    grade, cit_correct = framework.grade_response(q_unans, res_good_refusal)
    assert grade == "refused_correctly"
    assert cit_correct is True

    # 4. Unanswerable question answered (Hallucination)
    res_hallucination = QueryResponse(
        request_id="req4",
        question="How to bake cookies?",
        answer="Bake cookies at 375 degrees F for 10 minutes with flour and butter.",
        citations=[],
        metrics=ExecutionMetrics(
            total_ms=100, status="ok", tokens_in=50, tokens_out=20, retries=0
        ),
    )
    grade, cit_correct = framework.grade_response(q_unans, res_hallucination)
    assert grade == "wrong"
    assert cit_correct is False


def test_evaluator_compute_summary():
    """Verify that summary aggregation computes correct percentages and percentiles."""
    framework = EvaluationFramework()
    records = [
        QuestionEvaluationRecord(
            question_id="q1",
            category="direct_answer",
            setting_name="setting_a",
            question="Q1",
            expected_answer="A1",
            canonical_citations="C1",
            actual_answer="A1",
            citations_returned=["[Cit1]"],
            is_answerable=True,
            status="ok",
            answer_grade="correct",
            citation_correct=True,
            tokens_in=200,
            tokens_out=50,
            first_token_ms=150,
            total_ms=400,
        ),
        QuestionEvaluationRecord(
            question_id="q2",
            category="unanswerable_out_of_corpus",
            setting_name="setting_a",
            question="Q2",
            expected_answer="Refuse",
            canonical_citations="",
            actual_answer="Refused",
            citations_returned=[],
            is_answerable=False,
            status="ok",
            answer_grade="refused_correctly",
            citation_correct=True,
            tokens_in=0,
            tokens_out=0,
            first_token_ms=None,
            total_ms=5,
        ),
    ]

    summary = framework.compute_summary("setting_a", "Setting A Test", records)
    assert summary.questions_asked == 2
    assert summary.correct_count == 1
    assert summary.refused_correctly_count == 1
    assert summary.correct_and_cited_pct == 50.0
    assert summary.correctly_refused_pct == 50.0
    assert summary.wrong_pct == 0.0
    assert summary.citation_correct_pct == 100.0
    assert summary.tokens_per_answer == 125.0
