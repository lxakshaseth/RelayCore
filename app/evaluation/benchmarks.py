"""Benchmark question evaluation harness for research paper grounding."""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BenchmarkQuestion:
    """Legacy question representation for load testing runner."""
    id: str
    question: str
    target_paper: Optional[str]
    expected_answer_keywords: List[str]
    is_answerable: bool


@dataclass
class EvaluationQuestion:
    """Rich evaluation question for comprehensive retrieval and grounding assessment."""
    id: str
    category: str
    question: str
    target_papers: List[str]
    expected_answer: str
    canonical_citations: str
    expected_answer_keywords: List[str] = field(default_factory=list)
    is_answerable: bool = True
    trap_mechanism: Optional[str] = None


def load_benchmark_questions(filepath: str = "data/test_questions.json") -> List[BenchmarkQuestion]:
    """Load curated technical evaluation questions for load tests."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Benchmark file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        BenchmarkQuestion(
            id=item["id"],
            question=item["question"],
            target_paper=item.get("target_paper"),
            expected_answer_keywords=item.get("expected_answer_keywords", []),
            is_answerable=item.get("is_answerable", True),
        )
        for item in data
    ]


def load_evaluation_questions(filepath: str = "data/evaluation_questions.json") -> List[EvaluationQuestion]:
    """Load full 18-question evaluation benchmark covering all 5 required dimensions."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Evaluation questions file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        EvaluationQuestion(
            id=item["id"],
            category=item.get("category", "direct_answer"),
            question=item["question"],
            target_papers=item.get("target_papers", []),
            expected_answer=item.get("expected_answer", ""),
            canonical_citations=item.get("canonical_citations", ""),
            expected_answer_keywords=item.get("expected_answer_keywords", []),
            is_answerable=item.get("is_answerable", True),
            trap_mechanism=item.get("trap_mechanism"),
        )
        for item in data
    ]
