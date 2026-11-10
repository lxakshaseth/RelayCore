"""Benchmark question evaluation harness for research paper grounding."""

import json
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BenchmarkQuestion:
    id: str
    question: str
    target_paper: Optional[str]
    expected_answer_keywords: List[str]
    is_answerable: bool


def load_benchmark_questions(filepath: str = "data/test_questions.json") -> List[BenchmarkQuestion]:
    """Load curated technical evaluation questions."""
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
