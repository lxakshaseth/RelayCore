"""Run comprehensive evaluation framework across Setting A, B, and C."""

import argparse
import asyncio
import csv
import json
import os
import sys
from typing import List

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.evaluation.evaluator import (
    EvaluationFramework,
    QuestionEvaluationRecord,
    SettingEvaluationSummary,
    format_evaluation_table_markdown,
)


def save_summary_to_csv(summaries: List[SettingEvaluationSummary], filepath: str):
    """Save evaluation summary metrics to CSV."""
    rows = [
        {
            "setting": s.setting_name,
            "description": s.setting_description,
            "questions_asked": s.questions_asked,
            "correct_and_cited_pct": s.correct_and_cited_pct,
            "wrong_pct": s.wrong_pct,
            "correctly_refused_pct": s.correctly_refused_pct,
            "incorrectly_refused_pct": s.incorrectly_refused_pct,
            "citation_correct_pct": s.citation_correct_pct,
            "tokens_per_answer": s.tokens_per_answer,
            "cost_per_1000_answers": s.cost_per_1000_answers,
            "first_token_p50_ms": s.first_token_p50_ms,
            "first_token_p95_ms": s.first_token_p95_ms,
            "total_time_p50_ms": s.total_time_p50_ms,
            "total_time_p95_ms": s.total_time_p95_ms,
        }
        for s in summaries
    ]
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_records_to_json(records: List[QuestionEvaluationRecord], filepath: str):
    """Save detailed per-question evaluation logs to JSON."""
    raw = [
        {
            "question_id": r.question_id,
            "category": r.category,
            "setting_name": r.setting_name,
            "question": r.question,
            "expected_answer": r.expected_answer,
            "canonical_citations": r.canonical_citations,
            "actual_answer": r.actual_answer,
            "citations_returned": r.citations_returned,
            "is_answerable": r.is_answerable,
            "status": r.status,
            "answer_grade": r.answer_grade,
            "citation_correct": r.citation_correct,
            "tokens_in": r.tokens_in,
            "tokens_out": r.tokens_out,
            "first_token_ms": r.first_token_ms,
            "total_ms": r.total_ms,
        }
        for r in records
    ]
    with open(filepath, mode="w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)


async def main():
    parser = argparse.ArgumentParser(description="RelayCore Retrieval Evaluation Benchmark")
    parser.add_argument("--real", action="store_true", help="Execute against real Groq API")
    parser.add_argument("--delay", type=float, default=0.0, help="Inter-request spacing in seconds")
    args = parser.parse_args()

    print("=" * 80)
    print("RELAYCORE: SYSTEMATIC RETRIEVAL EVALUATION FRAMEWORK")
    print("Settings evaluated:")
    print("  A. Small chunks (150w), low top-k (k=2), no re-ranking")
    print("  B. Medium chunks (350w), moderate top-k (k=4), no re-ranking")
    print("  C. Larger chunks (800w), moderate top-k (k=3), with re-ranking")
    print(f"Backend: {'Real Groq API' if args.real else 'Calibrated Stand-in Simulator'}")
    print("=" * 80)

    framework = EvaluationFramework()
    summaries, records = await framework.run_full_evaluation(
        is_real=args.real, inter_request_delay_s=args.delay
    )

    os.makedirs("results/tables", exist_ok=True)
    table_md = format_evaluation_table_markdown(summaries)

    # Save CSV and Markdown table
    save_summary_to_csv(summaries, "results/tables/table_3_evaluation.csv")
    save_records_to_json(records, "results/evaluation_details.json")

    with open("results/tables/TABLE3_RETRIEVAL_EVALUATION.md", "w", encoding="utf-8") as f:
        f.write("# Table 3: Systematic Retrieval Configuration Evaluation\n\n")
        f.write(table_md)
        f.write("\n\n## Per-Question Detailed Breakdown\n\n")
        for s in summaries:
            f.write(f"### {s.setting_description}\n\n")
            setting_records = [r for r in records if r.setting_name == s.setting_name]
            f.write("| ID | Category | Question | Grade | Citations Valid? | Total ms |\n")
            f.write("| --- | --- | --- | --- | --- | --- |\n")
            for r in setting_records:
                f.write(f"| {r.question_id} | {r.category} | {r.question[:45]}... | `{r.answer_grade}` | {'Yes' if r.citation_correct else 'No'} | {r.total_ms}ms |\n")
            f.write("\n")

    print("\n" + table_md + "\n")
    print("=" * 80)
    print("Evaluation completed successfully.")
    print("Artifacts generated:")
    print("  - results/tables/TABLE3_RETRIEVAL_EVALUATION.md")
    print("  - results/tables/table_3_evaluation.csv")
    print("  - results/evaluation_details.json")
    print("  - results/results.csv (updated)")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
