"""Master experiment script running all 5 load scenarios and generating Tables 2a, 2b, and 3."""

import asyncio
import csv
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.metrics.analyzer import PerformanceAnalyzer
from app.metrics.results_logger import get_results_logger
from app.simulator.config import StandinConfig
from app.simulator.service import StandinModelService
from load_tests.load_runner import LoadTestRunner


def save_table_to_csv(rows: List[Dict[str, Any]], filepath: str):
    """Save a list of dictionaries to a CSV file."""
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def format_markdown_table(rows: List[Dict[str, Any]]) -> str:
    """Format a list of dictionaries as a GitHub-flavored Markdown table."""
    if not rows:
        return "*No data collected*"
    headers = list(rows[0].keys())
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = [
        "| " + " | ".join(str(r.get(h, "")) for h in headers) + " |"
        for r in rows
    ]
    return "\n".join([header_line, separator_line] + body_lines)


async def run_all_experiments():
    print("=" * 80)
    print("RELAYCORE: MASTER PERFORMANCE & LOAD EXPERIMENTATION HARNESS")
    print("=" * 80)

    logger = get_results_logger()
    logger.clear()  # Fresh run
    runner = LoadTestRunner()

    # 1. Steady Load (Real Groq API - Modest 5 requests)
    print("\n>>> [1/6] Running Scenario 1: Steady Load on Real Groq Service (5 requests)...")
    try:
        await runner.run_steady_load(num_requests=5, is_real=True)
        print("    [Real Groq API] 5 requests completed successfully.")
    except Exception as e:
        print(f"    [Real Groq API warning]: {e}. Falling back to stand-in for real sweep.")

    # 2. Steady Load (Stand-in Model - 15 requests)
    print("\n>>> [2/6] Running Scenario 1: Steady Load on Stand-in Simulator (15 requests)...")
    await runner.run_steady_load(num_requests=15, is_real=False)

    # 3. Sudden Burst (25 requests)
    print("\n>>> [3/6] Running Scenario 2: Sudden Burst (25 concurrent requests)...")
    await runner.run_sudden_burst(num_requests=25)

    # 4. Sustained Overload (40 requests)
    print("\n>>> [4/6] Running Scenario 3: Sustained Overload (40 requests)...")
    await runner.run_sustained_overload(num_requests=40)

    # 5. Limit Halved (20 requests)
    print("\n>>> [5/6] Running Scenario 4: Rate Limit Halved (RPM=15, TPM=4000) (20 requests)...")
    await runner.run_limit_halved(num_requests=20)

    # 6. Daily Budget Exhausted (15 requests)
    print("\n>>> [6/6] Running Scenario 5: Daily Budget Exhausted (3 PM scenario) (15 requests)...")
    await runner.run_daily_budget_exhausted(num_requests=15)

    # 7. Retrieval Settings Sweep (for Table 3)
    print("\n>>> Running Retrieval Settings Benchmark (small_k6, large_k2)...")
    sim_model = StandinModelService(StandinConfig(seed=505, rpm_limit=30, tpm_limit=8000))
    pipeline = runner._create_pipeline(sim_model, rpm=30, tpm=8000)
    await runner.run_scenario("retrieval_sweep_small", pipeline, num_requests=10, retrieval_setting="small_k6")
    await runner.run_scenario("retrieval_sweep_large", pipeline, num_requests=10, retrieval_setting="large_k2")

    print("\n" + "=" * 80)
    print("ALL SCENARIOS COMPLETE. COMPILING PERFORMANCE TABLES...")
    print("=" * 80)

    analyzer = PerformanceAnalyzer()
    t_2a, t_2b = analyzer.compute_table_2a_and_2b()
    t_3 = analyzer.compute_table_3()

    os.makedirs("results/tables", exist_ok=True)
    save_table_to_csv(t_2a, "results/tables/table_2a.csv")
    save_table_to_csv(t_2b, "results/tables/table_2b.csv")
    save_table_to_csv(t_3, "results/tables/table_3.csv")

    with open("results/tables/TABLES_SUMMARY.md", "w", encoding="utf-8") as f:
        f.write("# The Relay Assignment: Compiled Results Tables\n\n")
        f.write("## Table 2a: Latency, one row per load scenario\n\n")
        f.write(format_markdown_table(t_2a))
        f.write("\n\n## Table 2b: Outcomes, same rows\n\n")
        f.write(format_markdown_table(t_2b))
        f.write("\n\n## Table 3: Retrieval settings\n\n")
        f.write(format_markdown_table(t_3))
        f.write("\n")

    print("\n--- TABLE 2a: LATENCY (ms) ---")
    print(format_markdown_table(t_2a))

    print("\n--- TABLE 2b: OUTCOMES ---")
    print(format_markdown_table(t_2b))

    print("\n--- TABLE 3: RETRIEVAL SETTINGS ---")
    print(format_markdown_table(t_3))

    print("\nTables saved to results/tables/TABLES_SUMMARY.md and results.csv")


if __name__ == "__main__":
    asyncio.run(run_all_experiments())

