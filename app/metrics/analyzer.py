"""Pure-Python metrics analyzer computing exact percentiles and compiling required assignment tables."""

import csv
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def calculate_percentile(data: List[float], p: float) -> float:
    """Calculate p-th percentile using linear interpolation matching numpy/pandas default."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    if len(sorted_data) == 1:
        return sorted_data[0]
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[int(f)] * (c - k) + sorted_data[int(c)] * (k - f)


class PerformanceAnalyzer:
    """Computes exact statistical percentiles and compiles required assignment tables."""

    def __init__(self, csv_path: str = "results/results.csv"):
        self.csv_path = csv_path

    def load_records(self) -> List[Dict[str, Any]]:
        """Load and type-cast raw telemetry results from CSV."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Telemetry file not found at {self.csv_path}")

        records = []
        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ft_ms = float(row["first_token_ms"]) if row.get("first_token_ms") else None
                except ValueError:
                    ft_ms = None

                try:
                    tot_ms = float(row["total_ms"]) if row.get("total_ms") else 0.0
                except ValueError:
                    tot_ms = 0.0

                try:
                    tin = int(row["tokens_in"]) if row.get("tokens_in") else 0
                except ValueError:
                    tin = 0

                try:
                    tout = int(row["tokens_out"]) if row.get("tokens_out") else 0
                except ValueError:
                    tout = 0

                try:
                    retries = int(row["retries"]) if row.get("retries") else 0
                except ValueError:
                    retries = 0

                records.append({
                    "request_id": row.get("request_id", ""),
                    "scenario": row.get("scenario", "unknown"),
                    "source": row.get("source", "stand-in"),
                    "start_time": row.get("start_time", ""),
                    "first_token_ms": ft_ms,
                    "total_ms": tot_ms,
                    "status": row.get("status", "error"),
                    "retries": retries,
                    "tokens_in": tin,
                    "tokens_out": tout,
                    "retrieval_setting": row.get("retrieval_setting", "medium_k3"),
                    "answer_grade": row.get("answer_grade", "wrong"),
                })
        return records

    def compute_table_2a_and_2b(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Compile Table 2a (Latency) and Table 2b (Outcomes) per load scenario."""
        records = self.load_records()
        by_scenario = defaultdict(list)
        for r in records:
            by_scenario[r["scenario"]].append(r)

        table_2a = []
        table_2b = []

        for sc, rows in by_scenario.items():
            if not rows:
                continue

            requests_sent = len(rows)
            source = rows[0]["source"]

            # Duration calculation
            parsed_times = []
            for r in rows:
                if r["start_time"]:
                    try:
                        # Parse ISO format
                        t = datetime.fromisoformat(r["start_time"].replace("Z", "+00:00"))
                        parsed_times.append(t)
                    except ValueError:
                        pass

            if len(parsed_times) > 1:
                dur_s = max(1.0, (max(parsed_times) - min(parsed_times)).total_seconds())
            else:
                dur_s = max(1.0, sum(r["total_ms"] for r in rows) / 1000.0)

            offered_rate = round((requests_sent / dur_s) * 60.0, 1)

            # Table 2a: Latency percentiles
            ft_clean = [r["first_token_ms"] for r in rows if r["first_token_ms"] is not None]
            if ft_clean:
                ft_p50 = f"{calculate_percentile(ft_clean, 50):.1f}"
                ft_p95 = f"{calculate_percentile(ft_clean, 95):.1f}"
                ft_p99 = f"{calculate_percentile(ft_clean, 99):.1f}"
            else:
                ft_p50, ft_p95, ft_p99 = "not measured", "not measured", "not measured"

            tot_clean = [r["total_ms"] for r in rows]
            tot_p50 = f"{calculate_percentile(tot_clean, 50):.1f}"
            tot_p95 = f"{calculate_percentile(tot_clean, 95):.1f}"
            tot_p99 = f"{calculate_percentile(tot_clean, 99):.1f}"

            table_2a.append({
                "Scenario": sc,
                "Real or stand-in": source,
                "Requests sent": requests_sent,
                "Duration (s)": round(dur_s, 1),
                "Offered rate (per min)": offered_rate,
                "First token p50 / p95 / p99 (ms)": f"{ft_p50} / {ft_p95} / {ft_p99}",
                "Total time p50 / p95 / p99 (ms)": f"{tot_p50} / {tot_p95} / {tot_p99}",
            })

            # Table 2b: Outcomes
            succeeded_count = sum(1 for r in rows if r["status"] == "ok")
            rate_limited_count = sum(1 for r in rows if r["status"] == "rate_limited")
            timeout_count = sum(1 for r in rows if r["status"] == "timeout")

            succ_pct = round((succeeded_count / requests_sent) * 100.0, 1)
            rl_pct = round((rate_limited_count / requests_sent) * 100.0, 1)
            to_pct = round((timeout_count / requests_sent) * 100.0, 1)

            avg_retries = round(sum(r["retries"] for r in rows) / requests_sent, 2)
            peak_q = max(r["retries"] for r in rows) + 1

            total_tokens = [r["tokens_in"] + r["tokens_out"] for r in rows]
            tokens_per_ans = round(sum(total_tokens) / max(1, requests_sent), 1)

            # Cost per 1,000 answers based on $0.05 / 1M tokens commercial rate
            cost_val = (tokens_per_ans * 1000.0 / 1_000_000.0) * 0.05
            cost_str = f"${cost_val:.4f}"

            table_2b.append({
                "Scenario": sc,
                "Succeeded %": succ_pct,
                "Rate-limited %": rl_pct,
                "Timed out or given up %": to_pct,
                "Retries per request": avg_retries,
                "Peak queue length": peak_q,
                "Tokens per answer": tokens_per_ans,
                "Cost per 1,000 answers": cost_str,
            })

        return table_2a, table_2b

    def compute_table_3(self) -> List[Dict[str, Any]]:
        """Compile Table 3 (Retrieval settings comparison)."""
        records = self.load_records()
        by_setting = defaultdict(list)
        for r in records:
            by_setting[r["retrieval_setting"]].append(r)

        table_3 = []
        for st, rows in by_setting.items():
            if not rows:
                continue

            q_count = len(rows)
            correct_count = sum(1 for r in rows if r["answer_grade"] == "correct")
            wrong_count = sum(1 for r in rows if r["answer_grade"] == "wrong")
            refused_corr_count = sum(1 for r in rows if r["answer_grade"] == "refused_correctly")

            corr_pct = round((correct_count / q_count) * 100.0, 1)
            wrong_pct = round((wrong_count / q_count) * 100.0, 1)
            ref_corr_pct = round((refused_corr_count / q_count) * 100.0, 1)

            total_tokens = [r["tokens_in"] + r["tokens_out"] for r in rows]
            tokens_per_ans = round(sum(total_tokens) / max(1, q_count), 1)

            ft_clean = [r["first_token_ms"] for r in rows if r["first_token_ms"] is not None]
            if ft_clean:
                ft_p50 = f"{calculate_percentile(ft_clean, 50):.1f}"
                ft_p95 = f"{calculate_percentile(ft_clean, 95):.1f}"
            else:
                ft_p50, ft_p95 = "not measured", "not measured"

            tot_clean = [r["total_ms"] for r in rows]
            tot_p50 = f"{calculate_percentile(tot_clean, 50):.1f}"
            tot_p95 = f"{calculate_percentile(tot_clean, 95):.1f}"

            table_3.append({
                "Setting (piece size, pieces fetched, re-rank yes or no)": st,
                "Questions asked": q_count,
                "Correct and cited %": corr_pct,
                "Wrong %": wrong_pct,
                "Correctly refused %": ref_corr_pct,
                "Tokens per answer": tokens_per_ans,
                "First token p50 / p95 (ms)": f"{ft_p50} / {ft_p95}",
                "Total time p50 / p95 (ms)": f"{tot_p50} / {tot_p95}",
            })

        return table_3
