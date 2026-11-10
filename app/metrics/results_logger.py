"""Metrics collection and non-blocking results.csv logger for RelayCore."""

import asyncio
import csv
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import List, Optional


@dataclass
class RequestMetricRecord:
    """Exact telemetry record meeting The Relay Assignment Raw File data contract."""

    request_id: str
    scenario: str
    source: str  # "real" or "stand-in"
    start_time: str
    first_token_ms: Optional[int]
    total_ms: int
    status: str  # "ok", "rate_limited", "timeout", "error"
    retries: int
    tokens_in: int
    tokens_out: int
    retrieval_setting: str
    answer_grade: str  # "correct", "wrong", "refused_correctly", "refused_wrongly"


CSV_COLUMNS = [
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade",
]


class ResultsCsvLogger:
    """Non-blocking, thread-safe writer persisting raw telemetry to results.csv."""

    def __init__(self, output_file: str = "results/results.csv"):
        self.output_file = output_file
        self._lock = asyncio.Lock()
        self._records: List[RequestMetricRecord] = []
        self._initialized = False

    def _ensure_file_header(self) -> None:
        """Create file and write header row if it doesn't exist."""
        os.makedirs(os.path.dirname(self.output_file) or ".", exist_ok=True)
        if not os.path.exists(self.output_file) or os.path.getsize(self.output_file) == 0:
            with open(self.output_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_COLUMNS)
        self._initialized = True

    async def log_request(self, record: RequestMetricRecord) -> None:
        """Asynchronously log an executed request record to results.csv."""
        if not self._initialized:
            self._ensure_file_header()

        async with self._lock:
            self._records.append(record)
            with open(self.output_file, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    record.request_id,
                    record.scenario,
                    record.source,
                    record.start_time,
                    record.first_token_ms if record.first_token_ms is not None else "",
                    record.total_ms,
                    record.status,
                    record.retries,
                    record.tokens_in,
                    record.tokens_out,
                    record.retrieval_setting,
                    record.answer_grade,
                ])

    def get_records(self) -> List[RequestMetricRecord]:
        """Return all in-memory records collected so far."""
        return list(self._records)

    def clear(self) -> None:
        """Clear in-memory and file records (for fresh test runs)."""
        self._records.clear()
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
        self._initialized = False


_default_logger: Optional[ResultsCsvLogger] = None


def get_results_logger() -> ResultsCsvLogger:
    """Singleton getter for ResultsCsvLogger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = ResultsCsvLogger()
    return _default_logger
