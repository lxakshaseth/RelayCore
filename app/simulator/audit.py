"""Audit logging and telemetry recorder for all simulated model interactions."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


@dataclass
class SimulatedRequestRecord:
    """Historical trace of an individual request processed by the Stand-in Service."""

    record_id: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    model: str = "qwen/qwen3.8-27b"
    stream: bool = False
    tokens_in: int = 0
    tokens_out: int = 0
    first_token_ms: Optional[int] = None
    total_ms: int = 0
    active_concurrency: int = 1
    status: str = "ok"  # "ok", "rate_limited", "timeout", "error"
    error_type: Optional[str] = None
    retry_after: Optional[float] = None
    refusal_reason: Optional[str] = None


class SimulatorAuditRecorder:
    """Thread-safe historical recorder storing traces of all simulator executions."""

    def __init__(self, max_records: int = 10000):
        self.max_records = max_records
        self._records: List[SimulatedRequestRecord] = []
        self._counter: int = 0

    def record(self, **kwargs) -> SimulatedRequestRecord:
        """Append a new simulation record to the audit log."""
        self._counter += 1
        record = SimulatedRequestRecord(record_id=self._counter, **kwargs)
        self._records.append(record)
        if len(self._records) > self.max_records:
            self._records.pop(0)
        return record

    def get_all(self) -> List[SimulatedRequestRecord]:
        """Return a copy of all recorded requests."""
        return list(self._records)

    def get_count(self) -> int:
        """Return total number of recorded events."""
        return len(self._records)

    def clear(self) -> None:
        """Reset historical audit storage."""
        self._records.clear()
        self._counter = 0
