"""Pydantic schemas for Question Answering requests, citations, and metrics."""

from typing import List, Optional
from pydantic import BaseModel, Field


class CitationItem(BaseModel):
    """Citation metadata validating source paper grounding."""

    paper_title: str = Field(description="Title of cited research paper")
    arxiv_id: str = Field(description="arXiv ID and version, e.g. arXiv:1706.03762v7")
    section: str = Field(description="Specific paper section or heading")
    page: Optional[int] = Field(default=None, description="Page number in original PDF")
    passage_snippet: str = Field(description="Exact snippet supporting the claim")


class ExecutionMetrics(BaseModel):
    """Telemetry data matching the assignment reporting contract."""

    first_token_ms: Optional[int] = Field(default=None, description="Time to first token in milliseconds")
    total_ms: int = Field(description="Total request turnaround time in milliseconds")
    tokens_in: int = Field(default=0, description="Prompt tokens consumed")
    tokens_out: int = Field(default=0, description="Completion tokens generated")
    retries: int = Field(default=0, description="Number of retries triggered")
    source: str = Field(description="'real' or 'stand-in'")
    retrieval_setting: str = Field(description="Retrieval configuration tag")
    answer_grade: Optional[str] = Field(
        default=None,
        description="'correct', 'wrong', 'refused_correctly', 'refused_wrongly'",
    )


class QueryRequest(BaseModel):
    """Incoming user or benchmark query."""

    question: str = Field(..., min_length=3, description="Technical engineering question")
    target_papers: Optional[List[str]] = Field(
        default=None,
        description="Optional list of arXiv IDs to restrict search scope",
    )
    retrieval_setting: Optional[str] = Field(
        default="medium_k3",
        description="Retrieval configuration strategy (small_k6, medium_k3, large_k2)",
    )
    timeout_ms: Optional[int] = Field(
        default=15000,
        description="Client deadline timeout in milliseconds",
    )
    priority: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Priority level for admission queue (higher is served first)",
    )


class QueryResponse(BaseModel):
    """Complete response returned to client."""

    request_id: str = Field(description="Globally unique request tracking identifier")
    question: str = Field(description="Original question")
    answer: str = Field(description="Synthesized answer or grounded refusal message")
    citations: List[CitationItem] = Field(
        default_factory=list,
        description="List of verified citations grounding the answer",
    )
    metrics: ExecutionMetrics = Field(description="Execution latency and token metrics")
