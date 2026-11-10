"""Data models for paper representations, chunking, and retrieval."""

from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A granular chunk of text extracted from a research paper."""

    chunk_id: str = Field(description="Unique chunk identifier, e.g. 1706.03762v7_p3_c1")
    paper_id: str = Field(description="arXiv ID and version, e.g. 1706.03762v7")
    paper_title: str = Field(description="Full official title of research paper")
    version: str = Field(description="arXiv version tag, e.g. v7")
    section: str = Field(description="Section heading or title, e.g. '3.1 Encoder and Decoder Stacks'")
    page_number: int = Field(description="1-indexed physical PDF page number")
    content: str = Field(description="Extracted and normalized chunk text")
    source_url: str = Field(description="Permanent arXiv PDF source link")
    word_count: int = Field(description="Number of words in this chunk")


class RetrievalResult(BaseModel):
    """An individual retrieved chunk with ranking score and citation."""

    chunk: DocumentChunk
    score: float = Field(description="Raw retrieval score (e.g. BM25)")
    rerank_score: Optional[float] = Field(default=None, description="Adjusted score if reranker was applied")
    citation: str = Field(description="Formatted human-readable citation string")


class RetrievalResponse(BaseModel):
    """The aggregate response returned by the RetrievalService."""

    query: str
    passages: List[RetrievalResult] = Field(default_factory=list)
    has_evidence: bool = Field(
        default=True,
        description="Whether sufficient grounding evidence was found to answer the query",
    )
    refusal_reason: Optional[str] = Field(
        default=None,
        description="Reason for refusing to answer when has_evidence is False",
    )
    retrieval_setting: str = Field(
        default="medium_k3",
        description="Setting configuration identifier (e.g. small_k6, medium_k3, large_k2)",
    )
    total_candidates_searched: int = Field(default=0)


class RetrievalSetting(BaseModel):
    """Configuration bundle for retrieval experiments (Table 3 in assignment)."""

    name: str = Field(description="Identifier for setting (e.g. small_k6, medium_k3, large_k2)")
    chunk_size_words: int = Field(default=350, description="Target chunk size in words")
    overlap_words: int = Field(default=50, description="Overlap between consecutive chunks")
    top_k: int = Field(default=3, description="Number of passages to retrieve")
    rerank: bool = Field(default=False, description="Whether to apply secondary re-ranking")
    min_score_threshold: float = Field(
        default=0.8,
        description="Minimum score required to consider a chunk as grounded evidence",
    )
