"""Unit and integration tests for RelayCore research paper retrieval system."""

import pytest
from app.retrieval.service import RetrievalService, get_retrieval_service


@pytest.fixture(scope="module")
def retrieval_service() -> RetrievalService:
    """Fixture providing initialized RetrievalService across all retrieval tests."""
    service = get_retrieval_service(data_dir="data/papers")
    service.initialize_indexes(["small_k6", "medium_k3", "medium_k3_rerank", "large_k2"])
    return service


def test_relevant_passage_retrieval_attention(retrieval_service: RetrievalService) -> None:
    """Verify relevant passages are accurately retrieved for Transformer questions."""
    query = "What is the dimension of the embedding layers dmodel in the Transformer?"
    res = retrieval_service.retrieve(query, setting_name="medium_k3_rerank")

    assert res.has_evidence is True
    assert len(res.passages) > 0
    assert res.refusal_reason is None

    top_chunk = res.passages[0].chunk
    assert top_chunk.paper_id == "1706.03762v7"
    assert "Attention Is All You Need" in top_chunk.paper_title
    assert "dmodel" in top_chunk.content.lower()


def test_relevant_passage_retrieval_moshi(retrieval_service: RetrievalService) -> None:
    """Verify relevant passages are retrieved for Moshi questions."""
    query = "How does the Mimi neural audio codec operate with discrete bottleneck?"
    res = retrieval_service.retrieve(query, setting_name="medium_k3_rerank")

    assert res.has_evidence is True
    assert len(res.passages) > 0

    # Ensure Moshi paper is prominently retrieved
    paper_ids = [p.chunk.paper_id for p in res.passages]
    assert "2410.00037v1" in paper_ids


def test_relevant_passage_retrieval_dynamic_chunking(retrieval_service: RetrievalService) -> None:
    """Verify relevant passages are retrieved for Dynamic Chunking questions."""
    query = "H-Net hierarchical architecture dynamic subsampling sequence modeling"
    res = retrieval_service.retrieve(query, setting_name="medium_k3")

    assert res.has_evidence is True
    assert len(res.passages) > 0

    top_chunk = res.passages[0].chunk
    assert top_chunk.paper_id == "2507.07955v2"
    assert "Dynamic Chunking" in top_chunk.paper_title


def test_empty_query_handling(retrieval_service: RetrievalService) -> None:
    """Verify empty or whitespace-only queries are cleanly refused."""
    for empty_q in ["", "   ", "\n\t"]:
        res = retrieval_service.retrieve(empty_q)
        assert res.has_evidence is False
        assert len(res.passages) == 0
        assert res.refusal_reason is not None
        assert "empty" in res.refusal_reason.lower()


def test_missing_evidence_refusal(retrieval_service: RetrievalService) -> None:
    """Verify queries completely absent from research papers trigger deterministic refusal."""
    out_of_domain_queries = [
        "What is the secret baking recipe for a chocolate strawberry birthday cake?",
        "How do you change the engine motor oil on a 2012 Honda Civic?",
        "What are the top travel destinations in Switzerland during winter?",
    ]

    for q in out_of_domain_queries:
        res = retrieval_service.retrieve(q, setting_name="medium_k3")
        assert res.has_evidence is False
        assert len(res.passages) == 0
        assert res.refusal_reason is not None
        assert "not contain" in res.refusal_reason.lower() or "insufficient" in res.refusal_reason.lower()


def test_target_paper_filter(retrieval_service: RetrievalService) -> None:
    """Verify target_papers restriction strictly isolates search results."""
    query = "encoder decoder architecture layers"
    res = retrieval_service.retrieve(
        query,
        setting_name="medium_k3",
        target_papers=["1706.03762v7"],
    )

    assert res.has_evidence is True
    assert len(res.passages) > 0
    for p in res.passages:
        assert p.chunk.paper_id == "1706.03762v7"


def test_citation_metadata_integrity(retrieval_service: RetrievalService) -> None:
    """Verify that every retrieved passage has complete, accurate citation metadata."""
    query = "What is multi-head self-attention mechanism?"
    res = retrieval_service.retrieve(query, setting_name="medium_k3")

    assert res.has_evidence is True
    assert len(res.passages) > 0

    for item in res.passages:
        chunk = item.chunk
        assert chunk.paper_title != ""
        assert chunk.version in ("v7", "v1", "v2")
        assert chunk.page_number >= 1
        assert chunk.section != ""
        assert chunk.source_url.startswith("https://arxiv.org/pdf/")
        assert chunk.word_count > 0

        # Check citation string formatting
        citation = item.citation
        assert chunk.paper_title in citation
        assert chunk.paper_id in citation
        assert str(chunk.page_number) in citation
        assert chunk.section in citation


def test_configurable_settings_presets(retrieval_service: RetrievalService) -> None:
    """Verify that small_k6 and large_k2 presets return their configured top-k counts."""
    query = "attention mechanism positional encoding transformer"

    res_small = retrieval_service.retrieve(query, setting_name="small_k6")
    assert res_small.has_evidence is True
    assert len(res_small.passages) <= 6

    res_large = retrieval_service.retrieve(query, setting_name="large_k2")
    assert res_large.has_evidence is True
    assert len(res_large.passages) <= 2
