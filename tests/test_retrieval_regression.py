"""Regression tests for paper retrieval, version suffix filtering, and diagnostic telemetry."""

import pytest
from app.retrieval.service import RetrievalService, get_retrieval_service


@pytest.fixture(scope="module")
def retrieval_service() -> RetrievalService:
    """Provide initialized RetrievalService for regression testing."""
    service = get_retrieval_service(data_dir="data/papers")
    service.initialize_indexes(["medium_k3", "setting_a", "setting_b", "setting_c"])
    return service


def test_transformer_question_retrieves_relevant_evidence(
    retrieval_service: RetrievalService,
) -> None:
    """1. Prove the Transformer question retrieves relevant evidence with target_papers=['1706.03762']."""
    query = "What is the main idea behind the Transformer architecture?"
    res = retrieval_service.retrieve(
        query=query,
        setting_name="medium_k3",
        target_papers=["1706.03762"],
    )

    # Must find grounding evidence and not refuse
    assert res.has_evidence is True, f"Expected evidence, got refusal: {res.refusal_reason}"
    assert res.refusal_reason is None
    assert len(res.passages) > 0, "Expected at least 1 retrieved passage"

    # Verify top chunk is grounded in Attention Is All You Need
    top_chunk = res.passages[0].chunk
    assert top_chunk.paper_id == "1706.03762v7"
    assert "Attention Is All You Need" in top_chunk.paper_title
    assert top_chunk.section in [
        "3 Model Architecture",
        "Abstract",
        "1 Introduction",
        "3.1 Encoder and Decoder Stacks",
    ]
    assert top_chunk.page_number in [1, 2, 3]

    # Verify diagnostic metadata completeness
    assert res.diagnostics is not None
    diag = res.diagnostics
    assert diag["indexed_documents"] == 3
    assert diag["chunks_before_filtering"] > 0
    assert diag["chunks_after_paper_filtering"] == 52
    assert diag["chunks_after_retrieval"] == len(res.passages)
    assert len(diag["top_5_retrieval_scores"]) == 5
    assert diag["top_5_retrieval_scores"][0] >= 4.0
    assert len(diag["top_5_passage_snippets"]) == 5
    assert diag["matched_paper_ids"] == ["1706.03762v7"]
    assert diag["keyword_coverage"] >= 0.40
    assert diag["confidence_threshold"] == 4.0
    assert diag["exact_refusal_reason"] is None


def test_unrelated_topic_correctly_refused(
    retrieval_service: RetrievalService,
) -> None:
    """2. Prove that questions about unrelated topics are correctly and safely refused."""
    unrelated_queries = [
        "What is the best recipe for baking chocolate chip cookies?",
        "How do you train an equestrian horse to jump over high hurdles?",
        "What are the official rules and boundary measurements of professional cricket?",
    ]

    for q in unrelated_queries:
        res = retrieval_service.retrieve(q, setting_name="medium_k3")
        assert res.has_evidence is False, f"Expected refusal for out-of-domain query '{q}'"
        assert len(res.passages) == 0
        assert res.refusal_reason is not None
        assert any(
            phrase in res.refusal_reason.lower()
            for phrase in ["not contain", "insufficient", "cannot ground"]
        )

        # Check diagnostics recording refusal
        assert res.diagnostics is not None
        assert res.diagnostics["chunks_after_retrieval"] == 0
        assert res.diagnostics["exact_refusal_reason"] is not None


def test_paper_filtering_with_and_without_version_suffixes(
    retrieval_service: RetrievalService,
) -> None:
    """3. Prove that paper filtering works seamlessly with and without version suffixes."""
    query = "What is the main idea behind the Transformer architecture?"

    # Target variants that MUST all resolve to Attention Is All You Need (1706.03762v7)
    valid_targets = [
        ["1706.03762"],               # Without version suffix
        ["1706.03762v7"],             # With exact version suffix
        ["arXiv:1706.03762"],         # With arXiv prefix
        ["arXiv:1706.03762v7"],       # With arXiv prefix and version
        ["Attention Is All You Need"], # By title fragment
    ]

    for target in valid_targets:
        res = retrieval_service.retrieve(
            query=query,
            setting_name="medium_k3",
            target_papers=target,
        )
        assert res.has_evidence is True, f"Failed to match target: {target}"
        assert len(res.passages) > 0
        for p in res.passages:
            assert p.chunk.paper_id == "1706.03762v7"
            assert "Attention Is All You Need" in p.chunk.paper_title

    # Target that does NOT exist in library
    res_nonexistent = retrieval_service.retrieve(
        query=query,
        setting_name="medium_k3",
        target_papers=["9999.99999"],
    )
    assert res_nonexistent.has_evidence is False
    assert res_nonexistent.diagnostics["chunks_after_paper_filtering"] == 0
    assert "not found" in res_nonexistent.refusal_reason.lower()


def test_citations_contain_valid_metadata(
    retrieval_service: RetrievalService,
) -> None:
    """4. Prove citations contain valid paper IDs, sections, and page numbers."""
    query = "What is the main idea behind the Transformer architecture?"
    res = retrieval_service.retrieve(
        query=query,
        setting_name="medium_k3",
        target_papers=["1706.03762"],
    )

    assert res.has_evidence is True
    assert len(res.passages) >= 2

    for p in res.passages:
        chunk = p.chunk
        # Verify physical PDF and catalog bounds
        assert chunk.paper_id == "1706.03762v7"
        assert chunk.paper_title == "Attention Is All You Need"
        assert chunk.version == "v7"
        assert chunk.section != ""
        assert isinstance(chunk.page_number, int)
        assert 1 <= chunk.page_number <= 15

        # Verify structured citation string
        citation_str = p.citation
        assert chunk.paper_title in citation_str
        assert chunk.paper_id in citation_str
        assert f"Section: '{chunk.section}'" in citation_str
        assert f"Page {chunk.page_number}" in citation_str
