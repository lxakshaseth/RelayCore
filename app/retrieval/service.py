"""Comprehensive research paper retrieval service with citation formatting and refusal guardrails."""

import logging
from typing import Dict, List, Optional, Set
from app.retrieval.bm25 import BM25Okapi, tokenize
from app.retrieval.chunker import DocumentChunker
from app.retrieval.models import (
    DocumentChunk,
    RetrievalResponse,
    RetrievalResult,
    RetrievalSetting,
)
from app.retrieval.parser import CANONICAL_PAPERS, PaperParser
from app.retrieval.reranker import PassageReranker

logger = logging.getLogger(__name__)


RETRIEVAL_PRESETS: Dict[str, RetrievalSetting] = {
    "small_k6": RetrievalSetting(
        name="small_k6",
        chunk_size_words=150,
        overlap_words=30,
        top_k=6,
        rerank=False,
        min_score_threshold=3.5,
    ),
    "medium_k3": RetrievalSetting(
        name="medium_k3",
        chunk_size_words=350,
        overlap_words=50,
        top_k=3,
        rerank=False,
        min_score_threshold=4.0,
    ),
    "medium_k3_rerank": RetrievalSetting(
        name="medium_k3_rerank",
        chunk_size_words=350,
        overlap_words=50,
        top_k=3,
        rerank=True,
        min_score_threshold=4.5,
    ),
    "large_k2": RetrievalSetting(
        name="large_k2",
        chunk_size_words=800,
        overlap_words=100,
        top_k=2,
        rerank=False,
        min_score_threshold=4.5,
    ),
}


class RetrievalService:
    """Manages paper parsing, multi-setting BM25 indexing, re-ranking, and citation generation."""

    def __init__(self, data_dir: str = "data/papers"):
        self.data_dir = data_dir
        self.parser = PaperParser(data_dir=data_dir)
        self.reranker = PassageReranker()

        self._chunks_by_setting: Dict[str, List[DocumentChunk]] = {}
        self._bm25_by_setting: Dict[str, BM25Okapi] = {}
        self._corpus_vocabulary: Set[str] = set()
        self._initialized: bool = False

    def initialize_indexes(self, settings_to_build: Optional[List[str]] = None) -> None:
        """Parse canonical papers and pre-build BM25 indexes for specified settings."""
        if settings_to_build is None:
            settings_to_build = list(RETRIEVAL_PRESETS.keys())

        logger.info("Parsing 3 canonical research papers from %s...", self.data_dir)
        parsed_papers = self.parser.parse_all_papers()
        all_blocks = []
        for blocks in parsed_papers.values():
            all_blocks.extend(blocks)

        for setting_key in settings_to_build:
            if setting_key not in RETRIEVAL_PRESETS:
                continue
            setting = RETRIEVAL_PRESETS[setting_key]
            logger.info("Chunking and indexing corpus for setting: %s...", setting.name)

            chunker = DocumentChunker(
                chunk_size_words=setting.chunk_size_words,
                overlap_words=setting.overlap_words,
            )
            chunks = chunker.chunk_all_blocks(all_blocks)
            self._chunks_by_setting[setting_key] = chunks

            # Build tokenized representation for BM25
            tokenized_corpus = [tokenize(f"{c.section} {c.content}") for c in chunks]
            bm25 = BM25Okapi(tokenized_corpus)
            self._bm25_by_setting[setting_key] = bm25
            self._corpus_vocabulary.update(bm25.doc_freqs.keys())
            logger.info("Setting %s indexed: %d chunks created.", setting_key, len(chunks))

        self._initialized = True

    def format_citation(self, chunk: DocumentChunk) -> str:
        """Construct exact human-readable and structured citation string."""
        return (
            f"[{chunk.paper_title} ({chunk.paper_id}), Section: '{chunk.section}', Page {chunk.page_number}]"
        )

    def retrieve(
        self,
        query: str,
        setting_name: str = "medium_k3",
        top_k: Optional[int] = None,
        target_papers: Optional[List[str]] = None,
        rerank: Optional[bool] = None,
        min_score_threshold: Optional[float] = None,
        min_coverage_ratio: float = 0.4,
    ) -> RetrievalResponse:
        """Retrieve grounded passages for a query under specified retrieval setting.

        Returns RetrievalResponse with has_evidence=False if the papers lack supporting text.
        """
        if not self._initialized:
            self.initialize_indexes([setting_name])

        clean_query = query.strip() if query else ""
        if not clean_query:
            return RetrievalResponse(
                query=query,
                passages=[],
                has_evidence=False,
                refusal_reason="Empty query provided; cannot retrieve passages.",
                retrieval_setting=setting_name,
                total_candidates_searched=0,
            )

        setting = RETRIEVAL_PRESETS.get(setting_name, RETRIEVAL_PRESETS["medium_k3"])
        effective_k = top_k if top_k is not None else setting.top_k
        effective_rerank = rerank if rerank is not None else setting.rerank
        threshold = (
            min_score_threshold if min_score_threshold is not None else setting.min_score_threshold
        )

        bm25_index = self._bm25_by_setting.get(setting_name)
        chunks = self._chunks_by_setting.get(setting_name, [])

        if not bm25_index or not chunks:
            self.initialize_indexes([setting_name])
            bm25_index = self._bm25_by_setting[setting_name]
            chunks = self._chunks_by_setting[setting_name]

        query_tokens = tokenize(clean_query)
        if not query_tokens:
            return RetrievalResponse(
                query=clean_query,
                passages=[],
                has_evidence=False,
                refusal_reason="Query contains no meaningful keywords after stopword filtering.",
                retrieval_setting=setting_name,
                total_candidates_searched=len(chunks),
            )

        # 1. Corpus Vocabulary Check: Detect keywords completely absent from library
        unique_query_tokens = set(query_tokens)
        missing_from_corpus = [t for t in unique_query_tokens if t not in self._corpus_vocabulary]
        
        # If more than half of the distinct query terms don't exist in the corpus at all
        if len(unique_query_tokens) >= 2 and (len(missing_from_corpus) / len(unique_query_tokens)) > 0.5:
            missing_terms_str = ", ".join(f"'{t}'" for t in missing_from_corpus)
            return RetrievalResponse(
                query=clean_query,
                passages=[],
                has_evidence=False,
                refusal_reason=(
                    f"The research papers do not contain references to: {missing_terms_str}. "
                    "Cannot ground an answer from the library."
                ),
                retrieval_setting=setting_name,
                total_candidates_searched=len(chunks),
            )

        # 2. Extract top candidates using BM25
        candidate_pool_size = effective_k * 3 if effective_rerank else effective_k
        top_candidates = bm25_index.get_top_k(clean_query, k=candidate_pool_size)

        results: List[RetrievalResult] = []
        for idx, score in top_candidates:
            chunk = chunks[idx]

            # Optional filter by paper_id or arxiv_id
            if target_papers:
                matches_target = any(
                    target in chunk.paper_id or target in chunk.source_url
                    for target in target_papers
                )
                if not matches_target:
                    continue

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=round(score, 4),
                    citation=self.format_citation(chunk),
                )
            )

        # 3. Apply secondary re-ranking if configured
        if effective_rerank and results:
            results = self.reranker.rerank(clean_query, results)

        final_passages = results[:effective_k]

        # 4. Evidence Validation & Refusal Guardrails
        if not final_passages:
            return RetrievalResponse(
                query=clean_query,
                passages=[],
                has_evidence=False,
                refusal_reason=f"The provided research papers do not contain passages matching '{clean_query}'.",
                retrieval_setting=setting_name,
                total_candidates_searched=len(chunks),
            )

        best_result = final_passages[0]
        best_score = best_result.rerank_score or best_result.score

        # Check coverage of query terms in the best chunk
        best_chunk_tokens = set(tokenize(f"{best_result.chunk.section} {best_result.chunk.content}"))
        matched_tokens = unique_query_tokens.intersection(best_chunk_tokens)
        coverage_ratio = len(matched_tokens) / len(unique_query_tokens) if unique_query_tokens else 0.0

        if best_score < threshold or (len(unique_query_tokens) >= 2 and coverage_ratio < min_coverage_ratio):
            return RetrievalResponse(
                query=clean_query,
                passages=[],
                has_evidence=False,
                refusal_reason=(
                    f"Insufficient grounding evidence in the library. Retrieval score ({best_score:.2f}) "
                    f"or keyword coverage ({coverage_ratio:.1%}) did not meet confidence threshold."
                ),
                retrieval_setting=setting_name,
                total_candidates_searched=len(chunks),
            )

        return RetrievalResponse(
            query=clean_query,
            passages=final_passages,
            has_evidence=True,
            refusal_reason=None,
            retrieval_setting=setting_name,
            total_candidates_searched=len(chunks),
        )


_service_instance: Optional[RetrievalService] = None


def get_retrieval_service(data_dir: str = "data/papers") -> RetrievalService:
    """Retrieve or instantiate singleton RetrievalService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = RetrievalService(data_dir=data_dir)
    return _service_instance
