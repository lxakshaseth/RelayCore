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
    # Assignment Presets:
    # A. Small chunks, low top-k, no re-ranking
    "setting_a": RetrievalSetting(
        name="setting_a",
        chunk_size_words=150,
        overlap_words=30,
        top_k=2,
        rerank=False,
        min_score_threshold=3.0,
    ),
    # B. Medium chunks, moderate top-k, no re-ranking
    "setting_b": RetrievalSetting(
        name="setting_b",
        chunk_size_words=350,
        overlap_words=50,
        top_k=4,
        rerank=False,
        min_score_threshold=3.5,
    ),
    # C. Larger chunks, moderate top-k, with re-ranking
    "setting_c": RetrievalSetting(
        name="setting_c",
        chunk_size_words=800,
        overlap_words=100,
        top_k=3,
        rerank=True,
        min_score_threshold=4.0,
    ),
    # Named Aliases
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

    @staticmethod
    def _chunk_matches_target(chunk: DocumentChunk, target: str) -> bool:
        """Check if a chunk matches a target paper identifier, with or without version."""
        clean_target = target.strip().lower()
        if clean_target.startswith("arxiv:"):
            clean_target = clean_target.split("arxiv:", 1)[1].strip()

        # Extract base arXiv IDs without version (e.g. '1706.03762' from '1706.03762v7')
        base_chunk_id = chunk.paper_id.lower().split("v")[0]
        base_target = clean_target.split("v")[0]

        return (
            clean_target in chunk.paper_id.lower()
            or base_target == base_chunk_id
            or clean_target in chunk.paper_title.lower()
            or clean_target in chunk.source_url.lower()
        )

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
        num_indexed_docs = len(CANONICAL_PAPERS)

        if not clean_query:
            return RetrievalResponse(
                query=query,
                passages=[],
                has_evidence=False,
                refusal_reason="Empty query provided; cannot retrieve passages.",
                retrieval_setting=setting_name,
                total_candidates_searched=0,
                diagnostics={
                    "indexed_documents": num_indexed_docs,
                    "chunks_before_filtering": 0,
                    "chunks_after_paper_filtering": 0,
                    "chunks_after_retrieval": 0,
                    "top_5_retrieval_scores": [],
                    "top_5_passage_snippets": [],
                    "matched_paper_ids": [],
                    "keyword_coverage": 0.0,
                    "confidence_threshold": 0.0,
                    "exact_refusal_reason": "Empty query provided; cannot retrieve passages.",
                },
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

        num_chunks_before = len(chunks)

        # 1. Target Paper Pre-Filtering (search space partitioning)
        eligible_indices: Optional[Set[int]] = None
        if target_papers:
            eligible_indices = {
                idx
                for idx, c in enumerate(chunks)
                if any(self._chunk_matches_target(c, t) for t in target_papers)
            }
            num_chunks_after_filter = len(eligible_indices)
        else:
            num_chunks_after_filter = num_chunks_before

        # If user targeted papers that do not exist in the collection
        if target_papers and num_chunks_after_filter == 0:
            refusal_reason = (
                f"Target paper(s) {target_papers} were not found in the indexed collection of {num_indexed_docs} documents."
            )
            self._log_diagnostics(
                query=clean_query,
                num_indexed_docs=num_indexed_docs,
                num_chunks_before=num_chunks_before,
                num_chunks_after_filter=0,
                num_chunks_after_retrieval=0,
                top5_scores=[],
                top5_snippets=[],
                matched_paper_ids=[],
                coverage_ratio=0.0,
                threshold=threshold,
                refusal_reason=refusal_reason,
            )
            return RetrievalResponse(
                query=clean_query,
                passages=[],
                has_evidence=False,
                refusal_reason=refusal_reason,
                retrieval_setting=setting_name,
                total_candidates_searched=0,
                diagnostics={
                    "indexed_documents": num_indexed_docs,
                    "chunks_before_filtering": num_chunks_before,
                    "chunks_after_paper_filtering": 0,
                    "chunks_after_retrieval": 0,
                    "top_5_retrieval_scores": [],
                    "top_5_passage_snippets": [],
                    "matched_paper_ids": [],
                    "keyword_coverage": 0.0,
                    "confidence_threshold": threshold,
                    "exact_refusal_reason": refusal_reason,
                },
            )

        query_tokens = tokenize(clean_query)
        if not query_tokens:
            refusal_reason = "Query contains no meaningful keywords after stopword filtering."
            self._log_diagnostics(
                query=clean_query,
                num_indexed_docs=num_indexed_docs,
                num_chunks_before=num_chunks_before,
                num_chunks_after_filter=num_chunks_after_filter,
                num_chunks_after_retrieval=0,
                top5_scores=[],
                top5_snippets=[],
                matched_paper_ids=[],
                coverage_ratio=0.0,
                threshold=threshold,
                refusal_reason=refusal_reason,
            )
            return RetrievalResponse(
                query=clean_query,
                passages=[],
                has_evidence=False,
                refusal_reason=refusal_reason,
                retrieval_setting=setting_name,
                total_candidates_searched=num_chunks_after_filter,
                diagnostics={
                    "indexed_documents": num_indexed_docs,
                    "chunks_before_filtering": num_chunks_before,
                    "chunks_after_paper_filtering": num_chunks_after_filter,
                    "chunks_after_retrieval": 0,
                    "top_5_retrieval_scores": [],
                    "top_5_passage_snippets": [],
                    "matched_paper_ids": [],
                    "keyword_coverage": 0.0,
                    "confidence_threshold": threshold,
                    "exact_refusal_reason": refusal_reason,
                },
            )

        # 2. Corpus Vocabulary Check: Detect keywords completely absent from library
        unique_query_tokens = set(query_tokens)
        missing_from_corpus = [t for t in unique_query_tokens if t not in self._corpus_vocabulary]

        # If more than half of the distinct query terms don't exist in the corpus at all
        if len(unique_query_tokens) >= 2 and (len(missing_from_corpus) / len(unique_query_tokens)) > 0.5:
            missing_terms_str = ", ".join(f"'{t}'" for t in missing_from_corpus)
            refusal_reason = (
                f"The research papers do not contain references to: {missing_terms_str}. "
                "Cannot ground an answer from the library."
            )
            self._log_diagnostics(
                query=clean_query,
                num_indexed_docs=num_indexed_docs,
                num_chunks_before=num_chunks_before,
                num_chunks_after_filter=num_chunks_after_filter,
                num_chunks_after_retrieval=0,
                top5_scores=[],
                top5_snippets=[],
                matched_paper_ids=[],
                coverage_ratio=0.0,
                threshold=threshold,
                refusal_reason=refusal_reason,
            )
            return RetrievalResponse(
                query=clean_query,
                passages=[],
                has_evidence=False,
                refusal_reason=refusal_reason,
                retrieval_setting=setting_name,
                total_candidates_searched=num_chunks_after_filter,
                diagnostics={
                    "indexed_documents": num_indexed_docs,
                    "chunks_before_filtering": num_chunks_before,
                    "chunks_after_paper_filtering": num_chunks_after_filter,
                    "chunks_after_retrieval": 0,
                    "top_5_retrieval_scores": [],
                    "top_5_passage_snippets": [],
                    "matched_paper_ids": [],
                    "keyword_coverage": 0.0,
                    "confidence_threshold": threshold,
                    "exact_refusal_reason": refusal_reason,
                },
            )

        # 3. Extract scored candidates using BM25 with pre-filtered indices
        candidate_pool_size = (
            min(num_chunks_after_filter, max(effective_k * 4, 15))
            if effective_rerank
            else effective_k
        )
        all_candidate_scores = bm25_index.get_scores_for_indices(
            query_tokens, eligible_indices=eligible_indices
        )

        top5_scores = [round(s, 4) for _, s in all_candidate_scores[:5]]
        top5_snippets = [
            chunks[idx].content[:80].replace("\n", " ").strip() + "..."
            for idx, _ in all_candidate_scores[:5]
        ]

        if not all_candidate_scores:
            refusal_reason = f"The provided research papers do not contain passages matching '{clean_query}'."
            self._log_diagnostics(
                query=clean_query,
                num_indexed_docs=num_indexed_docs,
                num_chunks_before=num_chunks_before,
                num_chunks_after_filter=num_chunks_after_filter,
                num_chunks_after_retrieval=0,
                top5_scores=[],
                top5_snippets=[],
                matched_paper_ids=[],
                coverage_ratio=0.0,
                threshold=threshold,
                refusal_reason=refusal_reason,
            )
            return RetrievalResponse(
                query=clean_query,
                passages=[],
                has_evidence=False,
                refusal_reason=refusal_reason,
                retrieval_setting=setting_name,
                total_candidates_searched=num_chunks_after_filter,
                diagnostics={
                    "indexed_documents": num_indexed_docs,
                    "chunks_before_filtering": num_chunks_before,
                    "chunks_after_paper_filtering": num_chunks_after_filter,
                    "chunks_after_retrieval": 0,
                    "top_5_retrieval_scores": [],
                    "top_5_passage_snippets": [],
                    "matched_paper_ids": [],
                    "keyword_coverage": 0.0,
                    "confidence_threshold": threshold,
                    "exact_refusal_reason": refusal_reason,
                },
            )

        top_candidates = all_candidate_scores[:candidate_pool_size]
        results: List[RetrievalResult] = [
            RetrievalResult(
                chunk=chunks[idx],
                score=round(score, 4),
                citation=self.format_citation(chunks[idx]),
            )
            for idx, score in top_candidates
        ]

        # 4. Apply secondary re-ranking if configured
        if effective_rerank and results:
            results = self.reranker.rerank(clean_query, results)

        final_passages = results[:effective_k]

        # 5. Evidence Validation & Refusal Guardrails
        best_result = final_passages[0]
        best_score = best_result.rerank_score or best_result.score

        # Check coverage of query terms in the best chunk
        best_chunk_tokens = set(tokenize(f"{best_result.chunk.section} {best_result.chunk.content}"))
        matched_tokens = unique_query_tokens.intersection(best_chunk_tokens)
        coverage_ratio = len(matched_tokens) / len(unique_query_tokens) if unique_query_tokens else 0.0
        matched_paper_ids = sorted(list({p.chunk.paper_id for p in final_passages}))

        if best_score < threshold or (len(unique_query_tokens) >= 2 and coverage_ratio < min_coverage_ratio):
            refusal_reason = (
                f"Insufficient grounding evidence in the library. Retrieval score ({best_score:.2f}) "
                f"or keyword coverage ({coverage_ratio:.1%}) did not meet confidence threshold "
                f"(threshold: {threshold:.2f}, min_coverage: {min_coverage_ratio:.1%})."
            )
            self._log_diagnostics(
                query=clean_query,
                num_indexed_docs=num_indexed_docs,
                num_chunks_before=num_chunks_before,
                num_chunks_after_filter=num_chunks_after_filter,
                num_chunks_after_retrieval=0,
                top5_scores=top5_scores,
                top5_snippets=top5_snippets,
                matched_paper_ids=matched_paper_ids,
                coverage_ratio=coverage_ratio,
                threshold=threshold,
                refusal_reason=refusal_reason,
            )
            return RetrievalResponse(
                query=clean_query,
                passages=[],
                has_evidence=False,
                refusal_reason=refusal_reason,
                retrieval_setting=setting_name,
                total_candidates_searched=num_chunks_after_filter,
                diagnostics={
                    "indexed_documents": num_indexed_docs,
                    "chunks_before_filtering": num_chunks_before,
                    "chunks_after_paper_filtering": num_chunks_after_filter,
                    "chunks_after_retrieval": 0,
                    "top_5_retrieval_scores": top5_scores,
                    "top_5_passage_snippets": top5_snippets,
                    "matched_paper_ids": matched_paper_ids,
                    "keyword_coverage": round(coverage_ratio, 4),
                    "confidence_threshold": threshold,
                    "exact_refusal_reason": refusal_reason,
                },
            )

        # Successful retrieval with verified grounding
        num_chunks_after_retrieval = len(final_passages)
        self._log_diagnostics(
            query=clean_query,
            num_indexed_docs=num_indexed_docs,
            num_chunks_before=num_chunks_before,
            num_chunks_after_filter=num_chunks_after_filter,
            num_chunks_after_retrieval=num_chunks_after_retrieval,
            top5_scores=top5_scores,
            top5_snippets=top5_snippets,
            matched_paper_ids=matched_paper_ids,
            coverage_ratio=coverage_ratio,
            threshold=threshold,
            refusal_reason=None,
        )

        return RetrievalResponse(
            query=clean_query,
            passages=final_passages,
            has_evidence=True,
            refusal_reason=None,
            retrieval_setting=setting_name,
            total_candidates_searched=num_chunks_after_filter,
            diagnostics={
                "indexed_documents": num_indexed_docs,
                "chunks_before_filtering": num_chunks_before,
                "chunks_after_paper_filtering": num_chunks_after_filter,
                "chunks_after_retrieval": num_chunks_after_retrieval,
                "top_5_retrieval_scores": top5_scores,
                "top_5_passage_snippets": top5_snippets,
                "matched_paper_ids": matched_paper_ids,
                "keyword_coverage": round(coverage_ratio, 4),
                "confidence_threshold": threshold,
                "exact_refusal_reason": None,
            },
        )

    def _log_diagnostics(
        self,
        query: str,
        num_indexed_docs: int,
        num_chunks_before: int,
        num_chunks_after_filter: int,
        num_chunks_after_retrieval: int,
        top5_scores: List[float],
        top5_snippets: List[str],
        matched_paper_ids: List[str],
        coverage_ratio: float,
        threshold: float,
        refusal_reason: Optional[str],
    ) -> None:
        """Log structured diagnostics for every retrieval query."""
        logger.info(
            "\n"
            "====================== RETRIEVAL DIAGNOSTICS ======================\n"
            "  Query                       : '%s'\n"
            "  Number of indexed documents : %d\n"
            "  Chunks before filtering     : %d\n"
            "  Chunks after paper filter   : %d\n"
            "  Chunks after retrieval      : %d\n"
            "  Top 5 retrieval scores      : %s\n"
            "  Top 5 passage snippets      :\n"
            "%s\n"
            "  Matched paper IDs           : %s\n"
            "  Keyword coverage            : %.1f%%\n"
            "  Confidence threshold        : %.2f\n"
            "  Exact refusal reason        : %s\n"
            "===================================================================",
            query,
            num_indexed_docs,
            num_chunks_before,
            num_chunks_after_filter,
            num_chunks_after_retrieval,
            top5_scores,
            "\n".join(f"    [{i+1}] {s}" for i, s in enumerate(top5_snippets))
            if top5_snippets
            else "    (None)",
            matched_paper_ids if matched_paper_ids else "[]",
            coverage_ratio * 100.0,
            threshold,
            refusal_reason or "None (Sufficient grounding evidence)",
        )


_service_instance: Optional[RetrievalService] = None


def get_retrieval_service(data_dir: str = "data/papers") -> RetrievalService:
    """Retrieve or instantiate singleton RetrievalService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = RetrievalService(data_dir=data_dir)
    return _service_instance
