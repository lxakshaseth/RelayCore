"""Re-ranking engine to refine initial BM25 retrieval candidates."""

import re
from typing import List
from app.retrieval.bm25 import tokenize
from app.retrieval.models import DocumentChunk, RetrievalResult


class PassageReranker:
    """Applies lexical coverage, section alignment, and exact-phrase boosts to top candidates."""

    def __init__(self, section_boost: float = 1.35, coverage_weight: float = 0.4, phrase_boost: float = 1.5):
        self.section_boost = section_boost
        self.coverage_weight = coverage_weight
        self.phrase_boost = phrase_boost

    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Re-scores and sorts candidate results."""
        if not results or not query.strip():
            return results

        query_tokens = tokenize(query)
        if not query_tokens:
            return results

        unique_query_tokens = set(query_tokens)
        query_lower = query.lower().strip()

        reranked_results: List[RetrievalResult] = []

        for item in results:
            chunk = item.chunk
            base_score = item.score

            # 1. Query Term Coverage (Fraction of unique query keywords matched in chunk)
            chunk_tokens = set(tokenize(chunk.content))
            matched_tokens = unique_query_tokens.intersection(chunk_tokens)
            coverage_ratio = len(matched_tokens) / len(unique_query_tokens) if unique_query_tokens else 0.0

            # 2. Section Title Alignment (Boost if query terms appear directly in section heading)
            section_tokens = set(tokenize(chunk.section))
            section_overlap = len(unique_query_tokens.intersection(section_tokens))
            section_multiplier = self.section_boost if section_overlap > 0 else 1.0

            # 3. Exact Phrase Match Boost (Look for 2+ word sequences from query in chunk text)
            phrase_multiplier = 1.0
            if len(query_tokens) >= 2:
                # Check for 2-word shingles
                for i in range(len(query_tokens) - 1):
                    pair = f"{query_tokens[i]} {query_tokens[i+1]}"
                    if pair in chunk.content.lower():
                        phrase_multiplier = self.phrase_boost
                        break

            # Composite adjusted score
            adjusted_score = (
                base_score * (1.0 + self.coverage_weight * coverage_ratio)
                * section_multiplier
                * phrase_multiplier
            )

            reranked_results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=base_score,
                    rerank_score=round(adjusted_score, 4),
                    citation=item.citation,
                )
            )

        # Sort descending by rerank_score
        reranked_results.sort(key=lambda x: (x.rerank_score or x.score), reverse=True)
        return reranked_results
