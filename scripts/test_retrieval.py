"""Test script to test RetrievalService indexing and query capabilities."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.retrieval.service import RetrievalService


def main():
    service = RetrievalService(data_dir="data/papers")
    print("Initializing indexes for medium_k3 and medium_k3_rerank...")
    service.initialize_indexes(["medium_k3", "medium_k3_rerank"])

    test_queries = [
        "What is the dimension of the embedding layers dmodel in Transformer?",
        "How does the Mimi audio codec work in Moshi?",
        "What is dynamic chunking in hierarchical sequence modeling?",
        "What is the recipe for chocolate cake?",  # Out of corpus
        "",  # Empty query
    ]

    for q in test_queries:
        print("\n" + "=" * 80)
        print(f"QUERY: '{q}'")
        res = service.retrieve(q, setting_name="medium_k3_rerank")
        print(f"HAS EVIDENCE: {res.has_evidence}")
        if not res.has_evidence:
            print(f"REFUSAL REASON: {res.refusal_reason}")
        else:
            print(f"Found {len(res.passages)} passages:")
            for idx, p in enumerate(res.passages):
                print(f"\n--- Passage {idx+1} [BM25={p.score}, Rerank={p.rerank_score}] ---")
                print(f"Citation: {p.citation}")
                print(f"Section: {p.chunk.section} | Page: {p.chunk.page_number}")
                print(f"Snippet: {p.chunk.content[:220]}...")


if __name__ == "__main__":
    main()
