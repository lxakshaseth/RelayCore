"""Self-contained Okapi BM25 indexer with term weighting and document length normalization."""

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
}


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric and hyphenated tokens, removing stopwords."""
    tokens = re.findall(r"\b[a-zA-Z0-9\-_]{2,}\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


class BM25Okapi:
    """Okapi BM25 ranking algorithm implementation."""

    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_tokens)
        self.doc_lengths = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 0.0

        # Term frequencies per document
        self.doc_term_freqs: List[Counter] = [Counter(doc) for doc in corpus_tokens]

        # Document frequencies
        self.doc_freqs: Dict[str, int] = Counter()
        for doc in corpus_tokens:
            for term in set(doc):
                self.doc_freqs[term] += 1

        # Inverse Document Frequency (IDF) lookup
        self.idf: Dict[str, float] = {}
        for term, df in self.doc_freqs.items():
            # Robertson-Spärck Jones formula with +1 smoothing
            self.idf[term] = math.log(((self.corpus_size - df + 0.5) / (df + 0.5)) + 1.0)

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Calculate BM25 score for a specific document against query tokens."""
        if not query_tokens or doc_idx >= self.corpus_size:
            return 0.0

        doc_len = self.doc_lengths[doc_idx]
        term_freqs = self.doc_term_freqs[doc_idx]
        denom_part = self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl)) if self.avgdl > 0 else 1.0

        score = 0.0
        for token in query_tokens:
            if token not in self.idf:
                continue

            tf = term_freqs.get(token, 0)
            if tf > 0:
                idf = self.idf[token]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + denom_part
                score += idf * (numerator / denominator)

        return score

    def get_top_k(self, query: str, k: int = 3) -> List[Tuple[int, float]]:
        """Retrieve the top-K document indices and scores for a given query."""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: List[Tuple[int, float]] = []
        for idx in range(self.corpus_size):
            s = self.score(query_tokens, idx)
            if s > 0.0:
                scores.append((idx, s))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
