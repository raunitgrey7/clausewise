"""Clause retrieval: BM25, TF-IDF cosine, dense cosine and reciprocal-rank-fusion hybrids.

Indexes are built per contract (a few dozen clauses), which is the real deployment shape: a reviewer asks
questions about one document. BM25 is implemented here rather than imported so that its tokenisation is
the same stemmed, stopword-free token stream used by the grounding checker and the answerer. Fusion uses
reciprocal rank rather than score interpolation because BM25 and cosine scores live on different scales.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from clausewise.embeddings import Embedder
from clausewise.schemas import Clause
from clausewise.segment import body_text
from clausewise.textutil import content_tokens

MODES: tuple[str, ...] = ("bm25", "tfidf", "dense", "hybrid")


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Fuse ranked lists of document indices; ties broken by the first ranking's order."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
    first_pos = {doc: i for i, doc in reversed(list(enumerate(rankings[0])))} if rankings else {}
    return sorted(scores, key=lambda d: (-scores[d], first_pos.get(d, 1_000_000), d))


class BM25Index:
    """Okapi BM25 over stemmed content tokens."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tokens = [content_tokens(d) for d in docs]
        self.lengths = np.array([len(t) for t in self.tokens], dtype=np.float64)
        self.avg_len = float(self.lengths.mean()) if len(docs) else 0.0
        self.tfs = [Counter(t) for t in self.tokens]
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document for ``query``."""
        out = np.zeros(len(self.tokens))
        for term in content_tokens(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tfs):
                f = tf.get(term, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self.lengths[i] / max(self.avg_len, 1e-9))
                    out[i] += idf * f * (self.k1 + 1) / denom
        return out


class TfidfIndex:
    """TF-IDF cosine similarity on the same token stream as BM25 (unigrams and bigrams)."""

    def __init__(self, docs: list[str]) -> None:
        self.vec = TfidfVectorizer(analyzer=self._analyze, sublinear_tf=True)
        self.matrix = self.vec.fit_transform(docs) if docs else None

    @staticmethod
    def _analyze(text: str) -> list[str]:
        toks = content_tokens(text)
        return toks + [f"{a} {b}" for a, b in itertools.pairwise(toks)]

    def scores(self, query: str) -> np.ndarray:
        """Cosine similarity of every document to ``query``."""
        if self.matrix is None:
            return np.zeros(0)
        q = self.vec.transform([query])
        return np.asarray((self.matrix @ q.T).todense()).ravel()


class DenseIndex:
    """Cosine similarity between unit-norm embeddings."""

    def __init__(self, docs: list[str], embedder: Embedder) -> None:
        self.embedder = embedder
        self.matrix = embedder.encode(docs) if docs else np.zeros((0, 1), dtype=np.float32)

    def scores(self, query: str) -> np.ndarray:
        """Dot product of unit vectors, so cosine similarity, for every document."""
        if len(self.matrix) == 0:
            return np.zeros(0)
        q = self.embedder.encode([query])[0]
        return self.matrix @ q


def _rank(scores: np.ndarray) -> list[int]:
    """Indices sorted by score descending, stable on ties (earlier clause first)."""
    return [int(i) for i in np.argsort(-scores, kind="stable")]


def index_text(clause: Clause) -> str:
    """The text a clause is indexed under: heading twice (it is a strong signal) plus the body."""
    heading = f"{clause.heading}. {clause.heading}. " if clause.heading else ""
    return heading + body_text(clause)


class ClauseRetriever:
    """Searchable index over one document's clauses.

    ``mode`` is one of bm25, tfidf, dense, hybrid. ``hybrid`` fuses BM25 with the dense index when an
    embedder is available and with TF-IDF otherwise; ``dense`` without an embedder degrades to TF-IDF.
    """

    def __init__(self, clauses: list[Clause], embedder: Embedder | None = None, mode: str = "hybrid") -> None:
        if mode not in MODES:
            raise ValueError(f"unknown retrieval mode {mode!r}; expected one of {MODES}")
        self.clauses = clauses
        docs = [index_text(c) for c in clauses]
        self.mode = mode
        self.bm25 = BM25Index(docs) if mode in {"bm25", "hybrid"} else None
        self.dense = DenseIndex(docs, embedder) if embedder is not None and mode in {"dense", "hybrid"} else None
        needs_tfidf = mode == "tfidf" or (mode in {"dense", "hybrid"} and self.dense is None)
        self.tfidf = TfidfIndex(docs) if needs_tfidf else None

    @property
    def backend(self) -> str:
        """The backend actually used, for reporting."""
        if self.mode == "hybrid":
            return "bm25+dense" if self.dense is not None else "bm25+tfidf"
        if self.mode == "dense" and self.dense is None:
            return "tfidf (dense unavailable)"
        return self.mode

    def rank(self, query: str) -> list[int]:
        """All clause indices ordered by relevance to ``query``."""
        if not self.clauses:
            return []
        rankings: list[list[int]] = []
        if self.bm25 is not None:
            rankings.append(_rank(self.bm25.scores(query)))
        if self.dense is not None:
            rankings.append(_rank(self.dense.scores(query)))
        if self.tfidf is not None:
            rankings.append(_rank(self.tfidf.scores(query)))
        return rankings[0] if len(rankings) == 1 else reciprocal_rank_fusion(rankings)

    def search(self, query: str, k: int = 5) -> list[Clause]:
        """Top ``k`` clauses for ``query``."""
        return [self.clauses[i] for i in self.rank(query)[:k]]
