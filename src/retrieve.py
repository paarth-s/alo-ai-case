"""Retrieval over the vector domains, with each technique switchable so it can be ablated.

modes:
  dense          naive top-k by cosine similarity (the baseline)
  bm25           lexical only
  hybrid         dense + BM25 fused with reciprocal rank fusion
  dense_rerank   dense candidates re-scored by a cross-encoder
  hybrid_rerank  hybrid candidates re-scored by a cross-encoder
"""
from dataclasses import replace
from pathlib import Path

from .bm25 import BM25
from .embeddings import DEFAULT, Embedder, index_dir_for
from .rerank import Reranker
from .store import Hit, VectorStore

VECTOR_DOMAINS = ("policy", "product")  # "customer" is a structured lookup, not a vector index
MODES = ("dense", "bm25", "hybrid", "dense_rerank", "hybrid_rerank")


def rrf(rankings: list[list[Hit]], weights: list[float] | None = None, k: int = 60) -> list[Hit]:
    """Weighted reciprocal rank fusion: score = sum of w/(k+rank). Uses ranks only, so dense cosine and BM25
    scores need no calibration. A smaller k rewards top ranks more steeply."""
    scores, first = {}, {}
    for hits, w in zip(rankings, weights or [1.0] * len(rankings)):
        for rank, h in enumerate(hits):
            scores[h.id] = scores.get(h.id, 0) + w / (k + rank + 1)
            first.setdefault(h.id, h)
    return [replace(first[i], score=s) for i, s in sorted(scores.items(), key=lambda x: -x[1])]


class Retriever:
    def __init__(self, index_dir: str | Path | None = None, reranker: str = "minilm", embedder: str = DEFAULT):
        self.embedder = Embedder(embedder)
        self.stores = {d: VectorStore(Path(index_dir or index_dir_for(embedder)), d) for d in VECTOR_DOMAINS}
        self.bm25 = {d: BM25(s.records) for d, s in self.stores.items()}
        self._reranker_name, self._reranker = reranker, None

    @property
    def reranker(self) -> Reranker:
        self._reranker = self._reranker or Reranker(self._reranker_name)  # load only when a rerank mode is used
        return self._reranker

    def retrieve(self, query: str, k: int = 5, domains=VECTOR_DOMAINS, where: dict | None = None,
                 mode: str = "dense", pool: int = 20, lex_weight: float = 1.0, rrf_k: int = 60,
                 include_archived: bool = False) -> list[Hit]:
        """Top-k chunks over the routed domains. `pool` is how many candidates each retriever contributes;
        `lex_weight` and `rrf_k` tune the hybrid fusion."""
        assert mode in MODES, mode
        qvec = self.embedder.embed_query(query)
        candidates = []
        for d in (d for d in domains if d in self.stores):
            # superseded or not-yet-effective policy versions are kept for audit but never shown, or the model sees conflicting rules
            w = {**(where or {}), "archived": False} if d == "policy" and not include_archived else where
            dense = self.stores[d].search(qvec, pool, w)
            lexical = self.bm25[d].search(query, pool, w)
            fused = rrf([dense, lexical], [1.0, lex_weight], rrf_k)
            candidates += {"dense": dense, "dense_rerank": dense, "bm25": lexical, "hybrid": fused, "hybrid_rerank": fused}[mode]
        if mode.endswith("rerank"):  # reranking pools every routed domain, so scores are directly comparable
            return self.reranker.rerank(query, candidates)[:k]
        return sorted(candidates, key=lambda h: -h.score)[:k]
