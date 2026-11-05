"""Retrieval over the vector domains, with each technique switchable so it can be ablated.

modes:
  dense   naive top-k by cosine similarity (the baseline)
  bm25    lexical only
  hybrid  dense + BM25 fused with reciprocal rank fusion
"""
from dataclasses import replace
from pathlib import Path

from src.retrieval.bm25 import BM25
from src.ingestion.embeddings import INDEX_DIR, Embedder
from src.ingestion.store import Hit, VectorStore

VECTOR_DOMAINS = ("policy", "product")  # "customer" is a structured lookup, not a vector index
MODES = ("dense", "bm25", "hybrid")


def rrf(rankings: list[list[Hit]], k: int = 60) -> list[Hit]:
    """Reciprocal rank fusion: score = sum of 1/(k+rank). Uses ranks only, so dense cosine and BM25
    scores need no calibration."""
    scores, first = {}, {}
    for hits in rankings:
        for rank, h in enumerate(hits):
            scores[h.id] = scores.get(h.id, 0) + 1 / (k + rank + 1)
            first.setdefault(h.id, h)
    return [replace(first[i], score=s) for i, s in sorted(scores.items(), key=lambda x: -x[1])]


class Retriever:
    def __init__(self, index_dir: str | Path = INDEX_DIR):
        self.embedder = Embedder()
        self.stores = {d: VectorStore(Path(index_dir), d) for d in VECTOR_DOMAINS}
        self.bm25 = {d: BM25(s.records) for d, s in self.stores.items()}

    def retrieve(self, query: str, k: int = 5, domains=VECTOR_DOMAINS, where: dict | None = None,
                 mode: str = "dense", pool: int = 20, include_archived: bool = False) -> list[Hit]:
        """Top-k chunks over the routed domains. `pool` is how many candidates each retriever contributes."""
        assert mode in MODES, mode
        qvec = self.embedder.embed_query(query)
        candidates = []
        for d in (d for d in domains if d in self.stores):
            # superseded or not-yet-effective policy versions are kept for audit but never shown, or the model sees conflicting rules
            w = {**(where or {}), "archived": False} if d == "policy" and not include_archived else where
            dense = self.stores[d].search(qvec, pool, w)
            lexical = self.bm25[d].search(query, pool, w)
            fused = rrf([dense, lexical])
            candidates += {"dense": dense, "bm25": lexical, "hybrid": fused}[mode]
        return sorted(candidates, key=lambda h: -h.score)[:k]
