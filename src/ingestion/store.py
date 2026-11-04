"""Persistent per-domain vector store on Chroma.

Chroma gives us what FAISS did not: metadata filtering, on-disk persistence and upsert/delete by id. What it
does not do is decide *what* changed, so this wrapper keeps the incremental sync: only new or changed chunks
are re-embedded, so a policy edit or catalog refresh never forces a full re-index.

Search is cosine over Chroma's HNSW index (approximate in general, in practice exact at this size).

Vectors are only reusable by the model that made them, so the collection records the embedder's name and
re-embeds everything if it changes (a model upgrade is the one case that does need a full re-index).
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings

from src.ingestion.embeddings import Embedder
from src.ingestion.ingest import Chunk

COLLECTION = "chunks"


@dataclass
class Hit:
    id: str
    score: float  # cosine similarity (vectors are L2-normalised)
    text: str
    meta: dict


def meta_hash(meta: dict) -> str:
    return hashlib.sha256(json.dumps(meta, sort_keys=True, default=str).encode()).hexdigest()[:16]


def matches_filter(meta: dict, where: dict | None) -> bool:
    """True if meta satisfies every field in `where` ({"doc": "returns"} or {"doc": ["returns", "loyalty"]})."""
    return all(meta.get(f) in (v if isinstance(v, list) else [v]) for f, v in (where or {}).items())


def _chroma_where(where: dict | None) -> dict | None:
    clauses = [{f: {"$in": v if isinstance(v, list) else [v]}} for f, v in (where or {}).items()]
    return None if not clauses else clauses[0] if len(clauses) == 1 else {"$and": clauses}


class VectorStore:
    def __init__(self, root: Path, domain: str):
        self.dir = Path(root) / domain
        self.dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.dir), settings=Settings(anonymized_telemetry=False))
        self.col = self.client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        self.model = (self.col.metadata or {}).get("model")
        self.records: list[dict] = self._records()  # id, hash, mhash, text, meta, in chunk order (BM25 reads these)

    def _records(self, with_vectors: bool = False) -> list[dict]:
        got = self.col.get(include=["documents", "metadatas"] + (["embeddings"] if with_vectors else []))
        rows = []
        for n, i in enumerate(got["ids"]):
            m = dict(got["metadatas"][n])
            row = {"id": i, "hash": m.pop("_hash"), "mhash": m.pop("_mhash"), "ord": m.pop("_ord"), "text": got["documents"][n], "meta": m}
            if with_vectors:
                row["vec"] = np.asarray(got["embeddings"][n], dtype="float32")
            rows.append(row)
        return sorted(rows, key=lambda r: r["ord"])

    def diff(self, chunks: list[Chunk], model: str | None = None) -> dict:
        """What would sync change? Compares source chunks with the index; embeds nothing, writes nothing. This is the
        index-drift check. `updated` = text changed (needs re-embedding); `meta_changed` = only metadata changed (stored
        metadata is stale, which can silently break filters such as 'current policy version', but no re-embedding is needed)."""
        reuse = model is None or self.model == model
        old = {r["id"]: (r["hash"], r.get("mhash")) for r in self.records} if reuse else {}
        new = {c.id: (c.content_hash, meta_hash(c.meta)) for c in chunks}
        return {"added": sorted(i for i in new if i not in old),
                "updated": sorted(i for i in new if i in old and old[i][0] != new[i][0]),
                "meta_changed": sorted(i for i in new if i in old and old[i][0] == new[i][0] and old[i][1] != new[i][1]),
                "removed": sorted(i for i in old if i not in new), "unchanged": sum(1 for i in new if old.get(i) == new[i])}

    def sync(self, chunks: list[Chunk], embedder: Embedder) -> dict:
        """Make the store match `chunks`. Embeds only chunks whose content hash is new."""
        reuse = bool(self.records) and self.model == embedder.name  # never reuse another model's vectors
        if not reuse and self.records:
            self.client.delete_collection(COLLECTION)
            self.col = self.client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        old = {r["id"]: r for r in self._records(with_vectors=True)} if reuse else {}

        to_embed = [c for c in chunks if c.id not in old or old[c.id]["hash"] != c.content_hash]
        too_long = [(c.id, n) for c in to_embed if (n := embedder.n_tokens(c.embed_text)) > embedder.max_tokens]
        if too_long:  # the model would silently cut these, so part of the chunk would never be searchable
            raise ValueError(f"embedded text exceeds the {embedder.max_tokens}-token window of {embedder.name}: {too_long}. "
                             "Shorten it, split the chunk, or embed a compact form (Chunk.embed_text).")
        fresh = dict(zip((c.id for c in to_embed), embedder.embed_passages([c.embed_text for c in to_embed])))

        new_ids = {c.id for c in chunks}
        if gone := [i for i in old if i not in new_ids]:
            self.col.delete(ids=gone)
        if chunks:
            self.col.upsert(
                ids=[c.id for c in chunks],
                embeddings=[(fresh[c.id] if c.id in fresh else old[c.id]["vec"]).tolist() for c in chunks],
                documents=[c.text for c in chunks],
                metadatas=[{**{k: v for k, v in c.meta.items() if v is not None},  # Chroma has no null metadata
                            "_hash": c.content_hash, "_mhash": meta_hash(c.meta), "_ord": n} for n, c in enumerate(chunks)])
        self.col.modify(metadata={"model": embedder.name})
        old_meta = {i: r.get("mhash") for i, r in old.items()}
        self.model, self.records = embedder.name, self._records()

        return {
            "added": sum(c.id not in old for c in to_embed),
            "updated": sum(c.id in old for c in to_embed),
            "removed": len(gone),
            "meta_refreshed": sum(1 for c in chunks if c.id in old and c.id not in fresh and old_meta.get(c.id) != meta_hash(c.meta)),
            "unchanged": len(chunks) - len(to_embed),
        }

    def search(self, query_vec: np.ndarray, k: int = 5, where: dict | None = None) -> list[Hit]:
        """Top-k by cosine similarity. `where` filters on metadata inside the search (see matches_filter)."""
        n = self.col.count()
        if not n:
            return []
        got = self.col.query(query_embeddings=query_vec.tolist(), n_results=min(k, n), where=_chroma_where(where),
                             include=["documents", "metadatas", "distances"])
        return [Hit(i, 1.0 - d, doc, {k_: v for k_, v in m.items() if not k_.startswith("_")})
                for i, d, doc, m in zip(got["ids"][0], got["distances"][0], got["documents"][0], got["metadatas"][0])]
