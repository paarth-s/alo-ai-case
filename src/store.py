"""Persistent per-domain vector store on FAISS.

FAISS only does nearest-neighbour search, so this wrapper adds what a vector DB would give us:
metadata filtering, on-disk persistence, and incremental sync (only new or changed chunks are
re-embedded, so a policy edit or catalog refresh never forces a full re-index).

Exact search (IndexFlatIP) is deliberate: at 15k SKUs x 384 dims it is ~25 MB and a few ms per
query, and it has no recall loss to explain. An ANN index (HNSW/IVF) is only worth it past ~1M vectors.

Vectors are only reusable by the model that made them, so the store records the embedder's name and
re-embeds everything if it changes (a model upgrade is the one case that does need a full re-index).
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from .embeddings import Embedder
from .ingest import Chunk


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


class VectorStore:
    def __init__(self, root: Path, domain: str):
        self.dir = Path(root) / domain
        self.records: list[dict] = []  # parallel to FAISS row order: id, hash, text, meta
        self.index = None
        self.model = None
        if (self.dir / "faiss.index").exists():
            self.index = faiss.read_index(str(self.dir / "faiss.index"))
            self.records = json.loads((self.dir / "records.json").read_text())
            self.model = (self.dir / "model.txt").read_text() if (self.dir / "model.txt").exists() else None

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
        reuse = self.index is not None and self.model == embedder.name  # never reuse another model's vectors
        old = {r["id"]: (r["hash"], i) for i, r in enumerate(self.records)} if reuse else {}
        old_vecs = self.index.reconstruct_n(0, self.index.ntotal) if reuse and self.index.ntotal else None

        to_embed = [c for c in chunks if old.get(c.id, (None,))[0] != c.content_hash]
        too_long = [(c.id, n) for c in to_embed if (n := embedder.n_tokens(c.embed_text)) > embedder.max_tokens]
        if too_long:  # the model would silently cut these, so part of the chunk would never be searchable
            raise ValueError(f"embedded text exceeds the {embedder.max_tokens}-token window of {embedder.name}: {too_long}. "
                             "Shorten it, split the chunk, or embed a compact form (Chunk.embed_text).")
        fresh = dict(zip((c.id for c in to_embed), embedder.embed_passages([c.embed_text for c in to_embed])))

        vecs = np.vstack([fresh[c.id] if c.id in fresh else old_vecs[old[c.id][1]] for c in chunks]) \
            if chunks else np.zeros((0, embedder.dim), "float32")
        old_meta = {r["id"]: r.get("mhash") for r in self.records} if reuse else {}
        self.records = [{"id": c.id, "hash": c.content_hash, "mhash": meta_hash(c.meta), "text": c.text, "meta": c.meta} for c in chunks]
        self.index, self.model = faiss.IndexFlatIP(embedder.dim), embedder.name
        self.index.add(vecs)

        self.dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.dir / "faiss.index"))
        (self.dir / "records.json").write_text(json.dumps(self.records, indent=1))
        (self.dir / "model.txt").write_text(embedder.name)

        new_ids = {c.id for c in chunks}
        return {
            "added": sum(c.id not in old for c in to_embed),
            "updated": sum(c.id in old for c in to_embed),
            "removed": sum(i not in new_ids for i in old),
            "meta_refreshed": sum(1 for c in chunks if c.id in old and c.id not in fresh and old_meta.get(c.id) != meta_hash(c.meta)),
            "unchanged": len(chunks) - len(to_embed),
        }

    def search(self, query_vec: np.ndarray, k: int = 5, where: dict | None = None) -> list[Hit]:
        """Top-k by cosine similarity. `where` filters on metadata (see matches_filter)."""
        if not self.records:
            return []
        params = None
        if where:
            allowed = [i for i, r in enumerate(self.records) if matches_filter(r["meta"], where)]
            if not allowed:
                return []
            params = faiss.SearchParameters()
            params.sel = faiss.IDSelectorBatch(np.array(allowed, dtype="int64"))  # filter inside the search
        scores, rows = self.index.search(query_vec, min(k, len(self.records)), params=params)
        return [Hit(self.records[r]["id"], float(s), self.records[r]["text"], self.records[r]["meta"])
                for s, r in zip(scores[0], rows[0]) if r != -1]
