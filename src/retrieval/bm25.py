"""Okapi BM25 over a store's records: exact-term matching that dense embeddings blur."""
import math
import re
from collections import Counter

from src.ingestion.store import Hit, matches_filter

STOPWORDS = {"a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are", "was", "be", "can", "i", "my",
             "me", "you", "your", "it", "this", "that", "with", "at", "by", "from", "as", "do", "does", "how", "what",
             "when", "which", "who", "will", "if", "there"}


def tokenize(text: str) -> list[str]:
    tokens = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]
    return [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t for t in tokens]  # leggings -> legging


class BM25:
    def __init__(self, records: list[dict], k1: float = 1.5, b: float = 0.75):
        self.records, self.k1, self.b = records, k1, b
        self.docs = [Counter(tokenize(r["text"])) for r in records]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.lengths), 1)
        df = Counter(t for d in self.docs for t in d)
        n = len(records)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 20, where: dict | None = None) -> list[Hit]:
        terms, scored = set(tokenize(query)), []
        for i, doc in enumerate(self.docs):
            if not matches_filter(self.records[i]["meta"], where):
                continue
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
            s = sum(self.idf[t] * doc[t] * (self.k1 + 1) / (doc[t] + norm) for t in terms if t in doc)
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [Hit(self.records[i]["id"], s, self.records[i]["text"], self.records[i]["meta"]) for s, i in scored[:k]]
