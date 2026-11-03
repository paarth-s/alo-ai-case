"""Build or refresh the vector indexes. Safe to re-run: unchanged chunks are not re-embedded.

    python -m src.ingestion.build_index                       # default embedder (bge-small) -> index/
    python -m src.ingestion.build_index --embedder bge-base   # another embedder -> index_bge-base/
    python -m src.ingestion.build_index --enrich              # add LLM-generated questions per chunk -> index_hyq/
    python -m src.ingestion.build_index --check               # drift check: list stale chunks, change nothing, exit 1 if the index is stale
    python -m src.ingestion.build_index --as-of 2025-04-15    # choose each policy's version in force on that date
"""
import argparse
import sys
import time
from pathlib import Path

from src.ingestion.embeddings import DEFAULT, SPECS, Embedder, index_dir_for
from src.ingestion.enrich import enrich
from src.ingestion.ingest import load_all, searchable
from src.ingestion.store import VectorStore

DOMAINS = ("policy", "product")  # customer data is a structured lookup, not a vector index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--embedder", choices=list(SPECS), default=DEFAULT)
    ap.add_argument("--enrich", action="store_true", help="index LLM-generated questions alongside each chunk")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--as-of", default=None, help="ISO date: index the policy versions in force on it (default: newest)")
    ap.add_argument("--check", action="store_true", help="report index drift against the source files and exit 1 if any")
    args = ap.parse_args()
    index_dir = args.index_dir or index_dir_for(args.embedder) + ("_hyq" if args.enrich else "")

    start = time.time()
    chunks = searchable(load_all(Path(args.data_dir), args.as_of))
    if args.check:
        drift = False
        for domain in DOMAINS:
            d = VectorStore(Path(index_dir), domain).diff([c for c in chunks if c.domain == domain], args.embedder)
            drift |= bool(d["added"] or d["updated"] or d["meta_changed"] or d["removed"])
            print(f"{domain:<8} text changed: {len(d['updated'])} {d['updated']} | metadata only: {len(d['meta_changed'])} {d['meta_changed']} | missing from index: {len(d['added'])} {d['added']} | "
                  f"no longer in source: {len(d['removed'])} {d['removed']} | up to date: {d['unchanged']}")
        print("INDEX IS STALE" if drift else "index matches the source files")
        sys.exit(1 if drift else 0)
    if args.enrich:
        chunks = enrich(chunks)
    embedder = Embedder(args.embedder)
    for domain in DOMAINS:
        stats = VectorStore(Path(index_dir), domain).sync([c for c in chunks if c.domain == domain], embedder)
        print(f"{domain:<8} {stats}")
    print(f"{args.embedder} -> {index_dir}/ done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
