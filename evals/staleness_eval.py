"""Staleness and index-drift eval: when a policy or the catalog changes, does the system notice, update only what changed,
and answer from the new facts?

    python -m evals.staleness_eval             # about a dozen model calls
    python -m evals.staleness_eval --no-model  # index and retrieval checks only, free

Everything runs in a scratch copy of the data and index; the real ones are never touched.

Scenario 1, a seasonal update lands on two policies and the catalog at once:
  returns window 30 -> 60 days (touches two sections), two-day shipping $15 -> $20, Moto Legging $128 -> $138.
  D  drift detection finds exactly the chunks whose text changed, and nothing when nothing changed
  I  an incremental update re-embeds exactly those chunks (counted at the embedder), not the whole index
  R  the new text is retrievable and the old text is not
  A  answers use the new facts, including ones that flip a customer's eligibility; an unrelated answer does not change
Scenario 2, two versions of a policy exist (Q1 in force, Q2 effective April 1):
  V  only the version in force on the as-of date is searchable, never both, and the switch re-embeds only what differs
  F  defence in depth: even if archived chunks are indexed by mistake, retrieval excludes them
"""
import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

from src.context.catalog import Catalog
from src.context.customers import CustomerStore
from src.ingestion.embeddings import Embedder
from src.generation.generate import Assistant
from src.ingestion.ingest import load_all, searchable
from src.retrieval.retrieve import Retriever
from src.ingestion.store import VectorStore

from .common import AS_OF, HELDOUT_PATH, QUESTIONS_PATH, ROOT, load_questions
from .generation_eval import hit

RES = ROOT / "evals" / "results"
RETURNS, SHIPPING = "returns_and_exchanges_policy.md", "shipping_policy.md"
WINDOW_EDITS = [("**within 30 days of the original purchase date**", "**within 60 days of the original purchase date**"),
                ("initiated **within 30 days of purchase**", "initiated **within 60 days of purchase**"),
                ("within the 30-day return window from the purchase date", "within the 60-day return window from the purchase date")]
SHIPPING_EDIT = ("- **Cost:** $15.00", "- **Cost:** $20.00")
# the old RETURN window wording. The defect-reporting rule ("outside the 30-day window", returns §7) is a different rule that
# legitimately stays at 30 days, so a bare "30-day window" must not count as stale.
OLD_WINDOW = ["30 days of the original", "30-day return window", "within 30 days of purchase"]

results: list[dict] = []
SCRATCH: list[Path] = []


def check(cid: str, desc: str, ok: bool, evidence: str = "") -> bool:
    results.append({"id": cid, "desc": desc, "ok": bool(ok), "evidence": evidence})
    print(f"  {'PASS' if ok else 'FAIL'} {cid:<5}{desc}" + (f"\n         {evidence}" if evidence and not ok else ""), flush=True)
    return ok


def scratch() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="stale_"))
    SCRATCH.append(tmp)
    shutil.copytree(ROOT / "data", tmp / "data")
    return tmp


def edit(path: Path, pairs: list[tuple[str, str]]) -> None:
    text = path.read_text()
    for old, new in pairs:
        assert old in text, f"the data changed: {old!r} is no longer in {path.name}"
        text = text.replace(old, new)
    path.write_text(text)


class CountingEmbedder:
    """Wraps the embedder and counts how many passages it is asked to embed: the honest measure of 'no full re-index'."""
    def __init__(self, inner: Embedder):
        self.inner, self.count = inner, 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def embed_passages(self, texts, *a, **k):
        self.count += len(texts)
        return self.inner.embed_passages(texts, *a, **k)


def build(data_dir: Path, index_dir: Path, emb: CountingEmbedder, as_of: str | None = None, include_archived: bool = False) -> dict:
    emb.count = 0
    chunks = load_all(data_dir, as_of)
    chunks = chunks if include_archived else searchable(chunks)
    stats = {d: VectorStore(index_dir, d).sync([c for c in chunks if c.domain == d], emb) for d in ("policy", "product")}
    return {"stats": stats, "embedded": emb.count}


def drift(data_dir: Path, index_dir: Path, as_of: str | None = None) -> dict:
    chunks = searchable(load_all(data_dir, as_of))
    out = {"updated": [], "meta_changed": [], "added": [], "removed": []}
    for d in ("policy", "product"):
        r = VectorStore(index_dir, d).diff([c for c in chunks if c.domain == d], "bge-small")
        for k in out:
            out[k] += r[k]
    return out


def questions() -> dict:
    return {q["id"]: q for q in load_questions(QUESTIONS_PATH) + load_questions(HELDOUT_PATH)}


def make_assistant(tmp: Path, index_dir: Path) -> Assistant:
    return Assistant(retriever=Retriever(index_dir), customers=CustomerStore(tmp / "data" / "customers" / "customer_order_history.json"),
                     catalog=Catalog(tmp / "data" / "products" / "alo_product_catalog.json"), as_of=AS_OF, k=6, mode="hybrid",
                     product_mode="tools", cache_path=str(tmp / "gen_cache.json"))  # a scratch cache: every answer is a real call


def ask(a: Assistant, qs: dict, qid: str | None, text: str | None = None, customer: str | None = None) -> str:
    q = qs.get(qid) if qid else None
    return a.answer(q["query"] if q else text, q["customer_id"] if q else customer).answer


def scenario_update(emb: CountingEmbedder, use_model: bool) -> None:
    print("\nSCENARIO 1: a seasonal update lands on two policies and the catalog")
    tmp, qs = scratch(), questions()
    index = tmp / "index"
    first = build(tmp / "data", index, emb)
    check("D0", "no drift right after a build", not any(drift(tmp / "data", index).values()), str(drift(tmp / "data", index)))

    before = {}
    if use_model:
        a = make_assistant(tmp, index)
        before = {"window": ask(a, qs, "L1"), "mat": ask(a, qs, "C8"), "bottle": ask(a, qs, "H22"), "control": ask(a, qs, "P2"),
                  "ship": ask(a, qs, "H11"), "moto": ask(a, qs, None, "How much is the Moto Legging?")}
        check("A0", "baseline answers use the old facts (30 days, $15, $128)",
              hit(before["window"], "30") and hit(before["ship"], "$15|15") and hit(before["moto"], "128"), json.dumps({k: before[k][:80] for k in ("window", "ship", "moto")}))

    # oracle: which chunks should change, from the chunk TEXT itself (independent of the sync machinery)
    old_chunks = {c.id: c for c in searchable(load_all(tmp / "data"))}
    swaps = WINDOW_EDITS + [SHIPPING_EDIT]
    def swapped(t: str) -> str:
        for old, new in swaps:
            t = t.replace(old, new)
        return t
    expected = sorted(i for i, c in old_chunks.items() if c.domain == "policy" and swapped(c.text) != c.text) + ["W3301R-BLK"]

    edit(tmp / "data" / "policies" / RETURNS, WINDOW_EDITS)
    edit(tmp / "data" / "policies" / SHIPPING, [SHIPPING_EDIT])
    cat_path = tmp / "data" / "products" / "alo_product_catalog.json"
    cat = json.loads(cat_path.read_text())
    cat["last_updated"] = "2025-03-15"  # a real catalog update bumps this, which changes every product's metadata
    next(p for p in cat["products"] if p["sku"] == "W3301R-BLK")["price_usd"] = 138
    cat_path.write_text(json.dumps(cat, indent=2, ensure_ascii=False))

    d = drift(tmp / "data", index)
    check("D1", "drift check finds exactly the chunks whose text changed", sorted(d["updated"]) == sorted(expected),
          f"found {sorted(d['updated'])}, expected {sorted(expected)}")
    check("D2", "a catalog date bump shows up as metadata-only drift, not as text changes",
          len(d["meta_changed"]) >= 29 and not set(d["meta_changed"]) & set(d["updated"]), f"{len(d['meta_changed'])} metadata-only")
    check("D3", "nothing is added or removed by a text edit", not d["added"] and not d["removed"], str(d))

    second = build(tmp / "data", index, emb)
    check("I1", f"incremental update embeds exactly the {len(expected)} changed chunks (of {sum(first['stats'][x]['added'] for x in first['stats'])} in the index)",
          second["embedded"] == len(expected), f"embedded {second['embedded']}, expected {len(expected)}")
    check("I2", "the metadata-only chunks were refreshed without re-embedding",
          sum(s["meta_refreshed"] for s in second["stats"].values()) >= 29, str({k: v["meta_refreshed"] for k, v in second["stats"].items()}))
    check("I3", "after the update the drift check is clean", not any(drift(tmp / "data", index).values()), str(drift(tmp / "data", index)))

    r = Retriever(index)
    hits = r.retrieve("What is your return window?", 6, ["policy"], mode="hybrid")
    texts = " ".join(h.text for h in hits)
    check("R1", "the new return window is retrievable", "within 60 days" in texts, f"retrieved {[h.id for h in hits]}")
    check("R2", "the old return window is no longer retrievable", not any(p in texts for p in OLD_WINDOW), f"retrieved {[h.id for h in hits]}")
    ship = " ".join(h.text for h in r.retrieve("How much is two-day shipping?", 6, ["policy"], mode="hybrid"))
    check("R3", "the new shipping price is retrievable and the old one is gone", "$20.00" in ship and "$15.00" not in ship)

    if not use_model:
        return
    a = make_assistant(tmp, index)
    after = {"window": ask(a, qs, "L1"), "mat": ask(a, qs, "C8"), "bottle": ask(a, qs, "H22"), "control": ask(a, qs, "P2"),
             "ship": ask(a, qs, "H11"), "moto": ask(a, qs, None, "How much is the Moto Legging?")}
    check("A1", "the return window answer uses 60 days and none of the old wording",
          hit(after["window"], "60") and not any(p in after["window"] for p in OLD_WINDOW), after["window"][:160])
    check("A2", "a customer whose eligibility flips (Warrior Mat, 45 days old) is now told the window runs to March 16",
          hit(after["mat"], "March 16|Mar 16|2025-03-16|03/16"), after["mat"][:200])
    check("A3", "a second flipped customer (water bottle, 38 days old) is now told the window runs to March 23",
          hit(after["bottle"], "March 23|Mar 23|2025-03-23|03/23"), after["bottle"][:200])
    check("A4", "the two-day shipping answer uses $20 and not $15", hit(after["ship"], "$20|20") and "$15" not in after["ship"], after["ship"][:160])
    check("A5", "the catalog tool returns the new price immediately, with no re-embedding (138, not 128)",
          hit(after["moto"], "138") and "128" not in after["moto"], after["moto"][:160])
    check("A6", "an unrelated answer is unchanged (Warrior Mat size)", hit(after["control"], "68") and hit(after["control"], "24"), after["control"][:160])


def scenario_versions(emb: CountingEmbedder, use_model: bool) -> None:
    print("\nSCENARIO 2: two versions of the returns policy (Q1 in force, Q2 effective April 1)")
    tmp, qs = scratch(), questions()
    v1 = (tmp / "data" / "policies" / RETURNS).read_text()
    v2 = v1.replace("**Document Version:** 2025-Q1", "**Document Version:** 2025-Q2").replace("**Effective Date:** January 1, 2025", "**Effective Date:** April 1, 2025")
    for old, new in WINDOW_EDITS:
        v2 = v2.replace(old, new)
    (tmp / "data" / "policies" / "returns_and_exchanges_policy_2025q2.md").write_text(v2)
    index = tmp / "index"

    chunks = load_all(tmp / "data", "2025-03-01")
    archived = [c for c in chunks if c.meta.get("archived")]
    check("V0", "the not-yet-effective Q2 version is archived under versioned ids, Q1 keeps the stable ids",
          archived and all(c.id.startswith("returns@2025-Q2§") for c in archived) and any(c.id == "returns§1" for c in chunks),
          f"{len(archived)} archived, e.g. {archived[0].id if archived else None}")
    before = build(tmp / "data", index, emb, "2025-03-01")
    texts = " ".join(h.text for h in Retriever(index).retrieve("What is your return window?", 6, ["policy"], mode="hybrid"))
    check("V1", "as of 2025-03-01 only the Q1 rule (30 days) is searchable, never both", "within 30 days" in texts and "60" not in texts.replace("2025", ""),
          texts[:120])
    if use_model:
        ans = ask(make_assistant(tmp, index), qs, "L1")
        check("V2", "the answer as of 2025-03-01 says 30 days", hit(ans, "30") and "60" not in ans, ans[:160])

    changed = sorted(i for i, c in {c.id: c for c in searchable(load_all(tmp / "data", "2025-04-15"))}.items()
                     if i in {x.id: x for x in searchable(chunks)} and c.text != {x.id: x for x in searchable(chunks)}[i].text)
    after = build(tmp / "data", index, emb, "2025-04-15")
    check("V3", f"flipping the as-of date to 2025-04-15 re-embeds only the {len(changed)} chunks whose text differs between versions",
          after["embedded"] == len(changed), f"embedded {after['embedded']}, expected {changed}")
    texts = " ".join(h.text for h in Retriever(index).retrieve("What is your return window?", 6, ["policy"], mode="hybrid"))
    check("V4", "as of 2025-04-15 only the Q2 rule (60 days) is searchable", "within 60 days" in texts and "within 30 days" not in texts, texts[:120])
    if use_model:
        ans = ask(make_assistant(tmp, index), qs, "L1")
        check("V5", "the answer as of 2025-04-15 says 60 days", hit(ans, "60") and not any(p in ans for p in OLD_WINDOW), ans[:160])

    # defence in depth: a misconfigured build that indexes archived chunks must still never surface them
    bad = tmp / "bad_index"
    build(tmp / "data", bad, emb, "2025-04-15", include_archived=True)
    r = Retriever(bad)
    shown = r.retrieve("What is your return window?", 10, ["policy"], mode="hybrid")
    everything = r.retrieve("What is your return window?", 10, ["policy"], mode="hybrid", include_archived=True)
    check("F1", "even with archived chunks wrongly indexed, retrieval returns no archived version",
          shown and not any(h.meta.get("archived") for h in shown) and any(h.meta.get("archived") for h in everything),
          f"shown {[h.id for h in shown][:5]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-model", action="store_true", help="skip the checks that need model calls")
    args = ap.parse_args()
    emb = CountingEmbedder(Embedder())
    try:
        scenario_update(emb, not args.no_model)
        scenario_versions(emb, not args.no_model)
    finally:
        for d in SCRATCH:
            shutil.rmtree(d, ignore_errors=True)
    passed = sum(r["ok"] for r in results)
    print(f"\n{passed}/{len(results)} checks passed")
    (RES / "staleness.json").write_text(json.dumps({"passed": passed, "total": len(results), "checks": results}, indent=1))
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
