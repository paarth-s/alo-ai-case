"""Score retrieval against the gold anchors in the question files.

    python -m evals.retrieval_eval                          # routed dense top-10: the naive baseline
    python -m evals.retrieval_eval --mode hybrid            # dense bm25 hybrid dense_rerank hybrid_rerank
    python -m evals.retrieval_eval --embedder bge-base      # another embedder (build its index first)
    python -m evals.retrieval_eval --index-dir index_hyq    # score a specific index (e.g. LLM-enriched chunks)
    python -m evals.retrieval_eval --set heldout            # the held-out questions (not used for tuning)
    python -m evals.retrieval_eval --routing oracle         # use the labelled domains, to isolate retrieval from routing

Questions with no vector gold (customer lookups, out-of-scope) are excluded from the metrics.
A "random" row shows what a random ranking of the searched chunks would score, so no metric is read in isolation.
"""
import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

from src.customers import CustomerStore
from src.embeddings import DEFAULT, SPECS, index_dir_for
from src.rerank import MODELS as RERANKERS
from src.retrieve import MODES, VECTOR_DOMAINS, Retriever
from src.router.router import route

from .common import HELDOUT_PATH, QUESTIONS_PATH, ROOT, load_questions, validate

SHOULD_ANSWER = {"answerable", "trap", "partial"}


def recall_at_k(ranked, gold, k):
    return len(gold & set(ranked[:k])) / len(gold)


def mrr(ranked, gold):
    return next((1 / (i + 1) for i, r in enumerate(ranked) if r in gold), 0.0)


def ndcg_at_k(ranked, gold, k):  # binary relevance
    dcg = sum(1 / math.log2(i + 2) for i, r in enumerate(ranked[:k]) if r in gold)
    return dcg / sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))


def score(ranked, gold, k):
    """R@1, R@5, R@10 (those that fit within k), plus MRR and NDCG over the top k."""
    recalls = {f"R@{n}": recall_at_k(ranked, gold, n) for n in (1, 5, 10) if n <= k}
    return recalls | {f"MRR@{k}": mrr(ranked[:k], gold), f"NDCG@{k}": ndcg_at_k(ranked, gold, k)}


def random_score(n, g, k):
    """Exact expected metrics if the n searched chunks (g of them gold) were ranked uniformly at random."""
    total = math.comb(n, g)  # P(first gold at rank r) = C(n-r, g-1) / C(n, g)
    exp_mrr = sum(math.comb(n - r, g - 1) / total / r for r in range(1, min(k, n - g + 1) + 1))
    dcg = sum((g / n) / math.log2(r + 1) for r in range(1, min(k, n) + 1))
    ideal = sum(1 / math.log2(r + 1) for r in range(1, min(g, k) + 1))
    return ({f"R@{m}": min(m, n) / n for m in (1, 5, 10) if m <= k}
            | {f"MRR@{k}": exp_mrr, f"NDCG@{k}": dcg / ideal})


def summarize(rows, key="metrics"):
    return {m: statistics.mean(r[key][m] for r in rows) for m in rows[0][key]}


def print_table(title, groups):
    metrics = next(iter(groups.values()))[1].keys()
    print(f"\n{title:<12}{'n':>3}  " + "".join(f"{m:>8}" for m in metrics))
    for name, (n, summary) in groups.items():
        print(f"{name:<12}{n:>3}  " + "".join(f"{v:>8.2f}" for v in summary.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--set", choices=["dev", "heldout"], default="dev")
    ap.add_argument("--mode", choices=MODES, default="dense")
    ap.add_argument("--routing", choices=["router", "oracle", "none"], default="router")
    ap.add_argument("--embedder", choices=list(SPECS), default=DEFAULT)
    ap.add_argument("--reranker", choices=list(RERANKERS), default="minilm")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--lex-weight", type=float, default=1.0, help="hybrid: weight of BM25 relative to dense")
    ap.add_argument("--rrf-k", type=int, default=60, help="hybrid: reciprocal rank fusion constant")
    ap.add_argument("--with-customer", action="store_true",
                    help="append the session customer's loyalty tier to the retrieval query")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    index_dir = args.index_dir or index_dir_for(args.embedder)
    tag = Path(index_dir).name.removeprefix("index").lstrip("_")
    label = args.label or "_".join(filter(None, [
        args.mode, tag, "rr-" + args.reranker if "rerank" in args.mode and args.reranker != "minilm" else "",
        f"top{args.k}", args.routing, "cust" if args.with_customer else "",
        f"lw{args.lex_weight:g}" if args.lex_weight != 1 else "", f"rk{args.rrf_k}" if args.rrf_k != 60 else ""]))
    results_dir = ROOT / "evals" / "results" / ("heldout" if args.set == "heldout" else "")
    customers = CustomerStore(ROOT / "data" / "customers" / "customer_order_history.json")

    questions = load_questions(HELDOUT_PATH if args.set == "heldout" else QUESTIONS_PATH)
    if problems := validate(questions):
        raise SystemExit("invalid question set:\n  " + "\n  ".join(problems))

    retriever, rows = Retriever(index_dir, args.reranker, args.embedder), []
    pool_sizes = {d: len(s.records) for d, s in retriever.stores.items()}
    for q in questions:
        domains = {"router": lambda: route(q["query"]).domains,
                   "oracle": lambda: q["expected_domains"],
                   "none": lambda: list(VECTOR_DOMAINS)}[args.routing]()
        text = q["query"]
        if args.with_customer and "customer" in domains and (c := customers.get(q["customer_id"])):
            text += f" (asked by a {c['loyalty_tier']} loyalty member)"  # many policy answers depend on tier
        opts = {"mode": args.mode, "lex_weight": args.lex_weight, "rrf_k": args.rrf_k}
        start = time.perf_counter()
        hits = retriever.retrieve(text, args.k, domains, **opts) if set(domains) & set(VECTOR_DOMAINS) else []
        ms = (time.perf_counter() - start) * 1000
        top1 = retriever.retrieve(text, 1, **opts)  # unrouted, for the score-gap check below
        gold, ranked = set(q["gold_anchors"]), [h.id for h in hits]
        pool = sum(pool_sizes[d] for d in domains if d in pool_sizes)
        rows.append({"id": q["id"], "domain": q["domain"], "difficulty": q["difficulty"], "type": q["type"],
                     "query": q["query"], "routed": domains, "gold": sorted(gold), "retrieved": ranked,
                     "scores": [round(h.score, 3) for h in hits], "top1_score": round(top1[0].score, 3),
                     "ms": round(ms, 1), "pool": pool,
                     "metrics": score(ranked, gold, args.k) if gold else None,
                     "random": random_score(pool, len(gold), args.k) if gold and pool >= len(gold) else None})

    scored = [r for r in rows if r["metrics"]]
    print(f"config: {label} | {args.set} set | scored {len(scored)}/{len(rows)} questions (rest have no vector gold)")

    groups = defaultdict(list)
    for r in scored:
        groups["overall"].append(r)
        groups[r["domain"]].append(r)
        groups[r["difficulty"]].append(r)
    order = ["overall", "product", "policy", "customer", "adversarial", "easy", "medium", "hard"]
    summary = {g: (len(groups[g]), summarize(groups[g])) for g in order if g in groups}
    with_random = [r for r in scored if r["random"]]
    summary["random"] = (len(with_random), summarize(with_random, "random"))
    summary = {g: summary[g] for g in ["overall", "random", *order[1:]] if g in summary}
    print_table("group", summary)
    ceiling = statistics.mean(1 / len(r["gold"]) for r in scored)  # R@1 can't exceed 1/|gold| when several chunks are needed
    hit1 = statistics.mean(float(bool(r["retrieved"]) and r["retrieved"][0] in r["gold"]) for r in scored)
    print(f"R@1 ceiling {ceiling:.2f} (questions need several chunks) | top-1 is a gold chunk (hit@1): {hit1:.2f}")

    misses = [r for r in scored if r["metrics"][f"R@{args.k}"] < 1]
    print(f"\nincomplete recall@{args.k}: {len(misses)}/{len(scored)} questions")
    for r in misses:
        print(f"  {r['id']:<4} routed={r['routed']} missing={sorted(set(r['gold']) - set(r['retrieved']))}")
        print(f"       got {r['retrieved']}")

    answer = [r["top1_score"] for r in rows if r["type"] in SHOULD_ANSWER]
    decline = [r["top1_score"] for r in rows if r["type"] not in SHOULD_ANSWER]
    if answer and decline:
        print(f"\ntop-1 score ({args.mode}), should-answer: mean {statistics.mean(answer):.2f} (min {min(answer):.2f}) | "
              f"should-decline: mean {statistics.mean(decline):.2f} (max {max(decline):.2f})")
    times = sorted(r["ms"] for r in rows)
    latency = {"p50": round(statistics.median(times), 1), "p95": round(times[min(len(times) - 1, math.ceil(0.95 * len(times)) - 1)], 1)}
    print(f"retrieval latency: p50 {latency['p50']:.0f} ms | p95 {latency['p95']:.0f} ms")

    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"retrieval_{label}.json"
    out.write_text(json.dumps({"label": label, "set": args.set, "k": args.k, "mode": args.mode, "routing": args.routing,
                               "embedder": args.embedder, "reranker": args.reranker, "index_dir": index_dir,
                               "lex_weight": args.lex_weight, "rrf_k": args.rrf_k, "latency_ms": latency,
                               "r1_ceiling": round(ceiling, 3), "hit_at_1": round(hit1, 3),
                               "summary": {g: {"n": n, **s} for g, (n, s) in summary.items()}, "rows": rows}, indent=1))
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
