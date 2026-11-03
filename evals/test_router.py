"""Router accuracy on the labelled questions, dev and held-out.   python -m evals.test_router

A missed domain drops evidence (serious); an extra domain only costs a lookup (minor), except for out-of-scope
questions, where any routing at all is a failure. Also reports per-domain precision/recall and the router's own
latency and tokens (from the original call). Writes evals/results/router.json.
"""
import json
import statistics

from src.router.router import route

from evals.common import HELDOUT_PATH, QUESTIONS_PATH, ROOT, load_questions
from evals.pricing.pricing import cost
from evals.stats import pct

DOMAINS = ("customer", "policy", "product")


def evaluate(questions: list[dict]) -> dict:
    rows, failures = [], []
    for q in questions:
        r = route(q["query"])
        got, exp = set(r.domains), set(q["expected_domains"])
        rows.append({"id": q["id"], "got": sorted(got), "expected": sorted(exp), "missed": sorted(exp - got), "extra": sorted(got - exp),
                     "fallback": r.fallback, "meta": r.meta})
        if exp - got or (not exp and got):
            failures.append(rows[-1])
    n = len(rows)
    per = {}
    for d in DOMAINS:
        tp = sum(d in x["got"] and d in x["expected"] for x in rows)
        fp = sum(d in x["got"] and d not in x["expected"] for x in rows)
        fn = sum(d not in x["got"] and d in x["expected"] for x in rows)
        per[d] = {"precision": round(tp / (tp + fp), 3) if tp + fp else None, "recall": round(tp / (tp + fn), 3) if tp + fn else None}
    oos = [x for x in rows if not x["expected"]]
    metas = [x["meta"] for x in rows if x["meta"]]
    return {"n": n, "pass_rate": round(1 - len(failures) / n, 3), "exact_match": round(sum(not x["missed"] and not x["extra"] for x in rows) / n, 3),
            "missed_domains": sum(len(x["missed"]) for x in rows), "extra_domains": sum(len(x["extra"]) for x in rows),
            "out_of_scope": {"n": len(oos), "correct": sum(not x["got"] for x in oos)}, "per_domain": per, "fallbacks": sum(x["fallback"] for x in rows),
            "latency_ms": {"p50": round(pct([m["ms"] for m in metas], 50), 1), "p95": round(pct([m["ms"] for m in metas], 95), 1), "n": len(metas)} if metas else None,
            "tokens_per_call": {"in": round(statistics.mean(m["in"] for m in metas)), "out": round(statistics.mean(m["out"] for m in metas))} if metas else None,
            "failures": failures, "rows": rows}


def main() -> None:
    result = {name: evaluate(load_questions(path)) for name, path in (("dev", QUESTIONS_PATH), ("heldout", HELDOUT_PATH))}
    for name, r in result.items():
        print(f"{name:<8} n={r['n']} pass {r['pass_rate']:.2f} | exact match {r['exact_match']:.2f} | missed domains {r['missed_domains']} | "
              f"extra domains {r['extra_domains']} | out-of-scope handled {r['out_of_scope']['correct']}/{r['out_of_scope']['n']} | fallbacks {r['fallbacks']}")
        print("         " + "  ".join(f"{d}: P={v['precision']} R={v['recall']}" for d, v in r["per_domain"].items()))
        if r["latency_ms"]:
            print(f"         latency p50 {r['latency_ms']['p50']:.0f} ms / p95 {r['latency_ms']['p95']:.0f} ms (n={r['latency_ms']['n']}) | tokens/call {r['tokens_per_call']}")
        for f in r["failures"]:
            print(f"         FAIL {f['id']}: got {f['got']} expected {f['expected']}")
    (ROOT / "evals" / "results" / "router.json").write_text(json.dumps(result, indent=1))
    print("saved evals/results/router.json")


if __name__ == "__main__":
    main()
