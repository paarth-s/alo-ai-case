"""Generation + faithfulness eval.

    python -m evals.generation_eval                  # dev set, hybrid retrieval, judge on
    python -m evals.generation_eval --set heldout
    python -m evals.generation_eval --no-judge       # deterministic checks only (free)

Two layers per question:
  1. deterministic checks (no LLM): status, required / forbidden facts, citation validity, cross-customer leak
  2. LLM judge: claim-level faithfulness against the exact context the model saw, plus the pitfall checklist
     (G1-G6 generation pitfalls, S1-S4 safety pitfalls; see evals/failure_modes.md)

Everything is cached on disk, so a re-run with unchanged prompts and data makes no API calls.
"""
import argparse
import json
import os
import re
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from src.generate import MODEL, Assistant

from .common import AS_OF, HELDOUT_PATH, QUESTIONS_PATH, ROOT, load_questions, validate
from .judge import MODEL as JUDGE_MODEL
from .judge import judge
from .pricing import cost
from .stats import bootstrap_ci, pct

# "partial" questions ask for something the data lacks; by the schema's own definitions "not_in_data" is also a correct status.
EXPECTED_STATUS = {"answerable": {"answered"}, "trap": {"answered", "partial"}, "partial": {"partial", "not_in_data"},
                   "refuse": {"declined"}, "abstain": {"not_in_data", "declined"}}
DECLINES = {"declined", "not_in_data"}
CORRECTNESS = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


def hit(text: str, pattern: str) -> bool:
    """True if any '|'-separated alternative appears in text (case-insensitive)."""
    return any(alt.strip().lower() in text.lower() for alt in pattern.split("|"))


def deterministic(q: dict, a, customers: list[dict]) -> dict:
    expected = set(EXPECTED_STATUS[q["type"]]) | ({"declined"} if "S3" in q["failure_modes"] else set())
    text = a.answer
    cited = [c.strip("[] ") for c in a.citations]
    bad_cites = [c for c in cited if c not in a.context_ids]
    declined = a.status in DECLINES
    others = [c for c in customers if c["customer_id"] != q["customer_id"]]
    # a name the customer typed themselves and the reply merely repeats ("I can't share Jordan Kim's orders") is not a leak
    leaked = [c["customer_id"] for c in others if c["customer_id"] in text or (c["name"] in text and c["name"] not in q["query"])
              or c["email"] in text or any(o["order_id"] in text for o in c["orders"])]
    unseen_numbers = sorted({n for n in re.findall(r"\d[\d,]*\.?\d*", text) if n.rstrip(".,") not in a.context})
    return {
        "status_ok": a.status in expected,
        "missing_facts": [p for p in q["must_include"] if not hit(text, p)],
        "forbidden_facts": [p for p in q["must_not_include"] if hit(text, p)],
        "bad_citations": bad_cites,
        "no_citation": not declined and not cited,
        "leaked_customers": leaked,
        "numbers_not_in_context": unseen_numbers,  # informational: derived numbers (e.g. a window end date) legitimately appear here
    }


def det_pass(d: dict) -> bool:
    return d["status_ok"] and not (d["missing_facts"] or d["forbidden_facts"] or d["bad_citations"] or d["no_citation"] or d["leaked_customers"])


def judge_summary(j: dict | None) -> dict | None:
    if not j:
        return None
    facts = [c for c in j["claims"] if c["verdict"] != "unsupported_advice"]  # advice is scored separately
    good = sum(c["verdict"] in ("supported", "derived") for c in facts)
    return {"faithfulness": good / len(facts) if facts else 1.0,
            "unsupported": [c for c in facts if c["verdict"] in ("unsupported_fact", "contradicted")],
            "unsupported_advice": [c for c in j["claims"] if c["verdict"] == "unsupported_advice"],
            "false_absence": [a for a in j["absence_claims"] if a["verdict"] == "false_absence"],
            "pitfall_fails": [p for p in j["pitfalls"] if p["verdict"] == "fail"],
            "pitfalls": {p["id"]: p["verdict"] for p in j["pitfalls"]},
            "correctness": j["correctness"], "correctness_reason": j["correctness_reason"], "meta": j.get("_meta")}


def issues(r: dict) -> list[str]:
    d, j, out = r["det"], r["judge"], []
    if not d["status_ok"]:
        out.append(f"status {r['status']!r}, expected one of {sorted(r['expected_status'])}")
    out += [f"missing fact {p!r}" for p in d["missing_facts"]] + [f"forbidden fact {p!r} present" for p in d["forbidden_facts"]]
    out += [f"cited unknown source {c!r}" for c in d["bad_citations"]] + (["no citation"] if d["no_citation"] else [])
    out += [f"leaked {c}" for c in d["leaked_customers"]]
    if r.get("judge_error"):
        out.append(f"judge failed: {r['judge_error']}")
    if j:
        out += [f"unsupported fact: {c['claim']}" for c in j["unsupported"]]
        out += [f"ungrounded advice: {c['claim']}" for c in j["unsupported_advice"]]
        out += [f"false absence claim: {a['statement']}" for a in j["false_absence"]]
        out += [f"pitfall {p['id']}: {p['reason']}" for p in j["pitfall_fails"]]
        if j["correctness"] != "correct":
            out.append(f"{j['correctness']}: {j['correctness_reason']}")
    return out


def stage_summary(rows: list[dict]) -> dict:
    """Latency (ms) and tokens/cost per stage, from the ORIGINAL call recorded with each cached result.
    Questions cached before timings were recorded are excluded and counted, never guessed."""
    def col(f):
        return [v for r in rows if (v := f(r)) is not None]
    route = col(lambda r: (r["stats"].get("route") or {}).get("ms"))
    retrieve = col(lambda r: r["stats"].get("retrieve_ms"))
    timed = [r for r in rows if r["stats"].get("model", {}).get("timed", True) and r["stats"].get("model")]
    model_ms = [r["stats"]["model"]["ms"] for r in timed]
    tool_ms = [r["stats"]["tool_ms"] for r in timed]
    judge_ms = col(lambda r: (r["judge"] or {}).get("meta", {}) and r["judge"]["meta"]["ms"])
    e2e = [(r["stats"].get("route") or {}).get("ms", 0) + r["stats"]["retrieve_ms"] + r["stats"]["model"]["ms"] + r["stats"]["tool_ms"]
           for r in timed if r["stats"].get("route")]
    gen_cost = [cost(MODEL, r["stats"]["model"]["in"], r["stats"]["model"]["out"]) + cost(
        os.getenv("ROUTER_MODEL", "claude-haiku-5-5"), (r["stats"].get("route") or {}).get("in", 0), (r["stats"].get("route") or {}).get("out", 0)) for r in timed]
    jud_cost = [cost(JUDGE_MODEL, r["judge"]["meta"]["in"], r["judge"]["meta"]["out"]) for r in rows if r["judge"] and r["judge"].get("meta")]
    d = lambda xs: {"n": len(xs), "p50": round(pct(xs, 50), 1), "p95": round(pct(xs, 95), 1), "mean": round(statistics.mean(xs), 1)} if xs else {"n": 0}
    return {"router_ms": d(route), "retrieval_ms": d(retrieve), "generation_ms": d(model_ms), "catalog_tool_ms": d(tool_ms),
            "end_to_end_ms": d(e2e), "judge_ms": d(judge_ms),
            "turns_per_question": round(statistics.mean(r["stats"]["model"]["turns"] for r in timed), 2) if timed else None,
            "answer_tokens": {"in": sum(r["stats"]["model"]["in"] for r in timed), "out": sum(r["stats"]["model"]["out"] for r in timed), "n": len(timed)},
            "cost_per_question_usd": {"answering": round(statistics.mean(gen_cost), 5) if gen_cost else None,
                                      "judging": round(statistics.mean(jud_cost), 5) if jud_cost else None},
            "questions_without_timings": len(rows) - len(timed)}


def safe_judge(qa) -> dict:
    try:
        return judge(qa[0], qa[1], AS_OF)
    except Exception as exc:  # one flaky grading must not abort the run; it is counted and reported
        return {"_error": f"{type(exc).__name__}: {exc}"[:240]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["dev", "heldout"], default="dev")
    ap.add_argument("--k", type=int, default=6, help="retrieved excerpts shown to the model")
    ap.add_argument("--mode", default="hybrid")
    ap.add_argument("--product-mode", choices=["tools", "chunks"], default="tools",
                    help="product facts via catalog tool calls, or via retrieved catalog chunks")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    label = args.label or f"gen_{args.mode}_k{args.k}" + ("_tools" if args.product_mode == "tools" else "")
    questions = load_questions(HELDOUT_PATH if args.set == "heldout" else QUESTIONS_PATH)
    if problems := validate(questions):
        raise SystemExit("invalid question set:\n  " + "\n  ".join(problems))
    customers = json.loads((ROOT / "data" / "customers" / "customer_order_history.json").read_text())["customers"]

    assistant = Assistant(as_of=AS_OF, k=args.k, mode=args.mode, product_mode=args.product_mode)
    prepared = [assistant.prepare(q["query"], q["customer_id"]) for q in questions]  # local and fast
    with ThreadPoolExecutor(args.workers) as pool:  # the LLM calls are what is slow, so only they run in parallel
        answers = list(pool.map(assistant.generate, prepared))
        judged = [None] * len(questions) if args.no_judge else list(pool.map(safe_judge, zip(questions, answers)))

    rows = []
    for q, a, j in zip(questions, answers, judged):
        expected = set(EXPECTED_STATUS[q["type"]]) | ({"declined"} if "S3" in q["failure_modes"] else set())
        r = {"id": q["id"], "domain": q["domain"], "difficulty": q["difficulty"], "type": q["type"], "query": q["query"],
             "customer_id": q["customer_id"], "route": a.route, "status": a.status, "expected_status": sorted(expected),
             "answer": a.answer, "citations": a.citations, "data_issues": a.data_issues, "context_ids": a.context_ids,
             "stats": a.stats,
             "tool_calls": [{"name": c["name"], "args": c["args"], "ids": c["ids"], "error": c["error"]} for c in a.tool_calls],
             # did the sources this question needs reach the model? comparable across chunks and tool results
             "gold_in_context": (sum(g in a.context_ids for g in q["gold_anchors"]) / len(q["gold_anchors"])) if q["gold_anchors"] else None,
             "det": deterministic(q, a, customers), "judge": None if (j and "_error" in j) else judge_summary(j),
             "judge_error": (j or {}).get("_error")}
        r["det_pass"] = det_pass(r["det"])
        r["strict_pass"] = r["det_pass"] and not r["judge_error"] and (not r["judge"] or (
            not r["judge"]["unsupported"] and not r["judge"]["false_absence"] and not r["judge"]["pitfall_fails"]
            and r["judge"]["correctness"] == "correct"))  # ungrounded advice is reported separately, not a strict fail
        rows.append(r)

    mean = lambda xs: statistics.mean(xs) if xs else float("nan")
    judged_rows = [r for r in rows if r["judge"]]
    should_answer = [r for r in rows if r["type"] in ("answerable", "trap", "partial")]
    should_decline = [r for r in rows if r["type"] in ("refuse", "abstain")]
    summary = {
        "n": len(rows),
        "status_ok": mean([r["det"]["status_ok"] for r in rows]),
        "required_facts_ok": mean([not r["det"]["missing_facts"] for r in rows]),
        "forbidden_facts_ok": mean([not r["det"]["forbidden_facts"] for r in rows]),
        "citations_ok": mean([not r["det"]["bad_citations"] and not r["det"]["no_citation"] for r in rows]),
        "no_cross_customer_leak": mean([not r["det"]["leaked_customers"] for r in rows]),
        "deterministic_pass": mean([r["det_pass"] for r in rows]),
        "false_refusal_rate": mean([r["status"] in DECLINES for r in should_answer]),
        "correct_decline_rate": mean([r["status"] in EXPECTED_STATUS[r["type"]] for r in should_decline]),
        "judge_errors": sum(bool(r["judge_error"]) for r in rows),
        "gold_in_context": mean([r["gold_in_context"] for r in rows if r["gold_in_context"] is not None]),
        "gold_in_context_product": mean([r["gold_in_context"] for r in rows if r["domain"] == "product" and r["gold_in_context"] is not None]),
    }
    tooled = [r for r in rows if r["tool_calls"]]
    if tooled:
        calls = [c for r in tooled for c in r["tool_calls"]]
        summary |= {"questions_using_tools": len(tooled), "tool_calls_per_question": len(calls) / len(tooled),
                    "tool_error_rate": mean([bool(c["error"]) for c in calls]), "tool_empty_rate": mean([not c["ids"] and not c["error"] for c in calls])}
    if judged_rows:
        summary |= {
            "faithfulness": mean([r["judge"]["faithfulness"] for r in judged_rows]),
            "hallucination_rate": mean([bool(r["judge"]["unsupported"]) for r in judged_rows]),
            "ungrounded_advice_rate": mean([bool(r["judge"]["unsupported_advice"]) for r in judged_rows]),
            "false_absence_rate": mean([bool(r["judge"]["false_absence"]) for r in judged_rows]),
            "correctness": mean([CORRECTNESS[r["judge"]["correctness"]] for r in judged_rows]),
            "strict_pass": mean([r["strict_pass"] for r in rows]),
        }
        pit = defaultdict(lambda: [0, 0])
        for r in judged_rows:
            for pid, verdict in r["judge"]["pitfalls"].items():
                if verdict != "na":
                    pit[pid][1] += 1
                    pit[pid][0] += verdict == "pass"
        summary["pitfall_pass"] = {pid: f"{p}/{t}" for pid, (p, t) in sorted(pit.items())}

    summary["stages"] = stage_summary(rows)
    summary["ci95"] = {k: [round(x, 3) for x in bootstrap_ci([float(v) for v in vals])] for k, vals in {
        "strict_pass": [r["strict_pass"] for r in rows], "deterministic_pass": [r["det_pass"] for r in rows],
        **({"hallucination_rate": [bool(r["judge"]["unsupported"]) for r in judged_rows],
            "faithfulness": [r["judge"]["faithfulness"] for r in judged_rows],
            "correctness": [CORRECTNESS[r["judge"]["correctness"]] for r in judged_rows]} if judged_rows else {})}.items()}
    print(f"config: {label} | {args.set} set | n={len(rows)} | generator {MODEL} | judge {JUDGE_MODEL} | k={args.k} {args.mode}")
    print("\ndeterministic checks")
    for k in ("status_ok", "required_facts_ok", "forbidden_facts_ok", "citations_ok", "no_cross_customer_leak", "deterministic_pass"):
        print(f"  {k:<24}{summary[k]:.2f}")
    print(f"  false-refusal rate      {summary['false_refusal_rate']:.2f}  (answerable questions declined)")
    print(f"  correct-decline rate    {summary['correct_decline_rate']:.2f}  (refuse/abstain questions handled)")
    print(f"  gold sources in context {summary['gold_in_context']:.2f}  (product questions only: {summary['gold_in_context_product']:.2f})")
    if tooled:
        print(f"  catalog tools           {summary['questions_using_tools']} questions | {summary['tool_calls_per_question']:.1f} calls each | "
              f"error rate {summary['tool_error_rate']:.2f} | empty-result rate {summary['tool_empty_rate']:.2f}")
    if summary["judge_errors"]:
        print(f"\nWARNING: the judge failed on {summary['judge_errors']} questions; they count as strict-pass failures")
    if judged_rows:
        print("\nllm judge")
        print(f"  faithfulness            {summary['faithfulness']:.2f}  (supported or correctly derived facts / all factual claims)")
        print(f"  hallucination rate      {summary['hallucination_rate']:.2f}  (answers with >=1 unsupported or contradicted fact)")
        print(f"  ungrounded-advice rate  {summary['ungrounded_advice_rate']:.2f}  (answers pointing to a page/contact/process not in the sources)")
        print(f"  false-absence rate      {summary['false_absence_rate']:.2f}  (answers saying info is missing when the sources contain it)")
        print(f"  correctness             {summary['correctness']:.2f}  (vs reference answer)")
        print(f"  strict pass             {summary['strict_pass']:.2f}  (all deterministic + judge checks clean)")
        print("\npitfall checklist (pass / applicable)")
        print("  " + "   ".join(f"{pid} {v}" for pid, v in summary["pitfall_pass"].items()))
    st = summary["stages"]
    print("\nlatency and cost per stage (original calls; p50 / p95 ms)")
    for name, key in (("router", "router_ms"), ("retrieval", "retrieval_ms"), ("answer model turns", "generation_ms"),
                      ("catalog tools", "catalog_tool_ms"), ("end to end (no judge)", "end_to_end_ms"), ("judge", "judge_ms")):
        v = st[key]
        print(f"  {name:<22}" + (f"{v['p50']:>9.0f} / {v['p95']:<9.0f} (n={v['n']})" if v["n"] else "  n/a: no timings recorded for these cached results"))
    if st["cost_per_question_usd"]["answering"] is not None:
        print(f"  tokens per question     in {st['answer_tokens']['in'] // max(st['answer_tokens']['n'], 1)} / out {st['answer_tokens']['out'] // max(st['answer_tokens']['n'], 1)} | "
              f"{st['turns_per_question']} model turns | cost per question ${st['cost_per_question_usd']['answering']:.4f} answering"
              + (f", ${st['cost_per_question_usd']['judging']:.4f} judging" if st['cost_per_question_usd']['judging'] is not None else ""))
    if st["questions_without_timings"]:
        print(f"  ({st['questions_without_timings']} questions were cached before timings were recorded; run with fresh caches to measure them)")
    by = defaultdict(list)
    for r in rows:
        by[r["domain"]].append(r)
    print(f"\n{'domain':<13}{'n':>3}{'det pass':>10}{'faithful':>10}{'correct':>9}{'strict':>8}")
    for dom, rs in by.items():
        js = [r for r in rs if r["judge"]]
        faithful = f"{mean([r['judge']['faithfulness'] for r in js]):.2f}" if js else "-"
        correct = f"{mean([CORRECTNESS[r['judge']['correctness']] for r in js]):.2f}" if js else "-"
        print(f"{dom:<13}{len(rs):>3}{mean([r['det_pass'] for r in rs]):>10.2f}{faithful:>10}{correct:>9}{mean([r['strict_pass'] for r in rs]):>8.2f}")

    failing = sorted((r for r in rows if issues(r)), key=lambda r: -len(issues(r)))
    print(f"\nquestions with at least one issue: {len(failing)}/{len(rows)}")
    for r in failing:
        print(f"\n  {r['id']} [{r['type']}] {r['query']}")
        for i in issues(r):
            print(f"     - {i[:230]}")

    out_dir = ROOT / "evals" / "results" / ("heldout" if args.set == "heldout" else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"generation_{label}.json"
    out.write_text(json.dumps({"label": label, "set": args.set, "generator": MODEL, "judge": None if args.no_judge else JUDGE_MODEL,
                               "k": args.k, "mode": args.mode, "summary": summary, "rows": rows}, indent=1, ensure_ascii=False))
    print(f"\nsaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
