"""Failure attribution: for every question that does not pass cleanly, which stage is responsible?

    python -m evals.harness.attribution [--label gen_hybrid_k6_tools]

Causes are checked in pipeline order; a question can have several and the first is the primary one:
  router      a domain the question needs was never routed
  retrieval   a gold source never reached the model (policy chunks) or the catalog tools did not return it
  generation  routing was right and every needed source was present, yet the reply is unfaithful, incomplete or wrong
  eval        the reply looks acceptable but a check disagrees: only the status expectation failed, or a human /
              model label of the same reply says it is clean while the automated checks flagged it
Also ranks the worst failures by severity, with the stage trace needed to write them up.
"""
import argparse
import json
from collections import Counter

from evals.common import HELDOUT_PATH, QUESTIONS_PATH, ROOT, load_questions
from evals.suites.generation_eval import issues

CAL = ROOT / "evals" / "calibration"
CONTENT_FLAGS = ("unsupported", "unsupported_advice", "false_absence", "pitfall_fails")


def label_says_problem(lab: dict) -> bool:
    return (lab["unsupported_fact"] == "yes" or lab["ungrounded_advice"] == "yes" or lab["false_absence"] == "yes"
            or lab["correctness"] != "correct" or any(v == "fail" for v in lab["pitfalls"].values()))


def severity(r: dict) -> int:
    j, d = r["judge"] or {}, r["det"]
    if d["leaked_customers"] or any(p["id"].startswith("S") for p in j.get("pitfall_fails", [])):
        return 5  # a safety failure outranks everything
    if j.get("correctness") == "incorrect" or j.get("unsupported"):
        return 4
    if j.get("correctness") == "partially_correct" or j.get("false_absence") or j.get("pitfall_fails"):
        return 3
    if d["missing_facts"] or d["forbidden_facts"] or j.get("unsupported_advice"):
        return 2
    return 1


def attribute(rows: list[dict], questions: dict, labels: dict, frozen: dict) -> list[dict]:
    out = []
    for r in rows:
        if r["strict_pass"]:
            continue
        q = questions[(r["set"], r["id"])]
        causes, detail = [], {}
        missing_domains = sorted(set(q["expected_domains"]) - set(r["route"]))
        if missing_domains:
            causes.append("router"); detail["router"] = f"not routed to {missing_domains}"
        gold_missing = [g for g in q["gold_anchors"] if g not in r["context_ids"]]
        if gold_missing:
            tool_path = bool(r["tool_calls"])
            causes.append("catalog tools" if tool_path else "retrieval"); detail[causes[-1]] = f"gold not in context: {gold_missing}"
        j = r["judge"] or {}
        content_issue = any(j.get(k) for k in CONTENT_FLAGS) or r["det"]["missing_facts"] or r["det"]["forbidden_facts"] \
            or j.get("correctness") not in (None, "correct")
        if content_issue and not gold_missing and not missing_domains:
            causes.append("generation"); detail["generation"] = "all needed sources present; reply still flagged"
        # a labelled version of this very reply that a person or blind labeler called clean
        if r.get("judge_error"):
            causes.append("eval"); detail["eval"] = "the judge failed to grade this reply"
        key = (r["set"], r["id"])
        lab = labels.get(key)
        if lab and frozen.get(key) == r["answer"] and not label_says_problem(lab):
            causes.append("eval"); detail["eval"] = "a label of this exact reply says it is clean"
        elif not content_issue and not r["det"]["status_ok"]:
            causes.append("eval"); detail["eval"] = f"only the status expectation failed ({r['status']!r}, expected {r['expected_status']})"
        out.append({"set": r["set"], "id": r["id"], "query": r["query"], "type": r["type"], "primary": causes[0] if causes else "unclassified",
                    "causes": causes, "detail": detail, "severity": severity(r), "issues": issues(r), "route": r["route"],
                    "expected_domains": q["expected_domains"], "gold": q["gold_anchors"], "gold_missing": gold_missing, "status": r["status"],
                    "answer": r["answer"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="gen_hybrid_k6_tools")
    args = ap.parse_args()
    questions = {("dev", q["id"]): q for q in load_questions(QUESTIONS_PATH)} | {("heldout", q["id"]): q for q in load_questions(HELDOUT_PATH)}
    rows = []
    for sub, s in (("", "dev"), ("heldout", "heldout")):
        for r in json.loads((ROOT / "evals" / "results" / sub / f"generation_{args.label}.json").read_text())["rows"]:
            rows.append({**r, "set": s})
    n_total = len(rows)

    labels, frozen = {}, {}
    if (CAL / "items.json").exists() and (CAL / "combined_labels.json").exists():
        by_n = {i["n"]: i for i in json.loads((CAL / "combined_labels.json").read_text())["items"]}
        for it in json.loads((CAL / "items.json").read_text()):
            labels[(it["set"], it["id"])], frozen[(it["set"], it["id"])] = by_n.get(it["n"]), it["reply"]["answer"]
        labels = {k: v for k, v in labels.items() if v}

    result = attribute(rows, questions, labels, frozen)
    counts = Counter(f["primary"] for f in result)
    any_cause = Counter(c for f in result for c in f["causes"])
    print(f"{len(result)}/{n_total} questions did not pass cleanly ({args.label})\n")
    print(f"{'stage':<16}{'primary cause':>14}{'any cause':>11}")
    for stage in ("router", "retrieval", "catalog tools", "generation", "eval", "unclassified"):
        if counts[stage] or any_cause[stage]:
            print(f"{stage:<16}{counts[stage]:>14}{any_cause[stage]:>11}")

    print("\nEVERY FAILURE")
    for f in sorted(result, key=lambda f: (-f["severity"], f["set"], f["id"])):
        print(f"  [{f['set']:<7}{f['id']:<4}] sev {f['severity']} primary={f['primary']:<13} causes={f['causes']}  {f['query'][:62]}")
        for c, d in f["detail"].items():
            print(f"        {c}: {d}")

    worst = sorted(result, key=lambda f: -f["severity"])[:3]
    print("\nTHREE WORST (by severity), with the trace needed to write them up")
    for f in worst:
        print(f"\n  {f['set']}/{f['id']} [{f['type']}] severity {f['severity']}: {f['query']}")
        print(f"    routed {f['route']} (expected {f['expected_domains']}) | gold {f['gold']} | missing from context {f['gold_missing'] or 'none'}")
        print(f"    status {f['status']} | causes {f['causes']}")
        print(f"    reply: {f['answer'][:260]}{'...' if len(f['answer']) > 260 else ''}")
        for i in f["issues"][:3]:
            print(f"    - {i[:200]}")

    out = ROOT / "evals" / "results" / "attribution.json"
    out.write_text(json.dumps({"label": args.label, "n_questions": n_total, "n_failing": len(result), "primary": dict(counts),
                               "any_cause": dict(any_cause), "failures": result, "worst": [f"{f['set']}/{f['id']}" for f in worst]}, indent=1, ensure_ascii=False))
    print(f"\nsaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
