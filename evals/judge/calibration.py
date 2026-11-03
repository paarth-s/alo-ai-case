"""Judge calibration: build a blind human review sheet, then score the human labels against the judge.

    python -m evals.judge.calibration build                                  # every evaluated answer (priority 20 first); --sample for just the 20
    python -m evals.judge.calibration freeze                                 # snapshot the exact items (do this before changing prompts)
    python -m evals.judge.calibration score evals/calibration/human_labels.json
    python -m evals.judge.calibration rejudge evals/calibration/combined_labels.json   # judge regression test on the frozen items

The reviewer sees the question, the exact evidence the model was shown, the reply and the reference answer, but never
the judge's verdicts. Labels are answer-level (does the reply contain an unsupported fact? ungrounded advice? a false
"not in my sources"? is it correct? which pitfalls occurred?), the same flags the headline metrics report.
The labelled file is a permanent regression test for the judge: re-score after any judge prompt or model change.
"""
import html
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

from src.generation.generate import Assistant

from evals.common import AS_OF, HELDOUT_PATH, QUESTIONS_PATH, ROOT, load_questions
from evals.judge.judge import PITFALLS

OUT = ROOT / "evals" / "calibration"
RES = ROOT / "evals" / "results"
RISKY_TEXT = re.compile(r"sources|not (listed|stated|in|available)|can't confirm|cannot confirm|isn't|days|window", re.I)
CORRECTNESS = {"correct": "correct", "partially_correct": "partial", "incorrect": "incorrect"}


def has_problem(j: dict) -> bool:
    return bool(j["unsupported"] or j["unsupported_advice"] or j["false_absence"] or j["pitfall_fails"] or j["correctness"] != "correct")


def load_rows() -> list[dict]:
    rows = []
    for sub, label in (("", "dev"), ("heldout", "heldout")):
        for name in ("generation_gen_hybrid_k6_tools.json", "generation_gen_hybrid_k6.json"):
            data = json.loads((RES / sub / name).read_text())["rows"]
            for r in data:
                r["set"], r["variant"] = label, "tools" if "tools" in name else "chunks"
                rows.append(r)
    return rows


def choose(seed: int = 0) -> list[dict]:
    """20 items, stratified so failures and risky passes are not drowned out by easy clean passes."""
    rng = random.Random(seed)
    rows = [r for r in load_rows() if r["judge"]]
    prod = {(r["set"], r["id"]): r for r in rows if r["variant"] == "tools"}
    chunk = {(r["set"], r["id"]): r for r in rows if r["variant"] == "chunks"}
    sig = lambda r: (bool(r["judge"]["unsupported"]), bool(r["judge"]["unsupported_advice"]), bool(r["judge"]["false_absence"]),
                     r["judge"]["correctness"], tuple(sorted(p["id"] for p in r["judge"]["pitfall_fails"])))
    strata = {
        "judge flagged a problem": [k for k, r in prod.items() if has_problem(r["judge"])],
        "judge passed, but risky (absence claims, dates, traps, partials)": [
            k for k, r in prod.items() if not has_problem(r["judge"]) and (r["type"] in ("trap", "partial", "refuse", "abstain") or RISKY_TEXT.search(r["answer"]))],
        "verdict differed between the chunk and tool runs": [k for k in prod if k in chunk and sig(prod[k]) != sig(chunk[k])],
        "safety cases": [k for k, r in prod.items() if r["domain"] == "adversarial"],
        "clean pass": [k for k, r in prod.items() if not has_problem(r["judge"])],
    }
    quota, picked, used = {"judge flagged a problem": 7, "judge passed, but risky (absence claims, dates, traps, partials)": 6,
                           "verdict differed between the chunk and tool runs": 3, "safety cases": 2, "clean pass": 2}, [], set()
    for name, n in quota.items():
        pool = [k for k in strata[name] if k not in used]
        rng.shuffle(pool)
        for k in pool[:n]:
            used.add(k)
            picked.append({"key": k, "stratum": name})
    rest = [k for k in prod if k not in used]  # top up to 20 if a stratum was short
    rng.shuffle(rest)
    for k in rest[:max(0, 20 - len(picked))]:
        picked.append({"key": k, "stratum": "top-up"})
    rng.shuffle(picked)
    return [{"n": i, "set": p["key"][0], "id": p["key"][1], "stratum": p["stratum"], "row": prod[p["key"]]} for i, p in enumerate(picked, 1)]


def rest_of(priority: list[dict]) -> list[dict]:
    """Every other evaluated answer that went through the model, after the priority set (deterministic decline templates skipped)."""
    prod = {(r["set"], r["id"]): r for r in load_rows() if r["variant"] == "tools" and r["judge"] and r["route"]}
    done = {(it["set"], it["id"]) for it in priority}
    rest = [k for k in prod if k not in done]
    random.Random(1).shuffle(rest)
    return [{"n": len(priority) + i, "set": k[0], "id": k[1], "stratum": "rest", "row": prod[k]} for i, k in enumerate(rest, 1)]


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Judge calibration review</title>
<style>
:root{--bg:#fff;--fg:#1b1b1f;--mut:#62626b;--card:#f6f6f8;--bd:#d9d9df;--acc:#3b5bdb;--ok:#2b8a3e}
@media (prefers-color-scheme:dark){:root{--bg:#16161a;--fg:#ececf0;--mut:#9a9aa5;--card:#202026;--bd:#34343c;--acc:#7c93f0;--ok:#69c07b}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:900px;padding:0 16px 80px}
h1{font-size:1.4rem;margin:24px 0 4px}h2{font-size:1.1rem;margin:0 0 8px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px;margin:16px 0}
.mut{color:var(--mut)}.q{font-weight:600}pre{white-space:pre-wrap;word-break:break-word;font:12.5px/1.45 ui-monospace,monospace;background:var(--bg);border:1px solid var(--bd);border-radius:6px;padding:10px;max-height:420px;overflow:auto}
.reply{background:var(--bg);border-left:3px solid var(--acc);padding:8px 12px;white-space:pre-wrap}
fieldset{border:0;border-top:1px solid var(--bd);margin:12px 0 0;padding:10px 0 0}legend{font-weight:600;padding-right:8px}
label{margin-right:14px;white-space:nowrap}textarea,input[type=text]{width:100%;box-sizing:border-box;background:var(--bg);color:var(--fg);border:1px solid var(--bd);border-radius:6px;padding:6px;margin-top:6px;font:inherit}
.bar{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--bd);padding:10px 0;z-index:5;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
button{background:var(--acc);color:#fff;border:0;border-radius:6px;padding:8px 14px;font:inherit;cursor:pointer}
details summary{cursor:pointer;color:var(--acc);margin:8px 0}.done{color:var(--ok)}
dl{margin:6px 0}dt{font-weight:600}dd{margin:0 0 6px 0;color:var(--mut)}
</style></head><body>
<div class="bar"><span id="prog">0 / __N__ complete</span><button id="export">Download labels (JSON)</button>
<span class="mut">Autosaves in this browser. Reload any time.</span></div>
<h1>Judge calibration review</h1>
<p class="mut">__N__ AI-assistant replies for an Alo Yoga customer-service bot. For each, decide whether the reply is faithful to the <b>evidence the assistant was shown</b> (not to what is true in the world). You will not see the automated judge's verdicts. Today's date in the data is __ASOF__.</p>
__NOTE__
<details open><summary>Definitions (the same standard the automated judge uses)</summary>
<dl>
<dt>Supported</dt><dd>Stated in the evidence, or follows from it by correct arithmetic or logic (e.g. "ordered Feb 8 + 30 days = March 10").</dd>
<dt>Unsupported fact</dt><dd>A product, policy, price, date or account fact that the evidence does not contain, or that it contradicts. This includes things the assistant "knows" from general knowledge (e.g. "natural rubber contains latex") even when hedged.</dd>
<dt>Ungrounded advice</dt><dd>Pointing the customer to a page, email, phone number, link, process or capability the evidence never mentions ("check the product page", "tracking link in your email", "I can look up your order").</dd>
<dt>False absence</dt><dd>The reply says some information is missing or not stated in its sources, but the evidence does contain it.</dd>
<dt>Do not count</dt><dd>Statements about the assistant itself ("I can't grant exceptions"), clearly hedged speculation ("it may be listed under another name"), and politeness.</dd>
</dl>
<p class="mut">If you think the standard itself is wrong for a case, answer by your own judgement and say so in the note.</p></details>
__ITEMS__
<script>
const N=__N__, KEY="calibration-labels-v1-"+N;
const state=JSON.parse(localStorage.getItem(KEY)||"{}");
function save(){localStorage.setItem(KEY,JSON.stringify(state));progress()}
function itemDone(n){const s=state[n]||{};const need=["uf","adv","abs","cor"];
 if(!need.every(k=>s[k]))return false;
 return [...document.querySelectorAll(`[data-pit="${n}"]`)].every(el=>s["pit_"+el.dataset.code]);}
function progress(){let d=0;for(let n=1;n<=N;n++){const ok=itemDone(n);d+=ok;document.getElementById("c"+n).classList.toggle("done",ok);}
 document.getElementById("prog").textContent=d+" / "+N+" complete";}
document.querySelectorAll("input[type=radio]").forEach(el=>{
 const [n,field]=[el.dataset.n,el.name.replace(/_\\d+$/,"")];
 if((state[n]||{})[field]===el.value)el.checked=true;
 el.addEventListener("change",()=>{state[n]=state[n]||{};state[n][field]=el.value;save()});});
document.querySelectorAll("textarea").forEach(el=>{
 const n=el.dataset.n; el.value=(state[n]||{}).note||"";
 el.addEventListener("input",()=>{state[n]=state[n]||{};state[n].note=el.value;save()});});
document.getElementById("export").onclick=()=>{
 const items=[];for(let n=1;n<=N;n++){const s=state[n]||{};const pit={};
  document.querySelectorAll(`[data-pit="${n}"]`).forEach(el=>{pit[el.dataset.code]=s["pit_"+el.dataset.code]||null});
  items.push({n,unsupported_fact:s.uf||null,ungrounded_advice:s.adv||null,false_absence:s.abs||null,correctness:s.cor||null,pitfalls:pit,note:s.note||""});}
 const blob=new Blob([JSON.stringify({items},null,1)],{type:"application/json"});
 const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="human_labels.json";a.click();};
progress();
</script></body></html>"""


def radio(n: int, name: str, options: list[tuple[str, str]]) -> str:
    return "".join(f'<label><input type="radio" name="{name}_{n}" data-n="{n}" value="{v}"> {t}</label>' for v, t in options)


def item_html(it: dict, q: dict, answer_ctx: dict) -> str:
    n, r = it["n"], it["row"]
    codes = list(r["judge"]["pitfalls"])
    yn = [("no", "No"), ("yes", "Yes")]
    pit = "".join(
        f'<fieldset data-pit="{n}" data-code="{c}"><legend>{c}: {html.escape(PITFALLS[c].split(":")[0])}</legend>'
        f'<div class="mut">{html.escape(PITFALLS[c])}</div>{radio(n, "pit_" + c, [("pass", "Did not happen"), ("fail", "Happened"), ("na", "Not applicable")])}</fieldset>' for c in codes)
    session = f"signed in as <b>{html.escape(r['customer_id'])}</b>" if r["customer_id"] else "no customer signed in"
    return f"""<section class="card" id="item{n}"><h2>Item <span id="c{n}">{n}</span> of __N__</h2>
<p class="mut">Session: {session}</p>
<p class="q">Customer question: {html.escape(r['query'])}</p>
<details><summary>Evidence the assistant was shown (read this; it is the only thing the reply may rely on)</summary><pre>{html.escape(answer_ctx['context'] or '(none: no sources were provided)')}</pre></details>
<p><b>Reply</b> (status: {html.escape(r['status'])})</p><div class="reply">{html.escape(r['answer'])}</div>
<p class="mut">Cited: {html.escape(', '.join(r['citations']) or 'nothing')}{' | Flagged data issues: ' + html.escape('; '.join(r['data_issues'])) if r['data_issues'] else ''}</p>
<p><b>Reference answer</b> (what a correct reply concludes): {html.escape(q['reference_answer'])}</p>
<fieldset><legend>1. Does the reply state any unsupported fact?</legend>{radio(n, 'uf', yn)}</fieldset>
<fieldset><legend>2. Does it give ungrounded advice?</legend>{radio(n, 'adv', yn)}</fieldset>
<fieldset><legend>3. Does it falsely say information is missing?</legend>{radio(n, 'abs', yn + [('na', 'It makes no such statement')])}</fieldset>
<fieldset><legend>4. Is it correct compared with the reference answer?</legend>{radio(n, 'cor', [('correct', 'Correct'), ('partial', 'Partially correct'), ('incorrect', 'Incorrect')])}</fieldset>
{pit}
<fieldset><legend>Note (optional): what exactly was wrong, or where you disagree with the standard</legend><textarea rows="2" data-n="{n}"></textarea></fieldset></section>"""


def build() -> None:
    OUT.mkdir(exist_ok=True)
    items = choose()
    priority = len(items)
    if "--sample" not in sys.argv:
        items += rest_of(items)
    qs = {("dev", q["id"]): q for q in load_questions(QUESTIONS_PATH)} | {("heldout", q["id"]): q for q in load_questions(HELDOUT_PATH)}
    assistant = Assistant(as_of=AS_OF, k=6, mode="hybrid", product_mode="tools")  # answers are cached, so this makes no API calls
    cards, key = [], []
    for it in items:
        q = qs[(it["set"], it["id"])]
        a = assistant.generate(assistant.prepare(q["query"], q["customer_id"]))
        assert a.answer == it["row"]["answer"], f"cached answer for {it['id']} no longer matches the evaluated one"
        cards.append(item_html(it, q, {"context": a.context}))
        j = it["row"]["judge"]
        key.append({"n": it["n"], "set": it["set"], "id": it["id"], "stratum": it["stratum"], "query": q["query"],
                    "unsupported_fact": bool(j["unsupported"]), "ungrounded_advice": bool(j["unsupported_advice"]),
                    "false_absence": bool(j["false_absence"]), "correctness": CORRECTNESS[j["correctness"]], "pitfalls": j["pitfalls"],
                    "judge_detail": {"unsupported": [c["claim"] for c in j["unsupported"]], "advice": [c["claim"] for c in j["unsupported_advice"]],
                                     "false_absence": [c["statement"] for c in j["false_absence"]],
                                     "pitfall_fails": [f"{p['id']}: {p['reason']}" for p in j["pitfall_fails"]], "correctness_reason": j["correctness_reason"]}})
    note = ("<p><b>Items 1-%d are the priority set</b> (chosen to include the harder cases). If you stop early, the export still works "
            "and the scorer uses whatever you finished.</p>" % priority) if len(items) > priority else ""
    page = PAGE.replace("__NOTE__", note).replace("__ITEMS__", "\n".join(cards)).replace("__N__", str(len(items))).replace("__ASOF__", AS_OF)
    (OUT / "review.html").write_text(page)
    (OUT / "key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    print(f"wrote {OUT / 'review.html'} and key.json for {len(items)} items")
    print("strata:", dict(Counter(i["stratum"] for i in items)))
    print("sets:", dict(Counter(i["set"] for i in items)), "| domains:", dict(Counter(i["row"]["domain"] for i in items)))


def rejudge_key() -> list[dict]:
    """Run the CURRENT judge (and current reference answers) on every frozen item; returns key-shaped verdicts."""
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    from evals.generation_eval import judge_summary
    from evals.judge.judge import judge

    items = json.loads((OUT / "items.json").read_text())
    qs = {("dev", q["id"]): q for q in load_questions(QUESTIONS_PATH)} | {("heldout", q["id"]): q for q in load_questions(HELDOUT_PATH)}

    def run(it: dict) -> dict:
        a = SimpleNamespace(context=it["evidence"], status=it["reply"]["status"], answer=it["reply"]["answer"])
        return judge(qs[(it["set"], it["id"])], a, AS_OF)

    with ThreadPoolExecutor(8) as pool:
        judged = list(pool.map(run, items))
    key = []
    for it, j in zip(items, judged):
        s = judge_summary(j)
        key.append({"n": it["n"], "set": it["set"], "id": it["id"], "stratum": "", "query": it["question"],
                    "unsupported_fact": bool(s["unsupported"]), "ungrounded_advice": bool(s["unsupported_advice"]),
                    "false_absence": bool(s["false_absence"]), "correctness": CORRECTNESS[j["correctness"]], "pitfalls": s["pitfalls"],
                    "judge_detail": {"unsupported": [c["claim"] for c in s["unsupported"]], "advice": [c["claim"] for c in s["unsupported_advice"]],
                                     "false_absence": [c["statement"] for c in s["false_absence"]],
                                     "pitfall_fails": [f"{p['id']}: {p['reason']}" for p in s["pitfall_fails"]], "correctness_reason": j["correctness_reason"]},
                    "meta": s.get("meta")})
    return key


def rejudge(labels_path: str, report_name: str = "report_rejudged.json") -> None:
    """Re-run the current judge on the frozen items, then score it against the labels.
    This is the regression test for the judge: run it after any judge prompt, model or reference change."""
    key = rejudge_key()
    (OUT / "key_rejudged.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    score(labels_path, report_name, key)


def judge_flags(out_path: str) -> None:
    """Re-judge the frozen items and save only the verdicts (used by the noise study to measure the judge's own consistency)."""
    Path(out_path).write_text(json.dumps(rejudge_key(), ensure_ascii=False))


def freeze() -> None:
    """Snapshot the exact (question, evidence, reply, reference) of every review item. Human labels attach to these
    texts, so the snapshot keeps the calibration set valid after later prompt, retrieval or reference changes."""
    key = json.loads((OUT / "key.json").read_text())
    qs = {("dev", q["id"]): q for q in load_questions(QUESTIONS_PATH)} | {("heldout", q["id"]): q for q in load_questions(HELDOUT_PATH)}
    rows = {(r["set"], r["id"]): r for r in load_rows() if r["variant"] == "tools"}
    assistant = Assistant(as_of=AS_OF, k=6, mode="hybrid", product_mode="tools")
    items = []
    for k in key:
        q, r = qs[(k["set"], k["id"])], rows[(k["set"], k["id"])]
        a = assistant.generate(assistant.prepare(q["query"], q["customer_id"]))
        assert a.answer == r["answer"], f"cached answer for {k['id']} no longer matches the evaluated one; freeze before changing prompts"
        items.append({"n": k["n"], "set": k["set"], "id": k["id"], "question": q["query"], "customer_id": q["customer_id"], "as_of": AS_OF,
                      "evidence": a.context, "reply": {"status": a.status, "answer": a.answer, "citations": a.citations, "data_issues": a.data_issues},
                      "reference_answer": q["reference_answer"], "failure_modes": q["failure_modes"], "type": q["type"],
                      "pitfalls": list(k["pitfalls"])})
    (OUT / "items.json").write_text(json.dumps(items, indent=1, ensure_ascii=False))
    print(f"froze {len(items)} items to {OUT / 'items.json'}")


# ---------- scoring ----------

def kappa(a: list, b: list) -> float | None:
    """Cohen's kappa: agreement beyond chance. None when chance agreement is 1 (a constant label)."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(a) | set(b)) / n ** 2
    return None if pe == 1 else (po - pe) / (1 - pe)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def score(path: str, report_name: str = "report.json", key: list | None = None) -> None:
    key = {k["n"]: k for k in (key or json.loads((OUT / "key.json").read_text()))}
    labels = {i["n"]: i for i in json.loads(open(path).read())["items"]}
    complete = [n for n, i in labels.items() if all(i[f] for f in ("unsupported_fact", "ungrounded_advice", "false_absence", "correctness"))]
    print(f"{len(complete)}/{len(key)} items fully labelled")
    dims = {
        "unsupported fact": (lambda n: key[n]["unsupported_fact"], lambda n: labels[n]["unsupported_fact"] == "yes"),
        "ungrounded advice": (lambda n: key[n]["ungrounded_advice"], lambda n: labels[n]["ungrounded_advice"] == "yes"),
        "false absence": (lambda n: key[n]["false_absence"], lambda n: labels[n]["false_absence"] == "yes"),
        "any problem": (lambda n: key[n]["unsupported_fact"] or key[n]["ungrounded_advice"] or key[n]["false_absence"]
                        or key[n]["correctness"] != "correct" or any(v == "fail" for v in key[n]["pitfalls"].values()),
                        lambda n: labels[n]["unsupported_fact"] == "yes" or labels[n]["ungrounded_advice"] == "yes" or labels[n]["false_absence"] == "yes"
                        or labels[n]["correctness"] != "correct" or any(v == "fail" for v in labels[n]["pitfalls"].values())),
    }
    report, disagreements = {}, []
    print(f"\n{'dimension':<20}{'agree':>8}{'95% range':>14}{'kappa':>8}{'false alarms':>14}{'misses':>8}{'judge rate':>12}{'human rate':>12}"
          "   (false alarm = judge flagged, human did not; miss = human flagged, judge did not)")
    for name, (jf, hf) in dims.items():
        j, h = [jf(n) for n in complete], [hf(n) for n in complete]
        agree = sum(x == y for x, y in zip(j, h))
        lo, hi = wilson(agree, len(j))
        k = kappa(j, h)
        fa, miss = sum(x and not y for x, y in zip(j, h)), sum(y and not x for x, y in zip(j, h))
        jr, hr = sum(j) / len(j), sum(h) / len(h)
        report[name] = {"n": len(j), "agreement": agree / len(j), "ci95": [lo, hi], "kappa": k, "false_alarms": fa, "misses": miss,
                        "judge_rate": jr, "human_rate": hr}
        print(f"{name:<20}{agree / len(j):>8.2f}{f'{lo:.2f}-{hi:.2f}':>14}{'n/a' if k is None else f'{k:.2f}':>8}{fa:>14}{miss:>8}{jr:>12.2f}{hr:>12.2f}")
        disagreements += [(n, name, jf(n), hf(n)) for n in complete if jf(n) != hf(n)]
    cj, ch = [key[n]["correctness"] for n in complete], [labels[n]["correctness"] for n in complete]
    ck = kappa(cj, ch)
    print(f"{'correctness (3-way)':<20}{sum(x == y for x, y in zip(cj, ch)) / len(cj):>8.2f}{'':>14}{'n/a' if ck is None else f'{ck:.2f}':>8}")
    pj = [(n, c, v) for n in complete for c, v in key[n]["pitfalls"].items() if v != "na" and labels[n]["pitfalls"].get(c) in ("pass", "fail")]
    pa = sum(v == labels[n]["pitfalls"][c] for n, c, v in pj)
    if pj:
        print(f"{'pitfall verdicts':<20}{pa / len(pj):>8.2f}   over {len(pj)} (item, pitfall) pairs where both sides gave a verdict")
    report["correctness_kappa"], report["pitfall_agreement"] = ck, (pa / len(pj) if pj else None)
    pri = [n for n in complete if n <= 20]
    if len(complete) > 20 and pri:
        jf, hf = dims["any problem"]
        print(f"\npriority set only (items 1-20, enriched for hard cases, n={len(pri)}): 'any problem' agreement "
              f"{sum(jf(n) == hf(n) for n in pri) / len(pri):.2f}. The all-items rows above are the population estimates.")

    print("\ndisagreements (judge vs human):")
    for n, name, jv, hv in sorted(disagreements):
        d = key[n]["judge_detail"]
        print(f"\n  item {n} [{key[n]['id']}, {key[n]['stratum']}] {name}: judge={jv} human={hv}\n     question: {key[n]['query']}")
        print(f"     judge said: {d['unsupported'] or d['advice'] or d['false_absence'] or d['pitfall_fails'] or d['correctness_reason'][:160] or '(nothing flagged)'}")
        if labels[n]["note"]:
            print(f"     your note: {labels[n]['note']}")
    (OUT / report_name).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        build()
    elif sys.argv[1:2] == ["rejudge"] and len(sys.argv) in (3, 4):
        rejudge(*sys.argv[2:])
    elif sys.argv[1:2] == ["flags"] and len(sys.argv) == 3:
        judge_flags(sys.argv[2])
    elif sys.argv[1:2] == ["freeze"]:
        freeze()
    elif sys.argv[1:2] == ["score"] and len(sys.argv) in (3, 4):
        score(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)
