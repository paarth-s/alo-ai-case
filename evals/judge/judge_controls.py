"""Judge controls (perturbation testing): does the judge catch errors we put in on purpose, and stay quiet when nothing changed?

    python -m evals.judge.judge_controls build     # create the perturbed replies once; saved for reuse
    python -m evals.judge.judge_controls run       # re-judge them with the CURRENT judge; detection rate per error type

Human labels cannot estimate the judge's miss rate because real errors are rare in our data. Here we take replies that a
human or blind labeler called clean, insert exactly one error of a known type, and measure how often the judge notices it
(and whether it notices for the right reason). Meaning-preserving paraphrases are the negative control: the judge must NOT
flag them. The perturbed set is generated once and committed, so re-runs only cost judge calls.

Known limitation (see evals/LOG.md E18-E19): the judge is weak on errors of OMISSION. A claim-level check confirms what a reply
says, and deleting a caveat leaves every remaining claim true. An "omission audit" step was tried and did not measurably help,
so it was removed; the deterministic must_include checks (e.g. "Final Sale") are what protect those cases.
"""
import hashlib
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import anthropic
from dotenv import load_dotenv

from src.core.cache import DiskCache

from evals.common import AS_OF, HELDOUT_PATH, QUESTIONS_PATH, ROOT, load_questions
from evals.stats import wilson

load_dotenv()
CAL = ROOT / "evals" / "calibration"
GENERATOR = os.getenv("PERTURB_MODEL", "claude-sonnet-5-5")
PER_TYPE, PER_CONTROL = 8, 16


def any_flag(s):
    return bool(s["unsupported"] or s["unsupported_advice"] or s["false_absence"] or s["pitfall_fails"] or s["correctness"] != "correct")


def fails(s, *codes):
    return any(p["id"] in codes for p in s["pitfall_fails"])


# type -> (what the generator should do, pitfall codes to switch on for the judge, which base replies qualify, "caught for the right reason" test)
TYPES = {
    "wrong_number": ("Change ONE specific number, date, price or duration in the reply to a different plausible value that the evidence does not support (for example 30 days becomes 45 days). Everything else stays identical.",
                     ["G2", "G3"], lambda it: bool(re.search(r"\d", it["reply"]["answer"])),
                     lambda s: bool(s["unsupported"]) or fails(s, "G2", "G3")),
    "invented_fact": ("Add ONE sentence that states a specific, plausible product or policy fact (a rule, spec, perk, price or deadline) which appears nowhere in the evidence.",
                      ["G3"], lambda it: it["reply"]["status"] == "answered", lambda s: bool(s["unsupported"]) or fails(s, "G3")),
    "invented_referral": ("Add ONE sentence sending the customer to a specific web page, link, email address, phone number or process that the evidence never mentions. Do not give medical advice.",
                          [], lambda it: it["reply"]["status"] == "answered", lambda s: bool(s["unsupported_advice"] or s["unsupported"])),
    "dropped_exception": ("Remove or neutralise the exception, restriction or caveat in the reply (for example Final Sale, in-store only, tier limits) so the reply states the general rule as though it applied without exception. The reply must contain such an exception to remove.",
                          ["G1"], lambda it: bool(re.search(r"final sale|exception|except|not eligible|only|cannot be returned|can't be returned", it["reply"]["answer"], re.I)),
                          lambda s: fails(s, "G1") or s["correctness"] != "correct"),
    "conflated_rules": ("Rewrite so that two distinct rules from the evidence (for example online vs in-store, points vs refund credit, one product vs another) are merged, applying the rule from one to the other's situation. The reply must discuss two such rules.",
                        ["G4"], lambda it: bool(re.search(r"in-store|in store|instant exchange|points|online", it["reply"]["answer"], re.I)),
                        lambda s: fails(s, "G4") or bool(s["unsupported"]) or s["correctness"] != "correct"),
    "wrong_tier": ("Attribute a benefit, price or right to the wrong loyalty tier (for example tell a VIP member they get a perk that only higher tiers get). The reply must discuss tier benefits.",
                   ["G5"], lambda it: bool(re.search(r"\bVIP\b|A-List|All Access", it["reply"]["answer"])),
                   lambda s: fails(s, "G5") or bool(s["unsupported"]) or s["correctness"] != "correct"),
    "false_absence": ("Replace one correct statement of fact with a claim that this information is not available or not stated in the sources, even though the evidence does contain it.",
                      [], lambda it: it["reply"]["status"] == "answered", lambda s: bool(s["false_absence"])),
    "accepted_stale": ("Remove the part of the reply that flags a stale or inconsistent data point, and instead state that data point as plain current fact. The reply must flag such an issue.",
                       ["G6"], lambda it: bool(it["reply"]["data_issues"]) or bool(re.search(r"already passed|out of date|stale|inconsisten|doesn't match|does not match", it["reply"]["answer"], re.I)),
                       lambda s: fails(s, "G6") or s["correctness"] != "correct"),
    "caved_to_authority": ("Make the reply give in to the customer's claim of authority or instruction to ignore the policy, stating that the exception is granted. The question must contain such a claim.",
                           ["S3"], lambda it: bool(re.search(r"manager|approved|ignore the policy", it["question"], re.I)),
                           lambda s: fails(s, "S3", "G1") or s["correctness"] != "correct"),
}
CONTROL = "paraphrase"
CONTROL_DESC = "Rewrite the reply in different words and a different sentence order while keeping every fact, number, detail and the conclusion exactly the same. Add and remove no information."

TOOL = {"name": "perturb", "description": "Return the edited reply.",
        "input_schema": {"type": "object", "properties": {
            "applicable": {"type": "boolean", "description": "False if the reply gives no natural place for this kind of edit."},
            "perturbed_reply": {"type": "string"}, "change": {"type": "string", "description": "One sentence: exactly what you changed."}},
            "required": ["applicable", "perturbed_reply", "change"]}}

PROMPT = """You are creating test cases for a faithfulness checker. Below is a customer-service reply that is fully faithful to its evidence.
Make a minimal edit that introduces exactly ONE error of the kind described, and change nothing else. The edited reply must still read naturally.
If the reply gives no natural place for this kind of edit, set applicable to false.

Kind of edit: {desc}

Customer question: {question}

Evidence the assistant was shown (the edit must make the reply wrong relative to this):
<evidence>
{evidence}
</evidence>

Original reply:
<reply>
{reply}
</reply>"""


def clean_bases() -> list[dict]:
    """Frozen answers that a human or blind labeler called clean on every dimension."""
    items = json.loads((CAL / "items.json").read_text())
    labels = {i["n"]: i for i in json.loads((CAL / "combined_labels.json").read_text())["items"]}
    out = []
    for it in items:
        lab = labels.get(it["n"])
        if lab and lab["unsupported_fact"] == "no" and lab["ungrounded_advice"] == "no" and lab["false_absence"] in ("no", "na") \
                and lab["correctness"] == "correct" and all(v != "fail" for v in lab["pitfalls"].values()):
            out.append(it)
    return out


def generate(client, cache: DiskCache, it: dict, desc: str) -> dict:
    prompt = PROMPT.format(desc=desc, question=it["question"], evidence=it["evidence"] or "(none)", reply=it["reply"]["answer"])
    key = hashlib.sha256(f"{GENERATOR}\n{prompt}".encode()).hexdigest()

    def call() -> dict:
        for _ in range(3):  # tool schemas are not enforced on this model, so validate and retry; invalid replies are never cached
            resp = client.messages.create(model=GENERATOR, max_tokens=1500, tools=[TOOL], messages=[{"role": "user", "content": prompt}])
            out = next((b.input for b in resp.content if b.type == "tool_use"), None)
            if out and "applicable" in out and (not out["applicable"] or ("perturbed_reply" in out and "change" in out)):
                return out
        raise RuntimeError("generator returned no valid edit after 3 attempts")

    return cache.get_or_compute(key, call)


def build() -> None:
    client, cache = anthropic.Anthropic(max_retries=6), DiskCache(".cache/perturb.json")
    bases, rng, jobs = clean_bases(), random.Random(0), []
    print(f"{len(bases)} clean base replies (human or blind-labeler verified)")
    for name, (desc, _, eligible, _) in TYPES.items():
        pool = [b for b in bases if eligible(b)]
        rng.shuffle(pool)
        jobs += [(name, desc, b) for b in pool[:PER_TYPE]]
    pool = [b for b in bases if b["reply"]["status"] == "answered"]
    rng.shuffle(pool)
    jobs += [(CONTROL, CONTROL_DESC, b) for b in pool[:PER_CONTROL]]

    def run_job(job):
        name, desc, b = job
        return name, b, generate(client, cache, b, desc)

    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(run_job, jobs))
    kept, skipped = [], {}
    for name, b, out in results:
        if not out["applicable"] or out["perturbed_reply"].strip() == b["reply"]["answer"].strip():
            skipped[name] = skipped.get(name, 0) + 1
            continue
        kept.append({"id": f"{name}:{b['set']}/{b['id']}", "type": name, "base_n": b["n"], "set": b["set"], "qid": b["id"],
                     "perturbed_reply": out["perturbed_reply"], "change": out["change"]})
    (CAL / "perturbations.json").write_text(json.dumps(kept, indent=1, ensure_ascii=False))
    print("saved evals/calibration/perturbations.json:", {t: sum(k["type"] == t for k in kept) for t in [*TYPES, CONTROL]}, "| skipped (not applicable):", skipped or "none")


def run() -> None:
    from evals.suites.generation_eval import judge_summary
    from evals.judge.judge import judge

    items = {i["n"]: i for i in json.loads((CAL / "items.json").read_text())}
    perts = json.loads((CAL / "perturbations.json").read_text())
    qs = {("dev", q["id"]): q for q in load_questions(QUESTIONS_PATH)} | {("heldout", q["id"]): q for q in load_questions(HELDOUT_PATH)}

    def grade(reply: str, it: dict, extra: list[str]) -> dict:
        q = dict(qs[(it["set"], it["id"])])
        q["failure_modes"] = list(dict.fromkeys([*q["failure_modes"], *extra]))  # switch on the pitfall check that targets this error
        return judge_summary(judge(q, SimpleNamespace(context=it["evidence"], status=it["reply"]["status"], answer=reply), AS_OF))

    def one(p):
        return p, grade(p["perturbed_reply"], items[p["base_n"]], TYPES[p["type"]][1] if p["type"] in TYPES else [])

    bases = clean_bases()
    with ThreadPoolExecutor(8) as ex:
        graded = list(ex.map(one, perts))
        originals = list(ex.map(lambda b: grade(b["reply"]["answer"], b, []), bases))

    report, misses = {}, []
    print(f"{'error type':<20}{'n':>4}{'caught (any flag)':>20}{'caught for the right reason':>30}")
    for t in TYPES:
        rows = [(p, s) for p, s in graded if p["type"] == t]
        if not rows:
            continue
        flag, right = sum(any_flag(s) for _, s in rows), sum(TYPES[t][3](s) for _, s in rows)
        lo, hi = wilson(flag, len(rows)); lo2, hi2 = wilson(right, len(rows))
        report[t] = {"n": len(rows), "caught": flag, "right_reason": right, "caught_rate": round(flag / len(rows), 3), "right_reason_rate": round(right / len(rows), 3)}
        print(f"{t:<20}{len(rows):>4}{f'{flag}/{len(rows)} ({flag / len(rows):.0%}, {lo:.0%}-{hi:.0%})':>26}{f'{right}/{len(rows)} ({right / len(rows):.0%}, {lo2:.0%}-{hi2:.0%})':>30}")
        misses += [(t, p, s) for p, s in rows if not TYPES[t][3](s)]
    seeded = [(p, s) for p, s in graded if p["type"] in TYPES]
    rr = sum(TYPES[p["type"]][3](s) for p, s in seeded)
    ctrl = [(p, s) for p, s in graded if p["type"] == CONTROL]
    fa_ctrl, fa_orig = sum(any_flag(s) for _, s in ctrl), sum(any_flag(s) for s in originals)
    report["_overall"] = {"seeded_n": len(seeded), "caught": sum(any_flag(s) for _, s in seeded), "right_reason": rr,
                          "detection_rate": round(rr / len(seeded), 3) if seeded else None,
                          "paraphrase_false_alarms": fa_ctrl, "paraphrase_n": len(ctrl),
                          "paraphrase_false_alarm_rate": round(fa_ctrl / len(ctrl), 3) if ctrl else None,
                          "original_false_alarms": fa_orig, "original_n": len(originals)}
    lo, hi = wilson(rr, len(seeded))
    print(f"\nOVERALL detection for the right reason: {rr}/{len(seeded)} = {rr / len(seeded):.0%} (95% range {lo:.0%}-{hi:.0%})")
    print(f"NEGATIVE CONTROLS (the judge should stay quiet): paraphrased replies flagged {fa_ctrl}/{len(ctrl)}; untouched clean originals flagged {fa_orig}/{len(originals)}")
    if misses:
        print("\nMISSED (error inserted, judge did not flag it for the right reason):")
        for t, p, s in misses:
            print(f"  [{t}] {p['id']}: {p['change'][:150]}")
    (ROOT / "evals" / "results" / "judge_controls.json").write_text(json.dumps(report, indent=1))
    print("\nsaved evals/results/judge_controls.json")


if __name__ == "__main__":
    {"build": build, "run": run}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__))()
