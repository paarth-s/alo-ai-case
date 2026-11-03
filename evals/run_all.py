"""The regression harness: one command runs the whole eval suite, compares it with a saved baseline, and fails loudly on a regression.

    python -m evals.run_all                       # everything; cached calls make it near-instant
    python -m evals.run_all --fresh               # no caches: a real re-measurement, true latency and cost
    python -m evals.run_all --generator claude-haiku-5-5 --judge-model claude-haiku-5-5   # try a new model or configuration
    python -m evals.run_all --save-baseline       # accept the current numbers as the baseline
    python -m evals.run_all --skip-judge-controls # skip the perturbation test (saves the judge calls)

Stages: question validation, router, retrieval (hybrid), customer-record harness, generation + faithfulness (dev and held-out),
judge vs human labels, judge perturbation controls, staleness / index drift, failure attribution.
Tolerances come from evals/tolerances.json (2 x the run-to-run standard deviation measured by evals.noise); safety metrics are hard gates.
Exit code 1 means a regression. Target: under 5 minutes uncached.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

from .common import HELDOUT_PATH, QUESTIONS_PATH, ROOT, load_questions, validate

RES = ROOT / "evals" / "results"
LOGS = RES / "run_all_logs"
BASELINE = ROOT / "evals" / "baseline.json"
TOLERANCES = ROOT / "evals" / "tolerances.json"
DEFAULT_TOL = 0.05
RETRIEVAL_LABEL, GENERATION_LABEL = "hybrid_top10_router", "gen_hybrid_k6_tools"


def sh(name: str, argv: list[str], env: dict) -> subprocess.Popen:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = open(LOGS / f"{name}.log", "w")
    p = subprocess.Popen([sys.executable, "-m", *argv], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    p.started, p.step = time.time(), name
    return p


def finish(procs: list[subprocess.Popen], timings: dict, errors: list) -> None:
    for p in procs:
        code = p.wait()
        timings[p.step] = round(time.time() - p.started, 1)
        if code != 0:
            errors.append(p.step)
        print(f"  {'ok  ' if code == 0 else 'FAIL'} {p.step:<28}{timings[p.step]:>7.1f}s", flush=True)


def jload(path):
    return json.loads(path.read_text())


def collect() -> dict:
    """Every number the harness compares, read from the stage outputs."""
    m: dict = {}
    router = jload(RES / "router.json")
    for s in ("dev", "heldout"):
        m[f"router.{s}.missed_domains"] = router[s]["missed_domains"]
        m[f"router.{s}.exact_match"] = router[s]["exact_match"]
        sub = "heldout" if s == "heldout" else ""
        ret = jload(RES / sub / f"retrieval_{RETRIEVAL_LABEL}.json")["summary"]["overall"]
        for k in ("R@5", "MRR@10", "NDCG@10"):
            m[f"retrieval.{s}.{k}"] = round(ret[k], 4)
        gen = jload(RES / sub / f"generation_{GENERATION_LABEL}.json")["summary"]
        for k in ("strict_pass", "deterministic_pass", "faithfulness", "hallucination_rate", "false_absence_rate", "correctness",
                  "false_refusal_rate", "citations_ok", "no_cross_customer_leak", "gold_in_context", "judge_errors"):
            if k in gen:
                m[f"generation.{s}.{k}"] = round(gen[k], 4)
        st = gen["stages"]
        m[f"info.{s}.end_to_end_p95_ms"] = st["end_to_end_ms"].get("p95")
        m[f"info.{s}.cost_per_question_usd"] = st["cost_per_question_usd"]["answering"]
    judge = jload(ROOT / "evals" / "calibration" / "report_rejudged_all.json")
    m["judge.agreement_any_problem"] = round(judge["any problem"]["agreement"], 4)
    m["judge.agreement_unsupported_fact"] = round(judge["unsupported fact"]["agreement"], 4)
    if (RES / "judge_controls.json").exists():
        o = jload(RES / "judge_controls.json")["_overall"]
        m["judge.control_detection_rate"] = o["detection_rate"]
        m["judge.paraphrase_false_alarm_rate"] = o["paraphrase_false_alarm_rate"]
    if (RES / "staleness.json").exists():
        st = jload(RES / "staleness.json")
        m["staleness.failed_checks"] = st["total"] - st["passed"]
    att = jload(RES / "attribution.json")
    m["info.failing_questions"] = att["n_failing"]
    return m


# metric -> (direction, hard gate?). 'higher' must not fall by more than the tolerance; 'lower' must not rise by more than it.
RULES = {
    "missed_domains": ("lower", True), "exact_match": ("higher", False), "R@5": ("higher", False), "MRR@10": ("higher", False),
    "NDCG@10": ("higher", False), "strict_pass": ("higher", False), "deterministic_pass": ("higher", False), "faithfulness": ("higher", False),
    "hallucination_rate": ("lower", False), "false_absence_rate": ("lower", False), "failed_checks": ("lower", True), "correctness": ("higher", False),
    "false_refusal_rate": ("lower", False), "citations_ok": ("higher", True), "no_cross_customer_leak": ("higher", True),
    "gold_in_context": ("higher", False), "judge_errors": ("lower", True), "agreement_any_problem": ("higher", False), "agreement_unsupported_fact": ("higher", False),
    "control_detection_rate": ("higher", False), "paraphrase_false_alarm_rate": ("lower", False),
}


def compare(now: dict, base: dict, tols: dict) -> tuple[list[dict], bool]:
    rows, failed = [], False
    for name, value in sorted(now.items()):
        if name.startswith("info.") or value is None:
            continue
        short = name.split(".")[-1]
        direction, hard = RULES.get(short, ("higher", False))
        scope = name.split(".")[1] if name.count(".") == 2 else ""
        tol = 0.0 if hard else tols.get(f"{scope}.{short}", DEFAULT_TOL if not name.startswith(("retrieval", "router")) else 0.02)
        if name.startswith("judge.control") or name.startswith("judge.paraphrase"):
            tol = max(tol, 0.10)  # small samples (about 60 and 16 items)
        before = base.get(name)
        if before is None:
            rows.append({"metric": name, "baseline": None, "now": value, "tol": tol, "status": "new"}); continue
        delta = value - before
        worse = delta < -tol - 1e-9 if direction == "higher" else delta > tol + 1e-9
        better = delta > tol + 1e-9 if direction == "higher" else delta < -tol - 1e-9
        status = ("FAIL (hard gate)" if hard else "WORSE") if worse else ("better" if better else "ok")
        failed |= worse
        rows.append({"metric": name, "baseline": before, "now": value, "tol": tol, "status": status})
    return rows, failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="use no caches: real calls, true latency and cost")
    ap.add_argument("--generator"); ap.add_argument("--judge-model"); ap.add_argument("--router-model")
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--skip-judge-controls", action="store_true")
    args = ap.parse_args()

    env = dict(os.environ)
    for flag, var in ((args.generator, "GENERATION_MODEL"), (args.judge_model, "JUDGE_MODEL"), (args.router_model, "ROUTER_MODEL")):
        if flag:
            env[var] = flag
    if args.fresh:
        from .noise import fresh_env
        env = {**env, **{k: v for k, v in fresh_env("harness").items() if k.endswith("_CACHE")}}

    started, timings, errors = time.time(), {}, []
    print(f"regression harness | generator {env.get('GENERATION_MODEL', 'default')} | judge {env.get('JUDGE_MODEL', 'default')} | "
          f"{'FRESH (no caches)' if args.fresh else 'cached where possible'}\n")

    problems = validate(load_questions(QUESTIONS_PATH)) + validate(load_questions(HELDOUT_PATH))
    if problems or not (ROOT / "index" / "policy" / "faiss.index").exists():
        raise SystemExit(f"preflight failed: {problems or 'index missing, run: python -m src.build_index'}")
    print("preflight ok: question files valid, index present")

    print("\nstage 1: local and cheap")
    for name, argv in (("router", ["evals.test_router"]), ("retrieval_dev", ["evals.retrieval_eval", "--mode", "hybrid"]),
                       ("retrieval_heldout", ["evals.retrieval_eval", "--mode", "hybrid", "--set", "heldout"]),
                       ("customer_harness", ["evals.customer_eval"])):
        finish([sh(name, argv, env)], timings, errors)

    print("\nstage 2: model calls (run concurrently)")
    group = [sh("generation_dev", ["evals.generation_eval", "--set", "dev", "--workers", "4"], env),
             sh("generation_heldout", ["evals.generation_eval", "--set", "heldout", "--workers", "4"], env),
             sh("judge_vs_labels", ["evals.calibration", "rejudge", "evals/calibration/combined_labels.json", "report_rejudged_all.json"], env)]
    group.append(sh("staleness", ["evals.staleness_eval"], env))
    if not args.skip_judge_controls and (ROOT / "evals" / "calibration" / "perturbations.json").exists():
        group.append(sh("judge_controls", ["evals.judge_controls", "run"], env))
    finish(group, timings, errors)

    print("\nstage 3: analysis")
    finish([sh("attribution", ["evals.attribution"], env)], timings, errors)

    total = time.time() - started
    if errors:
        print(f"\nSTAGE FAILURES: {errors}  (see {LOGS.relative_to(ROOT)}/)")
    now = collect()
    tols = jload(TOLERANCES)["tolerances"] if TOLERANCES.exists() else {}

    if args.save_baseline and errors:
        print("\nNOT saving a baseline: a stage failed, so the numbers are incomplete")
        failed = True
    elif args.save_baseline:
        BASELINE.write_text(json.dumps({"saved": datetime.now().isoformat(timespec="seconds"), "generator": env.get("GENERATION_MODEL", "default"),
                                        "judge": env.get("JUDGE_MODEL", "default"), "metrics": now}, indent=1))
        print(f"\nbaseline saved to {BASELINE.relative_to(ROOT)} ({len(now)} metrics)")
        failed = bool(errors)
    elif BASELINE.exists():
        rows, failed = compare(now, jload(BASELINE)["metrics"], tols)
        print(f"\n{'metric':<44}{'baseline':>10}{'now':>10}{'tol':>7}  status")
        for r in rows:
            b = "-" if r["baseline"] is None else f"{r['baseline']:.3f}"
            print(f"{r['metric']:<44}{b:>10}{r['now']:>10.3f}{r['tol']:>7.3f}  {r['status']}")
        failed = failed or bool(errors)
        warn = [(k, now[k], jload(BASELINE)["metrics"].get(k)) for k in ("info.dev.end_to_end_p95_ms", "info.heldout.end_to_end_p95_ms", "info.dev.cost_per_question_usd")
                if now.get(k) and jload(BASELINE)["metrics"].get(k) and now[k] > 1.5 * jload(BASELINE)["metrics"][k]]
        for k, a, b in warn:
            print(f"warning: {k} is {a} vs baseline {b} (more than 1.5x)")
    else:
        print("\nno baseline yet: run with --save-baseline to create one")
        failed = bool(errors)

    for s in ("dev", "heldout"):
        print(f"info {s}: end-to-end p95 {now.get(f'info.{s}.end_to_end_p95_ms')} ms | answer cost per question ${now.get(f'info.{s}.cost_per_question_usd')}")
    print(f"info: {now['info.failing_questions']} questions did not pass cleanly (see evals/results/attribution.json)")
    (RES / "run_all_latest.json").write_text(json.dumps({"finished": datetime.now().isoformat(timespec="seconds"), "fresh": args.fresh, "seconds": round(total, 1),
                                                         "step_seconds": timings, "errors": errors, "metrics": now}, indent=1))
    print(f"\ntotal {total:.0f}s ({'under' if total < 300 else 'OVER'} the 5-minute target) | {'REGRESSION' if failed else 'no regression'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
