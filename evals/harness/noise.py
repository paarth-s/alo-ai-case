"""Run-to-run noise: repeat the pipeline with fresh caches and measure how much the headline numbers move.

    python -m evals.harness.noise --repeats 3

Two sources of noise are separated:
  * pipeline noise: router + retrieval + answer model + judge, all uncached, repeated K times
  * judge noise:    the judge re-grading the SAME frozen answers K times (answers held fixed)
The spread sets the tolerances the regression harness uses, and the fresh runs record true uncached latency and cost.
Writes evals/results/noise.json and evals/tolerances.json.
"""
import argparse
import itertools
import json
import os
import statistics
import subprocess
import sys

from evals.common import ROOT
from evals.stats import pct

CACHE_DIR = ROOT / ".cache" / "noise"
RES = ROOT / "evals" / "results"
METRICS = ["strict_pass", "deterministic_pass", "status_ok", "faithfulness", "hallucination_rate", "ungrounded_advice_rate",
           "false_absence_rate", "correctness", "false_refusal_rate", "gold_in_context"]


def fresh_env(tag: str) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for name in ("GENERATION", "JUDGE", "ROUTER"):
        path = CACHE_DIR / f"{name.lower()}_{tag}.json"
        path.unlink(missing_ok=True)  # a missing file means an empty cache: every call is real
        env[f"{name}_CACHE"] = str(path)
    return env


def run_pipeline(i: int) -> None:
    """One uncached pipeline run over dev and held-out. Resumes (skips sets already done) and retries a failed set once."""
    env = fresh_env(f"run{i}")
    todo = [(s, RES / ("heldout" if s == "heldout" else "") / f"generation_noise{i}.json") for s in ("dev", "heldout")]
    todo = [(s, out) for s, out in todo if not out.exists()]
    for attempt in (1, 2):
        procs = []
        for s, out in todo:
            log = open(CACHE_DIR / f"run{i}_{s}.log", "w")
            procs.append((s, out, subprocess.Popen([sys.executable, "-m", "evals.generation_eval", "--set", s, "--label", f"noise{i}"],
                                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)))
        todo = [(s, out) for s, out, p in procs if p.wait() != 0 or not out.exists()]
        if not todo:
            return
        print(f"  run {i}: {[s for s, _ in todo]} failed on attempt {attempt}; see {CACHE_DIR}/run{i}_<set>.log", flush=True)
    raise SystemExit(f"pipeline run {i} failed twice")


def run_judge_only(j: int) -> list[dict]:
    out = CACHE_DIR / f"flags_{j}.json"
    if not out.exists():
        r = subprocess.run([sys.executable, "-m", "evals.judge.calibration", "flags", str(out)], cwd=ROOT, env=fresh_env(f"judge{j}"),
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"judge-only run {j} failed:\n{r.stderr[-800:]}")
    return json.loads(out.read_text())


def load(i: int, sub: str) -> dict:
    return json.loads((RES / sub / f"generation_noise{i}.json").read_text())


def spread(xs: list[float]) -> dict:
    return {"mean": statistics.mean(xs), "sd": statistics.stdev(xs) if len(xs) > 1 else 0.0, "min": min(xs), "max": max(xs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--skip-run", action="store_true", help="analyse existing noise runs without making new calls")
    args = ap.parse_args()
    K = args.repeats
    if not args.skip_run:
        for i in range(1, K + 1):
            print(f"pipeline run {i}/{K} (fresh caches)...", flush=True)
            run_pipeline(i)
        flags = []
        for j in range(1, K + 1):
            print(f"judge-only run {j}/{K} (same answers, fresh judge cache)...", flush=True)
            flags.append(run_judge_only(j))
    else:
        flags = [json.loads((CACHE_DIR / f"flags_{j}.json").read_text()) for j in range(1, K + 1)]

    runs = {"dev": [load(i, "") for i in range(1, K + 1)], "heldout": [load(i, "heldout") for i in range(1, K + 1)]}
    report: dict = {"repeats": K, "metrics": {}, "unstable": {}, "tolerances": {}}

    print(f"\nPIPELINE NOISE over {K} uncached runs (mean ± sd | min..max)")
    print(f"{'metric':<24}" + "".join(f"{s:>30}" for s in ("dev", "held-out")))
    for m in METRICS:
        cells = []
        for s in ("dev", "heldout"):
            vals = [r["summary"][m] for r in runs[s] if m in r["summary"]]
            if not vals:
                cells.append("n/a"); continue
            sp = spread(vals)
            report["metrics"][f"{s}.{m}"] = sp
            report["tolerances"][f"{s}.{m}"] = round(max(0.03, 1.5 / runs[s][0]["summary"]["n"], 2 * sp["sd"]), 3)  # never tighter than 1.5 questions
            cells.append(f"{sp['mean']:.3f} ± {sp['sd']:.3f} | {sp['min']:.2f}..{sp['max']:.2f}")
        print(f"{m:<24}" + "".join(f"{c:>30}" for c in cells))

    print("\nPER-QUESTION STABILITY (does a question's outcome change between identical runs?)")
    for s in ("dev", "heldout"):
        by_q: dict = {}
        for r in runs[s]:
            for row in r["rows"]:
                by_q.setdefault(row["id"], []).append(row)
        unstable = {q: rows for q, rows in by_q.items() if len({x["strict_pass"] for x in rows}) > 1}
        status_flip = [q for q, rows in by_q.items() if len({x["status"] for x in rows}) > 1]
        route_flip = [q for q, rows in by_q.items() if len({tuple(x["route"]) for x in rows}) > 1]
        report["unstable"][s] = {"strict_pass": sorted(unstable), "status": status_flip, "route": route_flip, "n": len(by_q)}
        print(f"  {s:<8} strict-pass flips in {len(unstable)}/{len(by_q)} questions {sorted(unstable)} | status changes {len(status_flip)} {status_flip} | routing changes {len(route_flip)} {route_flip}")

    print(f"\nJUDGE NOISE: the judge re-grading the SAME {len(flags[0])} frozen answers {K} times")
    dims = {"unsupported_fact": lambda k: k["unsupported_fact"], "ungrounded_advice": lambda k: k["ungrounded_advice"],
            "false_absence": lambda k: k["false_absence"], "correctness (3-way)": lambda k: k["correctness"],
            "any problem": lambda k: k["unsupported_fact"] or k["ungrounded_advice"] or k["false_absence"] or k["correctness"] != "correct"
            or any(v == "fail" for v in k["pitfalls"].values())}
    report["judge_noise"] = {}
    for name, f in dims.items():
        series = [[f(k) for k in fl] for fl in flags]
        pairs = list(itertools.combinations(range(K), 2))
        agree = statistics.mean(sum(a == b for a, b in zip(series[x], series[y])) / len(series[x]) for x, y in pairs)
        flipped = sum(len({series[r][n] for r in range(K)}) > 1 for n in range(len(series[0])))
        report["judge_noise"][name] = {"pairwise_agreement": round(agree, 3), "items_that_flip": flipped}
        print(f"  {name:<22} pairwise agreement {agree:.3f} | {flipped}/{len(series[0])} items change verdict between repeats")

    print("\nLATENCY AND COST (uncached; mean across runs of each run's p50 / p95, ms)")
    stages = {}
    for key in ("router_ms", "retrieval_ms", "generation_ms", "catalog_tool_ms", "end_to_end_ms", "judge_ms"):
        for s in ("dev", "heldout"):
            xs = [r["summary"]["stages"][key] for r in runs[s] if r["summary"]["stages"][key].get("n")]
            if xs:
                stages[f"{s}.{key}"] = {"p50": round(statistics.mean(x["p50"] for x in xs), 1), "p95": round(statistics.mean(x["p95"] for x in xs), 1)}
    for key in ("router_ms", "retrieval_ms", "generation_ms", "catalog_tool_ms", "end_to_end_ms", "judge_ms"):
        both = [stages[f"{s}.{key}"] for s in ("dev", "heldout") if f"{s}.{key}" in stages]
        if both:
            print(f"  {key:<18} p50 {statistics.mean(b['p50'] for b in both):>8.0f}   p95 {statistics.mean(b['p95'] for b in both):>8.0f}")
    answering = [r["summary"]["stages"]["cost_per_question_usd"] for s in runs.values() for r in s]
    ca = [a["answering"] for a in answering if a["answering"] is not None]
    cj = [a["judging"] for a in answering if a["judging"] is not None]
    report["stages"] = stages
    report["cost_per_question_usd"] = {"answering": round(statistics.mean(ca), 5) if ca else None, "judging": round(statistics.mean(cj), 5) if cj else None}
    print(f"  cost per question: ${report['cost_per_question_usd']['answering']} answering (router + answer model), ${report['cost_per_question_usd']['judging']} judging")

    (RES / "noise.json").write_text(json.dumps(report, indent=1))
    (ROOT / "evals" / "tolerances.json").write_text(json.dumps(
        {"source": f"max(0.03, 1.5 questions, 2 x sample SD) over {K} uncached repeats; see evals/results/noise.json", "tolerances": report["tolerances"]}, indent=1))
    print("\nsaved evals/results/noise.json and evals/tolerances.json")


if __name__ == "__main__":
    main()
