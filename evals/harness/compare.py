"""Side-by-side comparison of saved retrieval runs, with per-question wins and regressions vs a baseline.

    python -m evals.harness.compare                                  # every saved dev run, baseline = dense_top10_router
    python -m evals.harness.compare dense_top10_router hybrid_top10_router
    python -m evals.harness.compare --heldout                        # the held-out runs instead
"""
import json
import sys

from evals.common import ROOT

BASELINE = "dense_top10_router"
COLUMNS = ["R@1", "R@5", "R@10", "MRR@10", "NDCG@10"]


def main(args):
    heldout = "--heldout" in args
    labels = [a for a in args if not a.startswith("--")]
    results_dir = ROOT / "evals" / "results" / ("heldout" if heldout else "")
    load = lambda label: json.loads((results_dir / f"retrieval_{label}.json").read_text())
    base_label = labels[0] if labels else BASELINE
    labels = labels or sorted(p.stem.removeprefix("retrieval_") for p in results_dir.glob("retrieval_*.json"))
    runs = {label: load(label) for label in labels}

    width = max(len(l) for l in runs) + 2
    print(f"{'run':<{width}}" + "".join(f"{c:>8}" for c in COLUMNS) + f"{'p50 ms':>9}{'p95 ms':>9}")
    for label, run in runs.items():
        o = run["summary"]["overall"]
        print(f"{label:<{width}}" + "".join(f"{o[c]:>8.2f}" for c in COLUMNS) + f"{run['latency_ms']['p50']:>9.0f}{run['latency_ms']['p95']:>9.0f}")
    if (first := next(iter(runs.values())))["summary"].get("random"):
        r = first["summary"]["random"]
        print(f"{'(random ranking)':<{width}}" + "".join(f"{r[c]:>8.2f}" for c in COLUMNS))

    if base_label not in runs:
        return
    base = {r["id"]: r for r in runs[base_label]["rows"] if r["metrics"]}
    for label, run in runs.items():
        if label == base_label:
            continue
        cur = {r["id"]: r for r in run["rows"] if r["metrics"]}
        better = [i for i in cur if cur[i]["metrics"]["MRR@10"] > base[i]["metrics"]["MRR@10"]]
        worse = [i for i in cur if cur[i]["metrics"]["MRR@10"] < base[i]["metrics"]["MRR@10"]]
        print(f"\n{label} vs {base_label} (MRR@10 per question):  better {better or '-'}  |  worse {worse or '-'}")


if __name__ == "__main__":
    main(sys.argv[1:])
