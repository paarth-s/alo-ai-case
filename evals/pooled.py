"""Pooled (dev + held-out) comparison with paired bootstrap confidence intervals against a baseline run.

    python -m evals.pooled dense_top10_router hybrid_top10_router hybrid_hyq_top10_router

The first label is the baseline. For each other run it prints the pooled mean and the mean per-question
difference vs the baseline with a 95% interval; an interval that contains 0 means the gap could be noise.
Each run must exist in both evals/results/ (dev) and evals/results/heldout/.
"""
import json
import sys

import numpy as np

from evals.common import ROOT

METRICS = ["R@1", "R@5", "MRR@10", "NDCG@10"]


def load(label: str) -> dict:
    rows = {}
    for sub in ("", "heldout"):
        for r in json.loads((ROOT / "evals" / "results" / sub / f"retrieval_{label}.json").read_text())["rows"]:
            if r["metrics"]:
                rows[r["id"]] = r["metrics"]
    return rows


def main(base_label: str, labels: list[str], resamples: int = 5000) -> None:
    base = load(base_label)
    ids = sorted(base)
    boot_idx = np.random.default_rng(0).integers(0, len(ids), size=(resamples, len(ids)))
    print(f"pooled n={len(ids)} questions (dev + held-out); baseline = {base_label}")
    print(f"{'run':<34}" + "".join(f"{m:>22}" for m in METRICS))
    for label in [base_label, *labels]:
        run, cells = load(label), []
        for m in METRICS:
            mean = np.mean([run[i][m] for i in ids])
            if label == base_label:
                cells.append(f"{mean:.2f}")
            else:
                diff = np.array([run[i][m] - base[i][m] for i in ids])
                lo, hi = np.percentile(diff[boot_idx].mean(axis=1), [2.5, 97.5])
                flag = "*" if lo > 0 or hi < 0 else " "
                cells.append(f"{mean:.2f} ({diff.mean():+.2f} [{lo:+.2f},{hi:+.2f}]){flag}")
        print(f"{label:<34}" + "".join(f"{c:>22}" for c in cells))
    print("* = 95% interval excludes 0 (difference unlikely to be noise)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
