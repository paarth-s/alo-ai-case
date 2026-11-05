# Evaluation Framework: Walkthrough

Maps the brief's five evaluation requirements to what was built. Run everything with `python -m evals.harness.run_all`. Sample results are from the committed fresh run (`evals/baseline.json`).

## 1. Question set

**What.** 61 questions: 36 dev (`evals/questions_dev.jsonl`) and 25 held-out (`evals/questions_heldout.jsonl`). 15 easy, 24 medium, 22 hard, across policy, product, customer and adversarial questions. Each has a reference answer, gold source ids, must-include and must-not-include facts, and the failure modes it probes (`evals/failure_modes.md`).

**Why.** Questions target known failure modes, not random coverage. The held-out set was written before further tuning, so it is the fairer number.

**Decisions it drove.** Every change below was accepted or rejected on these questions.

## 2. Retrieval metrics

**Metrics.** Recall@1/5/10, MRR@10, NDCG@10 (`evals/suites/retrieval_eval.py`), each shown next to a random-ranking baseline. Scored on policy questions only, because policy text is the only thing searched; products and orders are exact lookups.

**Why.** Recall@K asks whether the rule was found at all. MRR and NDCG ask how high it ranked, which matters because only 6 excerpts reach the model. On a 45-chunk corpus R@10 saturates, so MRR and NDCG are the ones to watch.

**Decisions.**
- Merged the 30-day rule with its eligibility section after it was outscored: dev recall@5 0.80 to 0.93.
- Chose bge-small over four other embedders (no gain from larger ones, at about 3x the latency).
- Rejected two cross-encoder rerankers (MRR 0.75 and 0.80 vs 0.87 without), LLM-written chunk questions, and fusion tuning.
- Scoring product questions against an embedded product index inflated these numbers, so that index was removed (LOG E23).

**Sample (hybrid, policy only).**

| | R@1 | R@5 | MRR@10 | NDCG@10 |
|---|---|---|---|---|
| Dev (21 q) | 0.58 | 0.94 | 0.87 | 0.87 |
| Held-out (16 q) | 0.53 | 0.94 | 0.74 | 0.79 |
| Random | 0.02 | 0.11 | 0.09 | 0.11 |

Dense alone scores MRR 0.83 dev and 0.88 held-out, so hybrid is not clearly better.

## 3. Generation metrics

**Metrics.** Faithfulness (every claim supported by the sources), correctness vs the reference, hallucination rate, false refusal, and **strict pass** (all deterministic checks and the judge clean). Deterministic checks cover citation validity, other customers' identifiers and required or forbidden facts. A Haiku judge grades claim by claim (`evals/judge/judge.py`).

**Why.** Faithfulness and hallucination catch invented facts. Correctness catches wrong or missing answers. Strict pass is the harshest single number, so it is what the harness gates on.

**Decisions.** Haiku as the answer model scored 10 to 15 points lower on strict pass, so Sonnet stayed. The harness caught that swap and exposed two bugs in my own checks. The router fails wide on error because a missed domain is worse than an extra lookup.

**Sample.**

| | Dev | Held-out |
|---|---|---|
| Strict pass | 0.86 | 0.92 |
| Correctness | 0.99 | 0.94 |
| Faithfulness | 1.00 | 1.00 |
| Hallucination rate | 0.06 | 0.00 |
| Valid citations / cross-customer leaks | 1.00 / 0 | 1.00 / 0 |

Other checks: router exact match 0.81 / 0.88 with 0 missed domains, and 25/25 staleness checks. Identical runs move dev strict pass by about ±0.03, so differences under 0.05 are noise.

## 4. Can the judge be trusted?

**Method.** Agreement with 58 human and assisted labels, 50 planted errors, and repeat grading (`evals/judge/`, `evals/calibration/`).

**Sample.** Agreement 0.86 (0.93 in the previous run; 4 of 58 items flip between repeats). Planted errors caught: 0.90. False alarms on harmless paraphrases: 0.06. It is weak on omitted caveats (0 to 2 of 5), so fixed Final Sale checks cover those.

**Decision.** An omission check I tried did not help and was removed.

## 5. Failure analysis

`evals/FAILURE_ANALYSIS.md` covers the three questions that fail in every run, with the actual replies quoted.
- **H24:** "send back" does not match "return", so the 30-day rule is never retrieved and the reply cannot answer.
- **H15:** the policy contradicts itself on stacking, and the reply leads with a hedge.
- **A5:** a hedged latex-allergy reply that labels itself `answered`.

It also lists the four intermittent failures and what the eval cannot see.

## 6. Regression harness

**What.** `evals/harness/run_all.py` runs 10 stages (router, retrieval dev and held-out, customer checks, generation, judge vs labels, staleness, planted errors, attribution) and compares 42 metrics to `evals/baseline.json`. It exits non-zero on a regression.

**Why this design.** Tolerances come from three uncached repeat runs (`evals/tolerances.json`), so noise does not raise false alarms. Citations, leaks and staleness are hard gates with zero tolerance. `--generator`, `--judge-model` and `--router-model` swap models.

**Sample.** 166 s fresh (about $1.50) and about 100 s cached; the limit is 5 minutes. The Haiku swap fails it, and an unchanged system passes.
