# Reliability and Production Readiness Memo

**Bottom line.** The assistant is accurate on the data it has (faithfulness 0.99, no cross-customer leak in 61 questions, 24/24 staleness checks) but is not production-ready. Its main weakness is retrieval, not generation: when the right policy section is not found, the model says it cannot answer. That is safe, but unhelpful.

| Top risk | Covered in this repo | Still open |
|---|---|---|
| Right policy section not retrieved ("send back" vs "return") | Hybrid search; honest `not_in_data` | Query rewrite, always-on core rules, support contact |
| Stale or inconsistent policy text | Incremental sync, drift check, versions (tested) | Scheduled drift job |
| Model or prompt change lowers quality unnoticed | Regression harness with measured tolerances | Run it in CI |
| Another customer's data exposed | Session-bound lookup, leak check | Check every live reply |
| Latency and cost at scale | p50 5.5 s, p95 9.1 s, $0.015 per question | Streaming, faster router |
| The quality judge is wrong | Validated against humans and planted errors | Weak on omissions |

## 1. Staleness and index drift

Chunks are keyed by a content hash, so an update re-embeds only what changed. A drift check detects an index that no longer matches its source, catalog facts are read live, and policies are versioned by effective date.

- Tested in `evals/suites/staleness_eval.py` (24 of 24 checks). A simulated update across two policies and the catalog re-embedded 4 of 75 chunks, and answers followed at once: an order 45 days old flipped from "outside the window" to "returnable through March 16".
- `python -m src.ingestion.build_index --check` lists stale chunks and exits non-zero. Building it exposed a bug: metadata changes were invisible to a text-only hash. Metadata is now hashed separately.
- Open: a rule repeated in several sections must be edited in each; an embedder upgrade needs a full re-embed (detected); version flips need a scheduled job.

## 2. Retrieval failure modes

Out-of-scope questions are declined before retrieval. Otherwise retrieval always returns its top 6, so the safeguard is the model's structured status (`partial` or `not_in_data`), verified by a judge and a citation check.

- A similarity threshold does not work: a nonexistent product scored 0.80 against a 0.71 average for real questions, and reranker scores overlap too (LOG E6, E7).
- Questions outside the data are handled (false-refusal rate 3 to 4%).
- It fails safely but unhelpfully when the section exists and was not found. For H24 the 30-day rule ranked 11th to 20th, so the reply said it could not confirm when the answer is "no, 138 days". This is the largest quality gap.
- To close it: always include the core return rules, retry once with a rewritten query on `not_in_data`, and add a grounded support contact (designed, not built).

## 3. Latency budget

Measured uncached (mean of three runs of 61 questions): **p50 5.5 s, p95 9.1 s**. The answer model is about 84% of that, the router 16%, retrieval negligible.

- By stage (p50 / p95): router 0.85 / 1.2 s; retrieval 8 to 22 / 15 to 60 ms; answer model 3.9 to 4.6 / 6.6 to 8.8 s.
- Cost: $0.015 per question to answer, $0.0009 to judge.
- Levers, biggest first: stream the reply; a smaller or rule-based router; cache the system prompt; cap tool turns. Haiku as the answerer is 14x cheaper but 10 to 15 points lower on strict pass.

## 4. Guardrails and out-of-scope queries

Layers: out-of-scope questions get a fixed decline with no model call; customer data loads only for the signed-in session; the prompt forbids guessing, other customers' data and policy exceptions; every reply is checked for valid citations and other customers' identifiers.

- Results: 0 leaks and 100% valid citations in the baseline run (one later run had a single invalid citation label, LOG E19). The competitor, other-customer, nonexistent-product and "my manager approved it" cases all behaved correctly.
- Gaps: only 5 adversarial questions; instructions hidden in data fields are untested; the judge is an offline measure, not a runtime guard. Next step: run the cheap deterministic checks on every live reply.

## 5. Observability

- **Per request:** router decision and latency, retrieved ids and scores, tool calls with result counts and errors, tokens and cost, reply status, citation validity, policy and index version.
- **Rates:** shares of answered / partial / not_in_data / declined (a rising `not_in_data` signals a retrieval or content gap), false refusals, tool errors.
- **Quality:** judge about 5% of replies. Alert on a drop larger than twice the measured run-to-run spread (about 0.06 on strict pass), not on any dip. Schedule `build_index --check` and review top `not_in_data` queries weekly.
