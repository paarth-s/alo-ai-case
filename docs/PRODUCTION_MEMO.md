# Reliability and Production Readiness Memo

**Bottom line.** The assistant is accurate on the data it has (faithfulness 0.99, no cross-customer leak in 61 questions, 24/24 staleness checks) but is not production-ready. Its main weakness is retrieval, not generation: when the right policy section is not retrieved, the model honestly says it cannot answer. That is safe, but unhelpful.

| Top risk | Likelihood / impact | Covered in this repo | Still open |
|---|---|---|---|
| Right policy section not retrieved ("send back" vs "return") | Likely / medium | Hybrid search; honest `not_in_data` | Query rewrite, always-on core rules, support contact |
| Stale or inconsistent policy text | Likely over time / high | Incremental sync, drift check, versions (tested) | Scheduled drift job; cross-section consistency check |
| Model or prompt change lowers quality unnoticed | Likely / high | Regression harness with measured tolerances | Run it in CI on every change |
| Another customer's data exposed | Unlikely / severe | Session-bound lookup, leak check | Run the check on every live reply |
| Latency and cost at scale | Certain / medium | Measured: p50 5.5 s, p95 9.1 s, $0.015 per question | Streaming, faster router |
| The quality judge is wrong | Possible / medium | Validated against humans and planted errors | Weak on omissions |

## The five required questions

### Q1. Staleness and index drift: how do you handle policy updates or catalog changes without a full re-index?

**Answer.** Chunks are keyed by a hash of their content, so an update re-embeds only what changed; a drift check detects an index that no longer matches its source; catalog facts are read live; and policies are versioned by effective date.

**Why.** Tested in `evals/staleness_eval.py` (24 of 24 checks) on a scratch copy:
- A simulated update across two policies and the catalog re-embedded exactly 4 of 75 chunks. Answers followed at once: an order 45 days old flipped from "outside the window" to "returnable through March 16"; an unrelated answer did not change.
- `python -m src.ingestion.build_index --check` lists stale chunks and exits non-zero. Building it exposed a real bug: metadata changes were invisible to a text-only hash, and the "current policy" filter silently returned nothing. Metadata is now hashed separately.
- The policy version in force keeps stable ids; others are archived and not embedded. A version switch re-embedded only the 2 chunks whose text differed.
- Open: one rule spelled in several sections must be edited consistently ("30 days" appears in two different rules); an embedder upgrade needs a full re-embed (detected); version flips need a scheduled job.

### Q2. Retrieval failure modes: what happens when nothing relevant is retrieved, and how do you detect and handle it?

**Answer.** Out-of-scope questions are declined before retrieval. Otherwise retrieval always returns its top 6, so the safeguard is the model's structured status (`partial` or `not_in_data`), verified by a judge and a citation check, with human escalation as the planned fallback.

**Why.**
- A similarity threshold does not work: a nonexistent product scored 0.80 against a 0.71 average for real questions, and reranker scores overlap too.
- For questions outside the data it works: all three refuse/abstain questions were handled, and the false-refusal rate is 3 to 4%.
- It fails safely but unhelpfully when the section exists and was not found. For H24 ("send back the leggings from my first order") the 30-day rule ranked 11th to 20th, so the reply said it could not confirm, when the answer is "no, 138 days". This is the largest quality gap.
- To close it: always include the few core rules every return question needs, retry once with a rewritten query on `not_in_data`, and give a grounded support contact (designed, not built).

### Q3. Latency budget: what are your estimated p50 and p95 latencies, where are the bottlenecks, and how would you optimise?

**Answer.** Measured uncached over 61 questions: p50 5.5 s and p95 9.1 s end to end. The answer model is about 84% of that, the router about 16%, and retrieval is negligible.

**Why.**
- p50 / p95 by stage: router 0.85 / 1.2 s; retrieval 8 to 22 / 15 to 60 ms; answer model 3.9 to 4.6 / 6.6 to 8.8 s (1.4 to 1.5 turns, since product questions call tools). The judge (about 4 s) is evaluation only.
- Cost: $0.015 per question to answer, $0.0009 to judge; a full uncached eval run is about $1.50.
- Levers, biggest first: stream the reply; a smaller or rule-based router; cache the system prompt; cap tool turns. A cheaper answer model is not free: Haiku 5.5 was 14x cheaper but 10 to 15 points lower on strict pass.

### Q4. Guardrails and out-of-scope queries: how do you prevent the system from confidently answering what it should not?

**Answer.** Layers: the router sends out-of-scope questions to a fixed decline with no model call; customer data loads only for the signed-in session; the prompt forbids guessing, other customers' data and policy exceptions; every reply is checked for valid citations and for other customers' identifiers.

**Why.**
- Results: 0 cross-customer leaks and 100% valid citations in 61 questions; the competitor, other-customer, nonexistent-product and "my manager approved it" cases all behaved correctly. Asking about "Jordan Kim" in Maya's session loads only Maya's record.
- The harness caught a cheaper answer model by its strict-pass drop, and exposed a false positive in my own leak check (a refusal repeating the name the user typed), now fixed. Model choice is gated by the harness, not by price.
- Gaps: only 5 adversarial questions; instructions hidden in data fields (an order note, a product description) are untested; the judge is an offline measure, not a runtime guard. Next step: run the cheap deterministic checks on every live reply.

### Q5. Observability: what would you instrument and monitor in production?

**Answer.** A trace per request, a few rate metrics with alert thresholds taken from the measured noise, a sampled quality score, and an index-freshness gauge.

**Why.**
- Per request: router decision and latency; retrieved ids and scores; each tool call with its arguments, result count and errors; turns, tokens and cost; reply status; citation validity; the policy and index version that answered.
- Rates: answered / partial / not_in_data / declined shares (a rising `not_in_data` means a retrieval or content gap), false refusals, tool errors, escalations.
- Quality: judge about 5% of replies. Alert on a drop larger than twice the run-to-run spread measured here (about 0.06 on strict pass), not on any dip. Run `build_index --check` on a schedule and alert on drift. Review the top `not_in_data` queries weekly.
