# Eval & Experiment Log

Running record of what we built, what broke, what we tried and what we decided. Append as we go; this is the raw material for the ADR (`docs/`) and the failure analysis. Newest entries go at the bottom of each section.

Conventions: eval "now" is **2025-03-01**; retrieval is scored on the 30/35 questions that have vector-index gold anchors (`evals/questions_dev.jsonl`); result files live in `evals/results/`.

---

> Cleanup note: the rejected cross-encoder rerankers (E3, E7), index-time question enrichment (E10) and the fusion-weight and `--with-customer` flags (E11) were removed from the code after these experiments. Entries below are history and keep their numbers; the commands and files they name for those experiments no longer exist.

## 1. Environment & tooling issues

| # | Issue | What happened | Resolution |
|---|---|---|---|
| T1 | `chromadb` would not install | pip spent 10+ min compiling a native C++ dependency (no wheel for Intel Mac / Python 3.10) | Switched to **FAISS** (prebuilt wheel). FAISS is search-only, so we wrote metadata filtering, persistence and incremental sync ourselves (`src/ingestion/store.py`, ~60 lines). |
| T2 | `fastembed` unusable | Only version with a wheel (0.1.3) downloads models from a Google Cloud bucket that now returns a permission error | Dropped it. Run BGE's published ONNX file directly with `onnxruntime` + `tokenizers` (`src/ingestion/embeddings.py`). No PyTorch needed. |
| T3 | `onnxruntime` crashed on import | `onnxruntime 1.16` is built against NumPy 1.x; pip had installed NumPy 2 | Pinned `numpy==1.26.4`. |
| T4 | Router silently widened to all domains | `temperature` is not accepted by the Haiku 5.5 endpoint; the fail-wide fallback hid the error behind plausible-looking output | Removed the parameter. Lesson: fallback paths need a loud log line, which it has (`router fallback: ...` on stderr). Also found the router cache key ignored the prompt text, so a prompt edit would have reused stale routes. Key now includes the prompt. |

## 2. Data findings (source files are never edited)

- **Date range**: orders run to 2025-02-10 and the catalog says `last_updated: 2025-03-01`, so the eval "now" is 2025-03-01 (not 2025-01-15 as first proposed).
- **Lena's rejection note says "34 days after purchase"**; the dates give 39 days from purchase, 34 from delivery. The policy counts from purchase. Late either way.
- **`unit_price` semantics differ**: pre-discount in Maya's order, post-discount in Priya's. "Price paid per item" has no reliable answer.
- **Stale statuses**: Chloe's order is "In Transit" with an ETA of 2025-02-03 (past); Ethan's return has been "Pending" since 2024-05-30.
- **Discounts not in totals**: Sofia and Chloe used welcome discounts but totals equal full price.
- **7/8 Airlift has no XL inseam** (`M_L` key only, XL is sold).
- **`is_final_sale_eligible` is `false` on all 24 SKUs** yet customers bought them as Final Sale. Final Sale depends on the discount at purchase (policy returns §6), so we dropped the field from the chunks.
- **Loyalty doc scope**: the title block says some benefits are US-only. Lost when the title block was skipped; fixed with a `{doc}§scope` chunk per policy.
- README says 22 SKUs; the catalog has 24.

## 3. Design decisions so far

| Decision | Choice | Reasoning |
|---|---|---|
| Router | One Haiku call, forced tool-use, `domains` list; flags derived from it; empty list = out of scope; fail wide on error | A missed domain drops evidence; an extra one only costs a lookup. |
| Embeddings | `BAAI/bge-small-en-v1.5`, local ONNX, query prefix | Free, 7 ms per small batch on CPU, strong retrieval benchmarks. |
| Vector store | Chroma (switched from FAISS in E20), one collection per domain | 15k SKUs x 384 dims is ~25 MB; exact search has no recall loss to explain. ANN only past ~1M vectors. |
| Policy chunking | Split at heading boundaries; chunk text = doc title + heading path + body; no overlap | Sections are 26-156 words, far under the 512-token limit; headings are the semantic boundaries; the prefix makes each chunk standalone. |
| Product chunking | One chunk per SKU + one per fabric; fabric composition joined into the product chunk; sizes/colors excluded from the embedded text but shown to the LLM | Long variant lists carry little semantic signal. |
| Staleness | Chunks keyed by content hash; sync re-embeds only new/changed chunks | Tested: editing one policy line re-embedded 1 of 79 chunks; unchanged rebuild embedded 0. |
| Gold labels | Chunk ids as source anchors (exact match) | Caveat: re-chunking at a finer granularity would need hierarchical matching. |
| Customer data | Direct structured lookup keyed by the **session** customer id, never by the query text | Domain 3 is per-user structured data; embedding it risks cross-customer leakage (question A2). |

## 4. Experiments

### E0 - Router accuracy (2026-11-02)
- First pass on 19 hand-written queries: 15/19. Fixed one genuine bug (competitor products routed to `product`: prompt now says `product` means Alo's own catalog). Deliberately did **not** tune for the other three to avoid contaminating the eval.
- Metric changed to report **missed** vs **extra** domains separately (missed = serious, extra = minor).
- On the final 35 eval questions: **34/35 pass**, 1 missed domain (P10 misses `policy`), 7 extra domains (mostly "my/I" questions that don't need the customer record).
- The router test now reads `questions_dev.jsonl` so there is one source of truth.

### E1 - Naive dense top-5 baseline (2026-11-02)
Routed dense top-5 over the 30 scored questions.

| R@1 | R@3 | R@5 | MRR | NDCG@5 | P@5 |
|---|---|---|---|---|---|
| 0.53 | 0.74 | 0.81 | 0.81 | 0.77 | 0.28 |

- Routing barely matters at this corpus size: oracle routing = 0.81, no routing = 0.79. The errors are retrieval errors, not routing errors.
- 8/30 questions miss a gold chunk:
  - `returns§1` (30-day rule) missing from L1, C5, C7, C8. Chunk is intact; it ranks ~7th of 49 for "What's your return window?" because every returns chunk scores 0.58-0.62. **Hypothesis (untested):** the shared doc-title prefix pulls all returns chunks together.
  - Glossary chunks lose to product chunks (`fabric:Airlift` in P4, `fabric:Alosoft` in P8).
  - `loyalty§5.vip` loses to sibling tier chunks (C10).
  - `shipping§8` never surfaces for P10 even with oracle routing (product chunks dominate the pooled ranking).
- **Raw cosine cannot gate refusals**: the nonexistent "Airlift Cloud Legging" (A3) scores 0.80, above the should-answer mean of 0.71 (min 0.57). A similarity threshold would reject real questions before it caught fake ones.

### Metric change (2026-11-02)
At the user's request the retrieval metrics are now **R@1, R@5, R@10, MRR@10, NDCG@10** (default k=10). Context precision@5 was dropped from the table, and the earlier top-5 result files were deleted (superseded; E1's top-5 numbers above are kept for history). All runs below are k=10, routed, n=30 scored questions. **One question = 0.033 on a recall metric, so differences under ~0.05 are within noise.**

### E2-E6 - Step 5 ablation (2026-11-02)

| Run | R@1 | R@5 | R@10 | MRR@10 | NDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| **dense** (baseline) | 0.53 | 0.81 | 0.90 | 0.82 | 0.80 | 8 | 18 |
| bm25 only | 0.48 | 0.72 | 0.92 | 0.76 | 0.76 | 10 | 15 |
| **hybrid** (dense + BM25, RRF k=60, pool 20) | 0.57 | 0.80 | **0.96** | **0.87** | **0.84** | 8 | 10 |
| dense + MiniLM reranker | 0.44 | 0.81 | 0.90 | 0.75 | 0.77 | 587 | 1787 |
| hybrid + MiniLM reranker | 0.44 | 0.81 | 0.90 | 0.75 | 0.77 | 782 | 2004 |
| dense, policy embedded without doc title | 0.53 | **0.85** | 0.93 | 0.82 | 0.82 | 8 | 11 |
| hybrid, policy embedded without doc title | 0.54 | 0.84 | 0.96 | 0.85 | 0.83 | 9 | 12 |
| dense + tier-aware query | 0.53 | 0.81 | 0.91 | 0.82 | 0.81 | 19 | 31 |
| hybrid + tier-aware query | 0.56 | 0.80 | 0.89 | 0.84 | 0.82 | 25 | 56 |
| hybrid + notitle + tier-aware query | 0.53 | 0.83 | 0.89 | 0.83 | 0.81 | 24 | 82 |

**E2 - Hybrid (BM25 + dense, RRF).** Best overall: R@10 +0.06 (two more questions fully covered: L1 `returns§1` and P8's glossary chunk), MRR +0.05, NDCG +0.04. R@5 is flat (-1 question, C10). Real but modest; BM25 alone has the best raw R@10 among the single retrievers (0.92) and the worst R@1, so the two are complementary. No bug found when `hybrid_rerank` matched `dense_rerank` exactly: the two differ on 2 of 35 questions because the reranker mostly picks the same top chunks from either candidate pool.

**E3 - MiniLM cross-encoder reranker: not adopted.** Hurts R@1 (0.53 -> 0.44) and MRR (0.82 -> 0.75), and costs 70-100x latency (p50 ~600-800 ms, p95 ~2 s on CPU). It fixes P4 (glossary chunks to rank 2) and C5 but pushes `returns§1` out of the top 10 for C7/C8 and hurts L6 and P6. Most likely a domain mismatch (ms-marco is web search; our chunks are short policy/catalog text). Larger reranker tried next (E7).

**E4 - Title-prefix hypothesis: partly supported.** Embedding policy chunks without the doc-title prefix lifts dense R@5 0.81 -> 0.85 (C8's `returns§1`: rank 7 -> 3; L1: miss -> rank 6). For hybrid it is neutral (R@5 +0.04, R@1/MRR -0.02/-0.03). `returns§1` is still ranked poorly for C5/C7. Evidence is 1-2 questions, so **not adopted yet**; a confirming test needs more policy questions.

**E5 - Tier-aware retrieval query: not adopted (negative result).** Appending "(asked by a VIP loyalty member)" was meant to surface `loyalty§5.vip` for C10 (Sofia never says she is VIP). It left dense unchanged and **cut hybrid R@10 from 0.96 to 0.89**: the extra words ("loyalty", "member") add lexical matches in BM25 that pull loyalty chunks into returns questions. Better route: give the tier to the generator through the customer record instead. Also noted: C10's gold may be over-specified, because `shipping§2.5` (the tier perks table) already states the VIP perk. Left as is; revisit in failure analysis rather than relabel to chase the metric.

**E6 - Chunks that no mode retrieves.**
- **P10 `shipping§8`**: a *routing* miss (router sent it to `product` only); the policy index is never searched.
- **C10 `loyalty§5.vip`**: the query never says the customer is VIP; retrieval cannot know (see E5).
- `returns§1` ranks 5-10 or misses for L1/C5/C7/C8 in every mode: the 43-word intro chunk is outscored by longer sibling sections.

**Caveat on the whole ablation.** The corpus is 79 chunks and the candidate pool is 20 per domain, so every retriever is looking at 40-67% of the corpus. Hybrid and reranking benefit is understated relative to 15,000 SKUs. A scale test (synthetic catalog expansion) is the first thing to do with two more weeks.

**Refusal signals do not separate.** Top-1 score for should-decline vs should-answer questions: dense cosine max 0.80 vs mean 0.71; MiniLM reranker logit max 5.15 vs mean 0.41 (min -10.4). Neither cosine, BM25 nor the reranker gives a usable "nothing relevant" threshold, so refusal has to come from the router (empty domain list), the generator prompt, and an explicit evidence check, evaluated in steps 6-7.

### E7 - Larger reranker (`BAAI/bge-reranker-base`) on the no-title index: not adopted
R@1 0.48, R@5 0.84, R@10 0.89, MRR@10 0.80, NDCG@10 0.80 (vs hybrid 0.57 / 0.80 / 0.96 / 0.87 / 0.84), at **p50 4.8 s, p95 12.4 s** on CPU. It improves a few sibling/glossary cases (P4, P8, L1, C8) but still loses R@1 and MRR to plain hybrid, and the latency is unusable for an interactive assistant. Its top-1 logit separates should-decline from should-answer questions *better* than the other scores (decline mean -3.91, max -0.27; answer mean -0.64, min -6.37) but the ranges still overlap, so it is not a clean refusal gate either. **Conclusion: neither cross-encoder earns its latency on this corpus; carry hybrid forward.**

### E7b - Is R@10 too high? (2026-11-02)
Yes, it is near saturation. Exact expected scores for random ranking over the searched chunks (mean 43 of 79 per question): **R@1 0.02, R@5 0.12, R@10 0.25, MRR@10 0.11, NDCG@10 0.13**. Dense scores 0.90 at R@10 against 0.25 random, so it is real signal, but retrieving 10 chunks covers ~23% of what is searched and leaves only 0.10 of headroom. R@1, MRR and NDCG have the headroom, so tuning targets those. The scorer now prints the random baseline as a row so no one reads R@10 in isolation.

### E8 - Customer harness (2026-11-02)
`src/context/customers.py` + `evals/suites/customer_eval.py`; 100% of deterministic checks pass (<1 s, no LLM).
- Record is looked up by the **session** customer id only. Question A2 (Maya asks about Jordan Kim) loads Maya's record, and a check confirms no other customer's name, id or order ids appear in any context.
- The prompt-facing render omits email, payment method and address.
- Adds **durations only**, no policy logic: `days ago` for order and delivery, and "N days after order, M after delivery" for return requests. The 30-day rule stays in the retrieved policy, so seasonal policy edits need no code change. This removes calendar counting (G2) from the model without hard-coding policy.
- Raw `notes` are kept, including Lena's incorrect "34 days after purchase". The context now shows "39 days after order, 34 days after delivery" next to it, so catching the discrepancy (G6) is still a test of the model.
- Order gold labels were verified to belong to the right customer.

### Ranking-improvement round (2026-11-02)
Prompted by "R@10 looks very high". Findings first, details below.

**Validity of the metrics**
- R@10 is near saturation (random ranking scores 0.25; dense 0.90), so rank-sensitive metrics are the ones to watch. The scorer prints a random-ranking row.
- **R@1 is capped by gold-set size**: with 1.67 gold chunks per question on average, the best possible dev R@1 is **0.725**. Hybrid's 0.57 is 79% of that ceiling, and its top-1 result is a gold chunk **83%** of the time (hit@1). R@1 looks worse than the retrieval really is. Both numbers are now printed.
- Wrote a **held-out set** (`evals/questions_heldout.jsonl`, 24 new questions, fresh phrasings) *before* running further experiments, and tune only on dev.

**E9 - Embedding model comparison (dev, k=10).** No embedder beats bge-small on the rank-sensitive metrics.

| Embedder | Params | dense R@1 / R@5 / R@10 / MRR / NDCG | hybrid R@1 / R@5 / R@10 / MRR / NDCG | p50 |
|---|---|---|---|---|
| bge-small (current) | 33M | .53 / .81 / .90 / .82 / .80 | .57 / .80 / .96 / **.87** / .84 | 8-11 ms |
| bge-base | 110M | .49 / .87 / .94 / .79 / .80 | .54 / .83 / .97 / .86 / .84 | 28 ms |
| gte-small | 33M | .47 / .84 / .95 / .79 / .80 | .54 / .79 / .94 / .85 / .82 | 8 ms |
| all-MiniLM-L6 | 22M | .43 / .86 / .94 / .76 / .76 | .53 / .84 / .95 / .85 / .82 | 4-6 ms |
| e5-small-v2 | 33M | .47 / .81 / .91 / .79 / .78 | .49 / .81 / .91 / .83 / .80 | 9 ms |

bge-base and MiniLM raise R@5/R@10 but lose R@1/MRR; bge-base is 3x slower for no net gain. Keep bge-small. (Side effect found: the index hash ignored the model, so swapping embedders would have silently reused stale vectors; the store now records the model name and re-embeds fully on change.)

**E10 - Index-time question enrichment** (`src/ingestion/enrich.py`: Haiku writes 4 questions per chunk; embedded and BM25-indexed alongside it; cached in `data/derived/chunk_questions.json`). Dev: dense NDCG 0.80 -> 0.82, R@10 0.90 -> 0.93; but hybrid MRR **0.87 -> 0.82** (generated questions add lexical noise). Held-out: hybrid+enrichment looked best of all (R@1 0.83, MRR 0.95). **Contradictory, so no claim.** See the diagnosis below.

**E11 - Fusion tuning.** BM25 weight 0.5/1/1.5/2 x RRF k 20/60: flat between weight 1 and 1.5; worse at 0.5 and 2. The default sits on a plateau; there is no free gain in the fusion knobs.

**E12 - Held-out vs dev disagree. Why?** (held-out dense MRR 0.87 / hybrid 0.84; dev dense 0.82 / hybrid 0.87)
1. **Different difficulty.** Dev: 15/30 questions need several chunks (1.67 gold each, R@1 ceiling 0.72). Held-out: 4/24 (1.17 each, ceiling 0.92). Hybrid's dev wins are multi-chunk cases (P4 glossary pair, L6, C10); held-out is mostly single-fact lookups where dense is already near its ceiling.
2. **Paraphrase breaks the lexical channel.** Held-out queries are deliberate paraphrases. For H13 ("cancel an order after placing it") and H23 ("perks ... membership level") BM25 ranked shipping/general loyalty chunks above the gold, because the query shares only generic words with it; the gold chunk says "cancelled/cancellations", which my plural-only stemmer does not connect to "cancel". RRF then let those bad lexical hits push the correct dense result down (H13 MRR 1.00 -> 0.33, H23 1.00 -> 0.14).
3. **Very few questions actually flip.** Dense vs hybrid differ on 7 dev and 6 held-out questions. Hybrid wins 5/loses 2 on dev, wins 2/loses 4 on held-out; pooled 7 vs 6. That is noise.
4. **Style leakage ruled out.** Concern: my held-out phrasing resembles the LLM-generated enrichment questions. Measured the closest generated question to each query: cosine 0.776 (dev) vs 0.772 (held-out), no difference.

**Conclusion.** Hybrid, embedder choice and enrichment are statistically indistinguishable on 54 questions; we should not tell a "hybrid improves ranking" story from this data. Hybrid is still the right default for a design reason the small corpus cannot show: exact tokens (SKUs, colour names, "Final Sale") at 15k SKUs. Report dense as the baseline and hybrid as the chosen config with this caveat. (A paired-bootstrap script for pooled dev + held-out was drafted but never run, so it was deleted.)

**The one robust finding: `returns§1`.** It is the gold chunk for L1, C5, C7, C8 (dev) and H10, H24 (held-out), and every method ranks it badly (dense MRR on these: 0.00-0.33). Two candidate causes, not yet separated:
- *Labelling*: the 30-day window is stated in `returns§1`, **`returns§1.1`** and `returns§7`. Retrieving §1.1 would let the model answer correctly, but the gold set only credits §1. The metric may be penalising valid evidence. (Same question as C10's `loyalty§5.vip`.) Changing gold labels after seeing results risks tuning the test, so this needs a deliberate decision, not a quiet edit.
- *Chunk design*: the 43-word §1 intro is outscored by longer sibling sections.

### Step 6 - Generation + faithfulness (2026-11-02)
Retrieval tuning stopped (hybrid + bge-small, k=6 excerpts shown to the model). Built `src/generation/generate.py` (route -> session customer record + retrieved excerpts -> one Sonnet 5.5 call, structured reply: `status`, `answer`, `citations`, `data_issues`) and `evals/suites/generation_eval.py` + `evals/judge/judge.py` (Haiku 5.5 judge). Dev set runs end to end in ~70 s uncached, seconds when cached.

**Prompt architecture.** Rules in the system prompt, evidence in the user message inside `<customer_record>` / `<source id=...>` tags that the model is told to treat as data; question last, after the evidence; customer record from the *session*, never the query; policy numbers deliberately not hard-coded in the prompt (seasonal changes need no prompt edit); empty route short-circuits to a fixed decline with no model call.

**The faithfulness check covers the common pitfalls** (all from `evals/failure_modes.md`):
- *Deterministic layer*: status vs expected, required/forbidden facts, citation validity (every cited id must have been shown), no other customer's data in the answer.
- *Judge layer*: claim-by-claim support against the exact context shown (supported / derived / unsupported fact / unsupported advice / contradicted), a separate audit of "my sources don't say X" statements (true vs false absence), and a pitfall checklist: G1 missing exception, G2 date arithmetic, G3 fabrication on a gap, G4 conflated rules, G5 entitlement error, G6 stale data accepted, S1 out of scope, S2 cross-customer leak, S3 instruction override, S4 over-refusal. Each pitfall is scored only when relevant to the question; G3 on every answer.

**Issues faced**
| # | Issue | Resolution |
|---|---|---|
| T5 | Sonnet 5.5 rejects a forced `tool_choice`, so the reply schema is not enforced | Prompt asks for the tool; reply is validated and retried up to 3x; invalid replies are never cached. |
| T6 | Haiku judge omitted required fields (`correctness_reason`) and the bad output was cached | Same validate-and-retry; purged malformed cache entries. |
| T7 | **First judge was badly calibrated**: hallucination rate 0.51, strict pass 0.40 | Read the answers by hand. The judge counted the assistant's statements about *itself* ("I can't grant exceptions"), hedges ("may be listed under another name"), and *true* statements that a source was missing as unsupported claims, and lumped harmless pointers with invented facts. Fixed the judge: exclude self-statements and hedges; split `unsupported_fact` from `unsupported_advice`; audit absence claims separately. |

**Attribution (important).** I changed the judge *and* the generation prompt between run 1 and run 2, which confounds them. Re-judging run 1's old answers with the new judge: hallucination **0.06** (not 0.51), ungrounded advice 0.09, false absence 0.03. So **~90% of the apparent improvement was fixing the judge, not the model**; the prompt change (explicit "only point to contacts/pages in the sources", clearer `partial` definition) moved ungrounded advice 0.09 -> 0.03 and hallucination 0.06 -> 0.03, i.e. 1-2 questions. The new judge is not merely lenient: on the old answers it still caught an invented "size chart on the product page" (P7), the tracking-email instruction (C3), and an unsupported points-expiry claim (C10).

**Results** (Sonnet 5.5 generator, Haiku 5.5 judge; the held-out set was run once, after all changes, with no tuning on it)

| | dev (n=35) | held-out (n=24) |
|---|---|---|
| status ok | 0.94 | 0.92 |
| required facts present | 0.94 | 0.96 |
| citations valid (no unknown source ids) | 1.00 | 1.00 |
| cross-customer leak | none | none |
| faithfulness (supported or derived facts) | 0.99 | 0.99 |
| hallucination rate (answers with an unsupported fact) | 0.03 | 0.04 |
| ungrounded-advice rate | 0.03 | 0.00 |
| false-absence rate | 0.03 | 0.04 |
| correctness vs reference | 0.91 | 0.94 |
| false-refusal rate | 0.09 | 0.04 |
| correct-decline rate (refuse/abstain) | 1.00 | n/a (none in set) |
| strict pass (everything clean) | 0.77 | 0.83 |

Pitfall checklist, dev (pass / applicable): G1 5/5, G2 4/4, G3 34/35, G4 3/3, G5 4/4, G6 1/2, S1 3/3, S2 1/1, S3 1/1, S4 2/2. Counts are small, so a "perfect" row means "not seen failing", not "solved". Safety cases (A1-A4: competitor, other customer, injection, nonexistent product) all behaved correctly.

**What actually went wrong (and why)**: mostly a *retrieval -> generation chain*, not model misbehaviour.
- `returns§1` (the 30-day rule) was not in the context for L1, C4, C6, H24. The model reported this honestly ("the standard window isn't in my sources"), which is faithful but unhelpful: L1 and L6 degraded to `partial`, H24 to `not_in_data` instead of the correct "no, 138 days". C6 and H23 got **false-absence** flags: the model said a window/perks weren't in its sources while a *different* source or the customer record did say it.
- P4 lacks Airlift's "sheen" because the Airlift glossary chunk was not retrieved (hybrid had it at rank > 10).
- P10 omits the Tuesday/Thursday drop days because the router never sent the question to `policy`.
- **Design tension, grounding vs. escalation**: forbidding contacts not in the sources removed C3's tracking-email hallucination but also removed the "contact support" next step the reference wants (judge marks G6 partially failed). A grounded assistant needs a *standing escalation source* (the contact chunk) in every prompt. Not implemented; recorded as the next change.
- Judge noise remains: H22 flagged "Alo can't grant exceptions" as an unsupported fact (a self-statement that slipped through). It is Haiku grading Sonnet on 59 questions with no human labels.

**Limitations to state out loud in the interview.** (1) The judge is a single small model with no human calibration; sample ~10 verdicts for human review or compare against a stronger judge. (2) "perfect" pitfall rows rest on 1-5 applicable questions. (3) Generation was tuned on dev (judge fixes, one prompt rule); held-out is the fairer number. (4) Retrieval failures cap generation quality; the 0.77/0.83 strict pass is mostly a retrieval ceiling.

### E14 - Product catalog via tool calls instead of chunks (2026-11-02)
**Decision (user's):** do not chunk the catalog for answering. Product questions are about exact attributes, lists and comparisons, which is where retrieval misses (P4 glossary outranked, P8 needs four chunks, P5 depends on top-k covering every match) and which breaks down at 15k SKUs with frequent attribute updates. `src/context/catalog.py` exposes three tools over the structured records: `get_product` (SKU or name), `search_products` (AND-combined filters: gender, category, fabric, best_for, feature, keyword, size, color, price range, sort), `get_fabric`. Results carry an `id` to cite and `total_matches`, so a truncated list cannot be mistaken for a complete one; bad calls return an error object, never raise. The model runs a small tool loop (max 6 turns) and finishes through `respond`. The chunked path stays behind `--product-mode chunks` as the baseline; policies are still retrieved (prose), and the product vector index is kept only for that baseline.

**Tools do not remove hallucination, they move it.** Exact values come from code, but the model can still pick a wrong filter, read an empty result as "does not exist", or go beyond what a result says. So tool results are passed to the faithfulness judge as evidence exactly like chunks, and the eval tracks tool-call error rate, empty-result rate and calls per question.

**A/B on the product questions (same prompt, judge, k):**

| | gold sources in context | strict pass | correctness |
|---|---|---|---|
| dev (10): chunks -> tools | 0.88 -> 0.95 | 0.80 -> 0.90 | 0.90 -> 0.95 |
| held-out (9): chunks -> tools | 1.00 -> 1.00 | 1.00 -> 0.89 | 1.00 -> 0.94 |

Whole-set strict pass: dev 0.77 -> 0.80, held-out 0.83 -> 0.79. Hallucination 0.03 -> 0.00 on dev, 0.04 -> 0.04 on held-out. **On 19 product questions this is a tie**, not a win: one question better on dev, one worse on held-out. The held-out loss is H02, where the reply answered what was asked and the judge docked it for omitting a capacity that only the reference answer mentions (judge strictness, not a tool fault).
- Tool behaviour was clean: error rate 0.00 on both sets; empty-result rate 0.15 on dev, all from the nonexistent-product question (A3), where the model tried three differently worded lookups and then correctly reported "not in the catalog" with no citations. 1.5-1.7 calls per product question.
- **Why adopt it anyway:** with 24 SKUs top-6 retrieval already covers most needs, so this corpus cannot show the benefit. The structural gains are completeness (a filter returns every match and says how many), exact comparisons and sorting, no embedding re-index when attributes change, and no glossary-vs-product ranking competition. That is an architectural argument for 15k SKUs, not a measured win here, and the ADR should say so.
- **Cost:** extra model turns per product question (latency and tokens not yet measured; folded into the step 7 latency/cost metrics).
- **Judge inconsistency seen:** A5 ("Is the Warrior Mat safe for latex allergies?") now answers "may not be a safe choice ... base is natural rubber ... can't confirm". Run 2 of the chunk path had the judge flag "natural rubber is latex-derived" as an unsupported fact; this run the same kind of inference passed. Same claim type, different verdicts: more evidence the judge needs calibration. The model also labelled it `answered` where `partial`/`not_in_data` was expected (a status-labelling miss, content is cautious).
- **Metric note:** the retrieval metrics (R@k, MRR) no longer describe the production path for products. The comparable measure across chunks and tools is `gold_in_context` (did the sources the question needs reach the model), now printed by the generation eval.

### E15 - Judge calibration and eval-system audit (2026-11-02)
**Setup.** `evals/judge/calibration.py` builds a blind review page (`review.html`: question, exact evidence, reply, reference; judge verdicts hidden) and a frozen snapshot of all 58 evaluated answers (`items.json`; freeze *before* changing prompts, because labels attach to specific reply texts). The user labelled the first 12 (`human_labels.json`). The other 46 were labelled by four independent subagents that were blind to the judge and given the user's standard (`assisted_labels.json`, **model-assigned and unreviewed by a human**; same model family as the system, so agreement with them is inter-model consistency, not validation; only 4 of 46 were flagged, so they cannot estimate the judge's miss rate).

**Original judge vs the user's 12 labels:** unsupported fact 12/12 agree, ungrounded advice 12/12, false absence 11/12, any problem 0.75 (kappa 0.44), correctness 0.67. The judge was reliable on faithfulness; it disagreed on **correctness**: three false alarms (P10, C10, H02) where the reply answered what was asked but the reference answer also listed extras (Tue/Thu drop days, a $15 price, a 24 oz capacity), plus one false-absence call on H23 where the evidence really lacked the A-List perks (a retrieval miss, so the absence statement was true).

**What the user's notes revealed (system, not judge):** item 5 / P10: restock questions never reach the policy index; item 6 / A5: the standard should allow (and the reference should state) a referral to a medical professional; item 9 / L1: the reply buried the 30-day window under exceptions (returns§1 not retrieved); item 12 / H23: A-List perks missing (the A-List chunk was not retrieved).

**Eval-system problems found and fixed**
| Problem | Fix |
|---|---|
| Reference answers mixed what a reply *must* say with nice-to-have extras | New optional `optional_details` field (P10, C10, C3, H02, H11); judge told never to penalise omitting them; correctness rubric (correct / partial / incorrect) written into the judge prompt |
| C3's reference demanded "contact support", which our own grounding rule forbids | Reference trimmed to the stale-status finding; contact advice moved to optional and conditional on the evidence containing contact details |
| C10's deterministic `must_include: "$15"` failed a correct reply | Removed; `$15` is optional detail |
| A5 reference lacked the medical referral | Added to reference and `must_include`; judge exempts medical-professional referrals from "ungrounded advice" |
| G6 judge penalised a reply for not recommending an action absent from the evidence | G6 now requires *flagging* stale data only |
| Router `policy` description omitted restock/new-drop timing | Added; P10 now routes to policy; dev exact 27 -> 28/35, held-out unchanged (21/24), 0 missed on both, no new extras |
| Reference numbers vs source data | Automated check: no factual errors (only derived values 39 and 138 days, verified by hand) |

**Revised judge, re-run on the frozen items (`python -m evals.judge.calibration rejudge ...`):** human 12: any-problem agreement 0.75 -> 0.92 (kappa 0.44 -> 0.75), correctness 0.67 -> 0.83, one disagreement left (C10). **This is in-sample: the revisions were informed by those same 12 items, so it is not a fair out-of-sample number.** Model-assigned 46: any-problem 0.87 -> 0.91, correctness 0.93 -> 0.98, pitfall verdicts 1.00 over 59 pairs.

**Borderline cases that need a human ruling** (the standard is ambiguous, not the judge): (a) item 53 H12 and item 58 L3: "Sign in to check your own account" / "if you sign in, I can check where you stand" when the evidence says no customer is signed in (labelers: ungrounded; judge: fine); (b) item 15 H22: "Alo can't grant exceptions to the return window", a policy claim in no source; (c) item 13 C6: "check the Returns Portal for the return's status"; (d) item 21 H15: the reply correctly said the two loyalty sections conflict on student-discount terms and the judge called it a false absence.

**Pipeline problems the audit exposed (not eval bugs, not yet fixed):** returns§1 / §1.1 missing from context for L1, H24 (so the reply cannot state the window: items 9, 38); no support-contact source in context for C3 (item 17); `shipping§2.5` (member free 2-day) missing for H11 (item 45); A-List perks chunk not retrieved for H23 (item 12). Proposed: merge the returns §1 intro into §1.1, a standing contact/escalation source in every prompt, and a self-contained tier-benefits chunk (A-List text currently says "All VIP benefits, plus:").

### E16 - Chunk merges: returns §1 + §1.1, and tier benefits into §4 (2026-11-02)
**Change (user's decision).** `src/ingestion/ingest.py` `MERGES`: (a) `returns§1` absorbs `returns§1.1`, so the 30-day rule and the eligibility conditions are one chunk; (b) `loyalty§4` absorbs the three tier-benefit sections (`loyalty§5.vip/.a-list/.all-access`), retitled "Membership Tiers and Benefits", because thresholds and benefits are always needed together. Tier inheritance is written out ("All VIP benefits, plus:" becomes the VIP benefits in full) so each tier's text stands alone. The merged tier chunk is ~420 words, over the embedder's 512-token window, so its last bullets are not in the embedding (they remain in the text shown to the model and in BM25). Gold labels remapped: C2, C10, H23 `loyalty§5.*` -> `loyalty§4` (same content, new location). Incremental sync re-embedded exactly 2 chunks and removed 4; the other 71 were untouched. The other embedder indexes and `index_hyq` are now stale; the E9/E10 tables describe the old chunking.

**Retrieval (hybrid, k=10).** Dev: R@5 0.80 -> 0.93, MRR 0.87 -> 0.90, NDCG 0.84 -> 0.89 (returns§1 rank: C7 7th -> 2nd, C8 9th -> 2nd, C5 9th -> 5th, L1 9th -> 6th). Held-out: R@5 0.92 -> 0.96, MRR 0.84 -> 0.85, NDCG flat at 0.87; **H24 still never retrieves returns§1** ("send back" shares no words with "return"; lexical match cannot help). Caveats: the merges were chosen from dev failures, so dev is optimistic and held-out is the fair number; C10's jump (missing -> rank 3) is partly the label remap, since its gold chunk is now the merged one that contains the VIP benefits.

**Generation: do not credit the merges with the headline gain.** Dev strict pass 0.80 -> 0.86 and held-out 0.79 -> 0.92 look like improvements, but the judge and reference answers also changed in this stretch (E15). Fair comparison, old answers vs new answers graded by the *same revised judge*: judge-clean 0.90 -> 0.90 overall (dev 0.91 -> 0.88, held-out 0.88 -> 0.92); correctness 0.96 -> 0.97. The merges produced two real fixes (L1 and H23: partial -> correct) and one new flag (L3, a "sign in and the assistant can answer from the account" phrasing, the borderline sign-in case from E15). Still failing: H24 (returns§1 not retrieved, reply says `not_in_data` instead of "no, 138 days") and C10 (partial). Net effect on generation is about zero; the value is in the targeted fixes and in the chunk now being self-contained.

### E17 - Fixing the over-window tier chunk: compact embedding + build guard + All Access questions (2026-11-02)
**Problem.** `loyalty§4` after the E16 merge embedded to 589 tokens against the 512-token window, so the last 77 tokens (end of A-List, all of All Access's own bullets) were silently cut from the embedding. It was the only chunk over the limit (next longest: 301).

**Changes (user's decision: A + E).**
- **A, embed compact / show expanded.** `Chunk.embed_text` for merged chunks is built from the source wording (nothing repeated); `text` (what the model reads, and what BM25 indexes) keeps the expanded, self-contained lists. `loyalty§4` now embeds at 386 tokens and shows 362 words. One chunk re-embedded.
- **E, build-time guard.** `VectorStore.sync` now raises if any chunk's embedded text exceeds the model window (`Embedder.n_tokens` counts without truncation). Tested: a 702-token chunk is refused with a clear message. Truncation can no longer be silent.
- **All Access questions added to both sets**, labelled `"added": "post-hoc ..."` because they were added after seeing results to cover a coverage gap the earlier eval could not detect: **L11** "What benefits do All Access members get?" (dev, policy) and **H25** "Besides free 2-day shipping, what other perks come with my membership?" (held-out, Priya, an All Access customer). Dev is now n=36 and held-out n=25, so headline numbers are not directly comparable with earlier runs; comparisons below use the original questions only.

**Result: the truncation had no measurable effect on retrieval.** On the *old truncated* index the new questions already ranked `loyalty§4` 1st (L11) and 2nd (H25); with the compact embedding the ranks are identical and similarity scores barely moved (0.712 -> 0.694, 0.654 -> 0.665). On the original questions, retrieval is unchanged on held-out and within one rank on dev (dense R@5 0.91 -> 0.93). So A and E are hygiene and a safeguard, not a rescue. The 13% cut only removed bullets that the rest of the chunk's wording already covered for these queries.

**Generation on the new questions.** L11: correct and complete; the expanded tier text let the model list VIP, A-List and All Access extras cleanly (strict pass). H25: correct and comprehensive, but flagged for one unsupported statement ("these [welcome gift and 10% first-order] perks apply from when the customer joined"). **A side effect of expanding inheritance:** "All VIP benefits" literally includes one-time onboarding perks (welcome gift, 10% off the first order), and the expanded list presents them to a member who joined years ago as if they were current. A better reply would omit or caveat them. This is an ADR-worthy tradeoff: self-contained expansion removes the entitlement-mixup risk (G5) but can surface perks that no longer apply to the customer. Original questions: same six dev issues as before (L3, L6, C6, C9, C10, A5; C10's flag text changed with a regenerated reply) and the same two held-out issues (H24, H15). Dev strict pass 0.86 (n=36), held-out 0.88 (n=25; the drop from 0.92 is the new H25).

### E18 - Regression harness, attribution, latency/cost, noise, judge controls (2026-11-02)
**Built:** `evals/harness/run_all.py` (the regression harness), `evals/harness/attribution.py`, `evals/harness/noise.py`, `evals/judge/judge_controls.py`, `evals/pricing/pricing.py`, `evals/stats.py`; router/judge/generator now store the original latency and token counts next to every cached result (cache keys unchanged, so nothing was invalidated). Prices (API reference, cached 2026-10-06): Sonnet 5.5 $2/$10 per MTok in/out, Haiku 5.5 $0.10/$0.50.

**Run-to-run noise (3 uncached repeats, `evals.harness.noise`).** Dev strict pass **0.898 ± 0.032 (0.86-0.92)**, held-out 0.920 ± 0.000. **7 of 36 dev questions flip pass/fail between identical runs** (A2, C10, C5, C6, L6, P6, P9), 0 of 25 held-out. Routing never changed between runs. Judge self-consistency on the same 58 frozen answers: any-problem pairwise agreement 0.954, 4/58 items change verdict; correctness 3/58. **Consequence: differences of 0.03-0.06 in strict pass between two runs are noise**, which includes several comparisons made earlier in this log (e.g. E16's 0.80 -> 0.86). Tolerances = max(0.03, 1.5 questions, 2 x sample SD) -> dev strict pass 0.064, held-out 0.06 (`evals/tolerances.json`; K=3 so the SD itself is rough).

**Latency and cost (uncached, mean over the 3 runs).** Router p50 0.88 s / p95 1.28 s; retrieval 8 ms / 15 ms; answer-model turns 4.6 s / 8.3 s; catalog tool execution ~0; **end to end (no judge) p50 5.5 s / p95 9.1 s**; judge p50 4.1 s / p95 5.6 s. The answer model is ~84% of user-facing latency and the router ~16%, so retrieval is not the bottleneck. Cost per question: **$0.0144 answering (router + answer model), $0.0009 judging**. Latency optimisation levers for the memo: streaming the answer, a faster/cheaper router or skipping it for obvious cases, and fewer tool turns.

**Failure attribution (`evals.harness.attribution`).** 8 of 61 questions did not pass cleanly in the cached run: retrieval 2 (C10: `loyalty§6.2` missing; H24: `returns§1` missing), generation 4 (H25, C6, C9, H15), eval-side 2 (A5: labelled clean; L6: only the status expectation failed), router 0. Known limitation: C6's reply says the return window is not in its sources, but C6's gold labels do not list `returns§1`, so attribution cannot see that retrieval gap; gold sets can be incomplete as well as strict.

**Judge perturbation controls (`evals.judge.judge_controls`).** 53 replies called clean by a human or blind labeler; one error inserted per reply (Sonnet generated, hand-checked); the judge must flag it for the right reason. **Overall 45/50 = 90% (95% range 79-96%)**. By type: wrong number 8/8, invented fact 8/8, invented referral 8/8, false absence 8/8, wrong tier 2/2, stale data accepted 3/3, caved to authority 1/1, conflated rules 6/7, **dropped exception 1/5**. Negative controls: paraphrased replies flagged 1/16, untouched clean originals flagged 3/53 (about 6%, the known borderline items). **The judge is nearly blind to errors of omission**: a claim-level check confirms what a reply says, and deleting a caveat leaves every remaining claim true. It missed 4 of 5 dropped exceptions even with the G1 pitfall switched on. One of those five (C1: dropping "both items are Final Sale" from an order-history answer) is arguably not an error for that question, so the miss rate on valid cases is about 3 of 4. It also missed one fabricated cross-channel rule (L5, "even if you bought it online"). Small samples for wrong_tier (2), stale (3) and caved (1). Consequence: the reported G1 "5/5 pass" is not evidence that exceptions are never omitted; the deterministic `must_include: "Final Sale"` checks (C4, C9, A4) are what protect that case. Mitigation not yet built: an explicit omission audit in the judge (list every exception or limit in the evidence relevant to the question, check each appears), to be re-measured on this same control set.

**The regression harness.** `python -m evals.harness.run_all` (cached, ~20 s), `--fresh` (no caches, **165 s**, under the 5-minute target), `--generator/--judge-model/--router-model` for a new model or configuration, `--save-baseline`. 41 baseline metrics; tolerances from the noise study; hard gates (zero tolerance) for missed router domains, citation validity, cross-customer leaks and judge failures; exit code 1 on a regression. **Verified in both directions:** unchanged system -> pass; answer model swapped to Haiku 5.5 -> detected: dev strict pass 0.86 -> 0.72 (0.75 after the fix below), held-out 0.92 -> 0.80, deterministic pass 0.94 -> 0.78, at **14x lower cost per question ($0.0011 vs $0.0157)**. So Haiku as the answer model trades roughly 10 points of strict pass for 14x lower cost; not acceptable for a customer-facing assistant without the omissions and format errors being fixed, but a useful number for the ADR.

**Bugs the harness and its fresh runs exposed (all fixed)**
| Bug | Cause | Fix |
|---|---|---|
| Runs crashed intermittently (dev, twice) | The Haiku judge returned a malformed grading: tool schemas are not enforced, and one reply put `"correct</correctness_reason>..."` into the `correctness` enum. Key-presence validation passed it, then the scorer crashed. | Validate every enum *value*; snap values that merely begin with a valid word; retry otherwise; fill missing optional fields; 5 attempts; record repairs in `_meta`. |
| One flaky judge call aborted a whole run | No per-question isolation | A judge failure is recorded per question, counts as a strict-pass failure, shows in a `judge_errors` hard gate; never treated as clean. |
| Baseline could be saved from a failed run | No guard | `--save-baseline` refuses when any stage failed. |
| Cross-customer "leak" flagged on a correct refusal | The check counted a name the user typed ("Jordan Kim") that the reply merely repeated. Sonnet's refusal happened not to echo it, so it never fired before. | A name counts only if it did not appear in the question. |
| Tolerance floor smaller than one question | 0.03 < 1/25 | Floor raised to 1.5 questions per set. |
| API clients gave up on transient errors | default 2 retries | `max_retries=6`. |

### E19 - Judge omission audit (tried, removed), staleness eval, versioned policies, deliverable docs (2026-11-02)
**Omission audit: no measurable gain, removed.** An explicit audit step (list the exceptions and conditions in the evidence that this customer's question depends on, flag essential ones the reply leaves out) was added to the judge and measured before and after on identical inputs. Original planted-error set: 45/50 before, 45/50 after (dropped exceptions 1/5 -> 2/5, in-sample). A fresh 22-case out-of-sample omission set: 10/22 before, 10/22 after (pure omissions only: 3/11 -> 4/11, one case, inside noise). False alarms on untouched originals 3/53 -> 2/53. The test set itself was unreliable for omissions: about half the fresh cases also flipped the verdict (any judge catches that), and in several the removed fact survives elsewhere in the reply or is arguably not essential to the question, so "judge is blind" and "this was not a real error" cannot be separated. The audit also coincided with more malformed judge replies. Decision (user's): remove the audit and the extra set, keep the 50-case set, document the weakness. The final uncached harness run caught 0/5 dropped exceptions (1/5 earlier), which is the judge's own noise. Mitigation that does work: deterministic `must_include: "Final Sale"` checks on C4, C9, A4.

**Staleness and update-propagation eval (`evals/suites/staleness_eval.py`, 24/24 checks, now a hard gate in the harness).** Scenario 1, a seasonal update across two policies and the catalog: drift check finds exactly the changed chunks; an incremental update embedded exactly 4 of 75 chunks (counted at the embedder); a catalog date bump refreshed 29 products' metadata with no embedding; new facts retrievable and old ones gone; answers flipped correctly (Warrior Mat order, 45 days old: "through March 16"; water bottle, 38 days: "through March 23"); the tool path returned the new price immediately; an unrelated answer was unchanged. Scenario 2, two policy versions: only the version in force on the as-of date is searchable; flipping the date re-embedded exactly the 2 chunks whose text differed; archived chunks wrongly indexed are still excluded at retrieval.

**Versioned policies (kept, user's decision).** `doc_slug`, `read_version`, `load_all(as_of)`, `searchable()`: the version in force keeps the stable ids; older or not-yet-effective versions are archived under `returns@2025-Q2§1`-style ids, are not embedded, and are filtered at retrieval as defence in depth.

**Bugs found by building the staleness eval**
| Bug | Effect | Fix |
|---|---|---|
| Content hash covered text only, so a metadata-only change was invisible to the drift check | After adding `archived` metadata, the "current policy" filter returned **no policy chunks at all** while `--check` reported the index up to date | Metadata hashed separately (`meta_changed`); sync refreshes metadata without re-embedding; `--check` exits 1 on either kind of drift |
| My old-wording check flagged "30-day window" | A different rule (defect reporting, returns §7) legitimately stays at 30 days | Narrowed to the return-window phrases; a reminder that the same number can mean two rules in one document |

**How many chunks to pass (user's question).** The pipeline already passes the top 6 policy chunks (`k=6`), not the top 1. Hybrid retrieval, share of needed chunks in the top k: dev 0.93 at k=5, 0.97 at k=6, flat after; held-out 0.96 from k=5. All needed chunks within the top 5: 51/56 questions; top 6: 52/56; top 8 and 10: 53/56; 3 questions never appear in the top 10 (H24's rule ranks 11-20). So k=6 buys one question over k=5 and larger k adds essentially nothing; the remaining misses need a targeted retry (wider search or rewritten query when the reply is `not_in_data`), not a bigger k. R@1 is low largely because many questions need several chunks (cap 0.73), which is why the harness gates on R@5, MRR and NDCG.

**Deliverables written:** `README.md`, `docs/ADR.md`, `docs/PRODUCTION_MEMO.md`, `docs/BRIEF_CHECKLIST.md`, `evals/FAILURE_ANALYSIS.md`, and `docs/review.html` (generated by `docs/build_review.py` from the same Q/A blocks). Fresh baseline saved: 42 metrics, uncached run 167 s.

---

## 5. Open questions / hypotheses

- [x] Does dropping the doc title from the embedded text fix the compressed returns scores? **Partly** (E4); not adopted.
- [x] Does BM25 fix exact-term misses without hurting policy recall? **Helps multi-chunk dev questions, hurts paraphrases on held-out** (E2, E12).
- [x] Does a cross-encoder fix sibling confusion and glossary-vs-product ranking? **No, both rerankers net negative** (E3, E7).
- [x] Is a reranker score a better refusal signal than cosine? **Slightly (bge-base), still overlapping** (E6, E7).
- [x] Customer harness: does direct lookup keep each session isolated (A2)? **Yes** (E8).
- [x] Does a bigger/different embedder improve ranking? **No** (E9).
- [x] Do LLM-generated chunk questions help? **Inconclusive** (E10, E12).
- [x] Paired CIs on the final candidates: dropped (the pooled-stats script was never run and has been deleted; the noise study covers it).
- [ ] Decide how to treat alternative evidence for the 30-day window (`returns§1` vs `returns§1.1`): accept equivalent gold sets, or fix the chunk. Needs a deliberate call.
- [ ] Improve the lexical channel for paraphrases (proper stemmer, e.g. cancel/cancelled), then re-test hybrid on both sets.
- [ ] Scale test: do hybrid/reranking gains grow with a 15k-SKU catalog?
- [ ] Is C10's `loyalty§5.vip` gold label over-specified?
- [ ] Add a standing escalation source (support contact chunk) to every prompt so refusals can still give a grounded next step.
- [ ] Human-check ~10 judge verdicts (or compare against a stronger judge) to estimate judge error.
- [ ] Failure analysis (step 7): pick the 3 worst end-to-end questions from `evals/results/generation_*.json` and trace each through router -> retrieval -> generation.

### E19 addendum: clean-room verification (2026-11-02)
Ran the README flow in a copy with no caches, index or results: index built in 8.3 s, demo answered, full harness 166 s, all 9 stages ran.
The harness reported REGRESSION on one hard gate: `generation.dev.citations_ok` 0.972 vs 1.000. Cause: on L9 the model cited `returns§Final Sale Policy` (a heading name, not a chunk id). This is real run-to-run variation (L9 is one of the intermittent questions), not a setup bug. Baseline was not re-saved. Possible fix, not built: map unknown citation labels back to ids by heading, or retry once on invalid citations.

### E20: swapped the FAISS store for Chroma (after the refactor)
Why: Chroma installed cleanly on the second attempt (same numpy/onnxruntime pins), and a reviewer finds an established store easier to accept than a hand-built file store.
Result: retrieval metrics identical to the FAISS baseline (dev R@5 0.933 / MRR 0.899 / NDCG 0.890; held-out 0.960 / 0.831 / 0.863). Search p50 8.4 -> 14.4 ms, p95 9.9 -> 20.4 ms. Index 200 KB -> 1.7 MB. Build 8.3 -> 9.1 s. `store.py` 112 -> 126 lines (the incremental sync is ours with either store). Staleness 24/24 on a standalone run; in the full harness one run crashed on a transient model failure ("no valid reply within 6 model turns") unrelated to the store.
Note for T1 above: the original FAISS wrapper filtered inside the search with an ID selector, it did not post-filter.

### E21: fresh full run after the Chroma switch and cleanup (baseline re-saved)
`run_all --fresh --save-baseline` on the final cleanup commit (150 s, all 10 stages ok, nothing cached). Retrieval, routing, staleness and customer harness are identical to the previous baseline. Compared with the old baseline using the harness tolerances, the only metric outside tolerance is held-out strict pass 0.92 -> 0.84 (tolerance 0.06; H15 and H19 newly fail, H18 and H24 still fail). Both are borderline calls on a 25-question set (one question = 0.04): H15 hedges over a real inconsistency between two policy sections and is judged "partially correct"; H19 is correct but the judge flags one inferential sentence. Nothing in retrieval, prompts or routing changed. Dev: strict pass 0.89 -> 0.86, correctness 0.97 -> 0.99. Judge planted-error detection 0.88 -> 0.94 (dropped exceptions 2/5). This fresh run is now the baseline; the old one is in git history.

### E22: back to FAISS (install time)
Clean-venv install of requirements.txt on this machine (Intel Mac, Python 3.10): Chroma 650 s cold and 755 s with a warm package cache (it compiles a native dependency, as in T1), FAISS 24 s warm. With about 3 minutes for the eval run that left a Chroma reviewer no margin against the brief's 15-minute limit. Retrieval results were identical, Chroma saved no code, so the store is FAISS again (same `VectorStore` interface). The Chroma version is in git history (commit f2ada78).

### E23: removed the product vector index (products are structured, so they are looked up, not searched)
Why: `generate.py` skipped the product index whenever a question was routed to products (catalog tools answer those), so nothing used it at answer time. But `retrieval_eval` still scored product questions against it, which inflated the retrieval numbers (product-only MRR 1.0).
Change: ingestion now embeds policy sections only (45 chunks); `VECTOR_DOMAINS = ("policy",)`; retrieval is scored on policy anchors only (21 dev, 16 held-out questions); catalog and fabric ids are valid gold anchors but are scored by gold-in-context. Staleness eval: a catalog edit must cause no drift and no re-embedding (D2, I2); added V3b for metadata-only refresh. 25 checks, all pass.
Retrieval, hybrid, policy only: dev R@5 0.94 / MRR 0.87 / NDCG 0.87 (was 0.93 / 0.90 / 0.89); held-out 0.94 / 0.74 / 0.79 (was 0.96 / 0.83 / 0.86). Dense alone: dev 0.95 / 0.83 / 0.85, held-out 1.00 / 0.88 / 0.90. Hybrid is not better than dense on policy text and was worse on held-out; kept for now, flagged in the ADR.
Fresh full run (166 s, 10 stages ok): strict pass 0.86 dev (unchanged) / 0.92 held-out (was 0.84); judge agreement 0.86 (was 0.93, 4 of 58 items, within judge noise). Stable failures unchanged: H24, H15, A5 fail in all four uncached runs; C10, C6, L6 still fail; C5, H18, H19 now pass; P6 is intermittent. Baseline re-saved.
