# Architecture Decision Record

**System.** A natural-language assistant over three retail domains. A small router decides which sources a question needs. Policies (Markdown) are searched with hybrid retrieval. Products (JSON catalog) are answered through exact lookup tools. Customer orders (JSON) are loaded directly by the signed-in session's ID, never searched. One Claude call writes the answer in a structured, citable format. Every number comes from `evals/` and is reproducible with `python -m evals.harness.run_all`; `evals/LOG.md` records the experiments, including failed ones.

**How to read the evidence.** The eval set is small (36 dev + 25 held-out questions), and identical runs disagree on about 1 in 5 dev questions (strict pass 0.898 ± 0.032 over three uncached runs). Differences under about 0.05 are noise, and the text says so where it applies.

## The five required questions

### Q1. Chunking strategy: how did you decide chunk size and overlap, and what tradeoffs did you consider?

**Answer.** Policies are split at their own heading boundaries, one chunk per section (15 to 124 words, no overlap). A few sections that are always needed together are merged. For example, information about loyalty and the details of each tier's perks, which were different sections but were nearly always used together. Products are not chunked for answering; they are served by catalog tools.

**Why.**
- Size follows document structure, not a sweep. Each section is one self-contained rule, far below the embedder's 512-token window, and a build-time guard refuses any chunk that would be truncated. Overlap is zero because headings are the semantic boundaries; each chunk is prefixed with its title and heading path so it stands alone.
- Two merges came from measured failures. The 43-word 30-day rule (`returns§1`) was outscored by longer sibling sections, so I merged it with the eligibility conditions: dev recall@5 went 0.80 to 0.93, held-out 0.92 to 0.96. Tier thresholds and benefits are one chunk, with inherited benefits written out so each tier stands alone.
- Costs: the merged tier chunk is 589 tokens, so I embed a 386-token form while the model reads the full text. Writing out inheritance also made the assistant list one-time welcome perks to a long-standing member (H25).
- Not done: a chunk-size sweep. With 45 sections it would measure noise.

### Q2. Embedding model selection: why this model, and which alternatives did you evaluate and reject?

**Answer.** `BAAI/bge-small-en-v1.5` (33M parameters, 384 dimensions), run locally through ONNX.

**Why.**
- I compared bge-base, gte-small, all-MiniLM-L6 and e5-small on the dev set. None beat bge-small on the rank-sensitive metrics (hybrid MRR@10: 0.87 vs 0.86, 0.85, 0.85, 0.83). bge-base improved recall@5/10 slightly but lost on R@1 and MRR at about 3x the latency (28 ms vs 8 to 11 ms).
- The gaps are inside the noise on 30 scored questions, so the tie-break is the cheapest, fastest and simplest option: free, no network, deterministic, and no data leaves the machine.
- Rejected: hosted embedding APIs (cost, latency, data egress, no measured gain) and bge-large (1.3 GB, no evidence it helps here).
- Caveat: that comparison ran on an earlier chunking, though the direction was consistent. The index stores the model name and re-embeds fully on a change.

### Q3. Retrieval approach: what failure modes does your technique address that naive top-k does not?

**Answer.** The router narrows what is searched. Policies use hybrid search (BM25 plus dense, fused by reciprocal rank) over the policy version in force. Products use exact catalog tools. Customer data is a direct lookup.

**Why.** Naive top-k failed in five observable ways, and I address four:
- **Wrong corpus pollutes results:** the router has recall 1.0 on every domain across all 61 questions.
- **List, compare and sort questions** ("which leggings have no side seams") need every match. Catalog tools return the full filtered set plus `total_matches`.
- **Superseded policy text:** an archived-version filter ensures the model never sees two conflicting rules.
- **Exact terms** that embeddings blur ("Final Sale", SKU codes) are why BM25 is in the fusion. Honestly, hybrid vs dense is not distinguishable here (dev MRR 0.90 vs 0.87, held-out 0.83 vs 0.88). I kept it for scale reasons (SKU codes and colour names at 15,000 products), not because it won.
- **Not solved: vocabulary mismatch.** "Send back" shares no words with "return", so H24 never retrieves the 30-day rule.
- Rejected after measuring: two cross-encoder rerankers (MRR 0.75 and 0.80 vs 0.87 for hybrid alone, at 0.7 to 4.8 s per query), fusion-weight tuning (flat), and LLM-generated chunk questions (inconclusive).

### Q4. Prompt architecture: how is context injected, and how do you handle long or low-relevance retrieved context?

**Answer.** Rules live in the system prompt. Evidence goes in the user message inside tagged blocks (`<customer_record>`, `<source id=…>`) that the model treats as data, with the question last. The reply must come back through a structured `respond` tool: `status`, `answer`, `citations`, `data_issues`.

**Why.**
- The customer record comes from the session, never from the question text, so naming someone else cannot change whose data loads. Code computes durations ("45 days ago"); the policy supplies the window, so a policy change needs no code or prompt edit.
- Context is capped at 6 excerpts in rank order (about 1,000 tokens). There is no similarity cutoff, because scores cannot separate answerable from unanswerable questions (a nonexistent product scored 0.80 against 0.71 for real ones). Instead the model must say what is missing (`partial` or `not_in_data`), and a judge verifies that claim.
- Product questions run a short tool loop (about 1.5 model turns, at most 6). Results carry ids and counts so lists cannot be mistaken for complete.
- Known weaknesses: the model sometimes adds advice no source supports, and it rejects forced tool choice, so replies are validated and retried.

### Q5. What would you do differently with two more weeks and a real production dataset?

**Answer.** Fix the vocabulary gap, test at real scale, and replace my hand-built evaluation data with real traffic.

**Why.** In priority order:
1. **Scale test.** Today every retriever sees 40 to 67% of a 75-chunk corpus, so hybrid and reranking are understated. Generate a 15,000-SKU catalog and a larger policy set and re-run the ablations. Grow the customer set too (8 supplied; the brief suggests 20 to 50).
2. **Mine real queries** for "send back" type mismatches, then fix with query rewriting or an embedder fine-tuned on (query, chunk) pairs.
3. **A standing support-contact source** in every prompt, so a decline still gives a grounded next step.
4. **A stronger judge:** 200+ human labels and a check for omitted caveats (my one attempt did not help), run on sampled live traffic.
5. **Lower latency:** stream replies, a faster router, a cached system prompt.
6. **Decide the model on evidence.** Haiku as the answerer is 14x cheaper but 10 to 15 points lower on strict pass.

## Other decisions

| Decision | Choice | Why |
|---|---|---|
| Vector store | FAISS, exact search (`IndexFlatIP`) | At this size exact search is simple and fast: 15,000 vectors is about 25 MB and a few ms, so approximate search adds only recall risk. Filtering, persistence and incremental sync are about 100 lines of my own code. I also built it on Chroma: identical retrieval metrics, no code saved, and an 11 to 12 minute install (vs 24 s), which would eat the reviewer's 15-minute budget. Production would swap in pgvector by changing only `VectorStore`. |
| Framework | None; raw SDK calls | The corpus is tiny and every stage needs to be inspectable in a 45-minute walkthrough. |
| Router | Haiku 5.5, forced tool call, empty list means out of scope, fails wide on error | A missed domain drops evidence; an extra one costs a lookup. |
| Answer model | Sonnet 5.5 | Haiku scored 0.75 vs 0.86 strict pass and cost 14x less; not acceptable for customer-facing answers without more work. |
| Staleness | Content-hash incremental sync, metadata drift check, versioned policies | A three-file update re-embeds 4 of 75 chunks. See the production memo. |
| Judge | Haiku 5.5, claim-level, validated against humans and planted errors | See `docs/BRIEF_CHECKLIST.md` (E6, E7). |
