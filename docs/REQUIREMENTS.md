# Brief Requirements Checklist

Each requirement from the case brief, as a question with a direct answer and the evidence. Companion to `ADR.md` and `PRODUCTION_MEMO.md`.

## System: RAG implementation

### S1. Does the ingestion pipeline process at least two different document or data types?

**Answer.** Yes, three: Markdown policies, a JSON product catalog, and JSON customer order records.

**Why.** `src/ingestion/ingest.py` splits the three policy files at heading boundaries (45 chunks) and renders the 24 products and 6 fabrics generically from every field (30 chunks). `src/context/customers.py` loads orders as structured records (not embedded). Rebuilds are incremental by content hash (`python -m src.ingestion.build_index`, about 8 s from scratch). The data is the synthetic package supplied with the case (8 customers, 24 SKUs, 3 policies). The brief suggests 20 to 50 customers; I used the supplied 8 and wrote the 61 questions myself.

### S2. Is there an embedding and vector store setup, with written justification?

**Answer.** Yes: `bge-small-en-v1.5` run locally through ONNX, stored in Chroma (local, persistent).

**Why.** Justification and the four rejected embedders (code removed, numbers in `evals/LOG.md` E9) are in ADR Q2. I built the store on FAISS first, then swapped in Chroma and re-ran every eval: retrieval metrics were identical (dev R@5 0.93, MRR 0.90; held-out R@5 0.96, MRR 0.83), search is about 6 ms slower, and the staleness checks pass 24 of 24. Chroma gives persistence and in-search metadata filtering; model-change detection and incremental sync (re-embed only changed chunks) are our code in `src/ingestion/store.py`. Production would use pgvector behind the same `VectorStore` interface.

### S3. Does retrieval use at least one technique beyond naive top-k?

**Answer.** Yes, four: query routing, hybrid BM25 plus dense search fused by reciprocal rank, exact catalog tools, and an effective-date policy-version filter.

**Why.** What each one fixes, and the two that were tried and rejected (cross-encoder rerankers, LLM-written chunk questions; code removed, numbers kept in `evals/LOG.md`), is in ADR Q3. Honest note: hybrid vs dense is within noise on this small corpus.

### S4. Does generation use a deliberate prompt architecture?

**Answer.** Yes: rules in the system prompt, tagged evidence blocks, question last, and a structured reply with citations and a status.

**Why.** Described in ADR Q4 and implemented in `src/generation/generate.py`. Sonnet 5.5 writes the answer; Haiku 5.5 routes and judges. Policy numbers are never written into the prompt.

### S5. Is there a working demo interface?

**Answer.** Yes, a CLI: `python -m src.generation.generate "<question>" [--customer CUST-10254]`.

**Why.** It prints the status, answer, citations, flagged data issues, routing and each catalog tool call. Example: `python -m src.generation.generate "Can I return the leggings I bought during Cyber Monday?" --customer CUST-10254` answers "No, Final Sale" and cites the order and the policy sections. Customers: `CUST-10041` (Maya), `CUST-10082` (Jordan), `CUST-10103` (Sofia), `CUST-10177` (Ethan), `CUST-10254` (Priya), `CUST-10312` (Lena), `CUST-10389` (Marcus), `CUST-10421` (Chloe). The brief allows a CLI as the interface.

### S6. Is the stack choice justified?

**Answer.** No framework: raw SDK calls plus four small libraries (Chroma, onnxruntime, tokenizers, numpy).

**Why.** The corpus is small and each stage has to be inspectable and measurable in a 45-minute debrief. A framework would hide exactly the decisions the brief asks me to defend (chunking, fusion, prompt assembly). The cost is more code of my own: about 3,400 lines across `src/` and `evals/`.

## Evaluation framework

### E1. Is there a question set of at least 20 queries across easy, medium and hard difficulty and all three domains?

**Answer.** Yes: 61 questions. 36 dev (`evals/questions.jsonl`) and 25 held-out (`evals/questions_heldout.jsonl`).

**Why.** 22 policy, 19 product, 15 customer or cross-domain, 5 adversarial; 15 easy, 24 medium, 22 hard. Each question maps to named failure modes (`evals/failure_modes.md`), has gold source ids, a reference answer, and must-include and must-not-include facts. The held-out set was written before further tuning and is the fairer number. Changes made after seeing results are logged in `evals/LOG.md`: two questions added (L11, H25, marked `added: post-hoc` in the file), six reference answers edited after the human review (E15), and three gold labels remapped after a chunk merge (E16).

### E2. Are there at least two retrieval metrics?

**Answer.** Yes: Recall@1/5/10, MRR@10 and NDCG@10, shown beside what a random ranking would score.

**Why.** Hybrid retrieval, dev: R@1 0.59, R@5 0.93, R@10 0.97, MRR 0.90, NDCG 0.89 (random: 0.03, 0.13, 0.25, 0.11, 0.13). Held-out: 0.68, 0.96, 0.96, 0.83, 0.86. R@10 looks high but is nearly saturated on a 75-chunk corpus, and R@1 is capped at 0.73 because many questions need several chunks, so MRR and NDCG are the metrics to watch. Product retrieval is scored by "gold sources in context" (0.99 dev, 0.96 held-out), since products use tools.

### E3. Are there at least two generation metrics?

**Answer.** Yes: faithfulness, hallucination rate, correctness, and a combined strict-pass rate, plus deterministic checks.

**Why.** Dev / held-out: faithfulness 0.99 / 1.00, hallucination rate 0.06 / 0.04, correctness 0.99 / 0.92, strict pass 0.86 / 0.84, valid citations 1.00, cross-customer leaks 0. Strict pass moves ±0.03 between identical runs on dev, so read differences under 0.06 as noise; on the 25 held-out questions one question is 0.04, so a swing of two is still noise (the fresh run's H15 and H19 are borderline judge calls, see LOG E21). Ten pitfall checks (missing exception, date arithmetic, fabrication, conflated rules, wrong tier, stale data, out-of-scope, privacy, injection, over-refusal) are scored per question.

### E4. Is there a failure analysis of the three worst examples?

**Answer.** Yes, in `evals/FAILURE_ANALYSIS.md`: a lookup failure (H24), an overclaim about a latex allergy (A5), and an overgeneralised rule (L9).

**Why.** They were chosen by the severity ranking in `evals/harness/attribution.py`, not by hand. Each is traced through router, retrieval and generation. Across the six questions that did not pass cleanly in the final run: 2 retrieval, 3 generation, 1 eval-side, 0 router.

### E5. Is there a regression harness that re-runs the evals against a new model or configuration in under five minutes?

**Answer.** Yes: `python -m evals.harness.run_all`, 167 seconds with no caches.

**Why.** It runs preflight, router, retrieval, customer harness, generation, judge-vs-labels, planted-error test, staleness and attribution, compares 42 metrics to `evals/baseline.json` with tolerances measured from run-to-run noise, and exits non-zero on a regression. `--generator`, `--judge-model` and `--router-model` swap models. I tested it both ways: unchanged passes; swapping the answerer to Haiku fails (strict pass 0.86 to 0.72) and surfaced two bugs in my own checks.

### E6. Which evaluation tooling did you choose, and why?

**Answer.** A custom claim-level LLM judge plus deterministic checks, not RAGAS, TruLens or DeepEval.

**Why.** The judge checks each claim against the exact evidence shown, and scores the ten domain pitfalls the brief's traps imply (Final Sale, window dates, tier perks). That is what the packaged metrics approximate, with less control over the rubric and caching. The judge is Haiku 5.5, about $0.0009 per question. Everything is disk-cached, so a repeat run makes no API calls.

### E7. Did you evaluate the judge itself?

**Answer.** Yes, three ways: against human labels, with planted errors, and for self-consistency.

**Why.**
- **Human labels:** 12 items labelled by me and 46 by blind labelers (not independent of the model family). Agreement on "any problem": 0.93 (kappa 0.56); on unsupported facts: 1.00.
- **Planted errors:** 44 to 47 of 50 caught for the right reason (88 to 94% across three runs); 1 to 3 of 53 untouched clean replies and 1 to 2 of 16 paraphrases wrongly flagged. It catches added errors almost perfectly and is weak on omissions (0 to 2 of 5 dropped exceptions caught); an omission check I tried did not help and was removed. Deterministic `Final Sale` checks cover that case.
- **Self-consistency:** the same 58 answers re-graded three times flip on 4 of 58 items (95.4% pairwise agreement).

### E8. Is the evaluation cost-efficient?

**Answer.** Yes: a full uncached run costs about $1.50 (estimated from the measured per-call costs) and every call is cached.

**Why.** The judge is the cheapest capable model, calls run concurrently, and cached results include the original latency and tokens so repeat runs are free. The noise study (three uncached repeats) cost a few dollars once; its outputs are saved.
