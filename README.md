# Retail RAG Assistant

A question-answering assistant over three retail domains: **product facts**, **policies** (returns, shipping, loyalty) and **a customer's own orders**. Every answer cites its sources. The emphasis is on measuring it honestly: a regression harness, a validated judge, and a written account of where it breaks.

**Read:** [ADR](docs/ADR.md) (design decisions) · [Production memo](docs/PRODUCTION_MEMO.md) · [Failure analysis](evals/FAILURE_ANALYSIS.md) (3 worst cases) · [Evaluation walkthrough](docs/EVALUATION.md) · [Brief checklist](docs/BRIEF_CHECKLIST.md) · [Lab notebook](evals/LOG.md)

## How it works

```
question ─► router (Haiku) ─► customer │ policy │ product   (none = out of scope: fixed decline)
  customer: the signed-in session's own record (never an id typed in the question)
  policy:   hybrid search (keywords + embeddings, merged by rank) over the policy version in force
  product:  exact catalog lookups (get_product, search_products, get_fabric), not search
all evidence ─► Sonnet 5.5 ─► one structured reply: status, answer, citations
```

Policies are split at headings into 45 chunks and stored in a FAISS index with local `bge-small` embeddings. Updates re-embed only changed chunks. Products and customers are structured data, so they are looked up, not searched.

## Setup (about 5 minutes; Python 3.10+, an Anthropic API key)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                       # put your key in ANTHROPIC_API_KEY=
python -m src.ingestion.build_index        # ~10 s; downloads the 130 MB embedding model once
```

## Try it

```bash
python -m src.web.app                      # web demo at http://127.0.0.1:8000 (interface only); /design shows choices and metrics
python -m src.generation.generate "Which leggings have no side seams?"
python -m src.generation.generate "Can I return my Cyber Monday leggings?" --customer CUST-10254
python -m src.generation.generate "What did Jordan Kim order?" --customer CUST-10041   # declines: not your account
python -m src.generation.generate "How much is Lululemon's Align legging?"             # declines: out of scope
```

Eight sample customers (`CUST-10041`, `10082`, `10103`, `10177`, `10254`, `10312`, `10389`, `10421`). "Today" is fixed at 2025-03-01 to match the data.

## Run the evals (about 3 minutes, about $1.50)

```bash
python -m evals.harness.run_all            # 10 stages, compared with evals/baseline.json; exit 1 = regression
python -m evals.harness.run_all --fresh    # ignore caches: real latency and cost (a cached run only checks the plumbing)
python -m evals.harness.run_all --generator claude-haiku-5-5   # try another answer model
```

It prints each metric with its baseline, new value and tolerance. Tolerances come from measured run-to-run noise, so one `WORSE` on a noisy metric is worth re-running. Single pieces: `evals.suites.retrieval_eval`, `evals.suites.generation_eval`, `evals.suites.staleness_eval`, `evals.judge.judge_controls run`, `src.ingestion.build_index --check`.

## Results (fresh run, 61 questions: 36 dev, 25 held-out)

| | Dev | Held-out |
|---|---|---|
| Policy retrieval R@5 / MRR@10 (21 dev, 16 held-out questions) | 0.94 / 0.87 | 0.94 / 0.74 |
| Faithfulness | 1.00 | 1.00 |
| Correctness vs reference | 0.99 | 0.94 |
| Strict pass (every check clean) | 0.86 | 0.92 |
| Cross-customer leaks / invalid citations | 0 / 0 | 0 / 0 |

Identical runs move dev strict pass by about ±0.03; on 25 questions one is worth 0.04. Retrieval covers policy text only (products and orders are exact lookups, scored in the generation eval). Hybrid search is not better than embeddings alone on this small policy set (held-out MRR 0.74 vs 0.88); the gains come from routing and exact lookups. Latency p50 5.5 s, p95 9.1 s; about $0.015 per question; staleness 25/25 checks.

## Where it is weak

- **Vocabulary gaps:** "send back" does not find the "return" rule (H24). Largest known gap.
- **The judge misses omissions** (0 to 2 of 5 dropped caveats); fixed Final Sale checks cover those.
- **Small data, no real traffic:** 45 policy chunks and 61 hand-written questions.

## Layout

```
src/   core/ (cache) · router/ · ingestion/ (chunking, embeddings, FAISS store, build_index)
       retrieval/ (BM25, hybrid) · context/ (catalog tools, customer lookup) · generation/ · web/
evals/ harness/ (run_all, noise, attribution) · judge/ · pricing/ · suites/ (the *_eval.py scripts) · questions_dev/heldout.jsonl
       baseline.json · results/ · calibration/ · LOG.md
data/  policies, product catalog, customer orders (the synthetic package supplied with the case)
docs/  ADR, production memo, evaluation walkthrough, brief checklist, review.html (the brief's questions with answers)
```

## Glossary

**RAG**: retrieve relevant text, then answer from it. **Chunk**: one searchable piece of a document. **Embedding**: text as numbers, so similar meanings sit close. **BM25**: keyword search. **Recall@K / MRR / NDCG**: how often, and how high, the right chunk is ranked. **Faithfulness**: every claim is supported by the sources. **Strict pass**: all automatic checks and the judge are clean. **Judge**: a second model that checks each answer against its sources. **Question codes**: P product, L policy, C customer, A adversarial, H held-out.
