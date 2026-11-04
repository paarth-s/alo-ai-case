# Retail RAG Assistant

A grounded question-answering assistant over three retail knowledge domains: **product facts**, **policies** (returns, shipping, loyalty) and **a customer's own orders**. The emphasis is on measuring it honestly: a regression harness, a validated judge, and a written account of where it breaks.

| Read this | For |
|---|---|
| [docs/ADR.md](docs/ADR.md) | The five design questions: chunking, embeddings, retrieval, prompts, what I'd do next |
| [docs/PRODUCTION_MEMO.md](docs/PRODUCTION_MEMO.md) | Staleness, retrieval failure, latency, guardrails, observability |
| [evals/FAILURE_ANALYSIS.md](evals/FAILURE_ANALYSIS.md) | The three worst failures, traced end to end |
| [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) | Every requirement in the brief, answered with evidence |
| [docs/review.html](docs/review.html) | The same questions and answers, colour-coded for review (open in a browser) |
| [evals/LOG.md](evals/LOG.md) | The lab notebook: every experiment, including the ones that failed |

## How it works

```
question ──► router (Haiku) ──► which of: customer │ policy │ product   (none = out of scope: fixed decline)
                                   │          │          │
              session customer id ─┘          │          └─► catalog tools: get_product, search_products, get_fabric
              (never from the question)       └─► hybrid search: BM25 + bge-small, fused by rank (RRF),
                                                  over the policy version in force on the as-of date
                          all evidence ──► Sonnet 5.5, one structured reply: status, answer, citations, data_issues
```

- **Policies** (Markdown) are split at heading boundaries into 45 chunks and indexed in Chroma with local `bge-small-en-v1.5` embeddings.
- **Products** (JSON, 24 products plus 6 fabrics) are not searched; the model calls exact lookup tools, so lists and comparisons are complete.
- **Customers** (JSON, the 8 synthetic customers supplied with the case) are loaded for the signed-in session only; durations are computed in code, the policy supplies the window.
- **Updates** re-embed only the chunks whose text changed (a 3-file update touched 4 of 75 chunks); policies are versioned by effective date.

## Setup (about 5 minutes, one model download)

Requires Python 3.10+ and an Anthropic API key.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then put your key in ANTHROPIC_API_KEY=
python -m src.ingestion.build_index       # ~10 s; downloads the 130 MB embedding model on first use
```

## Try it

```bash
python -m src.generation.generate "Which leggings have no side seams?"
python -m src.generation.generate "Can I return the leggings I bought during Cyber Monday?" --customer CUST-10254
python -m src.generation.generate "What did Jordan Kim order last month?" --customer CUST-10041     # declines: not your account
python -m src.generation.generate "How much does Lululemon's Align legging cost?"                   # declines: out of scope
```

Sample customers: `CUST-10041` Maya (Black Friday Final Sale), `CUST-10082` Jordan (All Access), `CUST-10103` Sofia (VIP), `CUST-10177` Ethan (A-List, pending return), `CUST-10254` Priya (Cyber Monday Final Sale), `CUST-10312` Lena (rejected return), `CUST-10389` Marcus, `CUST-10421` Chloe (in transit). The demo's "today" is fixed at 2025-03-01 to match the data.

## Run the whole eval suite (about 3 minutes, about $1.50)

```bash
python -m evals.harness.run_all            # all stages, compared with evals/baseline.json; exit code 1 = regression
python -m evals.harness.run_all --fresh    # ignore every cache: a true re-measurement with real latency and cost
# A cached run only checks the plumbing and scoring; use --fresh to measure a new model or prompt.
python -m evals.harness.run_all --generator claude-haiku-5-5     # try another answer model (also --judge-model, --router-model)
```

It prints one table of 42 metrics with the baseline, the new value, the tolerance and a status. Tolerances are twice the run-to-run spread I measured, so an occasional single `WORSE` on a noisy metric is chance; re-run before believing it. Hard gates (zero tolerance): missed router domains, citation validity, cross-customer leaks, judge failures, staleness checks.

Individual pieces:

```bash
python -m evals.retrieval_eval --mode hybrid [--set heldout]   # Recall@1/5/10, MRR, NDCG, with a random-ranking row
python -m evals.generation_eval [--set heldout]                # faithfulness, hallucination, 10 pitfall checks, latency, cost
python -m evals.staleness_eval                                 # policy / catalog / version updates, in a scratch copy
python -m evals.judge.judge_controls run                       # does the judge catch errors planted on purpose?
python -m evals.harness.attribution                            # which stage caused each failure
python -m src.ingestion.build_index --check                    # is the index stale? (exit 1 if so)
```

## Results at a glance

61 questions: 36 dev, 25 held-out (22 policy, 19 product, 15 customer, 5 adversarial). Numbers are from one uncached run; identical runs move strict pass by about ±0.03.

| | Dev | Held-out |
|---|---|---|
| Routing: missed domains | 0 | 0 |
| Retrieval (hybrid): R@5 / MRR@10 | 0.93 / 0.90 | 0.96 / 0.83 |
| Faithfulness | 0.99 | 1.00 |
| Correctness vs reference | 0.97 | 0.94 |
| Strict pass (every check clean) | 0.89 | 0.92 |
| Cross-customer leaks / invalid citations | 0 / 0 | 0 / 0 |

Retrieval note: hybrid is the default for exact-term reasons, but on held-out questions dense-only scored slightly higher (R@5 1.00, MRR 0.88), so the gains come from routing and exact lookup tools, not from hybrid. Differences this small are within noise.

Latency (uncached, end to end, mean of three runs): p50 5.5 s, p95 9.1 s. Cost: about $0.015 per question. Staleness: 24 of 24 checks.

## Where it is weak (honest list)

- **Retrieval vocabulary gaps.** "Send back" does not find the "return" rule (H24). This is the largest known quality gap.
- **The judge misses omissions.** It catches 44 to 45 of 50 planted errors (two runs) but only 0 to 1 of 5 dropped caveats; deterministic checks cover the Final Sale cases.
- **Small data.** 75 chunks and 61 hand-written questions. Hybrid search cannot show its value at this size, and several comparisons are within noise (the docs say so each time).
- **No real traffic.** All questions were written by me.

## Layout

```
src/
  core/        shared utilities (disk cache)
  router/      question -> which sources are needed
  ingestion/   chunking and versioning, embeddings, Chroma store, build_index (incremental sync)
  retrieval/   BM25, hybrid retrieve
  context/     catalog tool calls, session-keyed customer lookup
  generation/  prompt, answer, citations (python -m src.generation.generate)
evals/
  pricing/     cost model
  harness/     run_all (regression driver), compare, noise, attribution
  judge/       LLM judge, calibration code, planted-error controls
  *_eval.py, test_router.py, stats.py, common.py   the eval scripts
  questions*.jsonl, baseline.json, tolerances.json, results/, calibration/ (labels, planted errors), LOG.md
data/       the policies, product catalog and customer orders
docs/       ADR, production memo, requirements checklist, review page
```

## Glossary

- **RAG**: retrieve relevant text first, then have the model answer from it.
- **Chunk**: one searchable piece of a document (a policy section or a product).
- **Embedding / dense search**: text turned into numbers so similar meanings sit close together.
- **BM25**: classic keyword search. **RRF**: a way to merge two ranked lists by rank.
- **Recall@K**: how often the right chunk is in the top K results. **MRR / NDCG**: scores that reward ranking the right chunk higher.
- **Faithfulness**: every claim in the answer is supported by the retrieved text. **Hallucination**: a claim that is not.
- **Strict pass**: all automatic checks and the judge are clean for that question.
- **Judge**: a second model that checks each answer against its sources.
- **Chroma**: the local vector database that stores and searches the vectors.
- **Question codes**: P = product, L = policy, C = customer, A = adversarial; H = held-out set.
