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

- **Policies** (Markdown) are split at heading boundaries into 45 chunks and indexed in FAISS with local `bge-small-en-v1.5` embeddings.
- **Products** (JSON, 24 products plus 6 fabrics) are not searched; the model calls exact lookup tools, so lists and comparisons are complete.
- **Customers** (JSON, 8 synthetic) are loaded for the signed-in session only; durations are computed in code, the policy supplies the window.
- **Updates** re-embed only the chunks whose text changed (a 3-file update touched 4 of 75 chunks); policies are versioned by effective date.

## Setup (about 5 minutes, one model download)

Requires Python 3.10+ and an Anthropic API key.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then put your key in ANTHROPIC_API_KEY=
python -m src.build_index       # ~10 s; downloads the 130 MB embedding model on first use
```

## Try it

```bash
python -m src.generate "Which leggings have no side seams?"
python -m src.generate "Can I return the leggings I bought during Cyber Monday?" --customer CUST-10254
python -m src.generate "What did Jordan Kim order last month?" --customer CUST-10041     # declines: not your account
python -m src.generate "How much does Lululemon's Align legging cost?"                   # declines: out of scope
```

Sample customers: `CUST-10041` Maya (Black Friday Final Sale), `CUST-10082` Jordan (All Access), `CUST-10103` Sofia (VIP), `CUST-10177` Ethan (A-List, pending return), `CUST-10254` Priya (Cyber Monday Final Sale), `CUST-10312` Lena (rejected return), `CUST-10389` Marcus, `CUST-10421` Chloe (in transit). The demo's "today" is fixed at 2025-03-01 to match the data.

## Run the whole eval suite (about 3 minutes, about $1.50)

```bash
python -m evals.run_all            # all stages, compared with evals/baseline.json; exit code 1 = regression
python -m evals.run_all --fresh    # ignore every cache: a true re-measurement with real latency and cost
python -m evals.run_all --generator claude-haiku-5-5     # try another answer model (also --judge-model, --router-model)
```

It prints one table of 42 metrics with the baseline, the new value, the tolerance and a status. Tolerances are twice the run-to-run spread I measured, so an occasional single `WORSE` on a noisy metric is chance; re-run before believing it. Hard gates (zero tolerance): missed router domains, citation validity, cross-customer leaks, judge failures, staleness checks.

Individual pieces:

```bash
python -m evals.retrieval_eval --mode hybrid [--set heldout]   # Recall@1/5/10, MRR, NDCG, with a random-ranking row
python -m evals.generation_eval [--set heldout]                # faithfulness, hallucination, 10 pitfall checks, latency, cost
python -m evals.staleness_eval                                 # policy / catalog / version updates, in a scratch copy
python -m evals.judge_controls run                             # does the judge catch errors planted on purpose?
python -m evals.attribution                                    # which stage caused each failure
python -m src.build_index --check                              # is the index stale? (exit 1 if so)
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

Latency (uncached, end to end): p50 5.5 s, p95 9.1 s. Cost: about $0.015 per question. Staleness: 24 of 24 checks.

## Where it is weak (honest list)

- **Retrieval vocabulary gaps.** "Send back" does not find the "return" rule (H24). This is the largest known quality gap.
- **The judge misses omissions.** It catches 44 of 50 planted errors but 0 of 5 dropped caveats; deterministic checks cover the Final Sale cases.
- **Small data.** 75 chunks and 61 hand-written questions. Hybrid search and reranking cannot show their value at this size, and several comparisons are within noise (the docs say so each time).
- **No real traffic.** All questions were written by me.

## Layout

```
src/        router, ingest (chunking, versioning), embeddings, store (FAISS), retrieve, catalog tools, customers, generate
data/       the policies, product catalog and customer orders
evals/      questions, every eval script, results/, calibration/ (human + blind labels, planted errors), baseline.json
docs/       ADR, production memo, requirements checklist, review page
```
