# Failure Analysis

**What was analysed.** The final uncached run: 61 questions (36 dev, 25 held-out), 6 of which did not pass cleanly. `evals/harness/attribution.py` assigns each failure to the first stage that explains it (router, then retrieval, then generation, then the eval itself) and ranks them by severity. The three below are the top three by that ranking, not hand-picked. I rate their real severity myself, because the ranking treats any unsupported claim as serious.

| Stage | Failures | Which |
|---|---|---|
| Router | 0 | |
| Retrieval | 2 | H24, C10 |
| Generation | 3 | A5, L9, H18 |
| Eval-side | 1 | L6 (reply was fine; only the expected status label disagreed) |

**Read this with the noise in mind.** Identical runs disagree on about 1 in 5 dev questions, so the list of failures changes from run to run. Across the three uncached noise runs, H24 and A5 failed every time (stable), C10 flipped between pass and fail, and L9 and H18 passed all three times and failed only in the final run (intermittent).

---

## 1. H24: "Can I send back the leggings from my first order?" (high severity, stable)

**What happened.** The customer is Sofia (VIP). Her first order was 138 days ago, so the correct answer is "no, the 30-day window has passed." The assistant answered `not_in_data`: it could not confirm the return window and gave no yes or no. The judge marked it incorrect.

**Trace.**
- *Router:* correct (`customer` + `policy`).
- *Retrieval:* the rule that holds the 30-day window, `returns§1`, ranked between 11th and 20th. The six chunks shown to the model were three returns chunks about other topics (in-store, cancellation, exchange) and three shipping chunks (lost packages, tracking, address changes). The rule was in the index and never reached the model.
- *Generation:* correct given what it saw. It reported the order and the 133 days since delivery from the customer record, then said honestly that the window was missing from its sources.

**Root cause.** The question says "send back", and the policy says "return". There is no word overlap, so the keyword half of hybrid search could not help, and the word "order" pulled in shipping chunks that do match. The embedding half also ranked other returns chunks higher.

**Why the evals did not prevent it.** They found it. The weakness of `returns§1` has been in the log since the first baseline (E1), and H24 failed all three uncached noise runs. It has survived every retrieval change: chunk merges, a title-prefix test, hybrid search, two rerankers, and LLM-generated chunk questions.

**Fix I would make.** Do not rely on retrieval for the few rules every return question needs. Keep a small, always-present "policy essentials" source (the return window and the Final Sale rule, about 150 tokens), built from the policy files so it cannot go stale. In parallel, add a query-rewrite retry whenever the reply is `not_in_data`.

## 2. A5: "Is the Warrior Mat safe for latex allergies?" (medium severity, stable)

**What happened.** The reply opens with "The Warrior Mat is not a safe choice if you have a latex allergy." The catalog says only that the base is natural rubber, and says nothing about allergens. The rest of the reply is careful: it says the catalog does not say, and that the customer should ask a doctor.

**Trace.** Router and retrieval were correct (one `get_product` call returned the right record). The generation step turned "natural rubber" into a safety verdict using general knowledge. The prompt tells the model never to do that, and the safety instinct won anyway. The status was `answered`; the right status was `partial` or `not_in_data`.

**Root cause.** A health question pulls the model toward a protective answer, and a protective answer here is an unsupported fact. A human reviewer agreed that pointing to a medical professional is right; the problem is only the confident first sentence.

**Why the evals caught it unevenly.** It failed all three noise runs, always because the status was `answered`, but the judge flagged the overclaim itself in some runs and passed it in others, which is its measured noise (4 of 58 items change verdict between repeats).

**Fix.** Add one explicit rule for health and safety questions: state the material, say the sources do not cover allergens, refer to a medical professional, and never give a safe or unsafe verdict. Add a deterministic must-not-include check for "not a safe choice".

## 3. L9: "Can I return an item that was 25% off?" (low severity despite the ranking; intermittent)

**What happened.** The reply gets the conclusion right ("yes, in general") and lists every exclusion correctly, including bodysuits, intimates, gift cards and third-party items. But its second sentence says "Only items discounted 30% or more at the time of purchase are final sale," which the policy contradicts and which the reply itself contradicts two paragraphs later.

**Trace.** Router, retrieval and context were all correct (`returns§1.2` and `returns§6`, the exact sections, were in context). This is a pure generation slip: the model compressed a list of five final-sale rules into "only".

**Root cause.** The question is about a discount, so the model framed the answer around the discount threshold and over-generalised.

**Why it matters less than it looks.** The customer is not misled, because the exclusions follow immediately. It is still a real self-contradiction, and in a different sentence order it could matter, so the judge was right to flag it.

**Fix.** A prompt rule against "only" and "always" when summarising a list, and a regression case for it. I would not spend more effort here.

---

## Also failing

| Question | Stage | What |
|---|---|---|
| C10 (points and 2-day shipping) | Retrieval | `loyalty§6.2` (no redemption during sales) missing; the reply was correct but incomplete. Flips between runs |
| H18 (points on shipping fees or gift cards) | Generation | Correct "no", missing the full list of exclusions. Passed all three noise runs, so intermittent |
| L6 (UK return) | Eval | The reply was good; the status label `partial` vs expected `answered` is arguable |

## Patterns

1. **Retrieval, not generation, is the weak link for the failure that matters most.** H24 is a stable retrieval miss, and the model handles it honestly (`not_in_data`) instead of inventing an answer. Safe, but unhelpful.
2. **The model's errors are small wording slips**, not fabricated facts: an "only", a safety verdict. Faithfulness is 0.99.
3. **The judge is better at additions than omissions.** It caught 44 of 50 planted errors but 0 of 5 dropped exceptions, so a reply that quietly leaves out a caveat can pass. Deterministic checks protect the Final Sale cases.

## What this eval cannot see

- Real customer phrasing: all 61 questions were written by me, and H24 shows how much that matters.
- Anything about scale: the corpus is 75 chunks.
- Whether an omitted caveat would have mattered to a real customer.
- Multi-turn conversations, which are out of scope.
