# Failure Analysis

**What was analysed.** The committed fresh run (`evals/results/`, `evals/baseline.json`): 61 questions (36 dev, 25 held-out), of which 9 did not pass strictly. `evals/harness/attribution.py` assigns each to the first stage that explains it: 2 retrieval (C10, H24), 6 generation (A5, C5, C6, H15, H18, H19) and 1 eval-side (L6). Replies below are quoted from that run, trimmed with "…".

**How the three were chosen.** I compared the committed run with three more uncached runs. Three questions failed strict pass in all four: **H24, H15 and A5**. The rest are intermittent (table at the end). The attribution script's own severity ranking is tied between four questions and lists C5, A5 and H19, but C5 and H19 failed in only 2 and 1 of 4 runs, and C5's reply is correct, so I rank by stability and impact instead.

---

## 1. H24: "Can I send back the leggings from my first order?" (high severity, fails every run)

**Customer:** Sofia (CUST-10103, VIP). **Reference answer:** "No. Her first order was placed Oct 14, 2024, far outside the 30-day window (138 days ago)."

**What the assistant said** (status `not_in_data`):
> I can't confirm whether the leggings can still be returned, because the policy excerpts I have don't state the return window or eligibility rules for online orders. … It was delivered on 2024-10-19, which was 133 days before today (2025-03-01). … Without the return window, I can't say whether 133 days is inside or outside it.

The judge marked it incorrect: the answer is a clear "no" and the reply never says it.

**Trace.**
- *Router:* correct (`customer` + `policy`).
- *Retrieval:* the rule with the 30-day window, `returns§1`, did not reach the top 10. The six excerpts shown were `returns§2.2`, `returns§8`, `returns§4.1`, `shipping§6`, `shipping§4` and `shipping§5`: returns chunks on other topics and three shipping chunks. Gold-in-context: 0.
- *Generation:* correct given what it saw. It computed the 133 days from the customer record and said honestly that the window was missing.

**Root cause.** The question says "send back"; the policy says "return". With no shared word, the keyword half of hybrid search cannot help, and "order" pulled in shipping chunks. The embedding half also ranked other returns chunks higher.

**Why the evals did not prevent it.** They found it and have flagged it since the first baseline (LOG E1). It has survived every retrieval change: chunk merges, hybrid search, two rerankers and LLM-generated chunk questions (the last three were tried and removed; numbers in `evals/LOG.md` E3, E7, E10).

**Fix I would make.** Do not rely on retrieval for the few rules every return question needs: keep a small, always-present "policy essentials" source (return window, Final Sale rule, about 150 tokens) built from the policy files so it cannot go stale. Add a query-rewrite retry whenever the reply is `not_in_data`.

## 2. H15: "Do students get a discount?" (medium severity, fails every run)

**Reference answer:** "Yes: verified students get 15% off full-price items through Student Beans."

**What the assistant said** (status `answered`):
> Yes. Students verified through Student Beans get 15% off full-price items. Two of our sources describe it a little differently: [Community Discounts] … can't be stacked with other promotions or Alo Access redemptions. [How to Join] … says the 15% is on top of standard Access benefits. It also says it is one-time use per student account. Because of this difference, I can't say for certain whether the discount can be used more than once.

The judge scored it `partially_correct`: the conclusion is right but buried, and the "can't say for certain" statements are false absences, because the one-time-use rule is stated in the sources.

**Trace.** Router and retrieval were correct (`loyalty§9` and `loyalty§2` were both in context). Generation noticed that the two sections disagree about stacking and hedged on everything, including the one thing both sections agree on.

**Root cause.** The policy text contradicts itself, and the prompt tells the model to flag conflicts. That is the right instinct, applied too broadly: it doubted facts that were not in dispute.

**Why the evals caught it.** The judge's false-absence check is built for exactly this. Whether hedging on a real conflict should count as a failure is arguable, and I kept the strict reading.

**Fix.** Tell the model to lead with the answer, hedge only on the specific point in conflict, and put the conflict in `data_issues`. The real fix is upstream: send the contradiction to whoever owns the policy.

## 3. A5: "Is the Warrior Mat safe for latex allergies?" (low severity, fails every run)

**Reference answer:** "…natural rubber base… The catalog has no allergen information, so safety for latex allergies cannot be confirmed; the customer should consult a medical professional."

**What the assistant said** (status `answered`):
> The Warrior Mat may not be safe for a latex allergy. Its base is natural rubber, and the catalog doesn't say whether it's latex-free or hypoallergenic. … Natural rubber is a source of latex, but the catalog doesn't address allergen safety, so I can't confirm it's safe. If your allergy is significant, check with a medical professional before buying.

Across the four runs the reply is always hedged and always sends the customer to a professional. It still fails every time for one or two reasons:
- **Status:** the reply labels itself `answered`; the right label is `partial` or `not_in_data`. All four runs.
- **A check that misfires:** the question forbids the word "hypoallergenic", and this reply uses it inside a negation ("doesn't say whether it's … hypoallergenic"). A string match cannot tell the difference. Fixable in the check.
- **A general-knowledge claim:** "natural rubber is a source of latex" is not in the catalog. The judge flagged it in two runs, and it is the reason a health question is the hardest kind to keep grounded.

An earlier analysis of mine described a reply that opened "not a safe choice"; the current prompt no longer produces that, so I rewrote this entry.

**Fix.** One explicit rule for health and safety questions: state the material, say the sources do not cover allergens, set status `partial`, refer to a medical professional, and give no verdict. Change the forbidden-phrase check to look for an affirmative claim.

---

## The other six (intermittent, in order of how often they failed)

| ID | Stage | Failed | Note |
|---|---|---|---|
| C10 | retrieval | 3 of 4 | see `evals/results/attribution.json` |
| C5 | generation | 2 of 4 | **Judge false alarm.** The reply is correct (return rejected: 39 days from purchase, 34 from delivery, outside the 30-day window) and even catches an inconsistency in the customer's record. The judge marked its quote of that record "contradicted". |
| C6 | generation | 2 of 4 | not analysed in depth |
| L6 | eval | 2 of 4 | an eval-side issue, not a system failure |
| H19 | generation | 1 of 4 | Reply is right (contact channels, 30 days, photos, refund or replacement) but adds "that rule doesn't apply to an item that arrived damaged" as policy, which the source implies but does not state. Mild. |
| H18 | generation | 1 of 4 | not analysed in depth |

L9 ("25% off") appeared in my earlier analysis. It passes in this run, though its reply still says final sale applies "only" at 30% or more, a sloppy word that the judge noted and let through.

## Patterns

1. **The one failure that matters most is a retrieval miss.** H24 is stable, and the model handles it honestly (`not_in_data`) instead of inventing an answer. Safe, but unhelpful.
2. **The model's errors are mostly hedging and labelling, not invention.** H15 hedges too much, A5 mislabels its own status, H19 states an inference as a rule. Faithfulness is 0.99.
3. **Some "failures" are the eval's own noise.** C5 and the A5 phrase check are false alarms. The judge flips on 4 of 58 repeated items, and the judge is better at additions than omissions (44 to 47 of 50 planted errors caught, 0 to 2 of 5 dropped exceptions).

## What this eval cannot see

- Real customer phrasing: all 61 questions were written by me, and H24 shows how much that matters.
- Anything about scale: the corpus is 75 chunks and 8 customers.
- Whether an omitted caveat would have mattered to a real customer.
- Multi-turn conversations, which are out of scope.
