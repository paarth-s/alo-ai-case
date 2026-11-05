# Failure Analysis

**What was analysed.** The committed fresh run (`evals/results/`, `evals/baseline.json`): 61 questions (36 dev, 25 held-out), 7 of which did not pass strictly. `evals/harness/attribution.py` assigns each to the first stage that explains it: 2 retrieval (C10, H24) and 5 generation (A5, C6, H15, L6, P6). Replies are quoted from that run, trimmed with "…".

**How the three were chosen.** I compared the run with three more uncached runs. Three questions failed strict pass in all four: **H24, H15 and A5**. They are the same three as before the product-index removal, so that change did not move the failure modes. The rest are intermittent (table below). The script's severity ranking lists P6, L6 and H24, but P6 and L6 failed in only 2 of 4 runs and their replies are substantially right, so I rank by stability and impact.

---

## 1. H24: "Can I send back the leggings from my first order?" (high severity, fails every run)

**Customer:** Sofia (CUST-10103, VIP). **Reference:** "No. Her first order was placed Oct 14, 2024, far outside the 30-day window (as of Mar 1 it was 138 days ago)."

**Reply** (status `partial`):
> I can't confirm whether you can still return the leggings, because the sources I have don't include the return window or the eligibility rules for online returns. … It was an online order placed 2024-10-14 and delivered 2024-10-19, which was 133 days ago. …

The judge marked it incorrect: the answer is a clear "no" and the reply never says it.

**Trace.**
- *Router:* correct (`customer` + `policy`).
- *Retrieval:* the 30-day rule, `returns§1`, missed the top 10. The six excerpts shown were `returns§2.2`, `returns§8`, `returns§4.1`, `shipping§6`, `shipping§4`, `shipping§5`. Gold-in-context: 0.
- *Generation:* correct given what it saw; it computed the 133 days and said the window was missing.

**Root cause.** The question says "send back"; the policy says "return". No shared word, so keyword search cannot help, and "order" pulled in shipping chunks. Embeddings also ranked other returns chunks higher. C6 loses the same `returns§1`.

**Why the evals did not prevent it.** They found it and have flagged it since early on. It has survived every retrieval change: chunk merges, hybrid search, two rerankers and LLM-generated chunk questions (the last three were tried and removed; numbers in `evals/LOG.md` E3, E7, E10).

**Fix.** Keep a small, always-present "policy essentials" source (return window, Final Sale rule, about 150 tokens) built from the policy files so it cannot go stale. Add a query-rewrite retry when the reply is `not_in_data` or `partial` on a return question.

## 2. H15: "Do students get a discount?" (medium severity, fails every run)

**Reference:** "Yes: verified students get 15% off full-price items through Student Beans."

**Reply** (status `answered`):
> Yes. Verified students get 15% off full-price items. You verify your student status through Student Beans. Two details in our policies don't fully match, so I can't give you a definite answer on either: **Use limit** … **Stacking** … Check the terms when you verify through Student Beans to see which applies to you.

The judge scored it `partially_correct`: right headline, buried under hedging.

**Trace.** Router and retrieval were correct (`loyalty§9` and `loyalty§2` in context). Generation noticed the two sections disagree and said so.

**Root cause.** The policy contradicts itself (stackable vs not, one-time vs ongoing), and the prompt tells the model to flag conflicts. That is the right instinct; the miss is leading with the hedge. In earlier runs it also hedged on the one-time-use rule, which the sources do state, and the judge logged false absences.

**Fix.** Lead with the answer, hedge only on the point in conflict, and put the conflict in `data_issues`. The real fix is upstream: send the contradiction to the policy owner.

## 3. A5: "Is the Warrior Mat safe for latex allergies?" (low severity, fails every run)

**Reference:** "…natural rubber base… The catalog has no allergen information, so safety for latex allergies cannot be confirmed; the customer should consult a medical professional."

**Reply** (status `answered`):
> The Warrior Mat may not be safe for a latex allergy. Its base is natural rubber, and the catalog doesn't say whether it is latex-free or hypoallergenic. The top layer is PU leather. Please check with your doctor before buying. I don't have any allergen or latex-safety information beyond the material listing.

In all four runs the reply is hedged and refers the customer to a professional. It still fails:
- **Status:** it labels itself `answered`; the right label is `partial` or `not_in_data`. All four runs.
- **A check that misfires:** the question forbids "hypoallergenic", and the reply uses it in a negation. A string match cannot tell. Fixable in the check.
- **A general-knowledge claim:** in one run the judge flagged "natural rubber is a source of latex" as not in the catalog.

**Fix.** A rule for health questions: state the material, say the sources don't cover allergens, set `partial`, refer to a professional, give no verdict. Make the forbidden-phrase check look for an affirmative claim.
