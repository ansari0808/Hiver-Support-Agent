# Report: AI Support Agent for @AmazonHelp

*(Fill in the [bracketed] parts after running the pipeline + eval. Keep to ≤6 pages.)*

## 1. Problem framing

- What "good" means for this brand: [e.g. "a reply is good if it would not need
  correction by a human agent before sending, and a routing decision is good if
  it never auto-sends something that needed a human."]
- What I chose NOT to build, and why:
  - No fine-tuning — [reason, see Decision Log #3]
  - No full multi-turn thread modelling — [reason, see Decision Log #5]
  - No neural embedding retrieval — [reason, see Decision Log #6]
  - [anything else you scoped out]

## 2. Results vs. baselines

[Paste the tables from `data/processed/eval_report.md` here.]

Headline number: **[e.g. "Agent gets 0.81 macro-F1 on intent vs 0.42 for the
simple baseline, and a 0.02 unsafe-auto-handle rate vs 0.11 for simple."]**

## 3. Failure analysis — top 5 failure modes

For each: 1-2 real examples (input, agent output, what a human would have done)
and a hypothesis for why it happens.

1. **[Failure mode name]** — example: [...] — hypothesis: [...]
2. ...
3. ...
4. ...
5. ...

## 4. What is misleading about my headline number (mandatory)

[This section exists because every eval has a way of looking better than it
is. Be specific. E.g.:
- The golden set is stratified by intent, so it over-represents rare intents
  relative to real traffic — real-world accuracy will be pulled toward
  whatever the majority intent (order_status) scores.
- TF-IDF retrieval favors examples that share literal words with the query;
  it will look "grounded" even when it retrieved a superficially similar but
  substantively different precedent — the LLM judge may not catch this either
  since it only sees what was retrieved, not whether better precedents existed.
- The LLM-judge sample (n=40) may not be representative of the full golden set.
- "Unsafe auto-handle rate" is measured against a hand-labelled ground truth
  that itself has only 1 labeller's judgment for ambiguous cases.]

## 5. What I'd do next with one more week

- [e.g. reconstruct true multi-turn threads instead of first-reply-only]
- [e.g. add a second human labeller + inter-annotator agreement on the golden set]
- [e.g. replace TF-IDF with a real embedding model once network/deps allow it]
- [e.g. active-learning loop: route the agent's low-confidence/escalated cases
  back into the golden set to keep it representative of real failure modes]
