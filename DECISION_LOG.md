# Decision Log

Plain list of non-obvious decisions and why. Add/edit as you actually build —
these are seeded with the reasoning baked into the starter code; replace
anything you change.

1. **Subsample, not full 3M tweets.** Reconstructing threads for one brand
   (~4k resolved threads) is enough to ground replies and is what the
   assignment explicitly expects ("a subsample is encouraged"). Full-dataset
   processing would blow the 15-minute reproduction budget for no eval benefit.

2. **One config.py and one llm_client.py, everything else is a thin function.**
   Keeps model names, thresholds, and retry logic in exactly one place each, so
   changing e.g. the classifier model or a routing threshold is a one-line diff
   you can point to live instead of hunting through files.

3. **No fine-tuning.** Prompting + retrieval on an off-the-shelf model gets a
   working, explainable v1 in the assignment's timeframe. Fine-tuning needs
   more labelled data than 150-250 golden examples can responsibly provide,
   and would make the "why did it say that" question in a live review much
   harder to answer.

4. **8 hand-authored intents, not an LLM-discovered taxonomy, not Banking77's 77.**
   Read ~150 raw AmazonHelp threads by hand first and clustered by eye. A fixed,
   human-legible taxonomy keeps intent labels stable across pipeline runs
   (LLM-discovered clusters drift run to run) and interpretable in the report's
   per-class F1 table. 77 intents (Banking77) is built for a different domain
   (banking) and would badly over-fragment Twitter support traffic.

5. **"Resolved thread" = customer's first tweet + brand's first reply**, not
   full multi-turn reconstruction. Twitter support threads branch and loop
   (customer re-tweets, brand asks clarifying questions, etc.); modelling that
   properly is a project on its own. First-reply pairs still capture "how did
   this brand actually respond to this kind of issue," which is what grounding
   needs.

6. **TF-IDF retrieval, not a neural embedding model.** No extra model
   download/dependency, fully deterministic (same query -> same neighbors
   every run, useful for the eval harness), fast enough for a few thousand
   threads, and easy to explain and inspect live ("here's literally the
   vocabulary overlap it matched on").

7. **Retrieval-similarity auto-handle threshold (0.18) picked empirically, not
   theoretically.** Set by looking at the similarity-score distribution over a
   sample of matches and eyeballing where "genuinely similar precedent" stops
   and "vaguely related tweet" starts. Documented as a knob in config.py
   specifically so it can be re-tuned against the golden set without touching
   logic.

8. **Routing is rule-based on top of model outputs, not an LLM "decide
   auto/escalate" call.** An auditable if/else chain over
   (intent risk class, classifier confidence, retrieval similarity,
   groundedness flag, safety keyword list) means every escalation has a
   reason a human can check, and the policy can be tightened for one intent
   without retraining/re-prompting anything.

9. **Two-tier model choice (Haiku for classify/judge, Sonnet for drafting).**
   Classification and judging are narrower, higher-volume tasks; drafting
   customer-facing text benefits more from a stronger model. Keeps the
   15-minute reproduction budget realistic.

10. **Force-escalate keyword list is a hard override, checked first.** Legal
    threats, fraud claims, and self-harm mentions bypass every other signal —
    intentionally conservative, since the cost of missing one of these is far
    higher than the cost of an unnecessary escalation.

11. **Golden set is stratified by (rough) intent, not randomly sampled.**
    Random sampling from real traffic would be dominated by order_status and
    give almost no signal on rarer intents like account_access — see Report
    section 4 for the tradeoff this creates.

12. **"Unsafe auto-handle rate" reported separately from overall routing
    accuracy.** A false auto-handle (should have escalated) and a false
    escalate (should have auto-handled) are not equally bad; collapsing them
    into one accuracy number would hide the metric that actually matters for
    trusting the agent in production.

13. **LLM-judge only scores a subsample (default 40), not the whole golden
    set.** Judging every example on every eval run is slow and costly for
    marginal signal once the sample is large enough to be stable; the
    subsample size is a CLI flag so it can be scaled up before a final run.

14. **Grounding is a model self-report (`is_grounded`), checked, not assumed.**
    The generation prompt requires the model to name which retrieved example(s)
    it actually used and to say explicitly if none fit — this is what lets
    the router catch "confidently made something up" cases instead of trusting
    fluent text at face value.
