# AI Support Agent — AmazonHelp (Customer Support on Twitter)

An AI agent that (1) classifies incoming customer tweets into intents, (2) drafts a
reply grounded in how the brand has historically resolved similar issues (RAG over
past resolved threads), and (3) decides auto-handle vs. escalate-to-human with a
stated reason.

Brand chosen: **@AmazonHelp** — highest volume brand in the dataset, wide intent
variety (orders, refunds, account, delivery, app bugs), and enough resolved
multi-turn threads to ground replies on.

## Why this structure

- `src/` — the agent pipeline (data prep → intent → retrieval → reply → routing).
- `eval/` — everything needed to prove the agent works: baselines, metrics,
  LLM-as-judge, judge-vs-human agreement.
- `data/golden/` — the 150–250 hand-labelled examples + labelling notes.
- `report/` — the write-up (framing, baselines, failure analysis, "what's misleading
  about my headline number", next steps).
- `DECISION_LOG.md` — the 10–15 non-obvious calls, in one place.

## Setup (5 min)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
```

## Get the data (one-time, not included in repo)

1. Download `twcs.csv` from Kaggle: `thoughtvector/customer-support-on-twitter`.
2. Place it at `data/raw/twcs.csv`.
3. Build the brand-filtered, thread-reconstructed subsample:

```bash
python -m src.data_prep --brand AmazonHelp --max_threads 4000
# writes data/processed/threads_AmazonHelp.jsonl
```

We deliberately subsample (a few thousand threads, not 3M tweets) — see
`DECISION_LOG.md` item 1 for why, and the assignment explicitly expects this.

## Reproduce headline results (< 15 min)

```bash
bash scripts/run_pipeline.sh      # runs agent over golden set, ~5-8 min for 200 examples
bash scripts/run_eval.sh          # metrics + LLM-judge + baselines, ~3-5 min
```

Outputs land in `data/processed/`:
- `predictions.jsonl` — per-example intent, drafted reply, routing decision + reason.
- `eval_report.md` — metrics table, baseline comparison, judge-agreement numbers.

## Pipeline at a glance

```
customer tweet
      │
      ▼
 intent classifier (few-shot LLM, fixed taxonomy)  ──┐
      │                                              │
      ▼                                              │
 retrieval: TF-IDF over historically-resolved        │
 threads for this brand+intent → top-k examples      │
      │                                              │
      ▼                                              │
 reply generator: LLM drafts reply grounded in        │
 retrieved resolutions, cites which one it followed   │
      │                                              │
      ▼                                              │
 router: auto-handle vs escalate, with reason ◄──────┘
 (based on intent risk class, retrieval confidence,
  and safety/urgency signals in the message)
```

## What this is NOT

- Not a fine-tuned model — everything is prompting + retrieval on top of an off-the-shelf
  LLM (Claude). See Decision Log #3 for why, and the Report's "what I chose not to build."
- Not evaluated on the full 3M-tweet dataset — a stratified subsample + a hand-labelled
  golden set is used throughout, per the assignment's own guidance.

## Repo map

```
src/
  config.py            paths, brand, intent taxonomy, model names
  llm_client.py         thin wrapper around the Anthropic API
  data_prep.py           twcs.csv -> cleaned, thread-reconstructed JSONL for one brand
  intents.py              intent taxonomy + few-shot LLM classifier
  retrieval.py            TF-IDF index over historically-resolved threads
  reply_generator.py      RAG prompt + generation, returns reply + grounding evidence
  router.py               auto-handle / escalate decision + reason
  pipeline.py              orchestrates the above end to end
eval/
  build_golden_set.py     stratified sampling script -> CSV for human labelling
  baselines.py             trivial baseline + simple (keyword) baseline
  metrics.py               intent accuracy/F1, routing accuracy, groundedness rate
  llm_judge.py              LLM-as-judge rubric for reply quality
  judge_agreement.py       correlation of judge scores vs human scores
scripts/
  run_pipeline.sh
  run_eval.sh
report/REPORT_TEMPLATE.md
DECISION_LOG.md
```
<img width="1512" height="982" alt="Screenshot 2026-09-10 at 10 10 19 PM" src="https://github.com/user-attachments/assets/d78cee80-d68c-4ab3-aaac-4f5d4e664b83" />
