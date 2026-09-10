"""Builds the golden evaluation set: samples real customer messages for a human
to hand-label with true intent, a reference-quality reply, and routing decision.

Sampling strategy (documented here so the report's "how sampled/labelled" section
can just point at this file):
  1. Pull all resolved threads for the brand.
  2. Run the (untuned) LLM classifier once to get a rough intent bucket for each.
  3. Stratified-sample roughly equally across the 8 intent buckets, so rare
     intents (e.g. account_access) aren't drowned out by common ones
     (order_status). This avoids a golden set that's 70% order_status and
     therefore useless for judging the other 7 intents.
  4. Add a small extra slice of the longest/most-punctuation-heavy messages —
     a cheap proxy for "angry/complex" messages, to make sure escalation logic
     gets tested against the hard cases too.

Output: data/golden/golden_set.csv with columns:
  example_id, text, model_predicted_intent (for reference only, NOT ground truth)
  human_intent, human_reference_reply, human_routing_decision, human_notes
The last four columns are left BLANK for a human to fill in by hand.
"""
import argparse
import csv
import json
import random

from tqdm import tqdm

from src import config
from src.intents import classify_intent

random.seed(42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", default=config.BRAND)
    ap.add_argument("--n_total", type=int, default=200)
    args = ap.parse_args()

    threads_path = config.DATA_PROCESSED / f"threads_{args.brand}.jsonl"
    threads = [json.loads(l) for l in open(threads_path)]
    random.shuffle(threads)

    # Cap how many we classify to keep this script fast/cheap; classifying the
    # full pool is unnecessary just to stratify a sample.
    pool = threads[: max(args.n_total * 6, 1200)]

    buckets = {label: [] for label in config.INTENT_LABELS}
    for t in tqdm(pool, desc="rough-classifying pool for stratification"):
        try:
            pred = classify_intent(t["customer_text"])
        except Exception:
            continue
        buckets[pred["intent"]].append(t)

    per_bucket = args.n_total // len(config.INTENT_LABELS)
    sampled = []
    for label, items in buckets.items():
        sampled.extend(items[:per_bucket])

    # Top up to n_total with a few "hard" examples: long + punctuation-heavy,
    # a cheap proxy for frustrated/complex messages.
    remaining = [t for t in pool if t not in sampled]
    remaining.sort(key=lambda t: (t["customer_text"].count("!") + t["customer_text"].count("?"),
                                   len(t["customer_text"])), reverse=True)
    needed = args.n_total - len(sampled)
    sampled.extend(remaining[:needed])

    out_path = config.DATA_GOLDEN / "golden_set.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "example_id", "text", "model_predicted_intent_REFERENCE_ONLY",
            "human_intent", "human_reference_reply", "human_routing_decision", "human_notes",
        ])
        for i, t in enumerate(sampled):
            try:
                ref_intent = classify_intent(t["customer_text"])["intent"]
            except Exception:
                ref_intent = ""
            writer.writerow([f"ex_{i:04d}", t["customer_text"], ref_intent, "", "", "", ""])

    print(f"Wrote {len(sampled)} candidate examples -> {out_path}")
    print("Next: open this CSV and fill in human_intent / human_reference_reply / "
          "human_routing_decision / human_notes by hand for each row.")


if __name__ == "__main__":
    main()
