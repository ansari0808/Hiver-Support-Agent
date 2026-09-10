"""End-to-end: message in -> {intent, reply, routing decision} out.

Run over the golden set:
    python -m src.pipeline --input data/golden/golden_set.csv --output data/processed/predictions.jsonl
"""
import argparse
import csv
import json

from tqdm import tqdm

from src import config, router
from src.intents import classify_intent
from src.reply_generator import generate_reply
from src.retrieval import load_retriever


def run_one(text: str, retriever) -> dict:
    intent_result = classify_intent(text)
    examples = retriever.retrieve(text, k=config.TOP_K_RETRIEVAL)
    top_sim = examples[0]["similarity"] if examples else 0.0

    reply_result = generate_reply(text, intent_result["intent"], examples)

    routing = router.route(
        text=text,
        intent=intent_result["intent"],
        intent_confidence=intent_result.get("confidence", 0.5),
        top_retrieval_similarity=top_sim,
        is_grounded=reply_result.get("is_grounded", False),
    )

    return {
        "input_text": text,
        "intent": intent_result["intent"],
        "intent_confidence": intent_result.get("confidence"),
        "intent_rationale": intent_result.get("rationale"),
        "retrieved_examples": [
            {"customer_text": e["customer_text"], "brand_reply_text": e["brand_reply_text"], "similarity": e["similarity"]}
            for e in examples
        ],
        "draft_reply": reply_result.get("reply"),
        "is_grounded": reply_result.get("is_grounded"),
        "grounding_notes": reply_result.get("notes"),
        "routing_decision": routing["decision"],
        "routing_reason": routing["reason"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with a 'text' column (e.g. golden set)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--brand", default=config.BRAND)
    args = ap.parse_args()

    retriever = load_retriever(args.brand)

    rows = list(csv.DictReader(open(args.input)))
    out_f = open(args.output, "w")
    for row in tqdm(rows, desc="running pipeline"):
        result = run_one(row["text"], retriever)
        result["example_id"] = row.get("example_id", "")
        out_f.write(json.dumps(result) + "\n")
    out_f.close()
    print(f"Wrote {len(rows)} predictions -> {args.output}")


if __name__ == "__main__":
    main()
