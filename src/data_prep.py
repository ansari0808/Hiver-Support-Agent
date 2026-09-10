"""Load the raw Kaggle 'Customer Support on Twitter' CSV, reconstruct
customer -> brand-agent threads, and write a clean, brand-filtered JSONL.

twcs.csv columns (Kaggle schema):
  tweet_id, author_id, inbound, created_at, text,
  response_tweet_id, in_response_to_tweet_id

`inbound == True`  -> tweet is FROM a customer TO a brand
`inbound == False` -> tweet is FROM the brand

A "resolved thread" here = a customer's first inbound tweet in a conversation
paired with the brand's first reply to it. This is a simplification (real
threads can be longer) — see DECISION_LOG.md #5 for why we chose this over
reconstructing full multi-turn threads for v1.
"""
import argparse
import json
import re

import pandas as pd
from tqdm import tqdm

from src import config


def clean_text(text: str) -> str:
    text = re.sub(r"@\w+", "", text)          # strip @mentions (incl. brand handle)
    text = re.sub(r"http\S+", "", text)        # strip links
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_brand_threads(brand: str, max_threads: int) -> list[dict]:
    df = pd.read_csv(config.RAW_TWCS_CSV, dtype={"tweet_id": str, "in_response_to_tweet_id": str,
                                                   "response_tweet_id": str, "author_id": str})
    df = df.set_index("tweet_id", drop=False)

    brand_replies = df[(df["author_id"] == brand) & (~df["inbound"])]
    brand_replies = brand_replies[brand_replies["in_response_to_tweet_id"].notna()]

    threads = []
    for _, reply_row in tqdm(brand_replies.iterrows(), total=len(brand_replies), desc=f"reconstructing {brand} threads"):
        cust_tweet_id = reply_row["in_response_to_tweet_id"]
        if cust_tweet_id not in df.index:
            continue
        cust_row = df.loc[cust_tweet_id]
        if isinstance(cust_row, pd.DataFrame):  # duplicate index edge case
            cust_row = cust_row.iloc[0]
        if not cust_row["inbound"]:
            continue  # the "customer" tweet is actually another brand tweet, skip

        cust_text = clean_text(str(cust_row["text"]))
        reply_text = clean_text(str(reply_row["text"]))
        if len(cust_text) < 8 or len(reply_text) < 8:
            continue

        threads.append({
            "thread_id": str(cust_row["tweet_id"]),
            "customer_tweet_id": str(cust_row["tweet_id"]),
            "customer_text": cust_text,
            "brand_tweet_id": str(reply_row["tweet_id"]),
            "brand_reply_text": reply_text,
            "created_at": cust_row["created_at"],
        })
        if len(threads) >= max_threads:
            break

    return threads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", default=config.BRAND)
    ap.add_argument("--max_threads", type=int, default=4000)
    args = ap.parse_args()

    if not config.RAW_TWCS_CSV.exists():
        raise FileNotFoundError(
            f"{config.RAW_TWCS_CSV} not found. Download twcs.csv from Kaggle "
            "(thoughtvector/customer-support-on-twitter) and place it there."
        )

    threads = build_brand_threads(args.brand, args.max_threads)
    out_path = config.DATA_PROCESSED / f"threads_{args.brand}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for t in threads:
            f.write(json.dumps(t) + "\n")

    print(f"Wrote {len(threads)} resolved threads for {args.brand} -> {out_path}")


if __name__ == "__main__":
    main()
