#!/usr/bin/env bash
set -euo pipefail

BRAND="${1:-AmazonHelp}"

if [ ! -f "data/processed/threads_${BRAND}.jsonl" ]; then
  echo "Building thread index for ${BRAND}..."
  python -m src.data_prep --brand "${BRAND}" --max_threads 4000
fi

if [ ! -f "data/golden/golden_set.csv" ]; then
  echo "No golden set found — building candidate golden set (still needs human labelling)..."
  python -m eval.build_golden_set --brand "${BRAND}" --n_total 200
  echo "NOTE: data/golden/golden_set.csv was just created with BLANK human_* columns."
  echo "Fill those in by hand before running eval — see README."
fi

echo "Running agent pipeline over golden set..."
python -m src.pipeline \
  --input data/golden/golden_set.csv \
  --output data/processed/predictions.jsonl \
  --brand "${BRAND}"

echo "Done. Predictions at data/processed/predictions.jsonl"
