#!/usr/bin/env bash
set -euo pipefail

python -m eval.run_eval \
  --golden data/golden/golden_set.csv \
  --predictions data/processed/predictions.jsonl \
  --output data/processed/eval_report.md \
  --judge_sample_n 40

echo "---"
cat data/processed/eval_report.md
