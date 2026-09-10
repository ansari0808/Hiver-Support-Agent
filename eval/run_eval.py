"""Runs baselines + metrics + LLM-judge over the golden set and predictions.jsonl,
writes data/processed/eval_report.md.

Usage:
    python -m eval.run_eval \
        --golden data/golden/golden_set.csv \
        --predictions data/processed/predictions.jsonl \
        --output data/processed/eval_report.md
"""
import argparse
import csv
import json

from tqdm import tqdm

from eval.baselines import compute_majority_intent, simple_baseline_predict, trivial_baseline_predict
from eval.llm_judge import judge_reply
from eval.metrics import groundedness_rate, intent_metrics, routing_metrics
from src import config


def load_golden(path):
    return list(csv.DictReader(open(path)))


def load_predictions(path):
    return [json.loads(l) for l in open(path)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(config.DATA_GOLDEN / "golden_set.csv"))
    ap.add_argument("--predictions", default=str(config.DATA_PROCESSED / "predictions.jsonl"))
    ap.add_argument("--output", default=str(config.DATA_PROCESSED / "eval_report.md"))
    ap.add_argument("--judge_sample_n", type=int, default=40,
                     help="LLM-judge every example is slow/costly; judge a random subset by default.")
    args = ap.parse_args()

    golden_rows = load_golden(args.golden)
    golden_by_id = {r["example_id"]: r for r in golden_rows}
    preds = load_predictions(args.predictions)
    preds_by_id = {p["example_id"]: p for p in preds}

    common_ids = [eid for eid in golden_by_id if eid in preds_by_id and golden_by_id[eid]["human_intent"]]
    print(f"{len(common_ids)} labelled examples with predictions available.")

    y_true_intent = [golden_by_id[i]["human_intent"] for i in common_ids]
    y_pred_agent_intent = [preds_by_id[i]["intent"] for i in common_ids]

    y_true_routing = [golden_by_id[i]["human_routing_decision"] for i in common_ids]
    y_pred_agent_routing = [preds_by_id[i]["routing_decision"] for i in common_ids]

    majority_intent = compute_majority_intent(args.golden)
    y_pred_trivial_intent, y_pred_trivial_routing = [], []
    y_pred_simple_intent, y_pred_simple_routing = [], []
    for i in common_ids:
        text = golden_by_id[i]["text"]
        triv = trivial_baseline_predict(text, majority_intent)
        simp = simple_baseline_predict(text)
        y_pred_trivial_intent.append(triv["intent"])
        y_pred_trivial_routing.append(triv["routing_decision"])
        y_pred_simple_intent.append(simp["intent"])
        y_pred_simple_routing.append(simp["routing_decision"])

    agent_intent_m = intent_metrics(y_true_intent, y_pred_agent_intent)
    trivial_intent_m = intent_metrics(y_true_intent, y_pred_trivial_intent)
    simple_intent_m = intent_metrics(y_true_intent, y_pred_simple_intent)

    agent_routing_m = routing_metrics(y_true_routing, y_pred_agent_routing)
    trivial_routing_m = routing_metrics(y_true_routing, y_pred_trivial_routing)
    simple_routing_m = routing_metrics(y_true_routing, y_pred_simple_routing)

    grounded_flags = [preds_by_id[i].get("is_grounded", False) for i in common_ids]
    grounded_rate = groundedness_rate(grounded_flags)

    # LLM-judge on a subsample (cost control)
    judge_ids = common_ids[: args.judge_sample_n]
    judge_scores = []
    for i in tqdm(judge_ids, desc="LLM-judging replies"):
        ref = golden_by_id[i].get("human_reference_reply", "")
        reply = preds_by_id[i].get("draft_reply", "")
        try:
            score = judge_reply(golden_by_id[i]["text"], reply, ref)
            judge_scores.append(score)
        except Exception as e:  # noqa: BLE001
            print(f"judge failed on {i}: {e}")

    avg_overall = sum(s["overall"] for s in judge_scores) / len(judge_scores) if judge_scores else float("nan")
    avg_grounded = sum(s["groundedness"] for s in judge_scores) / len(judge_scores) if judge_scores else float("nan")

    report = f"""# Evaluation Report

n examples evaluated: {len(common_ids)}
Judge subsample size: {len(judge_scores)}

## Intent classification (accuracy / macro-F1)

| System   | Accuracy | Macro F1 |
|----------|----------|----------|
| Trivial  | {trivial_intent_m['accuracy']:.3f} | {trivial_intent_m['macro_f1']:.3f} |
| Simple   | {simple_intent_m['accuracy']:.3f} | {simple_intent_m['macro_f1']:.3f} |
| **Agent**| **{agent_intent_m['accuracy']:.3f}** | **{agent_intent_m['macro_f1']:.3f}** |

## Routing (auto-handle vs escalate)

| System   | Accuracy | Unsafe auto-handle rate (⚠️ lower=better) | Over-escalation rate |
|----------|----------|---------------------------------------------|------------------------|
| Trivial  | {trivial_routing_m['accuracy']:.3f} | {trivial_routing_m['unsafe_autohandle_rate']:.3f} | {trivial_routing_m['over_escalation_rate']:.3f} |
| Simple   | {simple_routing_m['accuracy']:.3f} | {simple_routing_m['unsafe_autohandle_rate']:.3f} | {simple_routing_m['over_escalation_rate']:.3f} |
| **Agent**| **{agent_routing_m['accuracy']:.3f}** | **{agent_routing_m['unsafe_autohandle_rate']:.3f}** | **{agent_routing_m['over_escalation_rate']:.3f}** |

Unsafe auto-handle rate is the number that matters most in production — it's
how often the agent would have auto-replied to something a human said should
have been escalated.

## Reply quality (LLM-judge, 1-5, subsample of {len(judge_scores)})

- Average overall score: {avg_overall:.2f}
- Average groundedness score: {avg_grounded:.2f}
- Groundedness rate (self-reported by generator): {grounded_rate:.2%}

See `eval/judge_agreement.py` for how well this judge tracks human ratings —
run it separately against a human-scored subsample.

## Per-intent F1 (agent)

{json.dumps(agent_intent_m['per_class_f1'], indent=2)}
"""

    with open(args.output, "w") as f:
        f.write(report)
    print(f"Wrote eval report -> {args.output}")


if __name__ == "__main__":
    main()
