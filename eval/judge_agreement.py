"""Measures how well the LLM judge agrees with human ratings.

Process (documented for the report):
  1. Take a subsample (~30-40) of golden-set examples.
  2. A human rates the SAME replies on the SAME 1-5 "overall" scale the judge uses
     (this file expects those human scores in a CSV — see human_overall_scores.csv template).
  3. Compute Spearman correlation (rank-based — we care about relative ordering,
     not exact score matching) and mean absolute difference.
  4. Report both. A judge that's rank-correlated but systematically 1 point
     higher than humans is still useful for comparing systems; report that
     nuance rather than collapsing to one number.
"""
import csv

from scipy.stats import spearmanr


def compute_agreement(judge_scores: list[float], human_scores: list[float]) -> dict:
    assert len(judge_scores) == len(human_scores), "score lists must be same length"
    rho, p_value = spearmanr(judge_scores, human_scores)
    mean_abs_diff = sum(abs(j - h) for j, h in zip(judge_scores, human_scores)) / len(judge_scores)
    exact_match_rate = sum(1 for j, h in zip(judge_scores, human_scores) if round(j) == round(h)) / len(judge_scores)
    return {
        "n": len(judge_scores),
        "spearman_rho": rho,
        "spearman_p_value": p_value,
        "mean_abs_diff": mean_abs_diff,
        "exact_match_rate": exact_match_rate,
    }


def load_scores_from_csv(path) -> tuple[list[float], list[float]]:
    """Expects columns: example_id, judge_overall_score, human_overall_score"""
    rows = list(csv.DictReader(open(path)))
    judge = [float(r["judge_overall_score"]) for r in rows]
    human = [float(r["human_overall_score"]) for r in rows]
    return judge, human


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_csv", required=True,
                     help="CSV with columns example_id, judge_overall_score, human_overall_score")
    args = ap.parse_args()
    judge, human = load_scores_from_csv(args.scores_csv)
    print(compute_agreement(judge, human))
