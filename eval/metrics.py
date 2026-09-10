"""Automated (non-LLM-judge) metrics computed against the human-labelled golden set."""
from collections import defaultdict

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def intent_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "per_class_f1": dict(zip(labels, f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0))),
        "n": len(y_true),
    }


def routing_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """y_true/y_pred are 'auto_handle' / 'escalate'.

    We report both plain accuracy AND the two error types separately, because
    they are NOT equally costly: auto-handling something that should have been
    escalated (a false auto-handle) is much worse than over-escalating.
    """
    tp_auto = sum(1 for t, p in zip(y_true, y_pred) if t == "auto_handle" and p == "auto_handle")
    fp_auto = sum(1 for t, p in zip(y_true, y_pred) if t == "escalate" and p == "auto_handle")
    fn_auto = sum(1 for t, p in zip(y_true, y_pred) if t == "auto_handle" and p == "escalate")
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == "escalate" and p == "escalate")

    n = len(y_true)
    unsafe_autohandle_rate = fp_auto / n if n else 0.0  # the metric that matters most
    over_escalation_rate = fn_auto / n if n else 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "unsafe_autohandle_rate": unsafe_autohandle_rate,   # should have escalated, agent didn't -> DANGEROUS
        "over_escalation_rate": over_escalation_rate,        # should have auto-handled, agent escalated -> just inefficient
        "confusion": {"tp_auto": tp_auto, "fp_auto": fp_auto, "fn_auto": fn_auto, "tn_escalate": tn},
    }


def groundedness_rate(is_grounded_flags: list[bool]) -> float:
    return sum(bool(x) for x in is_grounded_flags) / len(is_grounded_flags) if is_grounded_flags else 0.0
