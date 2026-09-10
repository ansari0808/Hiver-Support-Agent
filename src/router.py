"""Decides auto-handle vs escalate-to-human, with a stated reason.

Rule-based on top of model outputs, deliberately — a black-box "LLM decides
auto vs escalate" call would be unauditable and is exactly the kind of thing
this brand would want a transparent, overridable policy for.
See DECISION_LOG.md #8.
"""
from src import config


def route(
    text: str,
    intent: str,
    intent_confidence: float,
    top_retrieval_similarity: float,
    is_grounded: bool,
) -> dict:
    text_lower = text.lower()

    for kw in config.FORCE_ESCALATE_KEYWORDS:
        if kw in text_lower:
            return _escalate(f"Safety/legal keyword matched: '{kw}'")

    if intent in config.HIGH_RISK_INTENTS:
        return _escalate(f"Intent '{intent}' is in the high-risk category (billing/account/complaint)")

    if intent_confidence < 0.6:
        return _escalate(f"Low intent-classification confidence ({intent_confidence:.2f} < 0.60)")

    if top_retrieval_similarity < config.MIN_RETRIEVAL_SIMILARITY_FOR_AUTOHANDLE:
        return _escalate(
            f"No sufficiently similar historical precedent found "
            f"(best similarity {top_retrieval_similarity:.2f} < {config.MIN_RETRIEVAL_SIMILARITY_FOR_AUTOHANDLE})"
        )

    if not is_grounded:
        return _escalate("Drafted reply was not grounded in any retrieved precedent")

    return {
        "decision": "auto_handle",
        "reason": (
            f"Low-risk intent '{intent}' with high classifier confidence "
            f"({intent_confidence:.2f}) and a grounded reply backed by a similar "
            f"past resolution (similarity {top_retrieval_similarity:.2f})"
        ),
    }


def _escalate(reason: str) -> dict:
    return {"decision": "escalate", "reason": reason}
