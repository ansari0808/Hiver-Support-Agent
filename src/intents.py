"""Few-shot intent classifier over the fixed taxonomy in config.py.

We use a fixed, human-authored taxonomy (not LLM-discovered clusters) so that
labels are stable across runs and interpretable in the eval report.
See DECISION_LOG.md #4.
"""
from src import config
from src.llm_client import call_llm_json

_SYSTEM = (
    "You are an intent classifier for customer support tweets sent to a brand "
    "on Twitter. Classify the customer's message into exactly one of the given "
    "intents. Reply with strict JSON only: "
    '{"intent": "<one_of_the_labels>", "confidence": <0-1 float>, "rationale": "<one short sentence>"}'
)

_FEWSHOT = """
Intents and definitions:
{taxonomy}

Examples:
Message: "Where is my package? It said it would arrive yesterday and tracking hasn't updated."
-> {{"intent": "order_status", "confidence": 0.95, "rationale": "Asking about delivery ETA/tracking"}}

Message: "You charged me twice for the same order, please fix this now."
-> {{"intent": "billing_or_charge", "confidence": 0.93, "rationale": "Duplicate charge complaint"}}

Message: "This is the third time I've emailed about my broken order and no one replies. Ridiculous."
-> {{"intent": "complaint_escalation", "confidence": 0.85, "rationale": "Repeated unresolved issue with frustration"}}

Now classify this message:
Message: "{text}"
"""


def build_taxonomy_block() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in config.INTENT_TAXONOMY.items())


def classify_intent(text: str) -> dict:
    prompt = _FEWSHOT.format(taxonomy=build_taxonomy_block(), text=text)
    result = call_llm_json(prompt, model=config.CLASSIFIER_MODEL, system=_SYSTEM, max_tokens=200)

    intent = result.get("intent", "general_inquiry")
    if intent not in config.INTENT_LABELS:
        # LLM hallucinated a label outside the taxonomy -> fall back safely.
        intent = "general_inquiry"
        result["confidence"] = min(result.get("confidence", 0.5), 0.5)
    result["intent"] = intent
    return result
