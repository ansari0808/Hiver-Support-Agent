"""Two baselines to compare the agent against, as required by the assignment.

TRIVIAL baseline: always predicts the majority intent from the golden set,
and always replies with one generic canned message. Always routes to escalate
(the safest trivial policy). This is the floor — if the agent doesn't clearly
beat this, nothing else in the report matters.

SIMPLE baseline: keyword/regex-based intent classifier (no LLM), templated
reply per intent (no grounding/retrieval), and a fixed rule for routing
(auto-handle only for order_status and general_inquiry, escalate otherwise).
This represents "what you could ship in an afternoon without any LLM at all."
"""
import csv
import re
from collections import Counter

from src import config

_KEYWORD_RULES = [
    ("order_status", re.compile(r"\b(where.?s? my order|track|tracking|delivery|shipped|eta)\b", re.I)),
    ("refund_or_return", re.compile(r"\b(refund|return|send it back|wrong item|damaged)\b", re.I)),
    ("account_access", re.compile(r"\b(login|log in|password|locked out|can.?t sign in|2fa|verification code)\b", re.I)),
    ("billing_or_charge", re.compile(r"\b(charged|charge|billing|subscription|invoice|double.?charged)\b", re.I)),
    ("product_defect_or_quality", re.compile(r"\b(broken|defective|doesn.?t work|stopped working|poor quality)\b", re.I)),
    ("app_or_website_bug", re.compile(r"\b(app crash|error message|website (down|broken)|won.?t load|bug)\b", re.I)),
    ("complaint_escalation", re.compile(r"\b(ridiculous|unacceptable|worst|never again|third time|no one (responds|replies))\b", re.I)),
]

_TEMPLATE_REPLIES = {
    "order_status": "Thanks for reaching out — we'd like to look into your order's status. Please DM us your order number.",
    "refund_or_return": "Sorry to hear that. Please DM us your order number and we'll help with a return or refund.",
    "account_access": "Sorry for the trouble accessing your account. Please DM us so we can verify and help you back in.",
    "billing_or_charge": "We understand billing concerns are urgent. Please DM us your order/account details so we can review the charge.",
    "product_defect_or_quality": "Sorry to hear about the product issue. Please DM us your order number so we can help.",
    "app_or_website_bug": "Thanks for flagging this. Please DM us details (device, screenshot) so we can investigate.",
    "general_inquiry": "Thanks for your message! Happy to help — could you share a bit more detail?",
    "complaint_escalation": "We're sorry for the frustration. Please DM us so a specialist can look into this directly.",
}

_TRIVIAL_CANNED_REPLY = "Thanks for reaching out! Please DM us your order number so we can help."


def trivial_baseline_predict(text: str, majority_intent: str) -> dict:
    return {
        "intent": majority_intent,
        "reply": _TRIVIAL_CANNED_REPLY,
        "routing_decision": "escalate",
        "routing_reason": "Trivial baseline always escalates (safest default with zero understanding).",
    }


def simple_baseline_predict(text: str) -> dict:
    intent = "general_inquiry"
    for label, pattern in _KEYWORD_RULES:
        if pattern.search(text):
            intent = label
            break

    routing = "auto_handle" if intent in {"order_status", "general_inquiry"} else "escalate"
    reason = (
        f"Simple rule: intent '{intent}' is in the always-auto-handle allowlist"
        if routing == "auto_handle"
        else f"Simple rule: intent '{intent}' is not in the always-auto-handle allowlist"
    )
    return {
        "intent": intent,
        "reply": _TEMPLATE_REPLIES[intent],
        "routing_decision": routing,
        "routing_reason": reason,
    }


def compute_majority_intent(golden_csv_path) -> str:
    rows = list(csv.DictReader(open(golden_csv_path)))
    counts = Counter(r["human_intent"] for r in rows if r["human_intent"])
    return counts.most_common(1)[0][0] if counts else "order_status"
