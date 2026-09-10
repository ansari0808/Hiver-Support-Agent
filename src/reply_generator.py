"""Drafts a reply grounded in retrieved historical resolutions.

The prompt forces the model to (a) point at which retrieved precedent(s) it's
following, and (b) say explicitly if none of the retrieved precedents actually
fit — this is what lets the router later detect "ungrounded" drafts.
"""
from src import config
from src.llm_client import call_llm_json

_SYSTEM = (
    "You are drafting a customer support reply on behalf of a brand's Twitter "
    "support account. You must ground your reply in the provided past resolved "
    "examples where possible — match their tone, policy, and typical next step. "
    "Never invent policy details (refund amounts, timelines, order specifics) "
    "that aren't supported by the examples or the customer's own message. "
    "Reply with strict JSON only: "
    '{"reply": "<drafted reply, <280 chars, no @mentions>", '
    '"grounded_in_examples": [<list of integer indices of examples you actually used, empty list if none>], '
    '"is_grounded": <true/false>, '
    '"notes": "<one short sentence on what you based this on, or why nothing fit>"}'
)

_PROMPT = """
Customer's intent: {intent}

Customer message:
"{text}"

Past resolved examples for this brand (most similar first):
{examples_block}

Draft the reply now.
"""


def _format_examples(examples: list[dict]) -> str:
    if not examples:
        return "(none retrieved)"
    lines = []
    for i, ex in enumerate(examples):
        lines.append(
            f"[{i}] (similarity={ex['similarity']:.2f}) "
            f"Customer said: \"{ex['customer_text']}\" -> Brand replied: \"{ex['brand_reply_text']}\""
        )
    return "\n".join(lines)


def generate_reply(text: str, intent: str, examples: list[dict]) -> dict:
    prompt = _PROMPT.format(
        intent=intent,
        text=text,
        examples_block=_format_examples(examples),
    )
    result = call_llm_json(prompt, model=config.GENERATION_MODEL, system=_SYSTEM, max_tokens=400)
    result.setdefault("reply", "")
    result.setdefault("grounded_in_examples", [])
    result.setdefault("is_grounded", bool(result["grounded_in_examples"]))
    return result
