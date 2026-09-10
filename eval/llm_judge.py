"""LLM-as-judge: scores a drafted reply against the customer message and (when
available) the human reference reply, on a fixed rubric.

Rubric (1-5 each), chosen to map onto things that actually break in support
replies rather than vague "quality":
  - relevance:    does it actually address what the customer asked?
  - groundedness: does it avoid inventing policy/facts not supported by
                   context (order details, refund amounts, timelines)?
  - tone:         appropriately empathetic/professional for the situation?
  - actionability: does it give the customer a clear next step?
"""
from src import config
from src.llm_client import call_llm_json

_SYSTEM = (
    "You are an expert customer-support QA reviewer. Score the drafted reply "
    "on four dimensions, each 1-5 (5 = best). Be strict — a reply that sounds "
    "nice but invents details or dodges the actual question should score low "
    "on groundedness or relevance respectively. "
    "Reply with strict JSON only: "
    '{"relevance": <1-5>, "groundedness": <1-5>, "tone": <1-5>, "actionability": <1-5>, '
    '"overall": <1-5>, "justification": "<1-2 sentences>"}'
)

_PROMPT = """
Customer message:
"{text}"

Drafted reply to evaluate:
"{reply}"

{reference_block}

Score the drafted reply now.
"""


def judge_reply(text: str, reply: str, reference_reply: str = "") -> dict:
    reference_block = (
        f'Human reference reply (for comparison, not the only acceptable answer):\n"{reference_reply}"'
        if reference_reply else
        "(no human reference reply available for this example)"
    )
    prompt = _PROMPT.format(text=text, reply=reply, reference_block=reference_block)
    return call_llm_json(prompt, model=config.JUDGE_MODEL, system=_SYSTEM, max_tokens=300)
