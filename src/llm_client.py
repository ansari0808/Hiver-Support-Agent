"""Thin wrapper around the Anthropic API used by every stage of the pipeline.

Centralising this means the classifier, retriever-grounded generator, and judge
all get the same retry/timeout/parsing behaviour for free. See DECISION_LOG.md #2.
"""
import json
import os
import re
import time

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

_client = None


def get_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill it in."
            )
        _client = Anthropic(api_key=api_key)
    return _client


def call_llm(
    prompt: str,
    model: str,
    system: str = "",
    max_tokens: int = 800,
    temperature: float = 0.0,
    retries: int = 3,
) -> str:
    """Single-turn call, returns raw text. Retries with backoff on transient errors."""
    client = get_client()
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(
                block.text for block in resp.content if block.type == "text"
            )
        except Exception as e:  # noqa: BLE001 — deliberately broad, this is a thin retry shim
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {retries} retries: {last_err}")


def call_llm_json(prompt: str, model: str, system: str = "", max_tokens: int = 800) -> dict:
    """Calls the LLM and parses a JSON object out of the response, tolerating
    stray markdown fences or preamble text the model sometimes adds."""
    raw = call_llm(prompt, model=model, system=system, max_tokens=max_tokens)
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    # If there's still leading/trailing chatter, grab the outermost {...}
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON from LLM response: {raw!r}") from e
