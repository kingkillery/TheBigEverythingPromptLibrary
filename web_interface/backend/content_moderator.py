from __future__ import annotations

"""Light-weight content moderation helper.

The goal is *not* to achieve enterprise-grade safety but to provide a fast first-pass
filter before the heavier LLM stages in the prompt-planting pipeline run.

If an environment variable ``OPENAI_API_KEY`` is present we will query the
OpenAI Moderation endpoint.  Otherwise we fall back to a simple keyword regex
scan so that the rest of the pipeline can still operate in offline / test
scenarios.

The public API intentionally mirrors the behaviour of OpenAI's response::

    ok, categories = moderate(prompt)

Where ``ok`` is ``True`` when the prompt is *safe* and therefore allowed to
pass to the next stage.  ``categories`` is a list of category strings that were
flagged (empty list when safe).
"""

from typing import List, Tuple
import os
import re
import json
import httpx

__all__ = ["moderate"]

# ---------------------------------------------------------------------------
# Regex fallback (VERY crude – just to keep tests self-contained)
# ---------------------------------------------------------------------------

_BAD_PATTERNS = [
    r"\b(?:kill|bomb|terror|shoot|murder)\b",
    r"\b(?:nazi|hitler|kkk)\b",
    r"\b(?:rape|sexual assault)\b",
    r"\b(?:self\s*harm|suicide)\b",
]

_BAD_REGEX = re.compile("|".join(_BAD_PATTERNS), flags=re.IGNORECASE)


async def _openai_moderation(prompt: str, api_key: str, model: str = "gpt-4o-mini") -> Tuple[bool, List[str]]:
    """Call OpenAI's moderation endpoint asynchronously.

    Returns (ok, categories).
    """

    url = "https://api.openai.com/v1/moderations"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "text-moderation-latest", "input": prompt[:10000]}  # hard-cut long prompts

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

    result = data["results"][0]
    flagged = result["flagged"]
    categories = [k for k, v in result["categories"].items() if v]
    return (not flagged, categories)


async def moderate(prompt: str) -> Tuple[bool, List[str]]:
    """Return (ok, categories).

    ``ok == True`` means the prompt is safe.
    ``categories`` is *non-empty* when a violation was found.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            return await _openai_moderation(prompt, api_key)
        except Exception as exc:  # network failure – fall back
            print(f"⚠️  Moderation API failed, falling back to regex: {exc}")

    # Fallback basic regex check
    if _BAD_REGEX.search(prompt):
        return False, ["regex_flag"]
    return True, [] 