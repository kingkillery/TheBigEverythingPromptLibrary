"""Light-weight content-moderation helper.

The helper provides a *fast* offline regex screen and, when an
``OPENAI_API_KEY`` is available, upgrades seamlessly to OpenAI's
text-moderation endpoint.

Public API
==========
``moderate(prompt: str) -> tuple[bool, list[str]]``
    Returns ``(ok, categories)`` where ``ok`` is *True* when the prompt is
    safe to pass to the next pipeline stage, and ``categories`` is the list
    of violation categories that were triggered (empty when safe).

Key Enhancements
----------------
* **Category-aware regex fallback** instead of a single mega-regex.
* **Aggressive caching** to avoid duplicate network requests in the same
  process (SHA-256 keyed).
* **Timeout-resilient**: hard 10 s timeout on the HTTP request; automatic
  downgrade to regex on *any* exception.
* **Explicit category mapping** to align the regex fallback with OpenAI's
  official moderation categories so downstream code can treat both the same.
* **Tiny footprint** – no heavyweight third-party deps; will work inside the
  existing repo's dependency set (just needs `httpx`).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from typing import Dict, List, Tuple

import httpx

__all__ = ["moderate"]

# ---------------------------------------------------------------------------
# Regex fallback — map patterns to categories
# ---------------------------------------------------------------------------
# The patterns are *not* exhaustive; they are deliberately broad to catch the
# most common unwanted content while maintaining extremely low latency.
# They become irrelevant when the OpenAI endpoint is available.
_REGEX_CATEGORIES: Dict[str, List[str]] = {
    "violence": [
        r"\b(?:kill|murder|bomb|shoot|terror|assassin(at|ing|ion)?)\b",
    ],
    "self_harm": [
        r"\b(?:suicide|self\s*harm|self\s*destruct|kill\s*myself)\b",
    ],
    "hate": [
        r"\b(?:nazi|hitler|kkk|white\s*power)\b",
    ],
    "sexual": [
        r"\b(?:rape|sexual\s+assault|child\s*sex)\b",
    ],
}

# Compile once for speed
_COMPILED_REGEX = {cat: re.compile("|".join(pats), re.IGNORECASE) for cat, pats in _REGEX_CATEGORIES.items()}

# ---------------------------------------------------------------------------
# In-memory result cache (hash -> (ok, categories)) so duplicate prompts in a
# single run don't hit the network repeatedly.
# ---------------------------------------------------------------------------
_CACHE: Dict[str, Tuple[bool, List[str]]] = {}


async def _call_openai(prompt: str, api_key: str) -> Tuple[bool, List[str]]:
    """Thin asynchronous wrapper around OpenAI's moderation endpoint."""

    url = "https://api.openai.com/v1/moderations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    json_payload = {
        "model": "text-moderation-latest",  # 2025-06 default tier
        "input": prompt[:20_000],  # safety-cut absurdly large prompts
    }

    async with httpx.AsyncClient(timeout=10.0) as client:  # 10 s hard timeout
        response = await client.post(url, headers=headers, json=json_payload)
        response.raise_for_status()
        data = response.json()

    first = data["results"][0]
    flagged = first["flagged"]  # type: ignore[index]
    categories = [cat for cat, flagged in first["categories"].items() if flagged]  # type: ignore[assignment]
    return (not flagged, categories)


def _regex_scan(prompt: str) -> Tuple[bool, List[str]]:
    """Ultra-fast local scan returning the same (ok, categories) tuple."""

    tripped: List[str] = []
    for cat, regex in _COMPILED_REGEX.items():
        if regex.search(prompt):
            tripped.append(cat)
    return (not tripped, tripped)


async def moderate(prompt: str) -> Tuple[bool, List[str]]:
    """Validate *prompt* and return ``(ok, categories)``.

    * The function is **asynchronous** so the caller can schedule many in
      parallel.
    * Results are cached in-process keyed by SHA-256 of the prompt text.
    * Falls back silently to the regex scanner on *any* exception from the
      API call.
    """

    key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if key in _CACHE:
        return _CACHE[key]

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            result = await _call_openai(prompt, api_key)
            _CACHE[key] = result
            return result
        except Exception as exc:  # noqa: BLE001 — we want *any* failure to downgrade
            # Log to stderr so ops can see failures but keep the pipeline alive
            print(f"⚠️  Moderation API failure → using regex fallback: {exc}")

    # Offline / failure path
    result = _regex_scan(prompt)
    _CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# Synchronous helper — optional convenience wrapper
# ---------------------------------------------------------------------------

def moderate_sync(prompt: str) -> Tuple[bool, List[str]]:  # pragma: no cover – simple wrapper
    """Blocking wrapper around ``moderate`` for codebases that aren't async-aware."""

    try:
        return asyncio.run(moderate(prompt))
    except RuntimeError:
        # If we are already inside an event-loop (e.g. inside a FastAPI handler)
        # just get the running loop and schedule a task.
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(moderate(prompt))
    