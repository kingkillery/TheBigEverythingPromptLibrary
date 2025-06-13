from __future__ import annotations

"""Prompt alignment checker.

Determines whether a user-supplied prompt fits the thematic scope of the library
(e.g. system prompts, jailbreaks, prompt-engineering, LLM meta-prompts, etc.).

The implementation prefers calling an LLM (via existing ``llm_connector``) but
includes a cheap regex fallback so the function works offline and inside unit
tests.

API::

    ok, reason = await check_alignment(prompt, llm_connector)

``ok``      – ``True`` when the prompt *belongs* in the library.
``reason``  – short human-readable explanation.
"""

from typing import Tuple, Optional
import re

__all__ = ["check_alignment"]

_ALLOWED_KEYWORDS = [
    r"prompt",
    r"jailbreak",
    r"system instruction",
    r"chain of thought",
    r"role play",
    r"ignore previous",
]

_DISALLOWED_KEYWORDS = [
    r"recipe",
    r"cooking",
    r"fitness",
    r"astrology",
    r"stock advice",
]

_ALLOWED_RE = re.compile("|".join(_ALLOWED_KEYWORDS), flags=re.IGNORECASE)
_DISALLOWED_RE = re.compile("|".join(_DISALLOWED_KEYWORDS), flags=re.IGNORECASE)


async def check_alignment(prompt: str, llm_connector=None, model: str = "mistral-7b") -> Tuple[bool, str]:
    """Return (ok, explanation)."""

    # Fast regex heuristic first – cheap early exit
    if _DISALLOWED_RE.search(prompt):
        return False, "Prompt topic appears unrelated to prompt-engineering scope."

    if _ALLOWED_RE.search(prompt):
        heuristic_yes = True
    else:
        heuristic_yes = False

    # If no LLM available fallback to heuristic
    if llm_connector is None:
        return heuristic_yes, "Heuristic check only"

    system_msg = (
        "You are a strict curator for an open-source prompt-engineering library. "
        "Answer with JSON containing `{'accept': true|false, 'reason': str}`. "
        "A prompt belongs if it teaches or demonstrates prompt-engineering, "
        "LLM usage techniques, jailbreaks, system instructions, meta-prompts, or evaluation hacks. "
        "Reject ordinary factual Q&A or unrelated topics."
    )

    user_msg = f"Prompt: \n'''\n{prompt}\n'''"

    try:
        resp = await llm_connector._make_request(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            model=model,
            temperature=0.0,
            max_tokens=60,
        )
        if not resp:
            raise RuntimeError("empty response")
        import json as _json
        data = _json.loads(resp.strip())
        return bool(data.get("accept")), data.get("reason", "")
    except Exception as exc:
        # fallback to heuristic
        return heuristic_yes, f"LLM failure – heuristic used ({exc})" 