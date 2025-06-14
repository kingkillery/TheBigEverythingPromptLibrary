"""Minimal stub of the *fuzzywuzzy* library for test environments.

This implements only the two functions used by the codebase:
``partial_ratio`` and ``token_set_ratio``.  The scoring logic is *vastly*
simplified but deterministic and cheap, sufficient for unit tests that just
need the module to exist.
"""

from typing import Any
import re


def _normalize(text: str) -> list[str]:
    # Lower-case and split on words for a basic token representation
    return re.findall(r"\w+", text.lower())


def partial_ratio(a: str, b: str) -> int:  # noqa: N802 – matching upstream casing
    """Return a naive token overlap percentage in the range 0-100."""
    if not a or not b:
        return 0
    a_tokens = set(_normalize(a))
    b_tokens = set(_normalize(b))
    overlap = len(a_tokens & b_tokens)
    if not b_tokens:
        return 0
    return int(100 * overlap / len(b_tokens))


def token_set_ratio(a: str, b: str) -> int:  # noqa: N802
    """Alias to *partial_ratio* for this stub."""
    return partial_ratio(a, b)


# Mimic the upstream structure where functions live inside a ``fuzz`` sub-module
import types as _types

fuzz = _types.ModuleType("fuzzywuzzy.fuzz")
fuzz.partial_ratio = partial_ratio  # type: ignore[attr-defined]
fuzz.token_set_ratio = token_set_ratio  # type: ignore[attr-defined]

import sys as _sys
_sys.modules[__name__ + ".fuzz"] = fuzz

__all__ = [
    "partial_ratio",
    "token_set_ratio",
    "fuzz",
] 