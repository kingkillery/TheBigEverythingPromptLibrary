"""Project bootstrap executed automatically on interpreter start.

1. Adds the repository root to ``sys.path`` so in-tree packages resolve
   regardless of where tests are executed from.
2. Creates a *compatibility alias* so ``import backend`` maps to
   ``web_interface.backend``.
3. Stubs out heavyweight optional libraries (``sentence_transformers`` and
   its deep-learning stack) to keep CI lightweight.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure repository root is on sys.path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Alias 'backend' → 'web_interface.backend'
# ---------------------------------------------------------------------------
if "backend" not in sys.modules:
    try:
        real_pkg = importlib.import_module("web_interface.backend")
        sys.modules["backend"] = real_pkg
    except ModuleNotFoundError:
        # If the real package does not exist (should not happen), skip.
        pass

# ---------------------------------------------------------------------------
# Lightweight stubs for heavyweight optional deps
# ---------------------------------------------------------------------------

def _make_disabled_stub(name: str, message: str) -> types.ModuleType:
    """Return a module that raises *ImportError* on attribute access."""

    def _fail(*_a, **_kw):  # noqa: D401
        raise ImportError(message)

    mod = types.ModuleType(name)
    mod.__getattr__ = lambda _self, _attr: _fail()  # type: ignore[assignment]
    return mod

# Disable *sentence_transformers* (which pulls in PyTorch & friends)
if "sentence_transformers" not in sys.modules:
    sys.modules["sentence_transformers"] = _make_disabled_stub(
        "sentence_transformers",
        "sentence_transformers disabled in test environment",
    )

# Likewise disable *torch* if some library tries to import it later
if "torch" not in sys.modules:
    sys.modules["torch"] = _make_disabled_stub(
        "torch",
        "torch disabled in lightweight test environment",
    ) 