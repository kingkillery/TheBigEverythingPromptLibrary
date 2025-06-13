"""Compatibility alias package.

This lightweight shim exists because several test modules (and possibly
other code) expect to import ``backend.*`` directly (e.g. ``from backend.app
import app``).  The actual implementation lives under
``web_interface.backend``.  Instead of duplicating code or rewriting all
imports, we forward the package machinery so that ``import backend``
behaves exactly like ``import web_interface.backend``.
"""

import importlib
import sys
from types import ModuleType

# Import the real package
_real_pkg: ModuleType = importlib.import_module("web_interface.backend")

# Re-export the real package under the alias name so subsequent
# ``import backend`` returns the same module object.
sys.modules[__name__] = _real_pkg

# Also copy reference for sub-module resolution (e.g. backend.app)
# Because _real_pkg is a *package* with a __path__, it already supports
# loading its sub-modules, so nothing more is required. 