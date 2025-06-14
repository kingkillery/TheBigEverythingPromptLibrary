"""Very small stub of *numpy* exposing only ``mean`` used in the codebase."""

def mean(values):  # type: ignore[override]
    """Return arithmetic mean of *values* or 0 if empty."""
    values = list(values)
    return sum(values) / len(values) if values else 0.0 