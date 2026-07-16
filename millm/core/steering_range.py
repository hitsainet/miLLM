"""
Single source of truth for the steering apply-time clamp (Feature 8).

The cluster-definition contract allows member strengths in [-300, 300] and a
lambda dial up to 2.0 (effective +/-600), while miLLM's steering range is
+/-200 (Neuronpedia scale, enforced on the per-request profile path). All
apply paths clamp effective values through this helper so cluster activation,
the per-request dial (Feature 10), and manual profiles cannot drift apart.
"""

STEERING_RANGE: float = 200.0


def clamp_steering(value: float) -> float:
    """Clamp an effective steering value to the supported range."""
    return max(-STEERING_RANGE, min(STEERING_RANGE, value))


def would_clamp(value: float) -> bool:
    """True when clamping would change the value (used for import warnings)."""
    return abs(value) > STEERING_RANGE


def declared_intensity_range(
    cluster_meta: "dict | None",
) -> "tuple[float, float] | None":
    """
    The cluster's authored budget.intensity_range as an ordered float pair,
    or None when absent/malformed.

    Single interpreter for the range document (review 010 R2: three services
    each parsed it differently). cluster_meta stores the RAW imported
    definition (lossless storage), so nothing about the shape, types, or
    ordering can be assumed: a swapped pair is normalized ascending, and
    non-numeric content degrades to None (callers fall back to the config
    envelope) rather than raising into a 500.
    """
    if not cluster_meta:
        return None
    budget = cluster_meta.get("budget") or {}
    candidate = budget.get("intensity_range")
    if not isinstance(candidate, list) or len(candidate) != 2:
        return None
    try:
        lo, hi = float(candidate[0]), float(candidate[1])
    except (TypeError, ValueError):
        return None
    if lo > hi:
        lo, hi = hi, lo
    return (lo, hi)
