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
