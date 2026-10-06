"""
Backpressure numbers: `Retry-After` values, the estimated wait and the batch backlog.

Feature 29 (029 FTDD §5.3, §7.3). This module owns NUMBERS, not responses: the error
builders, `millm_error_handler`, the readiness probe and the in-stream error event all ask
`retry_after_for` for the value they write, so every 503 answers from one policy.
`RetryAfterMiddleware` (millm/api/retry_after.py) is the safety net for a 503 that bypassed
it, and uses a DISTINCT fallback value so a forgotten builder is visible in a test.

Unmeasured is `None`, never `0`: an estimate with fewer than three samples, an estimate
while continuous batching serves requests outside the queue, and a backlog with no batch
API registered are all `None` (FR-29.7.3, FR-29.7.5).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from millm.core.config import settings
from millm.core.logging import get_logger

logger = get_logger(__name__)

#: Codes whose `details` may say an UNLOAD (not a load) is what the caller waits for.
_BUSY_CODES = frozenset({"MODEL_BUSY", "MODEL_LOADING"})


def _clamp_seconds(value: float) -> int:
    """HTTP `delay-seconds`: a whole number, at least 1."""
    return max(1, int(math.ceil(value)))


def _is_unload(details: dict[str, Any] | None) -> bool:
    """Whether a busy refusal is waiting on an unload, as its producers mark it.

    `InferenceService._unloading_refusal` sets `unloading`; `ModelService.load_model` sets
    `unloading_model_id` when the resident model is being unloaded under it.
    """
    if not isinstance(details, dict):
        return False
    return bool(details.get("unloading")) or details.get("unloading_model_id") is not None


def _queue_full_seconds() -> int:
    """The estimated wait rounded up and clamped, or the configured default without one."""
    try:
        from millm.api.dependencies import get_inference_service

        inference = get_inference_service()
        estimate = estimate_wait_seconds(inference.request_queue, inference._use_cbm())
    except Exception as e:  # noqa: BLE001 - a lookup failure must not lose the header
        logger.warning("retry_after_estimate_unavailable", error=str(e))
        estimate = None
    if estimate is None:
        return settings.RETRY_AFTER_QUEUE_DEFAULT_S
    return min(_clamp_seconds(estimate), settings.RETRY_AFTER_MAX_S)


def _hub_seconds(now: Callable[[], float] | None = None) -> int:
    """The hub breaker's remaining recovery time, at least 1.

    `HUB_UNAVAILABLE` is raised by the cluster hub service through its own breaker
    (`cluster_hub_circuit`); with the breaker open the remainder is
    `recovery_timeout - (now - last_failure_time)`. With it closed (a single network
    failure) there is nothing to wait out, so 1.
    """
    try:
        from millm.core.resilience import CircuitState
        from millm.services.cluster_hub_service import cluster_hub_circuit

        state = cluster_hub_circuit.state
        if state.state != CircuitState.OPEN:
            return 1
        clock = now or time.time
        remaining = cluster_hub_circuit.config.recovery_timeout - (
            clock() - state.last_failure_time
        )
        return _clamp_seconds(remaining)
    except Exception as e:  # noqa: BLE001
        logger.warning("retry_after_hub_unavailable", error=str(e))
        return settings.RETRY_AFTER_FALLBACK_S


def retry_after_for(code: str | None, details: dict[str, Any] | None = None) -> int:
    """Seconds a caller should wait before retrying a 503 with this code (T-88).

    | code                          | value                                                |
    |-------------------------------|------------------------------------------------------|
    | QUEUE_FULL                    | ceil(estimated wait) in [1, RETRY_AFTER_MAX_S]; else |
    |                               | RETRY_AFTER_QUEUE_DEFAULT_S                          |
    | MODEL_BUSY, MODEL_LOADING     | RETRY_AFTER_UNLOAD_S for an unload, else _LOAD_S     |
    | MODEL_NOT_LOADED              | RETRY_AFTER_NOT_LOADED_S                             |
    | INSUFFICIENT_MEMORY           | RETRY_AFTER_MEMORY_S                                 |
    | HUB_UNAVAILABLE               | the hub breaker's remaining recovery time            |
    | READINESS                     | RETRY_AFTER_READINESS_S                              |
    | anything else                 | RETRY_AFTER_FALLBACK_S                               |
    """
    normalised = (code or "").upper()
    if normalised == "QUEUE_FULL":
        return _queue_full_seconds()
    if normalised in _BUSY_CODES:
        return settings.RETRY_AFTER_UNLOAD_S if _is_unload(details) else settings.RETRY_AFTER_LOAD_S
    if normalised == "MODEL_NOT_LOADED":
        return settings.RETRY_AFTER_NOT_LOADED_S
    if normalised == "INSUFFICIENT_MEMORY":
        return settings.RETRY_AFTER_MEMORY_S
    if normalised == "HUB_UNAVAILABLE":
        return _hub_seconds()
    if normalised == "READINESS":
        return settings.RETRY_AFTER_READINESS_S
    return settings.RETRY_AFTER_FALLBACK_S


def estimate_wait_seconds(queue: Any, cbm_running: bool) -> float | None:
    """How long a request arriving now waits for a slot; None when there is nothing to base it on.

        median(recent slot-holding durations) × (queue_waiting + in_flight) / max_concurrent

    `queue_waiting = pending_count − holding_count` (interactive requests waiting), and
    `in_flight = holding_count + background_holding_count` (Feature 26's batch chunks hold the
    same slot). The batch backlog is NOT added: interactive requests go first at every chunk
    boundary (026 FTDD §7, constraint 2). None below three samples, and while continuous
    batching is running, whose requests hold no queue slot (T-90).
    """
    if cbm_running:
        return None
    median = queue.median_hold_seconds()
    if median is None:
        return None
    waiting = max(queue.pending_count - queue.holding_count, 0)
    in_flight = queue.holding_count + queue.background_holding_count
    result: float = float(median) * (waiting + in_flight) / max(int(queue.max_concurrent), 1)
    return result


# --- the batch backlog (Feature 26 registers the provider, 026 FR-26.4.7) -------------------

_BACKLOG_PROVIDER: dict[str, Callable[[], int] | None] = {"fn": None}


def register_backlog_provider(fn: Callable[[], int] | None) -> None:
    """Register the one callable that reports rows not yet run across in-progress batches.

    `None` unregisters it (tests; a batch runner shutting down).
    """
    _BACKLOG_PROVIDER["fn"] = fn


def backlog_rows() -> int | None:
    """The batch backlog in rows; None when no batch API is registered, or the provider failed.

    None means "no measurement", never zero: before Feature 26 there is no batch API, and a
    `0` would claim an empty backlog nobody measured (FR-29.7.3).
    """
    fn = _BACKLOG_PROVIDER["fn"]
    if fn is None:
        return None
    try:
        return int(fn())
    except Exception as e:  # noqa: BLE001 - the health read must not fail on a provider
        logger.warning("backlog_provider_failed", error=str(e), error_type=type(e).__name__)
        return None
