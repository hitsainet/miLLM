"""Export health per signal (FR-38.13; FTDD §5.7, §7.5) and the health route's `telemetry` block.

The state is COMPUTED WHEN READ (no clock on the request path):

* `disabled` — the switch is off (or `OTEL_SDK_DISABLED`);
* `unavailable` — a configuration refusal, a 401/403 not followed by a success, or records waiting
  `TELEMETRY_HEALTH_UNAVAILABLE_AFTER_S` with no success;
* `degraded` — a failure not yet followed by a success, a failure or a drop in the last
  `TELEMETRY_HEALTH_DEGRADED_WINDOW_S`, or the queue at least half full;
* `healthy` — otherwise.

⚠ NEVER A MESSAGE. `last_error` is `{type, at}`: the exception's CLASS name (or `HTTP401`), never
its text — a receiver's error body, a URL with a token in it and a refused header all live in
exception messages. `snapshot()` never raises; an internal failure reports `{"error": <class>}`.

No OpenTelemetry import (D38.17).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from millm.telemetry import allowlist

DELETION_NOTE = "Deleting probes or events in miLLM does not erase copies already exported."
STATES = ("disabled", "healthy", "degraded", "unavailable")


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class SignalHealth:
    """Counters and the timestamps the state is computed from, for one signal."""

    def __init__(self, signal: str, clock: Callable[[], float] = time.time) -> None:
        self.signal = signal
        self._clock = clock
        self._lock = threading.Lock()
        self.reset(enabled=False)

    def reset(self, *, enabled: bool, refusal: str | None = None, capacity: int = 0) -> None:
        with self._lock:
            self.enabled = enabled
            self.refusal = refusal
            self.capacity = capacity
            self.queue_used: Callable[[], int] = lambda: 0
            self.exported = 0
            self.failed = 0
            self.dropped = 0
            self.last_success_at: float | None = None
            self.last_error: tuple[str, float] | None = None
            self.last_problem_at: float | None = None
            self.auth_failed = False
            self.waiting_since: float | None = None
            if refusal is not None:
                self.last_error = (refusal, self._clock())

    # ── events (export thread / enqueue) ─────────────────────────────────────────

    def success(self, n: int) -> None:
        with self._lock:
            self.exported += n
            self.last_success_at = self._clock()
            self.auth_failed = False
            self.waiting_since = None

    def failure(self, error_class: str, n: int, *, auth: bool = False, final: bool = True) -> None:
        """A failed attempt. `final` counts the records as `failed` (the retry horizon passed or
        the error is not retryable); a retry only marks the time."""
        if not allowlist.ERROR_CLASS(error_class):
            error_class = "Error"
        with self._lock:
            now = self._clock()
            if final:
                self.failed += n
            self.last_error = (error_class, now)
            self.last_problem_at = now
            if auth:
                self.auth_failed = True
            if self.waiting_since is None:
                self.waiting_since = now

    def drop(self, n: int = 1) -> None:
        with self._lock:
            self.dropped += n
            self.last_problem_at = self._clock()

    # ── reading ──────────────────────────────────────────────────────────────────

    def state(self, degraded_window_s: float, unavailable_after_s: float) -> str:
        with self._lock:
            if not self.enabled:
                return "disabled"
            if self.refusal is not None or self.auth_failed:
                return "unavailable"
            now = self._clock()
            if self.waiting_since is not None and now - self.waiting_since >= unavailable_after_s:
                return "unavailable"
            used = _safe(self.queue_used)
            # A failure not yet followed by a success keeps the signal degraded (it becomes
            # unavailable above once the wait passes the window); a drop recovers after the window.
            if self.waiting_since is not None \
                    or (self.last_problem_at is not None
                        and now - self.last_problem_at < degraded_window_s) \
                    or (self.capacity and used * 2 >= self.capacity):
                return "degraded"
            return "healthy"

    def view(self, degraded_window_s: float, unavailable_after_s: float) -> dict[str, Any]:
        state = self.state(degraded_window_s, unavailable_after_s)
        with self._lock:
            return {
                "state": state,
                "exported": self.exported,
                "failed": self.failed,
                "dropped": self.dropped,
                "last_success_at": _iso(self.last_success_at),
                "last_error": ({"type": self.last_error[0], "at": _iso(self.last_error[1])}
                               if self.last_error else None),
                "queue": {"used": _safe(self.queue_used), "capacity": self.capacity},
            }


def _safe(fn: Callable[[], int]) -> int:
    try:
        return int(fn())
    except Exception:  # noqa: BLE001
        return 0


SIGNALS: dict[str, SignalHealth] = {s: SignalHealth(s) for s in ("traces", "metrics", "logs")}


def snapshot() -> dict[str, Any]:
    """The `telemetry` block of `GET /api/health/detailed` (FTDD §5.7). Never raises."""
    try:
        from millm import telemetry
        from millm.telemetry import counters

        cfg = telemetry.current_settings()
        window, after = cfg.degraded_window_s, cfg.unavailable_after_s
        return {
            "enabled": cfg.any_on,
            "config": cfg.public_view(telemetry.sdk_version(), telemetry.SEMCONV_SCHEMA_URL),
            "signals": {name: h.view(window, after) for name, h in SIGNALS.items()},
            "series_overflow": counters.STORE.series_overflow,
            "attributes_dropped": allowlist.attributes_dropped(),
            "deletion_note": DELETION_NOTE,
        }
    except Exception as exc:  # noqa: BLE001 - the block is reported, never the route failed
        return {"error": type(exc).__name__}
