"""Recording, emitting and reporting probe verdicts (FR-24.8, FR-24.9).

Modelled on `SensingService`'s flush: bounded retention on every write, a throttled fire-and-forget
socket emission, and a status block that always says why a probe is not scoring.

⚠ **THE SOCKET PAYLOAD NEVER CARRIES PROMPT OR CONTEXT TEXT.** The event row holds the decoded
window around the top firing position so a reviewer can fetch it from the detail route; the
broadcast carries none of it. This is not a general caution — this estate has already shipped a
socket that leaked user prompt text while 135/135 tests stayed green, because the test asserted
what the payload *contained* rather than what it must not. `test_probe_events.py` asserts the
**absence**, with a mutation control.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, Sequence

from millm.core.config import settings
from millm.core.probe_evidence import probe_rung_language, probe_rung_next_step

logger = logging.getLogger(__name__)

#: Keys that must never leave the process on the socket.
CONTEXT_PREFIX = "context_"


class ProbeEventService:
    """Persists verdicts, emits them, and answers "what are the probes doing?"."""

    _WS_MAX_PER_FLUSH = 5
    _WS_MIN_INTERVAL_S = 0.1

    def __init__(self, repository: Any, events: Any) -> None:
        self.repository = repository
        self.events = events
        self._last_ws_emit_ts = 0.0
        self._ws_dropped = 0
        self._last_request_overhead_ms: Optional[float] = None

    async def record(
        self,
        request_id: str,
        verdicts: Sequence[Any],
        *,
        overhead_ms: Optional[float] = None,
        contexts: Optional[dict[str, dict[str, Any]]] = None,
    ) -> int:
        """Write one event per verdict, prune, and emit. Never raises.

        A probe that produced no row for a request is indistinguishable from one that was never
        armed, so every verdict is written — including the not-scored ones, which carry their
        reason instead of a score.
        """
        if not verdicts:
            return 0
        if overhead_ms is not None:
            self.note_request_overhead(overhead_ms)

        rows: list[dict[str, Any]] = []
        for verdict in verdicts:
            extra = (contexts or {}).get(verdict.probe_id, {})
            rows.append(
                {
                    "probe_id": verdict.probe_id,
                    "request_id": request_id,
                    "scored": bool(verdict.scored),
                    "not_scored_reason": verdict.not_scored_reason,
                    "score": verdict.score,
                    "threshold": verdict.threshold,
                    "verdict": verdict.fires,
                    # Denormalised: the event must keep describing the evidence that was true
                    # WHEN IT WAS OBSERVED, not the probe's rung today.
                    "rung": verdict.rung,
                    "top_positions": list(verdict.top_positions or []),
                    "n_scored_tokens": verdict.n_scored_tokens,
                    "context_text": extra.get("context_text"),
                    "context_token_ids": extra.get("context_token_ids"),
                    "summary": _summary(verdict),
                }
            )

        try:
            await self.events.create_many(rows)
            for probe_id in {row["probe_id"] for row in rows}:
                await self.events.prune(
                    probe_id,
                    cap=settings.PROBE_MAX_EVENTS_PER_PROBE,
                    max_age_days=settings.PROBE_MAX_AGE_DAYS,
                )
        except Exception as exc:
            logger.warning("probe_event_persist_failed", extra={"error": str(exc)})
            return 0

        self._emit_events(rows)
        return len(rows)

    def _emit_events(self, payloads: list[dict[str, Any]]) -> None:
        """Fire-and-forget socket emission, throttled like sensing.

        The database rows are complete regardless of what is dropped here; the UI reconciles on
        refetch, and the drop count is reported in status rather than hidden.
        """
        now = time.monotonic()
        if now - self._last_ws_emit_ts < self._WS_MIN_INTERVAL_S:
            self._ws_dropped += len(payloads)
            return
        self._last_ws_emit_ts = now
        if len(payloads) > self._WS_MAX_PER_FLUSH:
            self._ws_dropped += len(payloads) - self._WS_MAX_PER_FLUSH
            payloads = payloads[: self._WS_MAX_PER_FLUSH]
        try:
            from millm.sockets.progress import progress_emitter as emitter

            for payload in payloads:
                emitter.emit_probe_event(strip_context(payload))
        except Exception as exc:
            logger.warning("probe_ws_emit_failed", extra={"error": str(exc)})

    def note_request_overhead(self, overhead_ms: float) -> None:
        self._last_request_overhead_ms = float(overhead_ms)
        if overhead_ms > settings.PROBE_MAX_OVERHEAD_MS:
            logger.warning(
                "probe_overhead_above_threshold overhead_ms=%.2f threshold_ms=%.2f",
                overhead_ms,
                settings.PROBE_MAX_OVERHEAD_MS,
            )

    async def status(self) -> dict[str, Any]:
        """What the probes are doing, and **why any of them is not scoring** (FR-24.9).

        ⚠ `paused_reason` is reported for every armed-but-not-scoring probe. "A probe never goes
        silently quiet" is the governing invariant, and a status block that lists a probe as armed
        with no further comment is exactly that silence.
        """
        probes = await self.repository.list()
        armed = [p for p in probes if p.armed]
        return {
            "armed": [
                {
                    "id": p.id,
                    "name": p.name,
                    "layer": p.layer,
                    "rule": p.rule,
                    "rung": p.rung,
                    "rung_language": probe_rung_language(p.rung),
                    "next_step": probe_rung_next_step(p.rung),
                    "streamable": p.streamable,
                    "basis": p.basis,
                    "paused_reason": p.paused_reason,
                }
                for p in armed
            ],
            "armed_count": len(armed),
            "max_armed": settings.PROBE_MAX_ARMED,
            "imported_count": len(probes),
            "paused_reasons": sorted(
                {p.paused_reason for p in armed if p.paused_reason}
            ),
            # Reported even when nothing is paused, so an operator can tell "nothing is wrong"
            # apart from "this field is missing".
            "last_request_overhead_ms": self._last_request_overhead_ms,
            "overhead_warn_threshold_ms": settings.PROBE_MAX_OVERHEAD_MS,
            "events_recorded": await self.events.count(),
            "socket_events_dropped": self._ws_dropped,
            "force_serial": settings.PROBE_FORCE_SERIAL,
        }


def strip_context(payload: dict[str, Any]) -> dict[str, Any]:
    """Everything except the `context_*` keys.

    ⚠ The single place prompt text is removed before a payload leaves the process. A leaked
    prompt once passed 135/135 tests in this estate, so the test that guards this asserts the
    ABSENCE of these keys rather than the presence of the others.
    """
    return {key: value for key, value in payload.items() if not key.startswith(CONTEXT_PREFIX)}


def _summary(verdict: Any) -> str:
    if not verdict.scored:
        return f"not scored: {verdict.not_scored_reason or 'unknown'}"
    if verdict.fires is None:
        return f"score {verdict.score:.4f} (no threshold placed — ranks, does not decide)"
    return f"score {verdict.score:.4f} {'above' if verdict.fires else 'below'} threshold"
