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
from datetime import datetime
from typing import Any, Optional, Sequence

from millm import telemetry
from millm.core.config import settings
from millm.core.probe_evidence import probe_rung_language, probe_rung_next_step
from millm.db.models.rule_set import RuleSignalResult

logger = logging.getLogger(__name__)

#: Keys that must never leave the process on the socket.
CONTEXT_PREFIX = "context_"


class ProbeEventService:
    """Persists verdicts, emits them, and answers "what are the probes doing?"."""

    #: 039 (TD39-28): the throttle counts REQUESTS. A request within this interval of the last
    #: emit is dropped WHOLE and counted; there is no per-result cap (the old `[:5]` slice that
    #: silently cut a request's results is gone).
    _WS_MIN_INTERVAL_S = 0.1

    def __init__(self, repository: Any, events: Any) -> None:
        self.repository = repository
        self.events = events
        self._last_ws_emit_ts = 0.0
        #: Dropped by the throttle: whole requests, and the results they carried.
        self._ws_dropped: dict[str, int] = {"requests": 0, "results": 0}
        self._last_request_overhead_ms: Optional[float] = None
        #: Declared HERE, not defaulted at the read site. A `getattr(self, x, 0)` in `status()`
        #: would report a plausible 0 for an attribute that is simply missing, which is a
        #: fabricated value wearing a fallback's clothes — and `None` vs `0` is exactly the
        #: distinction this payload already makes for "nothing scored yet".
        self._last_request_n_passes: int = 0
        self._last_overhead_ms_per_pass: Optional[float] = None
        #: The recorded results of the last write that failed (`not_persisted`), for S2's export.
        self.last_not_persisted: list[Any] = []

    async def record(
        self,
        request_id: str,
        verdicts: Sequence[Any],
        *,
        overhead_ms: Optional[float] = None,
        n_passes: int = 0,
        contexts: Optional[dict[str, dict[str, Any]]] = None,
        batch_row: Optional[tuple[str, int]] = None,
        session: Any = None,
        reply_text: Optional[str] = None,
        reply_incomplete: bool = False,
        n_choices: int = 1,
        results: Optional[Sequence[Any]] = None,
    ) -> int:
        """Write one event per verdict, prune, and emit. Never raises.

        A probe that produced no row for a request is indistinguishable from one that was never
        armed, so every verdict is written — including the not-scored ones, which carry their
        reason instead of a score.

        ⚠ 038 (FR-38.3, FR-38.8): each row stores the FINALIZED result the client received
        (`results`, one per verdict, from `_probe_finish`) as its `result`, and every contract
        column is a projection of that object written in the same statement (T-S1-10). A caller
        that passes no results (a test, a path whose builder failed) gets them built here from the
        verdicts by the same builder, never a second serializer.
        """
        if not verdicts:
            return 0
        from millm.services.probe_result import (
            legacy_columns,
            recorded,
            stored_result,
            summary_of,
        )

        # 039 S2 (FR-39.23): the request's rule results ride the same list; they are stored in
        # `rule_signal_results` in the SAME session and commit as the probe events below.
        rule_results = [r for r in (results or ()) if getattr(r, "kind", "probe") == "rule"]
        results = _results_for(verdicts, results, request_id, batch_row, session)
        if overhead_ms is not None:
            self.note_request_overhead(overhead_ms, n_passes)
        # Feature 36: a batch row never belongs to a session (BRD BR-001), whatever reached here.
        if batch_row is not None:
            session = None

        # 039 (FR-39.10): rows are written — and so read back, emitted and listed — in contract
        # order, the order every other surface already carries.
        from millm.services.probe_result_order import result_sort_key

        pairs = sorted(zip(verdicts, results, strict=True), key=lambda vr: result_sort_key(vr[1]))
        rows: list[dict[str, Any]] = []
        for verdict, result in pairs:
            window = verdict.window
            # ⚠ The key is the PAIR. See `contexts_for`: keyed by probe id alone, two windows of
            # one probe collide and the reader is shown the wrong evidence for one of them.
            extra = (contexts or {}).get((verdict.probe_id, window), {})
            rows.append(
                {
                    # ⚠ EVERY CONTRACT AND LEGACY COLUMN IS A PROJECTION OF `result` (038,
                    # `legacy_columns`): score, verdict, reason, revision, served model, origin,
                    # batch and session all come from the one object the client received, so a
                    # stored row can never say something the response did not. The revision is
                    # the ARMED probe's (from the verdict, never the row) as before.
                    **legacy_columns(result),
                    "top_positions": list(verdict.top_positions or []),
                    "context_text": extra.get("context_text"),
                    "context_token_ids": extra.get("context_token_ids"),
                    "summary": summary_of(result),
                    # The recorded object WITHOUT `source_event_id`: the id does not exist until
                    # this INSERT returns (FTID X-05, I-38.1 (a)); `recorded_from_row` adds it.
                    "result": stored_result(result),
                }
            )

        origin = "batch" if batch_row is not None else "live"
        try:
            if batch_row is not None:
                # A row re-run after a crash (its chunk was never recorded) must not record a
                # second event for the same (probe, line, window); the partial unique index is the
                # backstop, this is the skip.
                kept = await self.events.without_recorded_batch_events(rows)
                if len(kept) < len(rows):
                    # ⚠ I-38.4: the first run's record (another `result_id`) is the persisted one;
                    # these results are NOT re-published to the socket or telemetry. Counted.
                    logger.info("probe_batch_rerun_results_skipped count=%d",
                                len(rows) - len(kept))
                rows = kept
                if not rows:
                    return 0
                if rule_results:
                    rule_results = await _without_recorded_batch_rules(
                        self.events.session, rule_results)
            rule_rows = [RuleSignalResult(**rule_columns(r)) for r in rule_results]
            if rule_rows:
                self.events.session.add_all(rule_rows)
            persisted = await self.events.create_many(rows)
            # ⚠ The AGE rule is table-wide: once per record, not once per probe (037 hardware
            # record: N identical table-wide DELETEs per request with N probes armed). The
            # background retention pass (Feature 36) runs it too.
            await self.events.prune_aged(settings.PROBE_MAX_AGE_DAYS)
            for probe_id in {row["probe_id"] for row in rows}:
                await self.events.prune_to_cap(
                    probe_id,
                    # ⚠ SEPARATE CAPS (mutation control M10): a batch must never evict the live
                    # history, so each origin is pruned only against its own rows and its own cap.
                    cap=(
                        settings.PROBE_MAX_BATCH_EVENTS_PER_PROBE if origin == "batch"
                        else settings.PROBE_MAX_EVENTS_PER_PROBE
                    ),
                    origin=origin,
                )
            if rule_rows:
                from millm.db.repositories.rule_set_repository import prune_rule_results

                await prune_rule_results(
                    self.events.session,
                    max_age_days=settings.PROBE_MAX_AGE_DAYS,
                    live_cap=settings.PROBE_MAX_EVENTS_PER_PROBE,
                    batch_cap=settings.PROBE_MAX_BATCH_EVENTS_PER_PROBE,
                    keys={(r.rule_set_id, r.signal) for r in rule_rows},
                )
                await self.events.session.commit()
        except Exception as exc:
            # The CLASS only: a failed INSERT's message can quote the row's values.
            logger.warning("probe_event_persist_failed", extra={"error": type(exc).__name__})
            # ⚠ FR-38.8: a failed write is `not_persisted`, with no `source_event_id`, and the
            # request still succeeds. Those objects go to telemetry (S2's hook) — never to the
            # socket, whose rows would have no `id` to open.
            self._publish_not_persisted([
                recorded(r, event_id=None, persistence="not_persisted")
                for r in [*results, *rule_results]
            ])
            try:
                await self.events.session.rollback()
            except Exception:  # pragma: no cover - best effort
                pass
            return 0

        if batch_row is not None:
            # Batch events are NOT emitted on the live feed (FR-26.4.8): a labelling run would
            # otherwise flood the operator's live probe view. They DO go to telemetry, subject to
            # TELEMETRY_RESULT_LOG_ORIGINS (default `live,batch`).
            self._publish(persisted, None, live=False, rules=rule_rows)
            return len(rows)
        turn = await self._write_turn(
            session, request_id, verdicts, reply_text, reply_incomplete, n_choices
        )
        self._publish(persisted, turn, rules=rule_rows)
        return len(rows)

    def _publish(self, persisted: Sequence[Any], turn: Optional[dict[str, Any]],
                 *, live: bool = True, rules: Sequence[Any] = ()) -> None:
        """Where the RECORDED objects go (038 FR-38.3): telemetry (every origin, filtered there),
        then the socket and the session (live only, throttled as before).

        ⚠ The PERSISTED rows, not the dicts written: only these carry `id` and `created_at`, which
        the UI needs to key the list and open the event detail, and `source_event_id` is the id.
        """
        # ⚠ 038 FR-38.16: THE result-record adapter — one `mi.probe.result` log record per
        # recorded result, the stored object plus its `source_event_id` (control M16 deletes this).
        # A no-op unless the operator switched logs on; filtered by the allowlist before enqueue.
        if telemetry.any_signal_on():
            from millm.services.probe_result import recorded_from_row

            telemetry.emit_results(
                [r for r in (recorded_from_row(row) for row in [*persisted, *rules]) if r]
            )
        if not live:
            return
        # ⚠ 039 FR-39.14: ONE message for the whole request (control: delete this call → the
        # socket test goes red; restore a per-result slice → the 27-result test goes red).
        # 039 S2: the request's rule entries in the same message, in contract order beside the
        # probes (`result_sort_key`), each marked `kind: "rule"` with a count and never a value.
        from millm.services.probe_result_order import result_sort_key

        summaries = [event_summary(row) for row in persisted] + [rule_summary(r) for r in rules]
        summaries.sort(key=lambda s: result_sort_key(_SortView(s)))
        self._emit_request(
            getattr(persisted[0], "request_id", None) if persisted else None, summaries,
        )
        if turn is not None:
            self._emit_session(turn)

    def _publish_not_persisted(self, results: Sequence[Any]) -> None:
        """Recorded objects whose write failed (FR-38.8): `persistence: not_persisted`, no
        `source_event_id`. They still go to telemetry; never to the socket (no row `id`)."""
        self.last_not_persisted = list(results)
        telemetry.emit_results(self.last_not_persisted)

    async def _write_turn(
        self,
        session: Any,
        request_id: str,
        verdicts: Sequence[Any],
        reply_text: Optional[str],
        incomplete: bool,
        n_choices: int,
    ) -> Optional[dict[str, Any]]:
        """The session turn (and transcript while recording is on). Never raises; COUNTS.

        After the events, so a failure here leaves the verdicts written (with their session id,
        whose row the route already committed) and the request served — and the failure visible
        as `session_write_failures` in status, never silent.
        """
        if session is None:
            return None
        from millm.services import probe_sessions

        try:
            served = next(
                (m for m in (_served_model_of(v) for v in verdicts) if m is not None), None
            )
            return await probe_sessions.ProbeSessionService(self.events.session).write_turn(
                session,
                request_id=request_id,
                served_model=served,
                reply_text=reply_text,
                incomplete=incomplete,
                n_choices=n_choices,
            )
        except Exception as exc:
            probe_sessions.counters["session_write_failures"] += 1
            # Ids only: the exception text of a failed INSERT can quote the row's values.
            logger.warning(
                "probe_session_write_failed session_id=%s error=%s",
                getattr(session, "id", None), type(exc).__name__,
            )
            try:
                await self.events.session.rollback()
            except Exception:  # pragma: no cover - best effort
                pass
            return None

    def _emit_session(self, turn: dict[str, Any]) -> None:
        """`probe:session` — ids, counts and a timestamp. ⚠ NEVER TEXT (FR-36.8, BR-002).

        Built here from named fields rather than passed through, so a field added to the turn
        writer's return value cannot reach the socket by accident; `strip_session_text` is the
        backstop the privacy test mutates.
        """
        payload = strip_session_text({
            "session_id": turn.get("session_id"),
            "turn_index": turn.get("turn_index"),
            "request_id": turn.get("request_id"),
            "n_turns": turn.get("n_turns"),
            "has_transcript": bool(turn.get("has_transcript")),
            "at": time.time(),
        })
        try:
            from millm.sockets.progress import progress_emitter as emitter

            emitter.emit_probe_session(payload)
        except Exception as exc:
            logger.warning("probe_session_emit_failed", extra={"error": str(exc)})

    def _emit_request(self, request_id: Optional[str], payloads: list[dict[str, Any]]) -> None:
        """One `probe:results` message per request, throttled PER REQUEST (039 TD39-28).

        The database rows are complete regardless of what is dropped here; the UI reconciles on
        refetch, and both drop counts are reported in status rather than hidden.
        """
        if not payloads:
            return
        now = time.monotonic()
        if now - self._last_ws_emit_ts < self._WS_MIN_INTERVAL_S:
            self._ws_dropped["requests"] += 1
            self._ws_dropped["results"] += len(payloads)
            return
        self._last_ws_emit_ts = now
        try:
            from millm.sockets.progress import progress_emitter as emitter

            emitter.emit_probe_results({
                "request_id": request_id,
                # ⚠ `strip_context` PER RESULT: the single place prompt text leaves the payload.
                "results": [strip_context(payload) for payload in payloads],
            })
        except Exception as exc:
            logger.warning("probe_ws_emit_failed", extra={"error": str(exc)})

    def note_request_overhead(self, overhead_ms: float, n_passes: int = 0) -> None:
        """Record this request's probe cost, and warn PER FORWARD PASS.

        ⚠ THE DENOMINATOR IS THE POINT. The cost is per-call dispatch, so a request's total is
        just the pass count times a near-constant. Judging the total means warning on long
        answers — the old 5 ms per-request threshold fired above ~50 generated tokens on any
        model — while a 4k-token prompt with a short completion passes comfortably. Per pass,
        one number describes every request shape.

        `n_passes` of 0 means nothing was scored; there is then no rate to judge, and the
        absolute backstop still applies.
        """
        self._last_request_overhead_ms = float(overhead_ms)
        self._last_request_n_passes = int(n_passes)
        per_pass = (float(overhead_ms) / n_passes) if n_passes > 0 else None
        self._last_overhead_ms_per_pass = per_pass

        threshold = _per_pass_threshold()
        if per_pass is not None and per_pass > threshold:
            logger.warning(
                "probe_overhead_above_threshold overhead_ms_per_pass=%.4f threshold_ms=%.4f "
                "overhead_ms=%.2f n_passes=%d",
                per_pass,
                threshold,
                overhead_ms,
                n_passes,
            )
        if overhead_ms > settings.PROBE_MAX_OVERHEAD_MS:
            logger.warning(
                "probe_overhead_request_backstop overhead_ms=%.2f threshold_ms=%.2f n_passes=%d",
                overhead_ms,
                settings.PROBE_MAX_OVERHEAD_MS,
                n_passes,
            )

    async def status(self) -> dict[str, Any]:
        """What the probes are doing, and **why any of them is not scoring** (FR-24.9).

        ⚠ `paused_reason` is reported for every armed-but-not-scoring probe. "A probe never goes
        silently quiet" is the governing invariant, and a status block that lists a probe as armed
        with no further comment is exactly that silence.
        """
        # ⚠ 039 FR-39.15: the ARMED rows only, and the catalogue as COUNTS — this response must be
        # the same size at 10 imported probes as at 10,000 (it listed every probe before).
        armed = list(await self.repository.list(armed=True))

        # ⚠ THE ROW IS NOT THE AUTHORITY ON WHETHER A HOOK EXISTS. `armed` is a database column
        # and the hook lives in process memory, so every restart leaves rows claiming to be armed
        # with nothing installed — and this method reported them as armed, silently, which is the
        # precise silence its own docstring forbids. Observed live 2026-09-28: a rollout landed
        # under an armed probe, `/api/probes/status` kept listing it, and a real generation
        # produced no verdict header.
        #
        # So the live registry is consulted and disagreement is REPORTED rather than reconciled
        # here — a read path that quietly writes would hide how often this happens. `main.py`
        # reconciles at startup, which is where the state actually becomes wrong.
        live = set(_live_armed_ids())
        stale = [p.id for p in armed if p.id not in live]
        # ⚠ WHICH WINDOWS A PROBE IS ACTUALLY REPORTING, FROM THE LIVE REGISTRY.
        #
        # Arming takes a window choice and NOTHING reported it back — not this payload, not the
        # UI. An operator could choose the windows and then had no way to confirm what was
        # chosen; the picker shows an intention that resets on re-render, which is a different
        # thing from state. Reported 2026-10-01: "I see the one armed, but I don't see which
        # windows were selected when it was armed."
        #
        # Read from the REGISTRY rather than the row, for the same reason `hook_installed` is:
        # the windows live in process memory on the `ArmedProbe`, the row knows nothing about
        # them, and after a restart a row claiming to be armed is not scoring anything at all.
        windows_by_id = _live_armed_windows()
        bars_by_id = _live_armed_bars()
        costs_by_id, projected = _live_armed_costs()
        # Per-model limits (operator 2026-10-21): the RESIDENT model's, else the globals.
        from millm.services.probe_limits import effective_for_model, resident_model_id

        limits = await effective_for_model(
            getattr(self.repository, "session", None), resident_model_id()
        ) if getattr(self.repository, "session", None) is not None else None

        def _bar_disagreement(row: Any, bars: dict[str, tuple[Any, Any]]) -> Optional[str]:
            """A sentence when the live bar is not the row's, or `None`.

            Only meaningful for a probe armed in THIS process: a stale row has no live bar to
            disagree with, and `paused_reason` already says that it is not scoring at all.
            """
            live_bar = bars.get(row.id)
            if live_bar is None:
                return None
            row_revision = int(getattr(row, "threshold_revision", 1) or 1)
            if live_bar[1] == row_revision:
                return None
            return (
                f"the stored bar is revision {row_revision} and this probe is judging against "
                f"revision {live_bar[1]} ({live_bar[0]}) — the registry was not refreshed when "
                f"the threshold moved; re-arm it"
            )

        def _paused_reason(row: Any) -> Optional[str]:
            if row.paused_reason:
                return row.paused_reason
            if row.id in stale:
                return (
                    "no hook is installed — this process restarted since the probe was armed, "
                    "so it is NOT scoring; re-arm it"
                )
            return None

        session = getattr(self.repository, "session", None)
        behaviours: dict[str, Any] = {"behaviours": [], "armed_before_unload": []}
        catalogue = {"imported": 0, "behaviours": 0}
        if session is not None:
            from millm.services import probe_behaviours

            try:
                behaviours = await probe_behaviours.armed_summary(session)
                catalogue = await probe_behaviours.catalogue_counts(session)
            except Exception as exc:  # noqa: BLE001 - status reports, it never fails
                logger.warning("probe_status_behaviours_failed error=%s", type(exc).__name__)

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
                    "paused_reason": _paused_reason(p),
                    "hook_installed": p.id in live,
                    #: `None` for a row with no live ArmedProbe — it is not scoring any window,
                    #: which is a different claim from "it is scoring an empty set".
                    "windows": windows_by_id.get(p.id),
                    #: THE BAR ACTUALLY IN FORCE, and the row's, so a disagreement is visible.
                    #: `None` when nothing is armed here — not scoring is not a bar of zero.
                    "threshold": bars_by_id.get(p.id, (None, None))[0],
                    "threshold_revision": bars_by_id.get(p.id, (None, None))[1],
                    "row_threshold": p.threshold,
                    "row_threshold_revision": int(
                        getattr(p, "threshold_revision", 1) or 1
                    ),
                    "armed_threshold_revision": getattr(p, "armed_threshold_revision", None),
                    #: ⚠ REPORTED, NOT RECONCILED — the same rule as `stale`, and the same
                    #: reason: a read path that quietly writes hides how often this happens.
                    #: Deliberately NOT folded into `paused_reason`: a probe judging against a
                    #: previous bar IS still scoring, and conflating the two would dilute the
                    #: one field whose whole job is "a probe never goes silently quiet".
                    "threshold_disagreement": _bar_disagreement(p, bars_by_id),
                    #: Feature 37: what this probe costs per pass on its layer's card (by UUID),
                    #: from the LIVE registry. `None` when it is not armed here.
                    "cost": costs_by_id.get(p.id),
                }
                for p in armed
            ],
            #: Probes actually hooked and able to score. Deliberately NOT the number of rows
            #: saying `armed`, which is what this reported before.
            "armed_count": len(armed) - len(stale),
            #: Rows claiming to be armed, so the discrepancy is visible rather than netted out.
            "armed_rows": len(armed),
            "stale_armed": sorted(stale),
            "max_armed": limits.max_armed if limits else settings.PROBE_MAX_ARMED,
            "imported_count": catalogue["imported"],
            # 039 (FR-39.4, FR-39.5, FR-39.15): each armed behaviour with its partial state, the
            # behaviours armed before an unload / restart, the catalogue as counts, and the
            # limits in behaviours with their source and whether they are stale.
            "behaviours": behaviours["behaviours"],
            "armed_before_unload": behaviours["armed_before_unload"],
            "catalogue": catalogue,
            "limits": {
                # D83: no behaviour count. The keys stay (null) for older readers.
                "max_behaviours": None,
                "max_behaviours_source": NOT_ENFORCED,
                "slowdown_budget_pct": float(settings.PROBE_SLOWDOWN_BUDGET_PCT),
                "max_armed": limits.max_armed if limits else settings.PROBE_MAX_ARMED,
                "stale": bool(limits.stale) if limits else False,
                "approval": limits.approval if limits else None,
            },
            "paused_reasons": sorted(
                {r for r in (_paused_reason(p) for p in armed) if r}
            ),
            # Reported even when nothing is paused, so an operator can tell "nothing is wrong"
            # apart from "this field is missing".
            "last_request_overhead_ms": self._last_request_overhead_ms,
            "overhead_warn_threshold_ms": settings.PROBE_MAX_OVERHEAD_MS,
            # The rate and its budget — what actually decides the warning. The two absolute
            # figures above stay for the tile that already renders them, and because a
            # per-request backstop is still a real guard.
            "last_request_n_passes": self._last_request_n_passes,
            "last_overhead_ms_per_pass": self._last_overhead_ms_per_pass,
            "overhead_warn_threshold_ms_per_pass": _per_pass_threshold(),
            "events_recorded": await self.events.count(),
            # 039: whole REQUESTS dropped by the socket throttle, and the results they carried.
            "socket_events_dropped": dict(self._ws_dropped),
            "force_serial": settings.PROBE_FORCE_SERIAL,
            # Feature 36: session bookkeeping that failed, COUNTED rather than silent, and when
            # retention last ran (a dead loop would keep transcripts past their promise).
            **_session_counters(),
            # Feature 37 (FR-37.7, D47): the budget, and the armed set's projected cost against it.
            # Lowering a budget disarms nothing; `over_budget` says the set no longer fits.
            "budget": limits.budget() if limits else _budget_now(),
            "budget_sources": limits.sources if limits else None,
            "projected": {**projected, "over_budget": (
                _over_budget_against(projected, limits.budget()) if limits
                else _over_budget(projected))},
        }


#: Since D83 the behaviour count binds nothing; status says so in these words.
NOT_ENFORCED = "not enforced (D83: the cost budget decides)"


def _per_pass_threshold() -> float:
    """The per-pass overhead warning's threshold: the decode budget the armed set was ADMITTED
    against (D83: a share of the model's own time), else the absolute fallback. A fixed 0.25 ms
    would warn on every request of a model whose 5% admits more."""
    from millm.services.probe_runtime import ProbeRuntimeState

    admitted = getattr(ProbeRuntimeState(), "admitted_budget", None)
    if isinstance(admitted, dict) and admitted.get("decode_ms_per_pass"):
        return float(admitted["decode_ms_per_pass"])
    return float(settings.PROBE_MAX_OVERHEAD_MS_PER_PASS)


def _budget_now() -> dict[str, float]:
    from millm.services.probe_arming import _budget

    return _budget()


def _over_budget_against(projected: dict[str, float], budget: dict[str, float]) -> bool:
    from millm.services.probe_arming import over_budget

    return over_budget(projected, budget)


def _over_budget(projected: dict[str, float]) -> bool:
    from millm.services.probe_arming import over_budget

    return over_budget(projected)


def _live_armed_costs() -> tuple[dict[str, Any], dict[str, float]]:
    """`{probe_id: cost dict}` and the projection, from the live registry (not the rows)."""
    from millm.services.probe_cost import ProbeCost
    from millm.services.probe_runtime import ProbeRuntimeState

    state = ProbeRuntimeState()
    costs = {
        p.probe_id: p.cost.as_dict() if isinstance(getattr(p, "cost", None), ProbeCost) else None
        for p in state.armed()
    }
    return costs, state.projected_cost()


def _session_counters() -> dict[str, Any]:
    from millm.services.probe_sessions import counters

    return {
        "session_resolve_failures": counters["session_resolve_failures"],
        "session_write_failures": counters["session_write_failures"],
        "retention_failures": counters["retention_failures"],
        "last_retention_pass_at": counters["last_retention_pass_at"],
        "transcripts": settings.PROBE_TRANSCRIPTS,
    }


def _live_armed_ids() -> list[str]:
    """Probe ids with a hook installed in THIS process.

    Imported inside the function: `probe_runtime` pulls in torch, and `status()` must not be the
    reason a lightweight import graph grows a deep-learning dependency.
    """
    from millm.services.probe_runtime import ProbeRuntimeState

    return [p.probe_id for p in ProbeRuntimeState().armed()]


def _live_armed_bars() -> dict[str, tuple[Optional[float], int]]:
    """`{probe_id: (threshold, threshold_revision)}` from the live registry.

    ⚠ FROM THE REGISTRY, NOT THE ROW, AND FOR THE SAME REASON AS `windows` AND `hook_installed`.
    A threshold can now be re-cut on an armed probe, and the write order is row-then-registry: if
    the refresh fails, the row and every read path that serialises it show the new bar while every
    verdict is still judged against the old one. `GET /api/probes` serialises the ROW, so that
    divergence would be invisible-but-visible — the UI showing one number and the monitor using
    another. This is the only surface that can see it.
    """
    from millm.services.probe_runtime import ProbeRuntimeState

    return {
        p.probe_id: (p.threshold, int(getattr(p, "threshold_revision", 1) or 1))
        for p in ProbeRuntimeState().armed()
    }


def _live_armed_windows() -> dict[str, list[str]]:
    """`{probe_id: [window, ...]}` from the live registry.

    An `ArmedProbe` with an empty `windows` reports its own scope, which is what the runtime
    actually does — so that is what is reported here, rather than the empty tuple that encodes
    it. A reader asking "which windows is this scoring?" wants the answer, not the encoding.
    """
    from millm.services.probe_runtime import ProbeRuntimeState

    return {
        p.probe_id: list(p.windows) if p.windows else [p.scope]
        for p in ProbeRuntimeState().armed()
    }


def event_summary(event: Any) -> dict[str, Any]:
    """One probe event, WITHOUT its context — the single shape both readers get.

    ⚠ THE SOCKET AND THE REST LIST FEED THE SAME UI LIST, SO THEY MUST NOT HAVE TWO SHAPES.
    They did. `record()` emitted the dicts it had just built to INSERT, which carry no `id`
    (autoincrement) and no `created_at` (server default) because neither exists until the row is
    written. The REST list served these fields. So a live event arrived at the browser with
    `id: undefined`, the tile's `key={event.id}` collapsed every live row onto the same key, and
    `onClick={() => onOpen(event.id)}` could not open the one route that serves the prompt window.
    The list only looked right after a refetch — reported 2026-09-28 as "I have to refresh the
    browser for the verdicts to be displayed".

    `create_many` has returned the persisted rows all along; the service discarded them.

    Privacy: `context_text` and `context_token_ids` are absent BY CONSTRUCTION here rather than
    removed afterwards. `strip_context` is still applied at the emit site — a no-op on this shape,
    and the thing that keeps holding if a field is ever added.
    """
    created = getattr(event, "created_at", None)
    from millm.services.probe_result import recorded_from_row

    legacy = {
        "id": getattr(event, "id", None),
        "probe_id": event.probe_id,
        "request_id": event.request_id,
        "scored": event.scored,
        "not_scored_reason": event.not_scored_reason,
        "score": event.score,
        "threshold": event.threshold,
        "verdict": event.verdict,
        "rung": event.rung,
        "top_positions": event.top_positions,
        "n_scored_tokens": event.n_scored_tokens,
        "summary": event.summary,
        # Two verdicts from one probe are otherwise identical on this shape — same probe_id, same
        # name upstream, same request. Without `window` the UI shows two rows it cannot tell
        # apart, and `provisional` is the operator's only signal that one of them fired against a
        # bar never cut for it.
        "window": getattr(event, "window", "all"),
        "provisional": bool(getattr(event, "provisional", False)),
        # Which cut of the bar judged this. Two verdicts under different bars are distinguishable
        # by `threshold` alone only while the two cuts landed on different numbers — and a reader
        # watching `provisional` disappear between two events cannot otherwise tell whether a
        # window gained its own bar or never needed one.
        "threshold_revision": int(getattr(event, "threshold_revision", 1) or 1),
        # Which model's residual this verdict read (033 D-14, addendum D-36); None = not recorded.
        "served_model": _served_model_of(event),
        # Feature 36: the conversation and the turn this verdict belongs to (NULL = not recorded).
        "session_id": getattr(event, "session_id", None),
        "turn_index": getattr(event, "turn_index", None),
        # 039 FR-39.33 item 3: the behaviour the STORED result recorded — what was armed — on the
        # socket and the REST list alike (one shape; the list route no longer adds its own).
        # `None` for a row without one; the contract merge below carries the same value.
        "behaviour": (getattr(event, "result", None) or {}).get("behaviour")
        if isinstance(getattr(event, "result", None), dict) else None,
        # ⚠ isoformat, not the datetime: this goes through socket.io's JSON encoder, which has no
        # datetime support, and a raised encoder error is swallowed by the fire-and-forget wrapper.
        "created_at": created.isoformat() if hasattr(created, "isoformat") else created,
    }
    # 038 FR-38.3: the STORED contract object (the recorded copy, its `source_event_id` injected
    # from this row's own id) beside the legacy keys. Every shared key is a projection of the same
    # object, so the two cannot disagree; the contract's keys are merged last.
    return {**legacy, **(recorded_from_row(event) or {})}


class _SortView:
    """A summary dict seen through `result_sort_key`'s attribute reads."""

    def __init__(self, summary: dict[str, Any]) -> None:
        self.kind = summary.get("kind", "probe")
        behaviour = summary.get("behaviour") or {}
        self.behaviour = type("B", (), {"key": behaviour.get("key")})() if behaviour else None
        self.rule_set_id = summary.get("rule_set_id")
        self.signal = summary.get("signal")
        self.probe_name = summary.get("probe_name")
        self.probe_id = summary.get("probe_id")
        self._legacy_name = summary.get("name")
        self.window = summary.get("window", "")


def rule_columns(result: Any) -> dict[str, Any]:
    """The `rule_signal_results` row of one rule result: every column a projection of the stored
    object (written in the same statement), as `legacy_columns` is for probe events."""
    from millm.api.schemas.probe_result import wire

    data = wire(result.model_copy(update={"persistence": "persisted"}))
    data.pop("source_event_id", None)
    behaviour = data.get("behaviour") or {}
    return {
        "result": data,
        "result_id": data["result_id"],
        "trace_id": data.get("trace_id"),
        "span_id": data.get("span_id"),
        "event_time": datetime.fromisoformat(data["event_time"].replace("Z", "+00:00")),
        "outcome": data["outcome"],
        "reason": data.get("reason"),
        "status": data.get("status") or "final",
        "persistence": "persisted",
        "request_id": data.get("request_id"),
        "session_id": data.get("session_id"),
        "turn_index": data.get("turn_index"),
        "origin": data.get("origin") or "live",
        "batch_id": data.get("batch_id"),
        "batch_line": data.get("batch_line"),
        "behaviour_key": behaviour.get("key"),
        "rule_set_id": data["rule_set_id"],
        "rule_set_digest": data["rule_set_digest"],
        "signal": data["signal"],
        "form": data["form"],
        "window": data["window"],
        "fires": data.get("fires"),
        "count": data.get("count"),
    }


def rule_summary(row: Any) -> dict[str, Any]:
    """One recorded rule result, as the socket and the events list carry it (039 S2).

    ⚠ NO TEXT: the stored object has a count and never a value (FTDD §4.3), and `strip_context`
    still runs on it at the emit site.
    """
    from millm.services.probe_result import recorded_from_row

    created = getattr(row, "created_at", None)
    legacy = {
        "kind": "rule",
        "id": getattr(row, "id", None),
        "request_id": row.request_id,
        "rule_set_id": row.rule_set_id,
        "rule_set_digest": row.rule_set_digest,
        "signal": row.signal,
        "form": row.form,
        "window": row.window,
        "fires": row.fires,
        "count": row.count,
        "outcome": row.outcome,
        "reason": row.reason,
        "session_id": row.session_id,
        "turn_index": row.turn_index,
        "behaviour": (row.result or {}).get("behaviour") if isinstance(row.result, dict) else None,
        "created_at": created.isoformat() if hasattr(created, "isoformat") else created,
    }
    return {**legacy, **(recorded_from_row(row) or {})}


async def _without_recorded_batch_rules(session: Any, results: Sequence[Any]) -> list[Any]:
    """Batch rule results whose (set, signal, line, window) is not already recorded (a rerun)."""
    from sqlalchemy import select

    if not results:
        return []
    first = results[0]
    rows = await session.execute(
        select(RuleSignalResult.rule_set_id, RuleSignalResult.signal, RuleSignalResult.window)
        .where(RuleSignalResult.batch_id == first.batch_id,
               RuleSignalResult.batch_line == first.batch_line))
    seen = {tuple(r) for r in rows.all()}
    return [r for r in results if (r.rule_set_id, r.signal, r.window) not in seen]


def _results_for(
    verdicts: Sequence[Any], results: Optional[Sequence[Any]], request_id: str,
    batch_row: Optional[tuple[str, int]], session: Any,
) -> list[Any]:
    """The finalized result for each verdict: the caller's, or built here by THE builder.

    ⚠ PAIRED BY `(probe_id, window)`, NOT BY POSITION (039): the caller's results are in contract
    order (`probe_result_order`), which is not the verdicts' order, and a positional zip would
    store one probe's result on another's row.
    """
    if results is not None:
        by_key = {(getattr(r, "probe_id", None), getattr(r, "window", None)): r
                  for r in results if getattr(r, "kind", "probe") == "probe"}
        paired = [by_key.get((v.probe_id, v.window)) for v in verdicts]
        if all(r is not None for r in paired):
            return paired
    from types import SimpleNamespace

    from millm.services.probe_result import build_probe_results

    context = SimpleNamespace(request_id=request_id, probes=(), trace=None, interrupted=None,
                              session=None if batch_row is not None else session)
    built = build_probe_results(verdicts, context)
    if batch_row is None:
        return built
    out = []
    for result in built:
        moved = result.model_copy(update={
            "origin": "batch", "batch_id": batch_row[0], "batch_line": int(batch_row[1]),
            "session_id": None, "turn_index": None,
        })
        moved._legacy_name = result._legacy_name
        out.append(moved)
    return out


def _served_model_of(obj: Any) -> Optional[str]:
    """`served_model` of a verdict or an event row, only as text (None = not recorded)."""
    value = getattr(obj, "served_model", None)
    return value if isinstance(value, str) and value else None


#: Keys that carry conversation text. None of them may reach the socket (FR-36.8).
SESSION_TEXT_KEYS = frozenset({"user_text", "reply_text", "pairs", "messages"})


def strip_session_text(payload: dict[str, Any]) -> dict[str, Any]:
    """Everything except conversation text and `context_*` — the backstop for `probe:session`."""
    return {
        key: value for key, value in strip_context(payload).items()
        if key not in SESSION_TEXT_KEYS
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
