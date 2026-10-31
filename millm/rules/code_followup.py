"""Operator code, scored AFTER the response (039 S3; suite decision D41; FR-39.18 amended D44).

The live path never runs operator code and never waits for it. At serving, a `code_rule` signal and
every expression that depends on it report pending — `outcome: "skipped"`, `status: "unavailable"`,
`reason: "rule_worker_unavailable"` — never `fires: false` (`evaluator.serve`, S2). Then:

1. `rules_finish` leaves a :class:`CodeJob` on the request context: the attached sets with pending
   expressions and the request's window texts, in memory only (never logged, never stored).
2. The probe record writes every result as the client received it (pending entries included).
3. Only THEN, `InferenceService._probe_record` hands the job to :func:`spawn` — a tracked task the
   request does not await — which asks the `rule-runner` worker for each needed code rule on each
   window (`runner_client`), resolves the pending expressions the counts now decide
   (`CompiledRuleSet.resolve_pending`), REVISES the stored `rule_signal_results` rows in place
   (same `result_id`) and emits one `probe:rule_results` socket message for live traffic.

The STATELESS routes (`/api/probes/score`, `/api/rule-sets/{id}/evaluate`) have no live budget,
so they run operator code synchronously instead (suite decision D57; :class:`StatelessCode`): the
same calls and the same resolution (`_decide`), awaited by the request, nothing stored.

What stays pending, honestly: an expression whose code rule could not be scored (the worker
unreachable, its bundle unverified, the per-request call count spent, its document not held) and an
expression that needs a code rule's POSITIONS (a proximity anchor, a validator `span_of`, a
`within_n` operand), which a code rule never reports.

Reason codes (suite decision D56): every runner reason that `mi.probe.result/v1` publishes
(`probe_reasons.REASONS`) passes through as it is (:func:`published`) — `rule_timeout`,
`rule_error`, `rule_resource_limit`, `rule_output_invalid`, `rule_load_refused`,
`rule_input_too_large`; any other (a contract reason the list does not publish, e.g.
`rule_not_approved`) is reported as `rule_error`, and the precise code is logged with the signal
and hash — never text.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from millm.rules import runner_contract as rc
from millm.rules.evaluator import PENDING_REASON

logger = logging.getLogger(__name__)

UNPUBLISHED_AS = "rule_error"
#: Review R2 (D57): the most time one STATELESS request spends on operator code in all, from the
#: request's start. Past it no call starts and no reply is awaited; those expressions stay pending
#: (`rule_worker_unavailable`). Without it the worst case was `RULE_CODE_CALLS_PER_REQUEST` x
#: (slot wait + wall) plus loads — over 64 s at the defaults, past miStudio's 60 s client timeout.
STATELESS_CODE_BUDGET_S = 30.0
#: The runner's own "could not run": the expression stays pending under the code it already has.
STILL_PENDING = "rule_worker_unavailable"

#: Review R1 (M4): the most after-response jobs in flight at once. Each holds its request's window
#: texts and one worker connection until its calls end (up to `RULE_CODE_CALLS_PER_REQUEST` x the
#: wall time and the slot wait); with nothing bounding them, traffic faster than the worker grew
#: memory and worker threads without limit. A job past the bound is not started: its results stay
#: pending (`rule_worker_unavailable`), which is honest, and the count is logged.
MAX_JOBS_IN_FLIGHT = 64

_tasks: "set[asyncio.Task[int]]" = set()


@dataclass(frozen=True)
class CodeJob:
    """What one request's after-response scoring needs. Held in memory only."""

    #: `(behaviour key, windows, AttachedSet)` for every attached set with a pending expression.
    sets: tuple[tuple[str, tuple[str, ...], Any], ...]
    #: `{window: text}` for the windows this request has text for.
    texts: dict[str, str]


def job_for(work: Iterable[tuple[str, tuple[str, ...], Any]], texts: dict[str, str]
            ) -> Optional[CodeJob]:
    """A job when any attached set has a pending expression this request reports; else None."""
    sets = []
    for key, windows, attached in work:
        compiled = getattr(attached, "compiled", None)
        if compiled is None or attached.vectors_failed or not compiled.pending_expressions:
            continue
        if any(name in compiled.pending_expressions for name in attached.names()):
            sets.append((key, tuple(windows), attached))
    if not sets:
        return None
    # Only the windows these sets report: nothing more of the request's text is held.
    needed = {w for _key, windows, _attached in sets for w in windows}
    return CodeJob(tuple(sets), {w: t for w, t in texts.items() if w in needed})


def published(reason: str) -> str:
    """The runner's reason as `mi.probe.result/v1` reports it (D56): as it is when the published
    list (`probe_reasons.REASONS`, the authority — so the two cannot drift) and the runner
    contract both name it; any other is `rule_error`."""
    from millm.services.probe_reasons import REASONS

    return reason if reason in REASONS and reason in rc.reasons() else UNPUBLISHED_AS


def _pending(results: Iterable[Any]) -> dict[tuple[str, str, str], str]:
    """`{(rule_set_id, signal, window): result_id}` of the request's pending rule results."""
    out = {}
    for r in results or ():
        if getattr(r, "kind", "probe") != "rule":
            continue
        if r.reason == PENDING_REASON and r.status == "unavailable":
            out[(r.rule_set_id, r.signal, r.window)] = r.result_id
    return out


Key = tuple[str, str, str]


async def _decide(session: Any, job: CodeJob, pending: set[Key], cap: int, *,
                  unanswered_pending: bool) -> tuple[list[tuple[Key, Any]], int]:
    """Ask the worker for each needed code rule on each window and resolve what the counts decide.

    Returns `([((rule_set_id, signal, window), Outcome)], calls skipped by the cap)`. Shared by
    the after-response job (`score`) and the stateless routes (`StatelessCode`, D57), so both
    send the same calls and resolve the same way. `unanswered_pending`: a call the worker never
    answered within the client's wait leaves its expressions pending instead of `rule_timeout`
    (both callers pass True: D57 for the stateless routes, D63 after the response).
    """
    decided: list[tuple[Key, Any]] = []
    skipped = 0
    for _key, windows, attached in job.sets:
        compiled = attached.compiled
        names = [n for n in attached.names()
                 if any((attached.rule_set_id, n, w) in pending for w in windows)]
        needs = compiled.resolvable_code_signals(names)
        present = {w: job.texts[w] for w in windows if w in job.texts}
        documents = dict(attached.code_documents)
        wanted = dict.fromkeys(sid for sids in needs.values() for sid in sids)
        code_results: dict[tuple[str, str], tuple[Optional[int], Optional[str]]] = {}
        for signal in compiled.code_signals():
            sid = signal["id"]
            if sid not in wanted:
                continue
            text_doc = documents.get(signal["code_sha256"])
            if text_doc is None or signal.get("runner_contract") != rc.contract_format():
                continue  # nothing to run it with: stays pending
            for window, text in present.items():
                if session.calls >= cap:
                    skipped += 1
                    continue
                result = await session.call(signal["code_sha256"], text_doc,
                                            signal["function"], signal["params"], text)
                if result.reason == STILL_PENDING:
                    continue
                if unanswered_pending and not result.answered:
                    continue
                if result.reason is not None:
                    logger.info("rule_code_failed rule_set_id=%s signal=%s code_sha256=%s "
                                "reason=%s", attached.rule_set_id, sid,
                                signal["code_sha256"], result.reason)
                code_results[(sid, window)] = (
                    result.count, published(result.reason) if result.reason else None)
        if not code_results:
            continue
        # Off the event loop: resolving re-runs the set's declarative signals on every window
        # (validators over a long text), which must not stall the loop streaming other requests.
        outcomes = await asyncio.to_thread(compiled.resolve_pending, present, code_results,
                                           only=names)
        for outcome in outcomes:
            key = (attached.rule_set_id, outcome.signal, outcome.window)
            if key in pending:
                decided.append((key, outcome))
    return decided, skipped


async def score(job: CodeJob, results: Iterable[Any], *, settings: Any
                ) -> list[tuple[str, Any]]:
    """`[(result_id, Outcome)]` for every pending result the runner's answers now decide."""
    from millm.rules.runner_client import RunnerSession, call_limits, floor_violations

    pending = _pending(results)
    if not pending:
        return []
    below = floor_violations(settings)
    if below:
        # FTASKS 17.4: a host never lowers a floor. Code rules stay pending, loudly.
        logger.warning("rule_code_below_floor limits=%s", ",".join(below))
        return []
    cap = int(settings.RULE_CODE_CALLS_PER_REQUEST)
    async with RunnerSession(str(settings.RULE_RUNNER_SOCKET), call_limits(settings)) as session:
        # D63 (operator, 2026-10-24): a call the worker never answered within the client's wait
        # stays PENDING here too — the call may never have run. `rule_timeout` is final only when
        # the runner itself reports the guest overran its wall time.
        decided, skipped = await _decide(session, job, set(pending), cap,
                                         unanswered_pending=True)
    if skipped:
        logger.warning("rule_code_calls_capped skipped=%d cap=%d", skipped, cap)
    return [(pending[key], outcome) for key, outcome in decided]


class StatelessCode:
    """Operator code on the STATELESS routes, run synchronously (suite decision D57).

    `POST /api/probes/score` (and a Batch `probe` row, which runs the same service) and
    `POST /api/rule-sets/{id}/evaluate` have no live budget, so D41's reason for scoring code
    after the response does not apply: the request waits for the runner and returns the decided
    results. Each call is bounded by the client's wait (`2 x RULE_CODE_CALL_WALL_MS` + margin,
    `runner_client`), and one request makes at most `RULE_CODE_CALLS_PER_REQUEST` calls in all
    (shared across a score request's inputs), within :data:`STATELESS_CODE_BUDGET_S` from the
    request's start (review R2); the connection is opened on first need only.

    Pending (`rule_worker_unavailable`, never `fires: false`) only when nothing decided it: the
    runner is absent or never answered in time, its bundle unverified, the call cap or the
    request's code budget spent, the code document not held, or the expression needs a code
    rule's positions. The live chat
    paths never use this (D41): they leave a `CodeJob` for `spawn`.
    """

    def __init__(self, settings: Any) -> None:
        import time

        self.settings = settings
        self._session: Any = None
        self.skipped = 0
        self.deadline = time.monotonic() + STATELESS_CODE_BUDGET_S

    async def __aenter__(self) -> StatelessCode:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            await session.__aexit__(None, None, None)
            if session.past_deadline:
                logger.warning("rule_code_stateless_budget_spent not_started=%d budget_s=%s",
                               session.past_deadline, STATELESS_CODE_BUDGET_S)
        if self.skipped:
            logger.warning("rule_code_calls_capped skipped=%d cap=%d", self.skipped,
                           int(self.settings.RULE_CODE_CALLS_PER_REQUEST))
            self.skipped = 0

    async def resolve(self, work: Iterable[tuple[str, tuple[str, ...], Any]],
                      texts: dict[str, str], items: list[Any]) -> list[Any]:
        """`items` (`serving.RuleItem`, as `evaluate_work` returned them) with every pending
        entry the runner's answers decide replaced by its final result; order kept."""
        import dataclasses

        from millm.rules.runner_client import RunnerSession, call_limits, floor_violations

        pending = {(i.rule_set_id, i.signal, i.window): n for n, i in enumerate(items)
                   if i.reason == PENDING_REASON and i.status == "unavailable"}
        if not pending:
            return items
        job = job_for(work, texts)
        if job is None:
            return items
        below = floor_violations(self.settings)
        if below:
            logger.warning("rule_code_below_floor limits=%s", ",".join(below))
            return items
        if self._session is None:
            self._session = RunnerSession(str(self.settings.RULE_RUNNER_SOCKET),
                                          call_limits(self.settings), deadline=self.deadline)
            await self._session.__aenter__()
        decided, skipped = await _decide(
            self._session, job, set(pending), int(self.settings.RULE_CODE_CALLS_PER_REQUEST),
            unanswered_pending=True)
        self.skipped += skipped
        out = list(items)
        for key, outcome in decided:
            n = pending[key]
            completed = outcome.outcome == "completed"
            out[n] = dataclasses.replace(
                out[n], outcome=outcome.outcome, reason=None if completed else outcome.reason,
                fires=bool(outcome.fires) if completed else None,
                count=max(0, int(outcome.count)) if completed else 0, status="final",
                # A code rule reports a count, never positions: no spans are claimed.
                spans=None)
        return out


def _revised_result(stored: dict[str, Any], outcome: Any) -> dict[str, Any]:
    completed = outcome.outcome == "completed"
    return {
        **stored,
        "outcome": outcome.outcome,
        "reason": None if completed else outcome.reason,
        "fires": bool(outcome.fires) if completed else None,
        "count": max(0, int(outcome.count)) if completed else 0,
        "status": "final",
    }


async def revise(revised: list[tuple[str, Any]]) -> list[Any]:
    """Write each decided outcome onto its stored row (same `result_id`). Returns the rows."""
    from sqlalchemy import select

    import millm.db.base as db_base
    from millm.db.models.rule_set import RuleSignalResult

    if not revised:
        return []
    by_id = dict(revised)
    async with db_base.async_session_factory() as session:
        rows = list((await session.execute(
            select(RuleSignalResult).where(RuleSignalResult.result_id.in_(list(by_id)))
            # Review R1 (M5): only a row that is STILL pending is revised, so a revision lands
            # exactly once whatever runs twice; a final row is never rewritten.
            .where(RuleSignalResult.status == "unavailable",
                   RuleSignalResult.reason == PENDING_REASON)
            .order_by(RuleSignalResult.id))).scalars().all())
        for row in rows:
            result = _revised_result(dict(row.result or {}), by_id[row.result_id])
            row.result = result
            row.outcome = result["outcome"]
            row.reason = result["reason"]
            row.fires = result["fires"]
            row.count = result["count"]
            row.status = "final"
        await session.commit()
    return rows


def _emit(rows: list[Any]) -> None:
    from millm.services.probe_event_service import rule_summary, strip_context
    from millm.sockets.progress import progress_emitter

    if not rows:
        return
    progress_emitter.emit_probe_rule_results({
        "request_id": rows[0].request_id,
        "results": [strip_context(rule_summary(row)) for row in rows],
    })


async def run(job: CodeJob, results: Iterable[Any], *, live: bool) -> int:
    """Score, revise, emit. Never raises (an observer never fails anything)."""
    from millm.core.config import settings

    try:
        revised = await score(job, list(results or ()), settings=settings)
        rows = await revise(revised)
        if live:
            _emit(rows)
        return len(rows)
    except Exception as exc:  # noqa: BLE001 - the class only (FR-39.28)
        logger.warning("rule_code_followup_failed error=%s", type(exc).__name__)
        return 0


def spawn(job: Optional[CodeJob], results: Iterable[Any], *, live: bool
          ) -> Optional["asyncio.Task[int]"]:
    """Start the job as a TRACKED task the caller does not await (D41). None when no job."""
    if job is None:
        return None
    in_flight = sum(1 for t in _tasks if not t.done())
    if in_flight >= MAX_JOBS_IN_FLIGHT:
        logger.warning("rule_code_jobs_saturated in_flight=%d sets=%d", in_flight, len(job.sets))
        return None
    task = asyncio.get_running_loop().create_task(run(job, list(results or ()), live=live))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task


async def drain(timeout: float) -> int:
    """Wait (bounded) for jobs in flight at shutdown. Returns how many remain."""
    pending = {t for t in _tasks if not t.done()}
    if not pending:
        return 0
    _done, still = await asyncio.wait(pending, timeout=timeout)
    return len(still)
