"""Repositories for probes and their events (Feature 24).

Persistence is bounded by construction, as `SensingRepository` is: every batch of events prunes to
the per-probe cap and the age window, so the table cannot grow without bound even if nobody ever
calls the API. A monitor that quietly fills a disk is a monitor that eventually stops monitoring.

Neither repository manages transactions — that is the caller's responsibility, matching
`ProfileRepository` and `SensingRepository`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import case, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from millm.db.models.probe import Probe, ProbeEvent


# ⚠ **EVERY WRITE COMMITS, AND `flush()` IS NOT ENOUGH HERE.**
#
# `get_db` yields a session and closes it — it does NOT commit. So a repository that only
# flushes returns a populated object, the route serialises it happily, and the transaction
# rolls back when the request ends. Nothing persists, and nothing reports a failure.
#
# This shipped. On the node, `POST /api/probes/import` answered `{"success": true}` with a
# real probe id, and `GET /api/probes` returned **zero rows** a second later. Arming then
# failed with PROBE_NOT_FOUND for a probe that had just been created. Every probe write was
# affected: imports, the armed flag, parity reports, acknowledgements and events.
#
# No test could see it. Unit tests assert inside the same transaction, where the flush is
# sufficient and the row is visibly there; the integration tests use mocked repositories. It
# took a second HTTP request against a real database — which is to say, hardware acceptance.
#
# `circuit_repository.py` has committed since it was written; this file was the outlier.

class ProbeRepository:
    """Async CRUD for `Probe` rows."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, **fields: Any) -> Probe:
        probe = Probe(**fields)
        self.session.add(probe)
        await self.session.commit()
        return probe

    async def get(self, probe_id: str) -> Probe | None:
        return await self.session.get(Probe, probe_id)

    async def get_by_name(self, name: str) -> Probe | None:
        result = await self.session.execute(select(Probe).where(Probe.name == name))
        return result.scalar_one_or_none()

    async def list(self, *, armed: bool | None = None) -> list[Probe]:
        stmt = select(Probe).order_by(Probe.created_at.desc())
        if armed is not None:
            stmt = stmt.where(Probe.armed == armed)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def page(self, *, armed: bool | None = None, behaviour: str | None = None,
                   after: "list[Any] | None" = None, limit: int = 200) -> list[Probe]:
        """One keyset page, `ORDER BY behaviour_key NULLS LAST, id` (039 FR-39.15, TD39-29).

        `after` is the previous page's last `[behaviour_key, id]`. ⚠ THE PREDICATE MIRRORS THE
        ORDER, NULLS LAST INCLUDED: after a keyed row come the larger keys, then every NULL key;
        after a NULL-keyed row only NULL keys with a larger id. One query whatever the catalogue.
        """
        from sqlalchemy import and_, or_

        stmt = select(Probe).order_by(Probe.behaviour_key.asc().nulls_last(), Probe.id)
        if armed is not None:
            stmt = stmt.where(Probe.armed == armed)
        if behaviour is not None:
            stmt = stmt.where(Probe.behaviour_key == behaviour)
        if after is not None:
            key, last_id = after[0], str(after[1])
            if key is None:
                stmt = stmt.where(Probe.behaviour_key.is_(None), Probe.id > last_id)
            else:
                stmt = stmt.where(or_(
                    Probe.behaviour_key > key,
                    and_(Probe.behaviour_key == key, Probe.id > last_id),
                    Probe.behaviour_key.is_(None),
                ))
        result = await self.session.execute(stmt.limit(limit))
        return list(result.scalars().all())

    async def count(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(Probe))
        return int(result.scalar_one())

    async def list_armed(self) -> list[Probe]:
        return await self.list(armed=True)

    async def count_armed(self) -> int:
        result = await self.session.execute(
            select(func.count()).select_from(Probe).where(Probe.armed.is_(True))
        )
        return int(result.scalar_one())

    async def names_taken(self, prefix: str) -> set[str]:
        """Every existing name starting with `prefix`, for `on_conflict=rename` de-duplication."""
        result = await self.session.execute(
            select(Probe.name).where(Probe.name.like(f"{prefix}%"))
        )
        return set(result.scalars().all())

    async def update(self, probe: Probe, **fields: Any) -> Probe:
        """Set fields on one row and COMMIT (039 FTID §3.2, verified: one transaction per call)."""
        for key, value in fields.items():
            setattr(probe, key, value)
        await self.session.commit()
        return probe

    async def update_many(self, updates: "list[tuple[Probe, dict[str, Any]]]") -> None:
        """Set fields on several rows and commit ONCE (039 FR-39.2, D14).

        ⚠ THE TRANSACTION BOUNDARY OF A SET ARM. `update` commits per call, so N calls could leave
        rows 1..k armed when row k+1's write fails; this writes every row and commits once, and on
        any failure rolls the session back so no row of the set says `armed`. The caller removes
        the hooks it installed (`ProbeArmingService.commit`).
        """
        try:
            for probe, fields in updates:
                for key, value in fields.items():
                    setattr(probe, key, value)
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise

    async def set_by_ids(self, probe_ids: "list[str]", **values: Any) -> int:
        """One UPDATE over `probe_ids`, committed (039 `disarm_set`).

        ⚠ A STATEMENT, NOT ATTRIBUTES ON LOADED OBJECTS. A row object loaded before another
        request armed it still says `armed=False`; setting `armed=False` on it is "no change" to
        the ORM, which then emits no UPDATE at all — the database keeps `armed=True` while the
        hooks are gone (found by `test_a_disarm_waits_for_an_arm_in_flight`).
        """
        from sqlalchemy import update as sa_update

        if not probe_ids:
            return 0
        result = await self.session.execute(
            sa_update(Probe).where(Probe.id.in_(list(probe_ids))).values(**values),
            execution_options={"synchronize_session": "fetch"},
        )
        await self.session.commit()
        return int(result.rowcount or 0)

    async def by_behaviour(self, key: str) -> list[Probe]:
        """Every imported probe of one behaviour key, by id — ONE query on the
        `(behaviour_key, behaviour_version)` index (039 `probe_selection`)."""
        result = await self.session.execute(
            select(Probe).where(Probe.behaviour_key == key)
            .order_by(Probe.behaviour_version, Probe.id)
        )
        return list(result.scalars().all())

    async def delete(self, probe: Probe) -> None:
        """Delete a probe. Its events go with it (FK ON DELETE CASCADE)."""
        await self.session.delete(probe)
        await self.session.commit()

    async def disarm_all(self, reason: str) -> int:
        """Disarm every armed probe, recording why.

        ⚠ The reason is written, not just the flag cleared. A probe that stopped scoring because
        the model changed and a probe an operator disarmed are different facts, and the page has to
        be able to say which — "a probe never goes silently quiet" applies to disarming too.
        """
        armed = await self.list_armed()
        for probe in armed:
            probe.armed = False
            probe.paused_reason = reason
            probe.cost = None  # Feature 37: a cost is never trusted past a disarm
        await self.session.commit()
        return len(armed)


class ProbeEventRepository:
    """Async writes + retention for `ProbeEvent` rows."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_many(self, events: list[dict[str, Any]]) -> list[ProbeEvent]:
        rows = [ProbeEvent(**event) for event in events]
        self.session.add_all(rows)
        await self.session.commit()
        return rows

    async def list_events(
        self,
        *,
        probe_id: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        verdict: str | None = None,
        window: str | None = None,
        origin: str | None = None,
        behaviour: str | None = None,
        score_min: float | None = None,
        score_max: float | None = None,
        result_id: str | None = None,
        trace_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
        after: "list[Any] | None" = None,
    ) -> list[ProbeEvent]:
        """Recent events, newest first. Every filter narrows; none is ignored (FR-36.6).

        `verdict`: `fired` (decided above the bar), `not_fired` (scored and decided below it) or
        `not_scored`. A probe that ranked without a threshold decided nothing, so it is neither
        fired nor not fired. `behaviour` matches the probe's projected `behaviour_key`.
        """
        stmt = select(ProbeEvent).order_by(ProbeEvent.created_at.desc(), ProbeEvent.id.desc())
        if probe_id is not None:
            stmt = stmt.where(ProbeEvent.probe_id == probe_id)
        if request_id is not None:
            stmt = stmt.where(ProbeEvent.request_id == request_id)
        if session_id is not None:
            stmt = stmt.where(ProbeEvent.session_id == session_id)
        if since is not None:
            stmt = stmt.where(ProbeEvent.created_at >= since)
        if until is not None:
            stmt = stmt.where(ProbeEvent.created_at <= until)
        if verdict == "fired":
            stmt = stmt.where(ProbeEvent.verdict.is_(True))
        elif verdict == "not_fired":
            stmt = stmt.where(ProbeEvent.scored.is_(True), ProbeEvent.verdict.is_(False))
        elif verdict == "not_scored":
            stmt = stmt.where(ProbeEvent.scored.is_(False))
        if window is not None:
            stmt = stmt.where(ProbeEvent.window == window)
        if origin is not None:
            stmt = stmt.where(ProbeEvent.origin == origin)
        if behaviour is not None:
            stmt = stmt.where(ProbeEvent.probe_id.in_(
                select(Probe.id).where(Probe.behaviour_key == behaviour)
            ))
        if score_min is not None:
            stmt = stmt.where(ProbeEvent.score >= score_min)
        if score_max is not None:
            stmt = stmt.where(ProbeEvent.score <= score_max)
        # 038 FR-38.11: one indexed clause each. Without either, the filter would return every row.
        if result_id is not None:
            stmt = stmt.where(ProbeEvent.result_id == result_id)
        if trace_id is not None:
            stmt = stmt.where(ProbeEvent.trace_id == trace_id)
        if after is not None:
            # 039 keyset: strictly after the previous page's last (created_at, id), newest first.
            from sqlalchemy import and_, or_

            at = datetime.fromisoformat(str(after[0]))
            stmt = stmt.where(or_(
                ProbeEvent.created_at < at,
                and_(ProbeEvent.created_at == at, ProbeEvent.id < int(after[1])),
            ))
        result = await self.session.execute(stmt.limit(limit).offset(offset))
        return list(result.scalars().all())

    async def get(self, event_id: int) -> ProbeEvent | None:
        return await self.session.get(ProbeEvent, event_id)

    async def count(self, probe_id: str | None = None) -> int:
        stmt = select(func.count()).select_from(ProbeEvent)
        if probe_id is not None:
            stmt = stmt.where(ProbeEvent.probe_id == probe_id)
        result = await self.session.execute(stmt)
        return int(result.scalar_one())

    async def totals_by_probe(self, probe_ids: "list[str]") -> dict[str, dict[str, Any]]:
        """Per probe: every stored result, those that fired, and when the newest was recorded.

        ⚠ ONE GROUPED QUERY FOR THE WHOLE LIST, never one per probe. Served on every row of
        `GET /api/probes` (operator 2026-10-29: the tile showed no count until opened). The
        count it replaces came from the shared newest-results list and was wrong; this one is
        the same set `GET /api/probes/events?probe_id=` pages through (every probe row, rules
        excluded). `ix_probe_events_probe_created (probe_id, created_at)` serves the grouping and
        the newest time; `verdict` is read from the rows. A probe with no results is absent from
        the answer, and the caller reads that as zero.
        """
        if not probe_ids:
            return {}
        fired = func.sum(case((ProbeEvent.verdict.is_(True), 1), else_=0))
        stmt = (
            select(
                ProbeEvent.probe_id,
                func.count(ProbeEvent.id),
                fired,
                func.max(ProbeEvent.created_at),
            )
            .where(ProbeEvent.probe_id.in_(probe_ids))
            .group_by(ProbeEvent.probe_id)
        )
        rows = (await self.session.execute(stmt)).all()
        return {
            probe_id: {
                "results_total": int(total or 0),
                "results_fired": int(fired_n or 0),
                "last_result_at": last,
            }
            for probe_id, total, fired_n, last in rows
        }

    async def clear(self, probe_id: str | None = None) -> int:
        stmt = delete(ProbeEvent)
        if probe_id is not None:
            stmt = stmt.where(ProbeEvent.probe_id == probe_id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return int(result.rowcount or 0)

    async def prune_aged(self, max_age_days: int) -> int:
        """Delete events older than the window. `max_age_days <= 0` disables age pruning."""
        if max_age_days <= 0:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        # ⚠ `synchronize_session=False`, and it is not an optimisation.
        #
        # A criteria DELETE defaults to synchronizing the session by EVALUATING the WHERE clause in
        # Python against every matching object already loaded in the identity map. That comparison
        # is `loaded_created_at < cutoff`, and `cutoff` is timezone-aware. `created_at` is
        # `DateTime(timezone=True)`, which PostgreSQL honours and **SQLite ignores** — so any
        # session that has an event's `created_at` loaded raises
        # `TypeError: can't compare offset-naive and offset-aware datetimes` and retention silently
        # stops, because `record()` catches the failure and returns 0.
        #
        # It stayed hidden because `commit()` expires attributes, so the evaluator normally finds
        # nothing loaded to compare. It surfaced on 2026-09-28 the moment `record()` began reading
        # the persisted rows back (to emit `id` and `created_at` to the UI) — two `record()` calls
        # on one session were enough. Nothing about the delete needs the identity map refreshed
        # here, so the evaluation is simply not asked for.
        result = await self.session.execute(
            delete(ProbeEvent).where(ProbeEvent.created_at < cutoff),
            execution_options={"synchronize_session": False},
        )
        await self.session.commit()
        return int(result.rowcount or 0)

    async def without_recorded_batch_events(
        self, rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Drop batch-event dicts whose (probe, batch, line, window) is already recorded (T-70)."""
        keys = {(r["probe_id"], r["batch_id"], r["batch_line"], r["window"]) for r in rows}
        if not keys:
            return rows
        batch_ids = {k[1] for k in keys}
        existing = await self.session.execute(
            select(
                ProbeEvent.probe_id, ProbeEvent.batch_id, ProbeEvent.batch_line, ProbeEvent.window
            ).where(ProbeEvent.batch_id.in_(batch_ids))
        )
        seen = {tuple(r) for r in existing.all()}
        return [
            r for r in rows
            if (r["probe_id"], r["batch_id"], r["batch_line"], r["window"]) not in seen
        ]

    async def prune_to_cap(self, probe_id: str, cap: int, origin: str = "live") -> int:
        """Keep only the newest `cap` events for one probe.

        Selects the ids to keep and deletes the rest, rather than computing an offset and deleting
        "everything older than row N". An offset-based delete is wrong the moment two events share
        a timestamp — which they routinely do here, because every armed probe records an event for
        the same request within the same millisecond.
        """
        if cap <= 0:
            return 0
        # ⚠ BY ORIGIN (Feature 26, T-70): live and batch events have separate caps, and each cap
        # counts and deletes only its own origin's rows — so a batch never evicts live history.
        keep = select(ProbeEvent.id).where(
            ProbeEvent.probe_id == probe_id, ProbeEvent.origin == origin
        ).order_by(ProbeEvent.created_at.desc(), ProbeEvent.id.desc()).limit(cap)
        result = await self.session.execute(
            delete(ProbeEvent).where(
                ProbeEvent.probe_id == probe_id,
                ProbeEvent.origin == origin,
                ProbeEvent.id.not_in(keep.scalar_subquery()),
            ),
            # Same reason as `prune_aged`: no Python-side evaluation of the criteria. This one
            # cannot be evaluated in Python at all (it is a subquery), so it has been falling back
            # to a fetch — a second round trip for a synchronisation nothing here uses.
            execution_options={"synchronize_session": False},
        )
        await self.session.commit()
        return int(result.rowcount or 0)

    async def prune(
        self, probe_id: str, *, cap: int, max_age_days: int, origin: str = "live"
    ) -> int:
        """Both retention rules, as one call. Returns the total number of rows removed."""
        removed = await self.prune_aged(max_age_days)
        removed += await self.prune_to_cap(probe_id, cap, origin)
        return removed
