"""Queries for probe sessions, turns and transcripts (Feature 36).

Every write commits (the rule and its history are in `probe_repository.py`). Every criteria DELETE
passes `synchronize_session=False`, for the reason `ProbeEventRepository.prune_aged` records.

⚠ No query here selects `probe_transcripts.user_text` or `reply_text` except `transcript()`, which
backs the one route allowed to serve them. A list or timeline that joined the table would put user
text on a path nothing reviews for it.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import delete, func, select, update

from millm.db.models.probe import Probe, ProbeEvent
from millm.db.models.probe_session import ProbeSession, ProbeSessionTurn, ProbeTranscript

#: Rows per retention DELETE, so a pass never holds a long lock (FPRD §8).
RETENTION_BATCH = 5000


class ProbeSessionRepository:
    def __init__(self, session: Any) -> None:
        self.session = session

    # ── resolution ──────────────────────────────────────────────────────────────

    async def create_session(self, **fields: Any) -> ProbeSession:
        row = ProbeSession(**fields)
        self.session.add(row)
        await self.session.commit()
        return row

    async def get(self, session_id: str) -> Optional[ProbeSession]:
        return await self.session.get(ProbeSession, session_id)

    async def newest_by_client_id(self, client_session_id: str) -> Optional[ProbeSession]:
        result = await self.session.execute(
            select(ProbeSession)
            .where(ProbeSession.client_session_id == client_session_id)
            .order_by(ProbeSession.opened_at.desc(), ProbeSession.id.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def sessions_matching(self, column: str, value: str) -> list[ProbeSession]:
        """Distinct sessions owning a turn whose `column` equals `value`."""
        col = getattr(ProbeSessionTurn, column)
        result = await self.session.execute(
            select(ProbeSession)
            .where(ProbeSession.id.in_(
                select(ProbeSessionTurn.session_id).where(col == value).limit(50)
            ))
        )
        return list(result.scalars().all())

    async def touch(self, session_id: str, now: datetime) -> None:
        await self.session.execute(
            update(ProbeSession).where(ProbeSession.id == session_id).values(last_seen_at=now),
            execution_options={"synchronize_session": False},
        )
        await self.session.commit()

    # ── the turn writer ─────────────────────────────────────────────────────────

    async def add_turn(self, **fields: Any) -> ProbeSessionTurn:
        row = ProbeSessionTurn(**fields)
        self.session.add(row)
        await self.session.commit()
        return row

    async def add_transcript(self, **fields: Any) -> ProbeTranscript:
        row = ProbeTranscript(**fields)
        self.session.add(row)
        await self.session.commit()
        return row

    async def bump(self, session_id: str, now: datetime) -> int:
        """`n_turns` from the turns themselves (never `+= 1`: a retried write would double it)."""
        count = select(func.count()).select_from(ProbeSessionTurn).where(
            ProbeSessionTurn.session_id == session_id
        ).scalar_subquery()
        await self.session.execute(
            update(ProbeSession).where(ProbeSession.id == session_id)
            .values(n_turns=count, last_seen_at=now),
            execution_options={"synchronize_session": False},
        )
        await self.session.commit()
        result = await self.session.execute(
            select(ProbeSession.n_turns).where(ProbeSession.id == session_id)
        )
        return int(result.scalar_one_or_none() or 0)

    # ── reads ───────────────────────────────────────────────────────────────────

    async def list_sessions(
        self,
        *,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        served_model: Optional[str] = None,
        behaviour: Optional[str] = None,
        fired_only: bool = False,
        source: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        after: Optional[list[Any]] = None,
    ) -> list[ProbeSession]:
        stmt = select(ProbeSession).order_by(
            ProbeSession.last_seen_at.desc(), ProbeSession.id.desc()
        )
        if after is not None:
            # 039 keyset (FR-39.15): strictly after the previous page's last (last_seen_at, id).
            from sqlalchemy import and_, or_

            at = datetime.fromisoformat(str(after[0]))
            stmt = stmt.where(or_(
                ProbeSession.last_seen_at < at,
                and_(ProbeSession.last_seen_at == at, ProbeSession.id < str(after[1])),
            ))
        if since is not None:
            stmt = stmt.where(ProbeSession.last_seen_at >= since)
        if until is not None:
            stmt = stmt.where(ProbeSession.opened_at <= until)
        if served_model is not None:
            stmt = stmt.where(ProbeSession.served_model == served_model)
        if source is not None:
            stmt = stmt.where(ProbeSession.source == source)
        if behaviour is not None or fired_only:
            events = select(ProbeEvent.session_id).where(ProbeEvent.session_id.is_not(None))
            if behaviour is not None:
                events = events.join(Probe, Probe.id == ProbeEvent.probe_id).where(
                    Probe.behaviour_key == behaviour
                )
            if fired_only:
                events = events.where(ProbeEvent.verdict.is_(True))
            stmt = stmt.where(ProbeSession.id.in_(events))
        result = await self.session.execute(stmt.limit(limit).offset(offset))
        return list(result.scalars().all())

    async def transcribed_counts(self, session_ids: list[str]) -> dict[str, int]:
        if not session_ids:
            return {}
        result = await self.session.execute(
            select(ProbeSessionTurn.session_id, func.count())
            .where(ProbeSessionTurn.session_id.in_(session_ids),
                   ProbeSessionTurn.has_transcript.is_(True))
            .group_by(ProbeSessionTurn.session_id)
        )
        return {sid: int(n) for sid, n in result.all()}

    async def events_for_sessions(self, session_ids: list[str]) -> list[tuple[ProbeEvent, Any]]:
        """(event, probe) pairs for the sessions, oldest first. Context text is not read here."""
        if not session_ids:
            return []
        result = await self.session.execute(
            select(ProbeEvent, Probe)
            .join(Probe, Probe.id == ProbeEvent.probe_id)
            .where(ProbeEvent.session_id.in_(session_ids))
            .order_by(ProbeEvent.turn_index, ProbeEvent.id)
        )
        return [(e, p) for e, p in result.all()]

    async def turns(self, session_id: str) -> list[ProbeSessionTurn]:
        result = await self.session.execute(
            select(ProbeSessionTurn).where(ProbeSessionTurn.session_id == session_id)
            .order_by(ProbeSessionTurn.turn_index, ProbeSessionTurn.id)
        )
        return list(result.scalars().all())

    async def transcript(
        self, session_id: str, turn_index: Optional[int] = None
    ) -> list[tuple[ProbeSessionTurn, ProbeTranscript]]:
        """⚠ THE ONE READ OF USER TEXT. Backs `GET /api/probes/sessions/{id}/transcript` only."""
        stmt = (
            select(ProbeSessionTurn, ProbeTranscript)
            .join(ProbeTranscript, ProbeTranscript.turn_id == ProbeSessionTurn.id)
            .where(ProbeSessionTurn.session_id == session_id)
            .order_by(ProbeSessionTurn.turn_index, ProbeSessionTurn.id)
        )
        if turn_index is not None:
            stmt = stmt.where(ProbeSessionTurn.turn_index == turn_index)
        result = await self.session.execute(stmt)
        return [(t, x) for t, x in result.all()]

    # ── sample (036 phase 4, FR-36.12) ──────────────────────────────────────────

    async def sample_candidates(
        self,
        *,
        probe_ids: Optional[list[str]] = None,
        behaviour: Optional[str] = None,
        window: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        require_transcript: bool = True,
        limit: int = 200_000,
    ) -> list[tuple[ProbeEvent, Any, ProbeSessionTurn]]:
        """Scored live events that belong to a recorded turn, with their probe and turn.

        Bounded (events are capped per probe by retention, so this stays small); the stratified,
        seeded order is computed by the caller in Python, not SQL (FTDD §5.1, TC-7).
        """
        stmt = (
            select(ProbeEvent, Probe, ProbeSessionTurn)
            .join(Probe, Probe.id == ProbeEvent.probe_id)
            .join(ProbeSessionTurn, (ProbeSessionTurn.request_id == ProbeEvent.request_id)
                  & (ProbeSessionTurn.session_id == ProbeEvent.session_id))
            .where(ProbeEvent.scored.is_(True), ProbeEvent.session_id.is_not(None),
                   ProbeEvent.origin == "live")
        )
        if probe_ids:
            stmt = stmt.where(ProbeEvent.probe_id.in_(probe_ids))
        if behaviour is not None:
            stmt = stmt.where(Probe.behaviour_key == behaviour)
        if window is not None:
            stmt = stmt.where(ProbeEvent.window == window)
        if since is not None:
            stmt = stmt.where(ProbeEvent.created_at >= since)
        if until is not None:
            stmt = stmt.where(ProbeEvent.created_at <= until)
        if require_transcript:
            stmt = stmt.where(ProbeSessionTurn.has_transcript.is_(True))
        result = await self.session.execute(stmt.order_by(ProbeEvent.id).limit(limit))
        return [(e, p, t) for e, p, t in result.all()]

    # ── delete ──────────────────────────────────────────────────────────────────

    async def delete_session(self, session_id: str, *, delete_events: bool) -> dict[str, int]:
        """Remove a session, its turns and transcripts; its events go too only when asked.

        ⚠ ONE TRANSACTION, EVENTS FIRST. With `keep`, the FK's `SET NULL` detaches them and they
        keep `turn_index`; with `delete` they are removed explicitly before the session row.
        Counts are taken before the delete so the response says what went.
        """
        def _count(model: Any, column: Any) -> Any:
            return select(func.count()).select_from(model).where(column == session_id)

        counts = {
            "turns": int((await self.session.execute(
                _count(ProbeSessionTurn, ProbeSessionTurn.session_id))).scalar_one()),
            "transcripts": int((await self.session.execute(
                _count(ProbeTranscript, ProbeTranscript.session_id))).scalar_one()),
            "events": int((await self.session.execute(
                _count(ProbeEvent, ProbeEvent.session_id))).scalar_one()),
        }
        if delete_events:
            await self.session.execute(
                delete(ProbeEvent).where(ProbeEvent.session_id == session_id),
                execution_options={"synchronize_session": False},
            )
        else:
            # Explicit, not left to the FK: SQLite without the pragma would otherwise orphan them.
            await self.session.execute(
                update(ProbeEvent).where(ProbeEvent.session_id == session_id)
                .values(session_id=None),
                execution_options={"synchronize_session": False},
            )
        await self.session.execute(
            delete(ProbeTranscript).where(ProbeTranscript.session_id == session_id),
            execution_options={"synchronize_session": False},
        )
        await self.session.execute(
            delete(ProbeSessionTurn).where(ProbeSessionTurn.session_id == session_id),
            execution_options={"synchronize_session": False},
        )
        await self.session.execute(
            update(ProbeSession).where(ProbeSession.predecessor_id == session_id)
            .values(predecessor_id=None),
            execution_options={"synchronize_session": False},
        )
        await self.session.execute(
            delete(ProbeSession).where(ProbeSession.id == session_id),
            execution_options={"synchronize_session": False},
        )
        await self.session.commit()
        return {
            "turns": counts["turns"],
            "transcripts": counts["transcripts"],
            "events_deleted": counts["events"] if delete_events else 0,
            "events_kept": 0 if delete_events else counts["events"],
        }

    # ── retention ───────────────────────────────────────────────────────────────

    async def _delete_ids(self, model: Any, ids_query: Any) -> int:
        removed = 0
        while True:
            ids = list((await self.session.execute(ids_query.limit(RETENTION_BATCH))).scalars())
            if not ids:
                break
            result = await self.session.execute(
                delete(model).where(model.id.in_(ids)),
                execution_options={"synchronize_session": False},
            )
            await self.session.commit()
            removed += int(result.rowcount or 0)
            if len(ids) < RETENTION_BATCH:
                break
        return removed

    async def prune_transcripts(self, *, max_age: timedelta, max_turns: int, now: datetime) -> int:
        """Transcripts older than `max_age`, then the oldest beyond `max_turns`. The turn keeps
        existing (its verdicts are monitoring evidence) and is marked as having no text."""
        cutoff = now - max_age
        removed = await self._delete_ids(
            ProbeTranscript, select(ProbeTranscript.id).where(ProbeTranscript.created_at < cutoff)
        )
        if max_turns > 0:
            keep = select(ProbeTranscript.id).order_by(
                ProbeTranscript.created_at.desc(), ProbeTranscript.id.desc()
            ).limit(max_turns)
            removed += await self._delete_ids(
                ProbeTranscript,
                select(ProbeTranscript.id).where(ProbeTranscript.id.not_in(keep.scalar_subquery())),
            )
        if removed:
            await self.session.execute(
                update(ProbeSessionTurn)
                .where(ProbeSessionTurn.has_transcript.is_(True),
                       ProbeSessionTurn.id.not_in(select(ProbeTranscript.turn_id)))
                .values(has_transcript=False),
                execution_options={"synchronize_session": False},
            )
            await self.session.commit()
        return removed

    async def prune_sessions(
        self, *, max_age: timedelta, now: datetime, idle: timedelta = timedelta(minutes=30)
    ) -> int:
        """Sessions not seen within `max_age`, and sessions with no turn left once they are no
        longer open (an open one may be a request still generating, whose turn is not written yet).
        Their turns and transcripts go with them; their events detach (SET NULL)."""
        cutoff = now - max_age
        idle_cutoff = now - idle
        stale = select(ProbeSession.id).where(
            (ProbeSession.last_seen_at < cutoff)
            | (
                (ProbeSession.last_seen_at < idle_cutoff)
                & ProbeSession.id.not_in(select(ProbeSessionTurn.session_id))
            )
        )
        removed = 0
        while True:
            ids = list((await self.session.execute(stale.limit(RETENTION_BATCH))).scalars())
            if not ids:
                break
            for model, column in (
                (ProbeTranscript, ProbeTranscript.session_id),
                (ProbeSessionTurn, ProbeSessionTurn.session_id),
            ):
                await self.session.execute(
                    delete(model).where(column.in_(ids)),
                    execution_options={"synchronize_session": False},
                )
            await self.session.execute(
                update(ProbeEvent).where(ProbeEvent.session_id.in_(ids)).values(session_id=None),
                execution_options={"synchronize_session": False},
            )
            await self.session.execute(
                update(ProbeSession).where(ProbeSession.predecessor_id.in_(ids))
                .values(predecessor_id=None),
                execution_options={"synchronize_session": False},
            )
            result = await self.session.execute(
                delete(ProbeSession).where(ProbeSession.id.in_(ids)),
                execution_options={"synchronize_session": False},
            )
            await self.session.commit()
            removed += int(result.rowcount or 0)
            if len(ids) < RETENTION_BATCH:
                break
        return removed
