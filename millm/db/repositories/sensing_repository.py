"""
Repository for sensing_events (Feature 11).

Persistence is bounded by construction: every flush prunes to the
per-cluster cap and the age window, so the table cannot grow without bound
even if the API is never called.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from millm.db.models.sensing_event import SensingEvent


class SensingRepository:
    """Async CRUD + retention for SensingEvent rows.

    The repository does not manage transactions — that's the caller's
    responsibility (matches ProfileRepository).
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_many(self, events: list[dict[str, Any]]) -> list[SensingEvent]:
        """Insert a batch of events (one request's flush).

        026 F-1: batch events of a (profile, batch, line) that already has events are SKIPPED —
        the row is being re-run after a crash, and a line's events are written together, so
        recording them again would double that line's counts. Live events are never filtered.
        Returns only the rows inserted.
        """
        events = await self._without_recorded_batch_lines(events)
        rows = [SensingEvent(**event) for event in events]
        self.session.add_all(rows)
        await self.session.flush()
        return rows

    async def _without_recorded_batch_lines(
        self, events: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        lines = {(e["profile_id"], e["batch_id"], e["batch_line"])
                 for e in events if e.get("batch_id") is not None}
        if not lines:
            return events
        existing = await self.session.execute(
            select(SensingEvent.profile_id, SensingEvent.batch_id, SensingEvent.batch_line)
            .where(SensingEvent.batch_id.in_({line[1] for line in lines}))
            .distinct()
        )
        seen = {tuple(r) for r in existing.all()}
        return [e for e in events if e.get("batch_id") is None
                or (e["profile_id"], e["batch_id"], e["batch_line"]) not in seen]

    async def list_events(
        self,
        profile_id: str | None = None,
        limit: int = 50,
        since: datetime | None = None,
    ) -> list[SensingEvent]:
        """Newest-first event listing, optionally scoped to a profile."""
        stmt = select(SensingEvent).order_by(SensingEvent.created_at.desc(),
                                             SensingEvent.id.desc())
        if profile_id is not None:
            stmt = stmt.where(SensingEvent.profile_id == profile_id)
        if since is not None:
            stmt = stmt.where(SensingEvent.created_at > since)
        stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get(self, event_id: int) -> SensingEvent | None:
        result = await self.session.execute(
            select(SensingEvent).where(SensingEvent.id == event_id)
        )
        return result.scalar_one_or_none()

    async def count(self, profile_id: str | None = None) -> int:
        stmt = select(func.count(SensingEvent.id))
        if profile_id is not None:
            stmt = stmt.where(SensingEvent.profile_id == profile_id)
        result = await self.session.execute(stmt)
        return int(result.scalar_one())

    async def clear(self, profile_id: str | None = None) -> int:
        """Delete events (all, or one profile's). Returns rows deleted."""
        stmt = delete(SensingEvent)
        if profile_id is not None:
            stmt = stmt.where(SensingEvent.profile_id == profile_id)
        result = await self.session.execute(stmt)
        return int(result.rowcount or 0)

    async def prune_aged(self, max_age_days: int) -> int:
        """Age-window prune across ALL profiles (read-path retention)."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        result = await self.session.execute(
            delete(SensingEvent).where(SensingEvent.created_at < cutoff),
            execution_options={"synchronize_session": False},
        )
        return int(result.rowcount or 0)

    async def prune(
        self, profile_id: str, cap: int, max_age_days: int, origin: str = "live"
    ) -> int:
        """
        Enforce retention for one profile: drop rows older than the age
        window, then keep only the newest `cap`. Returns rows deleted.
        """
        deleted = 0

        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        result = await self.session.execute(
            delete(SensingEvent)
            .where(SensingEvent.profile_id == profile_id)
            .where(SensingEvent.created_at < cutoff),
            execution_options={"synchronize_session": False},
        )
        deleted += int(result.rowcount or 0)

        keep_ids = select(SensingEvent.id).where(
            SensingEvent.profile_id == profile_id, SensingEvent.origin == origin
        ).order_by(
            SensingEvent.created_at.desc(), SensingEvent.id.desc()
        ).limit(cap)
        result = await self.session.execute(
            delete(SensingEvent)
            .where(SensingEvent.profile_id == profile_id)
            .where(SensingEvent.origin == origin)
            .where(SensingEvent.id.not_in(keep_ids)),
            execution_options={"synchronize_session": False},
        )
        deleted += int(result.rowcount or 0)
        return deleted
