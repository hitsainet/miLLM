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

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from millm.db.models.probe import Probe, ProbeEvent


class ProbeRepository:
    """Async CRUD for `Probe` rows."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, **fields: Any) -> Probe:
        probe = Probe(**fields)
        self.session.add(probe)
        await self.session.flush()
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
        for key, value in fields.items():
            setattr(probe, key, value)
        await self.session.flush()
        return probe

    async def delete(self, probe: Probe) -> None:
        """Delete a probe. Its events go with it (FK ON DELETE CASCADE)."""
        await self.session.delete(probe)
        await self.session.flush()

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
        await self.session.flush()
        return len(armed)


class ProbeEventRepository:
    """Async writes + retention for `ProbeEvent` rows."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_many(self, events: list[dict[str, Any]]) -> list[ProbeEvent]:
        rows = [ProbeEvent(**event) for event in events]
        self.session.add_all(rows)
        await self.session.flush()
        return rows

    async def list_events(
        self,
        *,
        probe_id: str | None = None,
        request_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ProbeEvent]:
        stmt = select(ProbeEvent).order_by(ProbeEvent.created_at.desc(), ProbeEvent.id.desc())
        if probe_id is not None:
            stmt = stmt.where(ProbeEvent.probe_id == probe_id)
        if request_id is not None:
            stmt = stmt.where(ProbeEvent.request_id == request_id)
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

    async def clear(self, probe_id: str | None = None) -> int:
        stmt = delete(ProbeEvent)
        if probe_id is not None:
            stmt = stmt.where(ProbeEvent.probe_id == probe_id)
        result = await self.session.execute(stmt)
        await self.session.flush()
        return int(result.rowcount or 0)

    async def prune_aged(self, max_age_days: int) -> int:
        """Delete events older than the window. `max_age_days <= 0` disables age pruning."""
        if max_age_days <= 0:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        result = await self.session.execute(
            delete(ProbeEvent).where(ProbeEvent.created_at < cutoff)
        )
        await self.session.flush()
        return int(result.rowcount or 0)

    async def prune_to_cap(self, probe_id: str, cap: int) -> int:
        """Keep only the newest `cap` events for one probe.

        Selects the ids to keep and deletes the rest, rather than computing an offset and deleting
        "everything older than row N". An offset-based delete is wrong the moment two events share
        a timestamp — which they routinely do here, because every armed probe records an event for
        the same request within the same millisecond.
        """
        if cap <= 0:
            return 0
        keep = select(ProbeEvent.id).where(ProbeEvent.probe_id == probe_id).order_by(
            ProbeEvent.created_at.desc(), ProbeEvent.id.desc()
        ).limit(cap)
        result = await self.session.execute(
            delete(ProbeEvent).where(
                ProbeEvent.probe_id == probe_id,
                ProbeEvent.id.not_in(keep.scalar_subquery()),
            )
        )
        await self.session.flush()
        return int(result.rowcount or 0)

    async def prune(self, probe_id: str, *, cap: int, max_age_days: int) -> int:
        """Both retention rules, as one call. Returns the total number of rows removed."""
        removed = await self.prune_aged(max_age_days)
        removed += await self.prune_to_cap(probe_id, cap)
        return removed
