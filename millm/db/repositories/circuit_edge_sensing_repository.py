"""Repository for circuit edge sensing events (Feature 15).

Mirrors ``SensingRepository``: no transaction management — the caller commits.
Ordering is ``(created_at desc, id desc)`` everywhere, because one flush
inserts many rows sharing a ``created_at`` and the id tiebreak is what keeps
paging and retention deterministic.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from millm.db.models.circuit_edge_sensing_event import CircuitEdgeSensingEvent


class CircuitEdgeSensingRepository:
    """Persistence for observed up→down edge firings."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_many(
        self, events: list[dict[str, Any]]
    ) -> list[CircuitEdgeSensingEvent]:
        rows = [CircuitEdgeSensingEvent(**e) for e in events]
        self.session.add_all(rows)
        await self.session.flush()
        return rows

    async def list_events(
        self,
        circuit_id: Optional[str] = None,
        edge_key: Optional[str] = None,
        limit: int = 50,
        since: Optional[datetime] = None,
    ) -> list[CircuitEdgeSensingEvent]:
        stmt = select(CircuitEdgeSensingEvent)
        if circuit_id is not None:
            stmt = stmt.where(CircuitEdgeSensingEvent.circuit_id == circuit_id)
        if edge_key is not None:
            stmt = stmt.where(CircuitEdgeSensingEvent.edge_key == edge_key)
        if since is not None:
            stmt = stmt.where(CircuitEdgeSensingEvent.created_at >= since)
        stmt = stmt.order_by(
            CircuitEdgeSensingEvent.created_at.desc(),
            CircuitEdgeSensingEvent.id.desc(),
        ).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get(self, event_id: int) -> Optional[CircuitEdgeSensingEvent]:
        result = await self.session.execute(
            select(CircuitEdgeSensingEvent).where(
                CircuitEdgeSensingEvent.id == event_id
            )
        )
        return result.scalar_one_or_none()

    async def count(
        self,
        circuit_id: Optional[str] = None,
        edge_key: Optional[str] = None,
    ) -> int:
        stmt = select(func.count()).select_from(CircuitEdgeSensingEvent)
        if circuit_id is not None:
            stmt = stmt.where(CircuitEdgeSensingEvent.circuit_id == circuit_id)
        if edge_key is not None:
            stmt = stmt.where(CircuitEdgeSensingEvent.edge_key == edge_key)
        result = await self.session.execute(stmt)
        return int(result.scalar() or 0)

    async def clear(self, circuit_id: Optional[str] = None) -> int:
        stmt = delete(CircuitEdgeSensingEvent)
        if circuit_id is not None:
            stmt = stmt.where(CircuitEdgeSensingEvent.circuit_id == circuit_id)
        result = await self.session.execute(
            stmt, execution_options={"synchronize_session": False}
        )
        return int(result.rowcount or 0)

    async def prune_aged(self, max_age_days: int) -> int:
        """Drop rows older than the age window, across all circuits."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        result = await self.session.execute(
            delete(CircuitEdgeSensingEvent).where(
                CircuitEdgeSensingEvent.created_at < cutoff
            ),
            execution_options={"synchronize_session": False},
        )
        return int(result.rowcount or 0)

    async def prune(self, circuit_id: str, cap: int, max_age_days: int) -> int:
        """Enforce retention for one circuit: age window, then newest `cap`."""
        deleted = 0

        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        result = await self.session.execute(
            delete(CircuitEdgeSensingEvent)
            .where(CircuitEdgeSensingEvent.circuit_id == circuit_id)
            .where(CircuitEdgeSensingEvent.created_at < cutoff),
            execution_options={"synchronize_session": False},
        )
        deleted += int(result.rowcount or 0)

        keep_ids = (
            select(CircuitEdgeSensingEvent.id)
            .where(CircuitEdgeSensingEvent.circuit_id == circuit_id)
            .order_by(
                CircuitEdgeSensingEvent.created_at.desc(),
                CircuitEdgeSensingEvent.id.desc(),
            )
            .limit(cap)
        )
        result = await self.session.execute(
            delete(CircuitEdgeSensingEvent)
            .where(CircuitEdgeSensingEvent.circuit_id == circuit_id)
            .where(CircuitEdgeSensingEvent.id.not_in(keep_ids)),
            execution_options={"synchronize_session": False},
        )
        deleted += int(result.rowcount or 0)
        return deleted
