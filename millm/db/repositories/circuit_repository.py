"""
Circuit repository (Feature 13: Circuit Import).

CRUD over the ``circuits`` table plus the single-active guard: at most one
circuit may be active at a time (mirrors ProfileRepository.set_active).
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from millm.db.models.circuit import Circuit


class CircuitRepository:
    """Data access for imported circuits."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, **kwargs: Any) -> Circuit:
        """Insert a new circuit row."""
        circuit = Circuit(**kwargs)
        self.session.add(circuit)
        await self.session.commit()
        await self.session.refresh(circuit)
        return circuit

    async def get(self, circuit_id: str) -> Circuit | None:
        """Fetch a circuit by id."""
        result = await self.session.execute(
            select(Circuit).where(Circuit.id == circuit_id)
        )
        return result.scalar_one_or_none()

    async def get_by_name(self, name: str) -> Circuit | None:
        """Fetch a circuit by its unique name."""
        result = await self.session.execute(
            select(Circuit).where(Circuit.name == name)
        )
        return result.scalar_one_or_none()

    async def get_all(
        self,
        *,
        min_rung: int | None = None,
        serveable: bool | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Circuit]:
        """List circuits, newest first, with optional filters."""
        stmt = select(Circuit).order_by(Circuit.created_at.desc())
        if min_rung is not None:
            stmt = stmt.where(Circuit.rung >= min_rung)
        if serveable is not None:
            stmt = stmt.where(Circuit.serveable == serveable)
        if offset:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count(
        self, *, min_rung: int | None = None, serveable: bool | None = None
    ) -> int:
        """Total circuits matching the filters (for pagination meta)."""
        stmt = select(Circuit.id)
        if min_rung is not None:
            stmt = stmt.where(Circuit.rung >= min_rung)
        if serveable is not None:
            stmt = stmt.where(Circuit.serveable == serveable)
        result = await self.session.execute(stmt)
        return len(list(result.scalars().all()))

    async def get_active(self) -> Circuit | None:
        """The currently active circuit, if any."""
        result = await self.session.execute(
            select(Circuit).where(Circuit.is_active == True)  # noqa: E712
        )
        return result.scalar_one_or_none()

    async def update(self, circuit_id: str, **kwargs: Any) -> Circuit | None:
        """Patch a circuit row."""
        circuit = await self.get(circuit_id)
        if circuit is None:
            return None
        for key, value in kwargs.items():
            if hasattr(circuit, key):
                setattr(circuit, key, value)
        circuit.updated_at = datetime.now(timezone.utc)
        await self.session.commit()
        await self.session.refresh(circuit)
        return circuit

    async def set_active(
        self, circuit_id: str, *, serving_mode: str | None = None
    ) -> Circuit | None:
        """Activate a circuit, deactivating any other active circuit first.

        The partial unique index enforces one active row; deactivating first
        keeps the write ordering safe.
        """
        await self.deactivate_all()
        circuit = await self.get(circuit_id)
        if circuit is None:
            return None
        circuit.is_active = True
        if serving_mode is not None:
            circuit.serving_mode = serving_mode
        circuit.updated_at = datetime.now(timezone.utc)
        await self.session.commit()
        await self.session.refresh(circuit)
        return circuit

    async def deactivate(self, circuit_id: str) -> Circuit | None:
        """Deactivate one circuit and clear its serving mode."""
        circuit = await self.get(circuit_id)
        if circuit is None:
            return None
        circuit.is_active = False
        circuit.serving_mode = None
        circuit.updated_at = datetime.now(timezone.utc)
        await self.session.commit()
        await self.session.refresh(circuit)
        return circuit

    async def deactivate_all(self) -> int:
        """Deactivate every active circuit. Returns how many were changed."""
        result = await self.session.execute(
            select(Circuit).where(Circuit.is_active == True)  # noqa: E712
        )
        circuits = list(result.scalars().all())
        for circuit in circuits:
            circuit.is_active = False
            circuit.serving_mode = None
            circuit.updated_at = datetime.now(timezone.utc)
        if circuits:
            await self.session.commit()
        return len(circuits)

    async def delete(self, circuit_id: str) -> bool:
        """Delete a circuit. Returns True when a row was removed."""
        circuit = await self.get(circuit_id)
        if circuit is None:
            return False
        await self.session.delete(circuit)
        await self.session.commit()
        return True
