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

    async def list_active(self) -> list[Circuit]:
        """EVERY active circuit (Feature 19).

        Several circuits may serve at once when their claim sets are disjoint,
        so "what is active" is a LIST. Callers that must act on all of them —
        releasing co-tenants, reporting status — use this.
        """
        result = await self.session.execute(
            select(Circuit)
            .where(Circuit.is_active == True)  # noqa: E712
            .order_by(Circuit.updated_at.desc(), Circuit.id.desc())
        )
        return list(result.scalars().all())

    async def get_active(self) -> Circuit | None:
        """The most recently activated circuit, if any.

        F19 R1-04: this used `scalar_one_or_none()`, which RAISES
        `MultipleResultsFound` the moment two circuits are active — the exact
        state this feature exists to make possible. Verified by execution
        before the fix.

        The consequence was not a loud failure. `_active_full_circuit` catches
        broadly and returns None, so every chat request would have served
        UNSTEERED while both circuit rows read active, and `GET /circuits/active`
        would 500. The feature's SUCCESS state broke the feature.

        Returns the most recently updated one, matching what an operator most
        recently asked for and what the migration downgrade keeps. Callers that
        need all of them use `list_active()` — this exists for the surfaces
        that are genuinely singular (a status summary, a dial target).
        """
        result = await self.session.execute(
            select(Circuit)
            .where(Circuit.is_active == True)  # noqa: E712
            .order_by(Circuit.updated_at.desc(), Circuit.id.desc())
            .limit(1)
        )
        return result.scalars().first()

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
        """Activate a circuit. Does NOT disturb other active circuits.

        F19 R2-01: this called `deactivate_all()` first, and that single line
        defeated the entire feature at the DB layer while every owner-map and
        claim-registry test passed.

        Activating two DISJOINT circuits left exactly one active row — proven
        by execution: the claim gate passed, both circuits steered through the
        owner map, and the first circuit's row read `is_active=False` with
        `serving_mode=None`. The model was steering with nothing recording it,
        `GET /circuits/active` reported only the second, and no operator could
        stop the first through any surface that reads the row.

        R1-04 fixed the READER (`get_active` no longer raises on two rows, and
        `list_active` was added) and never touched the WRITER, so
        `list_active()` returned a list that could never hold more than one
        element.

        Exclusivity is now enforced where it belongs: `circuit_layer_claims`
        arbitrates per LAYER, which is the unit contention actually has. The
        single-active index was dropped by migration 013 for the same reason.
        """
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
