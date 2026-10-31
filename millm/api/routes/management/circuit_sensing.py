"""Circuit edge sensing routes (Feature 15).

Mirrors the Feature 11 sensing routes: the same envelope, the same read-path
prune throttle, and the same separation between persistent operator INTENT
(the ``sensing_enabled`` column) and runtime ``armed``.

House refusal style: no active circuit is a 200 with ``success: false`` in the
envelope, not a 404 — nothing is missing, the operation simply does not apply.
"""

import time
from datetime import datetime
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Path, Query
from sqlalchemy import select

from millm.api.schemas.circuit_sensing import (
    CircuitSensingEventListResponse,
    CircuitSensingEventResponse,
    CircuitSensingStatusResponse,
    CircuitSensingToggleResult,
)
from millm.api.schemas.common import ApiResponse
from millm.core.config import settings
from millm.core.errors import (
    CircuitNotFoundError,
    CircuitSensingEventNotFoundError,
    ValidationError,
)
from millm.core.logging import get_logger

logger = get_logger(__name__)

_PRUNE_ON_READ_INTERVAL_S = 600.0
_last_read_prune: list[float] = [0.0]

router = APIRouter(prefix="/api/circuit-sensing", tags=["circuit-sensing"])

CircuitId = Annotated[str, Path(description="Circuit ID")]


def _service():
    from millm.api.dependencies import get_circuit_sensing_service

    return get_circuit_sensing_service()


def _layer_saes() -> dict:
    """layer -> LoadedSAE, only for unambiguously attached layers."""
    from millm.services.sae_service import AttachedSAEState

    state = AttachedSAEState()
    out: dict = {}
    for entry in state.entries():
        resolved = state.by_layer(entry.layer)
        if resolved is not None:
            out[entry.layer] = resolved.sae
    return out


@router.get(
    "/status",
    response_model=ApiResponse[CircuitSensingStatusResponse],
    summary="Circuit edge sensing status",
)
async def circuit_sensing_status() -> ApiResponse[CircuitSensingStatusResponse]:
    from millm.db.base import async_session_factory
    from millm.db.models.circuit import Circuit

    data = _service().status(_layer_saes())

    # Persistent intent, reported DISTINCTLY from runtime armed: a circuit can
    # be enabled but unarmed because it is not active or its SAEs are absent.
    async with async_session_factory() as session:
        rows = await session.execute(
            select(Circuit.id, Circuit.name, Circuit.is_active).where(
                Circuit.sensing_enabled.is_(True)
            )
        )
        data["enabled_circuits"] = [
            {"id": r.id, "name": r.name, "is_active": bool(r.is_active)}
            for r in rows
        ]
    return ApiResponse.ok(CircuitSensingStatusResponse(**data))


@router.get(
    "/events",
    response_model=ApiResponse[CircuitSensingEventListResponse],
    summary="List observed edge firings",
)
async def list_circuit_sensing_events(
    circuit_id: Optional[str] = Query(None),
    edge_key: Optional[str] = Query(None, description="Filter to one edge"),
    limit: int = Query(50, ge=1, le=500),
    since: Optional[datetime] = Query(None),
) -> ApiResponse[CircuitSensingEventListResponse]:
    from millm.db.base import async_session_factory
    from millm.db.repositories.circuit_edge_sensing_repository import (
        CircuitEdgeSensingRepository,
    )

    async with async_session_factory() as session:
        repo = CircuitEdgeSensingRepository(session)

        # Age-prune on read, throttled — retention must not depend on write
        # traffic alone, or an idle deployment keeps events past the window.
        now = time.monotonic()
        if now - _last_read_prune[0] >= _PRUNE_ON_READ_INTERVAL_S:
            _last_read_prune[0] = now
            pruned = await repo.prune_aged(settings.CIRCUIT_SENSING_MAX_AGE_DAYS)
            if pruned:
                await session.commit()

        events = await repo.list_events(
            circuit_id=circuit_id, edge_key=edge_key, limit=limit, since=since
        )
        total = await repo.count(circuit_id=circuit_id, edge_key=edge_key)

    return ApiResponse.ok(
        CircuitSensingEventListResponse(
            total=total,
            events=[
                CircuitSensingEventResponse(**e.to_dict()) for e in events
            ],
        )
    )


@router.get(
    "/events/{event_id}",
    response_model=ApiResponse[CircuitSensingEventResponse],
    summary="One observed edge firing",
)
async def get_circuit_sensing_event(
    event_id: int,
) -> ApiResponse[CircuitSensingEventResponse]:
    from millm.db.base import async_session_factory
    from millm.db.repositories.circuit_edge_sensing_repository import (
        CircuitEdgeSensingRepository,
    )

    async with async_session_factory() as session:
        event = await CircuitEdgeSensingRepository(session).get(event_id)
    if event is None:
        raise CircuitSensingEventNotFoundError(
            f"Circuit sensing event {event_id} not found",
            details={"event_id": event_id},
        )
    return ApiResponse.ok(CircuitSensingEventResponse(**event.to_dict()))


@router.delete(
    "/events",
    response_model=ApiResponse[dict],
    summary="Clear observed edge firings",
)
async def clear_circuit_sensing_events(
    circuit_id: Optional[str] = Query(None),
) -> ApiResponse[dict]:
    from millm.db.base import async_session_factory
    from millm.db.repositories.circuit_edge_sensing_repository import (
        CircuitEdgeSensingRepository,
    )

    async with async_session_factory() as session:
        deleted = await CircuitEdgeSensingRepository(session).clear(
            circuit_id=circuit_id
        )
        await session.commit()
    return ApiResponse.ok({"deleted": deleted})


async def _toggle(circuit_id: str, enabled: bool) -> CircuitSensingToggleResult:
    """Persist the intent column, then live-arm/disarm when the circuit is
    the active one and its SAE set is attached."""
    from millm.db.base import async_session_factory
    from millm.db.repositories.circuit_repository import CircuitRepository

    service = _service()
    async with async_session_factory() as session:
        repo = CircuitRepository(session)
        circuit = await repo.get(circuit_id)
        if circuit is None:
            raise CircuitNotFoundError(
                f"Circuit '{circuit_id}' not found",
                details={"circuit_id": circuit_id},
            )
        circuit.sensing_enabled = enabled
        is_active = bool(circuit.is_active)
        meta = circuit.circuit_meta
        snapshot = _CircuitSnapshot(
            id=circuit.id, name=circuit.name, circuit_meta=meta
        )
        await session.commit()

    layer_saes = _layer_saes()
    unsensable: list = []
    message = ""
    if is_active:
        if enabled:
            from millm.api.schemas.circuit import CircuitDefinitionV1

            try:
                definition = CircuitDefinitionV1.model_validate(meta)
            except Exception as exc:
                raise ValidationError(
                    f"Sensing enabled but the stored definition is unreadable: {exc}",
                    details={"circuit_id": circuit_id},
                ) from exc
            try:
                unsensable = service.arm_for_circuit(
                    snapshot, definition, layer_saes
                )
            except ValueError as exc:
                # The column stays enabled — persistent intent survives — but
                # the caller is told exactly why arming refused rather than
                # getting a silent no-arm.
                raise ValidationError(
                    f"Sensing enabled but could not arm: {exc}",
                    details={"circuit_id": circuit_id},
                ) from exc
            if not service.is_armed:
                message = (
                    "Enabled, but no edge is currently sensable — see "
                    "unsensable_edges for why."
                )
        else:
            service.disarm(layer_saes)
    else:
        message = "Enabled; the circuit will arm when it is activated."

    return CircuitSensingToggleResult(
        circuit_id=circuit_id,
        enabled=enabled,
        armed=service.is_armed,
        unsensable_edges=[u.to_dict() for u in unsensable],
        message=message,
    )


class _CircuitSnapshot:
    """Detached view of the row, so arming never touches a closed session."""

    __slots__ = ("id", "name", "circuit_meta")

    def __init__(self, id: str, name: str, circuit_meta: Any):
        self.id = id
        self.name = name
        self.circuit_meta = circuit_meta


@router.post(
    "/{circuit_id}/enable",
    response_model=ApiResponse[CircuitSensingToggleResult],
    summary="Enable edge sensing for a circuit",
)
async def enable_circuit_sensing(
    circuit_id: CircuitId,
) -> ApiResponse[CircuitSensingToggleResult]:
    return ApiResponse.ok(await _toggle(circuit_id, True))


@router.post(
    "/{circuit_id}/disable",
    response_model=ApiResponse[CircuitSensingToggleResult],
    summary="Disable edge sensing for a circuit",
)
async def disable_circuit_sensing(
    circuit_id: CircuitId,
) -> ApiResponse[CircuitSensingToggleResult]:
    return ApiResponse.ok(await _toggle(circuit_id, False))
