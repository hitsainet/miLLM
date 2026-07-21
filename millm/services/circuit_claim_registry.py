"""
Circuit layer claim registry (Feature 19: Concurrent Circuit Serving).

Two circuits may serve at once IFF their claim sets are disjoint. This module
owns that decision and the bookkeeping behind it.

The central distinction — kept structurally separate here because collapsing it
is the defect this design most wants to avoid:

  * CONTENTION — two circuits want the same LAYER. Overridable, because
    additive composition on a layer is a coherent thing an operator might
    intend (a compounding study). It is refused BY DEFAULT because the GPU
    close-out measured two steered layers at strength 5 destroying generation,
    two orders of magnitude below the per-member clamp.

  * COLLISION — two circuits name the same (LAYER, FEATURE_IDX). NEVER
    overridable. `set_steering_batch` merges into one dict, so one strength
    silently overwrites the other and the served value belongs to NEITHER
    author. There is no honest composition of that case, so it has no override
    — and the check for it must run FIRST and UNCONDITIONALLY, before any
    `allow_layer_overlap` branch, or the override becomes a way to reach it.

Release is per-owner: deactivating a circuit clears exactly the
`(layer, feature_idx)` keys IT wrote, never the whole layer dict, which would
tear out a co-tenant's steering while its row still reads active.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import sqlalchemy as sa
import structlog
from sqlalchemy.exc import IntegrityError

from millm.db.models.circuit_layer_claim import CircuitLayerClaim

logger = structlog.get_logger()


@dataclass(frozen=True)
class LayerClaim:
    """One live claim, as the registry reports it."""

    circuit_id: str
    layer: int
    composed: bool = False
    steering_keys: tuple[int, ...] = ()
    circuit_name: Optional[str] = None


@dataclass(frozen=True)
class ContentionVerdict:
    """The outcome of assessing a proposed claim set against what is live.

    `has_contention` and `has_collision` are DELIBERATELY separate properties
    rather than one severity field. One is overridable and one is never, and a
    single field invites a caller to compare it with `>=` and accidentally let
    a collision through an override branch.
    """

    #: Layers wanted by the requester that an incumbent already holds.
    contended_layers: tuple[int, ...] = ()
    #: Incumbents on those layers: {circuit_id: (name, layers)}.
    incumbents: dict[str, tuple[Optional[str], tuple[int, ...]]] = field(
        default_factory=dict
    )
    #: (layer, feature_idx, incumbent_circuit_id) triples named by BOTH.
    colliding_keys: tuple[tuple[int, int, str], ...] = ()

    @property
    def has_contention(self) -> bool:
        return bool(self.contended_layers)

    @property
    def has_collision(self) -> bool:
        """Never overridable. See the module docstring."""
        return bool(self.colliding_keys)

    @property
    def is_clear(self) -> bool:
        return not self.has_contention and not self.has_collision


class CircuitClaimRegistry:
    """Assess, take and release per-layer claims."""

    def __init__(self, session: Any) -> None:
        self._session = session

    # ── Reads ──────────────────────────────────────────────────────────────

    async def live_claims(self) -> list[LayerClaim]:
        """Every claim not yet released."""
        rows = (
            await self._session.execute(
                sa.select(CircuitLayerClaim).where(
                    CircuitLayerClaim.released_at.is_(None)
                )
            )
        ).scalars().all()
        return [
            LayerClaim(
                circuit_id=r.circuit_id,
                layer=r.layer,
                composed=bool(r.composed),
                steering_keys=tuple(r.steering_keys or ()),
            )
            for r in rows
        ]

    async def assess(
        self,
        circuit_id: str,
        layers: set[int],
        *,
        steering_keys: Optional[dict[int, set[int]]] = None,
    ) -> ContentionVerdict:
        """Would claiming `layers` for `circuit_id` conflict with what is live?

        SELF-EXCLUDING (EC-19.3): a circuit re-activating, or extending its own
        claim set, does not contend with itself. Without this an idempotent
        re-activation would refuse against its own incumbent claim, which is
        both wrong and extremely confusing to debug.
        """
        live = [c for c in await self.live_claims() if c.circuit_id != circuit_id]
        if not live:
            return ContentionVerdict()

        names = await self._names_for({c.circuit_id for c in live})

        contended = sorted({c.layer for c in live if c.layer in layers})

        incumbents: dict[str, tuple[Optional[str], tuple[int, ...]]] = {}
        for claim in live:
            if claim.layer not in layers:
                continue
            name, held = incumbents.get(
                claim.circuit_id, (names.get(claim.circuit_id), ())
            )
            incumbents[claim.circuit_id] = (name, tuple(sorted(set(held) | {claim.layer})))

        # Collisions are computed PER HOLDER, and independently of contention:
        # a collision is still a collision on a layer the requester is being
        # allowed to compose onto.
        colliding: list[tuple[int, int, str]] = []
        if steering_keys:
            for claim in live:
                wanted = steering_keys.get(claim.layer)
                if not wanted:
                    continue
                for idx in sorted(wanted & set(claim.steering_keys)):
                    colliding.append((claim.layer, int(idx), claim.circuit_id))

        return ContentionVerdict(
            contended_layers=tuple(contended),
            incumbents=incumbents,
            colliding_keys=tuple(colliding),
        )

    async def _names_for(self, circuit_ids: set[str]) -> dict[str, Optional[str]]:
        if not circuit_ids:
            return {}
        from millm.db.models.circuit import Circuit

        rows = (
            await self._session.execute(
                sa.select(Circuit.id, Circuit.name).where(Circuit.id.in_(circuit_ids))
            )
        ).all()
        return {r[0]: r[1] for r in rows}

    # ── Writes ─────────────────────────────────────────────────────────────

    async def claim(
        self,
        circuit_id: str,
        layers: set[int],
        *,
        composed: bool = False,
        steering_keys: Optional[dict[int, set[int]]] = None,
    ) -> list[int]:
        """Take claims. Returns the layers claimed.

        The `IntegrityError` catch is the RACE HANDLER (EC-19.7), not
        defensive padding: `assess()` and this INSERT are a check-then-act
        pair, and two concurrent activations can both pass `assess` before
        either inserts. The partial unique index is what actually decides, and
        the loser is converted into an ordinary contention refusal so it reads
        identically to the sequential case.
        """
        keys = steering_keys or {}
        for layer in sorted(layers):
            self._session.add(
                CircuitLayerClaim(
                    circuit_id=circuit_id,
                    layer=int(layer),
                    composed=composed,
                    steering_keys=sorted(keys.get(layer, set())) or None,
                )
            )
        try:
            await self._session.flush()
        except IntegrityError as exc:
            await self._session.rollback()
            logger.warning(
                "circuit_claim_race_lost",
                circuit_id=circuit_id,
                layers=sorted(layers),
                detail=(
                    "another activation claimed one of these layers between "
                    "the assessment and the insert — the database index "
                    "decided, as designed"
                ),
            )
            from millm.core.errors import CircuitLayerContentionError

            raise CircuitLayerContentionError(
                contended_layers=sorted(layers),
                incumbent_id=None,
                incumbent_name=None,
                requested_id=circuit_id,
                requested_name=None,
                detail="lost a concurrent race for these layers",
            ) from exc
        return sorted(layers)

    async def release(self, circuit_id: str) -> list[int]:
        """Release THIS circuit's live claims. Returns the layers released.

        Scoped to one circuit_id on purpose. A blanket release is the
        highest-consequence defect available in this feature: a circuit the
        operator never touched silently stops steering while its row still
        reports active.
        """
        rows = (
            await self._session.execute(
                sa.select(CircuitLayerClaim).where(
                    CircuitLayerClaim.circuit_id == circuit_id,
                    CircuitLayerClaim.released_at.is_(None),
                )
            )
        ).scalars().all()
        now = datetime.now(timezone.utc)
        released = []
        for row in rows:
            row.released_at = now
            released.append(row.layer)
        await self._session.flush()
        return sorted(released)

    async def mark_composed(self, circuit_id: str, layers: set[int]) -> None:
        """Flip the requester's AND the incumbents' rows on `layers`.

        Both sides must be marked. The exclusive partial index only tolerates
        the co-tenancy if NEITHER row is exclusive, and — more importantly for
        honesty — the rung header is suppressed for any circuit sitting on a
        composed layer, which cannot be determined from the requester's rows
        alone.
        """
        if not layers:
            return
        await self._session.execute(
            sa.update(CircuitLayerClaim)
            .where(
                CircuitLayerClaim.layer.in_(sorted(layers)),
                CircuitLayerClaim.released_at.is_(None),
            )
            .values(composed=True)
        )
        await self._session.flush()

    async def reconcile(self, *, allow_concurrent: bool) -> dict[str, Any]:
        """Startup reconciliation. Runs UNCONDITIONALLY.

        Two jobs:

        * Drop ORPHAN claims (EC-19.4) — claims whose circuit is no longer
          active. A stale claim refuses activations forever for a circuit
          nobody can deactivate, because deactivating an inactive circuit is a
          no-op.
        * With the flag FALSE, demote to a single active circuit (EC-19.5).
          A database written while the flag was true must not keep serving two
          circuits after an operator turns it off — the flag would be a lie.

        Every demotion is logged. A silent demotion is indistinguishable from
        the single-active disarm this feature exists to replace.
        """
        from millm.db.models.circuit import Circuit

        result: dict[str, Any] = {"orphans_released": [], "demoted": []}

        live = await self.live_claims()
        if live:
            active_ids = set(
                (
                    await self._session.execute(
                        sa.select(Circuit.id).where(Circuit.is_active.is_(True))
                    )
                ).scalars().all()
            )
            orphan_ids = {c.circuit_id for c in live} - active_ids
            for circuit_id in sorted(orphan_ids):
                layers = await self.release(circuit_id)
                result["orphans_released"].append(
                    {"circuit_id": circuit_id, "layers": layers}
                )
                logger.warning(
                    "circuit_claim_orphan_released",
                    circuit_id=circuit_id,
                    layers=layers,
                    detail=(
                        "claims outlived their circuit's activation — they "
                        "would have refused every future activation on these "
                        "layers"
                    ),
                )

        if not allow_concurrent:
            actives = (
                await self._session.execute(
                    sa.select(Circuit.id, Circuit.name)
                    .where(Circuit.is_active.is_(True))
                    .order_by(Circuit.updated_at.desc(), Circuit.id.desc())
                )
            ).all()
            for circuit_id, name in actives[1:]:
                await self.release(circuit_id)
                await self._session.execute(
                    sa.update(Circuit)
                    .where(Circuit.id == circuit_id)
                    .values(is_active=False)
                )
                result["demoted"].append({"circuit_id": circuit_id, "name": name})
                logger.warning(
                    "circuit_demoted_by_flag",
                    circuit_id=circuit_id,
                    name=name,
                    detail=(
                        "CIRCUIT_ALLOW_CONCURRENT is false but this database "
                        "had several active circuits — demoted all but the "
                        "most recently updated"
                    ),
                )
            if result["demoted"]:
                await self._session.flush()

        return result
