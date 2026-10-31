"""
Circuit layer claims (Feature 19: Concurrent Circuit Serving).

A CLAIM is one circuit's hold on one layer. It exists because the unit of
contention is the LAYER, not the feature:

    modified = original + Σ(strength_i × W_dec[i])      # sae_wrapper.py:444

Two circuits steering *different* features on the same layer still contend —
both write into that layer's single steering dict and both contribute to the
same residual-stream sum. Nothing bounds that sum: the ±200 `clamp_steering`
bounds each member individually.

The GPU close-out (2026-07-20) measured what that costs, holding prompt, seed
and temperature fixed on LFM2.5-1.2B-Instruct:

    1 layer,  1 member @ strength 5   -> coherent, indistinguishable from base
    2 LAYERS, 1 member each @ 5       -> DEGENERATE (repeated " lé" tokens)

Cross-layer compounding destroys generation at two layers, two orders of
magnitude below the per-member clamp. So silent composition is not a
theoretical hazard — it is a reliable way to produce garbage the operator
cannot distinguish from a bad circuit. Hence: claim the layer, refuse by
default, and compose only on an explicit, loud, informed override.
"""

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from millm.db.base import Base

JSONVariant = JSON().with_variant(JSONB(), "postgresql")


class CircuitLayerClaim(Base):
    """One circuit's hold on one layer.

    A claim is LIVE while ``released_at IS NULL``. The partial unique index
    ``uq_circuit_layer_claim_live`` enforces at most one EXCLUSIVE live claim
    per layer, in the DATABASE rather than in service code — so two concurrent
    activations racing for the same layer are decided by the index, not by a
    check-then-act window that a second request can slip through.

    ``composed`` rows are excluded from that index deliberately: composition is
    the explicit override, and once an operator has accepted it, several
    circuits legitimately hold the same layer.

    Attributes:
        circuit_id: Owning circuit (CASCADE — deleting a circuit drops its
            claims; a claim outliving its circuit is an orphan that would
            refuse activations forever for a circuit nobody can deactivate).
        layer: The claimed transformer layer.
        claimed_at: When the claim was taken.
        released_at: NULL while live. Set on release rather than deleting, so
            the history of who held what survives a post-mortem.
        composed: True when this claim coexists with another on the same layer
            by explicit ``allow_layer_overlap``.
        steering_keys: The ``feature_idx`` values this circuit wrote on this
            layer. Release uses these to remove ONLY its own keys — the
            co-tenant's keys must survive, which a blanket
            ``clear_circuit_steering()`` would destroy.
    """

    __tablename__ = "circuit_layer_claims"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    circuit_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("circuits.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    layer: Mapped[int] = mapped_column(Integer, nullable=False)
    claimed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    released_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    composed: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    #: The feature indices this circuit wrote on this layer, so release can
    #: remove its own keys and leave a co-tenant's alone.
    steering_keys: Mapped[Optional[list[Any]]] = mapped_column(
        JSONVariant, nullable=True
    )

    __table_args__ = (
        # At most one EXCLUSIVE live claim per layer. Both dialect predicates
        # are required: without `sqlite_where` the index is unconditional on
        # SQLite, every released claim collides, and — worse for a review —
        # every contention test passes for the wrong reason.
        Index(
            "uq_circuit_layer_claim_live",
            "layer",
            unique=True,
            postgresql_where=(released_at.is_(None) & (composed == False)),  # noqa: E712
            sqlite_where=(released_at.is_(None) & (composed == False)),  # noqa: E712
        ),
        Index("idx_circuit_layer_claims_live", "layer", "released_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        state = "live" if self.released_at is None else "released"
        return (
            f"<CircuitLayerClaim {self.circuit_id} L{self.layer} "
            f"{state}{' composed' if self.composed else ''}>"
        )
