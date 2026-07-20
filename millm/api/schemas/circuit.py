"""Circuit serving/runtime schemas (Feature 12 partial; extended by Feature 13).

Feature 12 needs only the per-member serving shape — a circuit member keyed to
its layer and SAE with a frozen per-layer budget. Feature 013 (Circuit Import)
extends this module with the full ``mistudio.circuit-definition/v1`` document,
edges, and evidence rungs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CircuitMember(BaseModel):
    """One steerable member of a circuit at a given layer.

    The ``budget`` is the frozen per-layer strength authored in miStudio
    (γ=0 ⇒ B = B_dir); serving applies ``clamp_steering(budget · sign · λ)``
    where λ is the one global intensity. ``sae_id`` records which SAE the
    member's layer resolves to (advisory — resolution is by layer at serve
    time so an equivalent SAE on that layer still serves the member).
    """

    model_config = ConfigDict(extra="ignore")

    feature_idx: int = Field(..., ge=0, description="SAE feature index at this layer")
    layer: int = Field(..., ge=0, description="Transformer layer the member steers")
    budget: float = Field(
        ...,
        ge=-300.0,
        le=300.0,
        description="Frozen per-layer budget (B_dir); authored, not recomputed",
    )
    sign: Literal[1, -1] = Field(1, description="Steering direction")
    sae_id: str | None = Field(
        None, description="Advisory SAE id the layer resolves to (resolution is by layer)"
    )
    label: str | None = Field(None, max_length=200)
