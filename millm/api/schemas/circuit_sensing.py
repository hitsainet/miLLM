"""Circuit edge sensing API schemas (Feature 15).

``edge_rung_language`` is always the SERVER-rendered ladder phrase carried from
the moment of observation. Clients render it verbatim and never re-phrase it —
re-deriving evidence language client-side is how a rung-0 observation ends up
described as causal.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class EdgeEndpoint(BaseModel):
    """One end of an observed edge firing."""

    layer: int
    feature_idx: int
    pos: int = Field(..., description="Absolute token position within the request")
    act: float


class UnsensableEdgeInfo(BaseModel):
    """An edge that could not be watched, and why.

    Surfaced so a user never reads "no events" as "the edge never fired" —
    absence of observation is not evidence of absence.
    """

    edge_key: str
    reason: str = Field(
        ...,
        description=(
            "layer_not_attached | no_activation_threshold | endpoint_not_a_feature"
        ),
    )
    detail: str = ""


class CircuitSensingStatusResponse(BaseModel):
    """Runtime state, reconciled against the SAEs actually armed."""

    armed: bool
    paused_reason: Optional[str] = Field(
        default=None,
        description=(
            "Why an armed circuit is not observing right now (e.g. "
            "speculative decoding). Null when observing normally — an "
            "armed circuit reporting zero events must be able to say why."
        ),
    )
    circuit_id: Optional[str] = None
    circuit_name: Optional[str] = None
    layers: list[int] = Field(default_factory=list)
    sensable_edges: int = 0
    unsensable_edges: list[UnsensableEdgeInfo] = Field(default_factory=list)
    max_token_lag: int
    last_request_overhead_ms: float = 0.0
    truncated_layers: list[int] = Field(
        default_factory=list,
        description=(
            "Layers that dropped events in the last drained request (BR-006). "
            "Empty means every armed layer reported completely — which is a "
            "different statement from 'no events were observed', and the "
            "reason this names layers instead of being a boolean. An operator "
            "seeing an empty result needs to know whether the gap is where "
            "they are looking."
        ),
    )
    requests_sensed: int = Field(
        default=0,
        description=(
            "Request boundaries this armed circuit has actually observed since "
            "arming. ZERO while armed means no request reached sensing at all "
            "— a wiring or skip condition, not quiet traffic; check "
            "`paused_reason`. Without this the two readings are identical."
        ),
    )
    events_recorded: int = 0
    ws_dropped: int = 0
    #: Persistent operator intent, distinct from runtime `armed`.
    enabled_circuits: list[dict[str, Any]] = Field(default_factory=list)


class CircuitSensingEventResponse(BaseModel):
    """One observed up→down firing."""

    id: int
    circuit_id: str
    request_id: str
    phase: str
    edge_key: str
    up: EdgeEndpoint
    down: EdgeEndpoint
    token_lag: int
    edge_rung: int
    edge_rung_language: str = Field(
        ...,
        description="Server-rendered evidence phrase — render verbatim, never re-phrase",
    )
    edge_type: Optional[str] = None
    ambient_fired_count: Optional[int] = None
    summary: str
    truncated: bool
    created_at: Optional[str] = None
    context_text: Optional[str] = None
    context_token_ids: Optional[list[int]] = None
    context_parts: Optional[dict[str, str]] = None


class CircuitSensingEventListResponse(BaseModel):
    total: int
    events: list[CircuitSensingEventResponse]


class CircuitSensingToggleResult(BaseModel):
    circuit_id: str
    enabled: bool
    armed: bool
    unsensable_edges: list[UnsensableEdgeInfo] = Field(default_factory=list)
    message: str = ""
