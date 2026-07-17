"""
Sensing API schemas (Feature 11).
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class SensingEventResponse(BaseModel):
    """One persisted co-activation event."""

    id: int
    profile_id: str
    request_id: str
    phase: str
    pos_start: int
    pos_end: int
    fired_members: list[Any]
    fired_count: int
    score: float
    ambient_fired_count: Optional[int] = None
    context_text: Optional[str] = None
    context_token_ids: Optional[list[int]] = None
    context_parts: Optional[dict[str, str]] = None
    summary: str
    truncated: bool
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class SensingEventListResponse(BaseModel):
    events: list[SensingEventResponse]
    total: int


class SensingStatusResponse(BaseModel):
    """Runtime arm state + persistent intent, reported DISTINCTLY (FTID
    pitfall 8): sensing_enabled is the column; armed is live runtime state
    (active cluster + enabled + SAE attached)."""

    armed: bool
    profile_id: Optional[str] = None
    profile_name: Optional[str] = None
    member_count: int = 0
    sensable_count: int = 0
    min_k: Optional[int] = None
    threshold_mode: Optional[str] = None
    context_tokens: Optional[int] = None
    last_request_overhead_ms: float = 0.0
    overhead_warn_threshold_ms: float = 5.0
    events_recorded_since_start: int = 0
    ws_events_dropped: int = 0
    retention: dict[str, Any] = Field(default_factory=dict)
    # Persistent intent (the sensing_enabled column), distinct from `armed`
    enabled_clusters: list[dict[str, Any]] = Field(default_factory=list)


class SensingToggleResult(BaseModel):
    profile_id: str
    sensing_enabled: bool
    armed: bool


class SensingConfigRequest(BaseModel):
    """Runtime sensing overrides (miLLM-local; never exported)."""

    min_k: Optional[int] = Field(
        None, ge=1,
        description="Quorum: members that must co-fire for an event. "
                    "null clears the override (default: all sensable members)",
    )


class SensingConfigResult(BaseModel):
    profile_id: str
    min_k: Optional[int] = None
    effective_min_k: Optional[int] = None
    armed: bool
