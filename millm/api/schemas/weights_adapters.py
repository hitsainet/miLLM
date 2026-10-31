"""Adapter serving schemas (Feature 33, `/api/weights/adapter*`).

Requests forbid extra fields, so a misspelt key is a 422 rather than silently ignored.
`AdapterStatus` is the one adapter block: `GET /api/weights/adapter` returns it, and 031's
weights status embeds the same model (addendum D-28). Its `base_steps_since_load` is null when
not recorded — never 0, which would claim a count nobody took.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class AdapterLoadRequest(BaseModel):
    """`POST /api/weights/adapter/load`."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="The adapter directory, relative to a checkpoint root "
        "(WEIGHTS_CHECKPOINT_DIR or a WEIGHTS_CHECKPOINT_READ_ROOTS entry)",
    )


class AdapterUnloadRequest(BaseModel):
    """`POST /api/weights/adapter/unload`. The body is optional and carries nothing."""

    model_config = ConfigDict(extra="forbid")


class AdapterBase(BaseModel):
    """The base the adapter's manifest names."""

    name: str
    manifest_sha256: Optional[str] = None


class AdapterInfo(BaseModel):
    """What was applied."""

    name: str
    manifest_sha256: str
    weights_sha256: str
    base: AdapterBase
    rank: int
    alpha: float
    scaling: float
    target_count: int


class AdapterLoadResponse(BaseModel):
    adapter: AdapterInfo
    resolved_model_name: Optional[str] = Field(
        None, description="What /v1 now reports: the adapter's own name"
    )
    held_ms: float = Field(..., description="How long serving was paused")


class AdapterUnloaded(BaseModel):
    name: str
    manifest_sha256: str


class AdapterUnloadResponse(BaseModel):
    unloaded: AdapterUnloaded
    resolved_model_name: Optional[str] = Field(
        None, description="What /v1 now reports: the bare base's name"
    )
    held_ms: float


class AdapterStatus(BaseModel):
    """The adapter block (D-28, FR-33.4.5). Every field but `applied` is null when nothing is
    applied. Forbids extra fields, so a status dict that grew a key fails validation instead of
    reaching the unauthenticated status routes."""

    model_config = ConfigDict(extra="forbid")

    applied: bool
    name: Optional[str] = None
    base_name: Optional[str] = Field(
        None,
        description="The RESIDENT base's served name (what /v1 reports once the adapter is "
        "unloaded) — not the base the manifest names; that is manifest_base_name",
    )
    manifest_sha256: Optional[str] = Field(
        None, description="SHA-256 of the applied adapter's adapter_manifest.json"
    )
    manifest_base_name: Optional[str] = Field(
        None, description="The base the adapter's manifest names (base.name)"
    )
    manifest_base_sha256: Optional[str] = Field(
        None,
        description="The manifest's base.manifest_sha256; null when the manifest names the base "
        "by name only",
    )
    weights_sha256: Optional[str] = None
    rank: Optional[int] = None
    target_count: Optional[int] = None
    loaded_at: Optional[datetime] = None
    base_steps_since_load: Optional[int] = Field(
        None,
        description="Commits to the base since the adapter was applied; null = not recorded",
    )
