"""
Cluster-definition interchange contract + cluster API DTOs (Feature 8).

The interchange models mirror the FROZEN `mistudio.cluster-definition/v1`
contract (vendored JSON Schema: docs/schemas/cluster-definition-v1.json).
Field definitions must stay byte-compatible with the miStudio originals —
the sync test regenerates the JSON Schema from these models and compares it
to the vendored file. Never "fix" the vendored file; fix the mirror.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "1"
DEFINITION_KIND = "mistudio.cluster-definition"
BUNDLE_KIND = "mistudio.cluster-bundle"

MAX_MEMBERS = 20
MAX_BUNDLE = 50
MAX_NAME = 120
MAX_NARRATIVE = 10_000
MAX_IMPORT_BYTES = 1_048_576  # 1 MB — definitions are a few KB; near this is hostile


# ── Interchange contract (frozen v1 — mirrors miStudio exactly) ─────────────

class ProfileMember(BaseModel):
    """A cluster member snapshot with its tuned strength."""

    feature_idx: int = Field(..., ge=0)
    label: str | None = None
    similarity: float | None = Field(None, ge=0.0, le=1.0)
    activation_frequency: float | None = None
    max_activation: float | None = None
    strength: float = Field(..., ge=-300.0, le=300.0)
    sign: Literal[1, -1] = 1
    pinned: bool = False


class ProfileBudget(BaseModel):
    """Allocation snapshot from Feature 013 (self-describing: formula + constants travel)."""

    B: float | None = None
    B_dir: float | None = None
    G: float | None = None
    f_eff: float | None = None
    formula_id: str | None = None
    constants: dict[str, float] | None = None
    intensity: float = Field(1.0, ge=0.0, le=2.0)
    intensity_range: list[float] = Field(default_factory=lambda: [0.0, 2.0])


class DefinitionModelRef(BaseModel):
    hf_id: str | None = None
    mistudio_model_id: str | None = None


class DefinitionSAERef(BaseModel):
    mistudio_sae_id: str | None = None
    layer: int | None = None
    hook_type: str | None = None
    n_features: int | None = None
    d_model: int | None = None
    source_hint: str | None = Field(
        None, description="e.g. 'hf:repo/path' — NEVER an absolute local path"
    )

    @field_validator("source_hint")
    @classmethod
    def no_local_paths(cls, v: str | None) -> str | None:
        """Reject absolute/relative filesystem paths — the format must stay portable."""
        if v and (v.startswith("/") or v.startswith("~") or v.startswith("..") or ":\\" in v):
            raise ValueError("source_hint must not be a filesystem path")
        return v


class DefinitionProvenance(BaseModel):
    created_at: datetime | None = None
    exported_at: datetime | None = None
    mistudio_version: str | None = None
    source_note: str | None = Field(None, max_length=500)


class ClusterDefinitionV1(BaseModel):
    """One portable cluster definition (the mobile artifact — IDL-30)."""

    kind: Literal["mistudio.cluster-definition"] = DEFINITION_KIND
    schema_version: Literal["1"] = SCHEMA_VERSION
    name: str = Field(..., min_length=1, max_length=MAX_NAME)
    narrative: str | None = Field(None, max_length=MAX_NARRATIVE)
    display_token: str | None = None
    model: DefinitionModelRef = Field(default_factory=DefinitionModelRef)
    sae: DefinitionSAERef = Field(default_factory=DefinitionSAERef)
    members: list[ProfileMember] = Field(..., min_length=1, max_length=MAX_MEMBERS)
    budget: ProfileBudget | None = None
    provenance: DefinitionProvenance = Field(default_factory=DefinitionProvenance)


class ClusterBundleV1(BaseModel):
    """A multi-cluster export: an array of definitions in one file."""

    kind: Literal["mistudio.cluster-bundle"] = BUNDLE_KIND
    schema_version: Literal["1"] = SCHEMA_VERSION
    definitions: list[ClusterDefinitionV1] = Field(..., min_length=1, max_length=MAX_BUNDLE)


# ── Cluster API DTOs (miLLM-local; not part of the frozen contract) ─────────

class ClusterImportItem(BaseModel):
    """Per-definition import outcome (bundle imports aggregate these)."""

    name: str
    status: Literal["imported", "imported_unbound", "blocked", "error"]
    profile_id: str | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class ClusterImportResult(BaseModel):
    results: list[ClusterImportItem]
    imported: int
    blocked: int
    errors: int


class ClusterSummary(BaseModel):
    """One cluster-typed profile row for the Clusters page.

    Built manually by ClusterService._summarize (member_count/bound/warnings
    are derived, not ORM attributes) — deliberately NOT from_attributes.
    """

    id: str
    name: str
    description: str | None = None
    model_id: str | None = None
    sae_id: str | None = None
    layer: int | None = None
    is_active: bool
    intensity: float
    sensing_enabled: bool
    member_count: int = 0
    display_token: str | None = None
    bound: bool = False
    warnings: list[str] = Field(default_factory=list)
    hub_ref: dict[str, Any] | None = None
    intensity_range: list[float] | None = None
    budget_b: float | None = None
    formula_id: str | None = None
    created_at: datetime
    updated_at: datetime


class ClusterListResponse(BaseModel):
    clusters: list[ClusterSummary]
    active_cluster_id: str | None = None


class SetIntensityRequest(BaseModel):
    intensity: float = Field(..., ge=0.0, le=2.0)
    reapply: bool = Field(True, description="Re-apply steering now if this cluster is active")


class HubRepoInfo(BaseModel):
    repo_id: str
    likes: int = 0
    downloads: int = 0
    last_modified: datetime | None = None
    tags: list[str] = Field(default_factory=list)


class HubDefinitionRef(BaseModel):
    file: str
    name: str | None = None
    member_count: int | None = None
    base_model: str | None = None


class HubImportRequest(BaseModel):
    repo_id: str = Field(..., min_length=3, max_length=200)
    filename: str = Field(..., min_length=1, max_length=300)
    revision: str | None = None
    activate: bool = False
    # 009 R2: agents' dedupe guard — parity with the inline import route
    on_conflict: Literal["rename", "fail"] = "rename"
