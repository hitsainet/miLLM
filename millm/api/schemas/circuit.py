"""Circuit schemas: the ``mistudio.circuit-definition/v1`` contract mirror plus
runtime DTOs (Features 12 + 13).

The contract models mirror miStudio's frozen v1 schema (vendored at
``docs/schemas/circuit-definition-v1.json``; a sync test pins them together).
Sub-models shared with the cluster contract (members, budgets, SAE/model refs,
provenance — including the no-local-paths validator) are REUSED from
``cluster.py`` rather than duplicated, so a fix in one place holds for both.

``extra="allow"`` is deliberate on the contract models: a newer miStudio may
emit additive fields (e.g. Tier-2.5 position data) and those must survive a
round-trip rather than being silently stripped.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from millm.api.schemas.cluster import (
    DefinitionModelRef,
    DefinitionProvenance,
    DefinitionSAERef,
    ProfileBudget,
    ProfileMember,
)

# ── Contract constants (mirror miStudio's frozen v1) ────────────────────────
CIRCUIT_DEFINITION_KIND = "mistudio.circuit-definition"
CIRCUIT_SCHEMA_VERSION = "1"

MAX_SAES = 16
MAX_EDGES = 200
MAX_MEMBERS_PER_LAYER = 20
MAX_CIRCUIT_NAME = 200
MAX_CIRCUIT_NARRATIVE = 10_000
MAX_CIRCUIT_IMPORT_BYTES = 1_048_576  # 1 MB — same hostile-payload cap as clusters


# ── Contract models ─────────────────────────────────────────────────────────


class CircuitNodeRef(BaseModel):
    """One endpoint of an edge: a feature (or cluster) at a layer."""

    model_config = ConfigDict(extra="allow")

    layer: int = Field(..., ge=0)
    feature_idx: int | None = Field(None, ge=0)
    cluster_profile_id: str | None = None
    kind: str = "feature"


class EdgeCoactivation(BaseModel):
    model_config = ConfigDict(extra="allow")

    pmi: float | None = None
    lift: float | None = None
    spearman: float | None = None
    support: int | None = None
    null_percentile: float | None = None
    replicated_heldout: bool | None = None


class EdgeAttribution(BaseModel):
    model_config = ConfigDict(extra="allow")

    method: str | None = None
    score: float | None = None
    sign_consistency: float | None = None


class EdgePosition(BaseModel):
    """Tier-2.5 attention-mediation data (nullable; additive)."""

    model_config = ConfigDict(extra="allow")

    mediating_heads: list[Any] | None = None
    roles: list[str] | None = None


class CircuitEdge(BaseModel):
    """A typed, evidence-graded cross-layer edge."""

    model_config = ConfigDict(extra="allow")

    up: CircuitNodeRef
    down: CircuitNodeRef
    type: str = "computed"
    rung: int = Field(0, ge=0, le=3)
    tested_and_failed: list[int] = Field(default_factory=list)
    effect_size: float | None = None
    weight_prior: float | None = None
    coactivation: EdgeCoactivation | None = None
    attribution: EdgeAttribution | None = None
    position: EdgePosition | None = None
    type_signals: dict[str, Any] | None = None
    validation_manifest_ref: str | None = None


class CircuitMemberV1(BaseModel):
    """A circuit member at a layer: a feature ref, or a cluster supernode whose
    membership was frozen at export (``expanded_members``)."""

    model_config = ConfigDict(extra="allow")

    layer: int = Field(..., ge=0)
    member_kind: Literal["feature_ref", "cluster_ref"] = "feature_ref"
    feature: ProfileMember | None = None
    cluster_profile_id: str | None = None
    cluster_name: str | None = None
    expanded_members: list[ProfileMember] | None = None


class CircuitBudget(BaseModel):
    """Per-layer budgets under ONE global intensity (λ)."""

    model_config = ConfigDict(extra="allow")

    layers: dict[str, ProfileBudget] = Field(default_factory=dict)
    formula_id: str | None = None
    intensity: float = Field(1.0, ge=0.0, le=2.0)
    intensity_range: list[float] = Field(default_factory=lambda: [0.0, 2.0])


class CircuitFaithfulness(BaseModel):
    model_config = ConfigDict(extra="allow")

    necessity: float | None = None
    sufficiency: float | None = None
    metric_id: str | None = None
    manifest_ref: str | None = None


class CircuitDiscoveryProvenance(BaseModel):
    model_config = ConfigDict(extra="allow")

    mode: str | None = None          # seeded | open
    granularity: str | None = None   # feature | cluster
    corpus_ref: str | None = None
    thresholds: dict[str, Any] | None = None


class CircuitDefinitionV1(BaseModel):
    """One portable circuit definition — the multi-SAE mobile artifact.

    Consumed, never produced, by miLLM: import validates strictly, stores the
    RAW document for lossless re-export, and refuses unknown kinds / major
    versions.
    """

    model_config = ConfigDict(extra="allow")

    kind: Literal["mistudio.circuit-definition"] = CIRCUIT_DEFINITION_KIND
    schema_version: Literal["1"] = CIRCUIT_SCHEMA_VERSION
    name: str = Field(..., min_length=1, max_length=MAX_CIRCUIT_NAME)
    narrative: str | None = Field(None, max_length=MAX_CIRCUIT_NARRATIVE)
    model: DefinitionModelRef = Field(default_factory=DefinitionModelRef)
    saes: list[DefinitionSAERef] = Field(..., min_length=1, max_length=MAX_SAES)
    members: list[CircuitMemberV1] = Field(..., min_length=1)
    edges: list[CircuitEdge] = Field(default_factory=list, max_length=MAX_EDGES)
    budget: CircuitBudget | None = None
    faithfulness: CircuitFaithfulness | None = None
    discovery: CircuitDiscoveryProvenance | None = None
    provenance: DefinitionProvenance = Field(default_factory=DefinitionProvenance)

    @field_validator("members")
    @classmethod
    def cap_members_per_layer(
        cls, v: list[CircuitMemberV1]
    ) -> list[CircuitMemberV1]:
        """At most MAX_MEMBERS_PER_LAYER members on any single layer.

        A circuit spans layers, so the cluster's flat 20-member cap does not
        apply globally — but an unbounded per-layer member list is the same
        hostile-payload risk the cluster cap exists to stop.
        """
        # Count the DISTINCT features the projection/serving flatten produces
        # (expansion AND the member's own feature, deduped by feature_idx).
        # Counting pre-dedupe rejected contract-valid circuits whose cluster_ref
        # expansion overlapped its own feature; counting only one source let an
        # over-cap slice through. Measure exactly what the projection emits.
        per_layer: dict[int, set[int]] = {}
        for m in v:
            seen = per_layer.setdefault(m.layer, set())
            for feat in list(m.expanded_members or []):
                seen.add(feat.feature_idx)
            if m.feature is not None:
                seen.add(m.feature.feature_idx)
            if not (m.expanded_members or m.feature is not None):
                # A member carrying no feature at all still occupies a slot.
                seen.add(-(len(seen) + 1))
        counts = {layer: len(idxs) for layer, idxs in per_layer.items()}
        for layer, n in counts.items():
            if n > MAX_MEMBERS_PER_LAYER:
                raise ValueError(
                    f"layer {layer} has {n} members (max {MAX_MEMBERS_PER_LAYER})"
                )
        return v

    def layers(self) -> list[int]:
        """Sorted distinct layers this circuit references."""
        return sorted({m.layer for m in self.members})

    def sae_for_layer(self, layer: int) -> DefinitionSAERef | None:
        """The declared SAE ref for a layer, if the document names one."""
        for ref in self.saes:
            if ref.layer == layer:
                return ref
        return None


# ── API DTOs (Feature 13) ───────────────────────────────────────────────────


class PerSAEVerdict(BaseModel):
    """One layer's compatibility verdict from import/activation."""

    layer: int
    sae_id: str | None = None
    verdict: Literal["bind", "warn", "block", "unbound"]
    reason: str | None = None


class CircuitSummary(BaseModel):
    """List/detail row. Rung language is ALWAYS server-rendered."""

    id: str
    name: str
    description: str | None = None
    rung: int = Field(..., ge=0, le=3)
    rung_language: str = Field(
        ..., description="Server-rendered evidence phrase — never re-phrase client-side"
    )
    rung_next_step: str
    validated: bool = Field(
        ..., description="rung >= 2; below this the circuit is not causally validated"
    )
    edge_count: int
    layers: list[int]
    serveable: bool
    is_active: bool
    serving_mode: str | None = None
    intensity: float
    steering: bool | None = Field(
        default=None,
        description=(
            "Whether this circuit is ACTUALLY steering generation right now — "
            "is_active only says a row is marked active. Answered by the "
            "server's own predicate so clients never re-derive it (a client "
            "deriving it from is_active will overclaim on a slice-fallback, "
            "unparseable, or unattached circuit). Null when not evaluated."
        ),
    )
    per_sae_warnings: list[PerSAEVerdict] = Field(default_factory=list)
    created_at: Any = None
    updated_at: Any = None


class CircuitListResponse(BaseModel):
    circuits: list[CircuitSummary]
    active_circuit_id: str | None = None
    total: int = 0


class CircuitActivationResponse(CircuitSummary):
    """Activation outcome: how it is serving and what the user must know."""

    bound_layers: list[int] = Field(default_factory=list)
    slice_layer: int | None = None
    applied_per_layer: dict[str, dict[str, float]] | None = None
    hazards: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    acknowledged_unvalidated: bool = False


class SetCircuitIntensityRequest(BaseModel):
    intensity: float = Field(..., ge=0.0, le=2.0)
    reapply: bool = True
    acknowledge_unvalidated: bool = Field(
        False,
        description=(
            "Required to re-apply steering for a circuit whose evidence rung is "
            "below 2 — re-applying is a fresh arm, so the gate holds here too"
        ),
    )


class CircuitIntensityResponse(CircuitSummary):
    """Intensity change outcome.

    ``reapplied`` is load-bearing: a slice-fallback circuit records the new λ
    but does NOT push it to the model (the backing cluster profile owns its own
    intensity), so a client must not report the dial as applied.
    """

    reapplied: bool = False
    superseded: bool = Field(
        default=False,
        description=(
            "True when the intensity was written but immediately superseded by "
            "another authoritative steering change, so it is NOT live. "
            "Distinguishes that case from slice-fallback, where reapplied is "
            "also false but for an entirely different reason. Declared here "
            "deliberately: a field the service computes but the response model "
            "omits is silently dropped by Pydantic and never reaches a client."
        ),
    )
    warnings: list[str] = Field(default_factory=list)


class CircuitDeactivationResponse(CircuitSummary):
    cleared_steering: bool = False


# ── Runtime DTOs (Feature 12 serving) ───────────────────────────────────────


class CircuitMember(BaseModel):
    """One steerable member of a circuit at a given layer (serving shape).

    The ``budget`` is the frozen per-layer strength authored in miStudio
    (γ=0 ⇒ B = B_dir); serving applies the canonical sign rule then
    ``clamp_steering(directional_budget · λ)`` where λ is the one global
    intensity. ``sae_id`` names the SAE the member was authored against —
    serving prefers an exact ``(sae_id, layer)`` match so a member is never
    silently steered through a different feature basis.
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
        None, description="SAE id the member was authored against (exact-match preferred)"
    )
    label: str | None = Field(None, max_length=200)
