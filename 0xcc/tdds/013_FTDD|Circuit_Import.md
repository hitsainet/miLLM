# Technical Design Document: Circuit Import, Slice-Fallback & Evidence Ladder

## miLLM Feature 13

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**Feature Priority:** Core (Increment: Circuit Runtime)
**References:** `013_FPRD|Circuit_Import.md` · `000_PADR|miLLM.md` (v1.2) · `docs/mcp-contract.md` (v1.1) · miStudio `docs/schemas/circuit-definition-v1.json`

---

## 1. Executive Summary

Circuit Import adds a validation + graph-registration layer in front of two existing, battle-tested
paths: Feature 12's multi-SAE per-layer apply (for full serving) and Feature 8's cluster import (for the
per-layer slice fallback). A `mistudio.circuit-definition/v1` document becomes a `circuits` row; on
activation the service either drives Feature 12's per-layer steering (all SAEs bind) or materializes the
circuit's per-layer `cluster-definition/v1` slice as an ordinary cluster profile (incomplete SAE set) —
**never a member through a mismatched SAE**. Evidence rung is stored verbatim and rendered from a single
vocabulary module; the "causal" copy-audit and a schema sync test are first-class deliverables.

### Key Technical Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Storage | New `circuits` table (migration 011), not a `profiles` row | A circuit is a multi-layer graph with edges; single-active invariant now spans manual/cluster/circuit via a shared active-guard |
| Full serving | Delegate to Feature 12 per-layer apply (each member through its own layer's SAE) | Zero new steering math; one composition semantics, authored in miStudio |
| Slice fallback | Materialize the per-layer `cluster-definition/v1` slice through Feature 8 unchanged | A slice IS a valid v1 cluster; reuse the whole import→activate path, no fork |
| Rung vocabulary | Single `circuit_evidence.py` module mirroring miStudio's ladder verbatim | One source of `rung_language`; copy-audit test forbids "causal" < rung 2 |
| Budgets | Frozen as authored; not recomputed against local SAEs | Matches cluster-import + miStudio profile-load semantics |
| Validation | Pydantic mirror of the vendored JSON schema + sync test | Drift-proof against the frozen v1 contract |
| Compat | Per-referenced-SAE: bind / warn-bind / block / unbound | Serveable only when ALL bind; else slice-fallback, never wrong-decoder |

## 2. System Architecture

```
 ┌────────────┐  file/paste   ┌──────────────────┐     ┌───────────────────────────┐
 │CircuitsPage│ ─────────────►│ /api/circuits/*  │ ──► │ CircuitService             │
 │ (admin-ui) │               │ (circuits.py)    │     │  validate → per-SAE compat │
 └────────────┘ ◄─────────────└──────────────────┘     │  → register (circuits row) │
                                      │                 └────────┬──────────┬────────┘
                            activate  │                          │ full     │ incomplete
                                      ▼                 ┌─────────▼───┐  ┌───▼──────────────┐
                              ┌───────────────┐         │ Feature 12  │  │ to_layer_slice → │
                              │ EvidenceRung  │         │ MultiSAE    │  │ ClusterService   │
                              │ (rung_language│         │ per-layer   │  │ .import+activate │
                              │  verbatim)    │         │ apply (1 λ) │  │ (Feature 8 path) │
                              └───────────────┘         └─────────────┘  └──────────────────┘
```

## 3. Database Design

```python
# millm/db/migrations/versions/011_add_circuits_table.py
def upgrade() -> None:
    op.create_table(
        "circuits",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("model_id", sa.String(200), nullable=True),
        sa.Column("rung", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("layers", postgresql.JSONB(), nullable=False),
        sa.Column("edge_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("circuit_meta", postgresql.JSONB(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("serving_mode", sa.String(16), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.func.now()),
    )
    op.create_index("uq_circuits_active", "circuits", ["is_active"],
                    unique=True, postgresql_where=sa.text("is_active"))

def downgrade() -> None:
    op.drop_index("uq_circuits_active", table_name="circuits")
    op.drop_table("circuits")
```

`millm/db/models/circuit.py` maps the columns + a `serveable` property (all referenced SAEs bound at
import time, cached in `circuit_meta.per_sae_warnings`). `circuit_meta` stores the FULL original
definition verbatim (members keyed to layers, edges with rung/rung_language/statistics/attribution/
validation, per-layer budgets, provenance) plus import-time `per_sae_warnings[]` and optional
`hub_ref{repo_id, revision, path}`. The single-active partial unique index is the DB-level guard; the
cross-kind single-active invariant (a circuit deactivates any active profile/cluster, and vice versa)
is enforced in the service layer against `ProfileRepository` + `CircuitRepository`.

## 4. Service Design

```python
# millm/api/schemas/circuit.py — pydantic mirror of the frozen v1 contract
class CircuitSAERef(BaseModel):
    layer: int = Field(ge=0)
    mistudio_sae_id: str | None = None
    n_features: int | None = None
    d_model: int | None = None
    hook_type: str | None = None
    source_hint: str | None = None       # no-local-paths validator (reused from cluster.py)

class CircuitNodeRef(BaseModel):
    layer: int = Field(ge=0)
    kind: Literal["feature", "cluster"] = "feature"
    feature_idx: int | None = Field(None, ge=0)
    cluster_profile_id: str | None = None

class CircuitEdge(BaseModel):
    source: CircuitNodeRef
    target: CircuitNodeRef
    type: str | None = None
    rung: int = Field(ge=0, le=3)
    statistics: dict | None = None       # tolerant: unknown keys preserved
    attribution: dict | None = None
    validation: dict | None = None

class CircuitDefinitionV1(BaseModel):
    kind: Literal["mistudio.circuit-definition"]
    schema_version: Literal["1"]
    name: str = Field(min_length=1, max_length=120)
    narrative: str | None = Field(None, max_length=10_000)
    model: DefinitionModelRef            # reused from cluster.py
    saes: list[CircuitSAERef] = Field(min_length=1, max_length=16)
    members: list[CircuitMember] = Field(min_length=1)   # each carries a layer; ≤20/layer
    edges: list[CircuitEdge] = Field(default_factory=list, max_length=200)
    budgets: CircuitBudgets              # per-layer strengths + intensity/intensity_range
    provenance: DefinitionProvenance     # reused from cluster.py
    # model_config extra="allow" — Tier-2.5 nullable fields survive unknown-key round-trip
# Caps: raw payload ≤ 1 MB checked pre-parse; ≤20 members/layer enforced post-validate.
```

```python
# millm/core/circuit_evidence.py — the ONE rung vocabulary (mirrors miStudio evidence_ladder.py)
class EvidenceRung(IntEnum):
    MINED = 0
    ATTRIBUTION_SUPPORTED = 1
    CAUSALLY_VALIDATED = 2
    FAITHFULNESS_TESTED = 3

RUNG_LANGUAGE: dict[EvidenceRung, str] = {
    EvidenceRung.MINED: "associated",
    EvidenceRung.ATTRIBUTION_SUPPORTED: "suggested (attribution-supported)",
    EvidenceRung.CAUSALLY_VALIDATED: "causally validated (edge)",
    EvidenceRung.FAITHFULNESS_TESTED: "faithfulness-tested (circuit)",
}

def rung_language(rung: int) -> str:
    return RUNG_LANGUAGE[EvidenceRung(rung)]

def circuit_rung(edges: list[CircuitEdge]) -> EvidenceRung:
    """Circuit rung = MIN over edges; empty edges ⇒ MINED (0)."""
    return EvidenceRung(min((e.rung for e in edges), default=0))
```

```python
# millm/services/circuit_service.py
class CircuitService:
    def __init__(self, circuit_repo: CircuitRepository,
                 cluster_service: ClusterService,        # slice-fallback path (Feature 8)
                 multisae_service: MultiSAEService,       # full serving (Feature 12)
                 profile_repo: ProfileRepository): ...

    async def import_definition(self, defn: CircuitDefinitionV1, *,
                                on_conflict: str = "rename",
                                hub_ref: dict | None = None,
                                activate: bool = False,
                                acknowledge_unvalidated: bool = False) -> CircuitImportResult:
        per_sae = [self._assess_sae(s) for s in defn.saes]   # bind/warn/block/unbound per SAE
        rung = circuit_rung(defn.edges)
        name = await self._dedupe_name(defn.name, on_conflict)
        row = await self.circuit_repo.create(
            name=name, model_id=defn.model.hf_id or defn.model.mistudio_model_id,
            rung=int(rung), layers=sorted({m.layer for m in defn.members}),
            edge_count=len(defn.edges),
            circuit_meta={**defn.model_dump(mode="json"),
                          "per_sae_warnings": per_sae,
                          **({"hub_ref": hub_ref} if hub_ref else {})})
        ...

    def _assess_sae(self, ref: CircuitSAERef) -> dict:
        """Per referenced SAE: bind (attached SAE at layer matches n_features) /
        warn-bind (model|layer differ) / block/unbound (n_features mismatch or not attached).
        Serveable ⇔ every ref binds."""

    async def activate(self, circuit_id: str, *,
                       acknowledge_unvalidated: bool = False) -> ActivationResult:
        circ = await self.circuit_repo.get(circuit_id)          # else CIRCUIT_NOT_FOUND (404)
        if circ.rung < 2 and not acknowledge_unvalidated:
            return ActivationResult.refused("UNVALIDATED_CIRCUIT")   # 200 + success:false
        missing = self._unbound_saes(circ)                      # SAE_SET_INCOMPLETE candidates
        if not missing:
            await self._deactivate_any_active()                 # cross-kind single-active
            return await self.multisae_service.apply_circuit(circ)  # serving_mode="full"
        return await self._activate_slice_fallback(circ, missing)   # serving_mode="slice_fallback"

    async def _activate_slice_fallback(self, circ, missing) -> ActivationResult:
        # for each BOUND layer: to_layer_slice(defn, layer) → a cluster-definition/v1 doc →
        # ClusterService.import_definition(...) + activate — Feature 8 path, unchanged.
        # A slice is NEVER presented as the whole circuit (marker rides in slice name + source_note).

    async def export_definition(self, circuit_id: str) -> dict:
        # raw circuit doc rebuilt from circuit_meta — lossless (unknown fields survive)
```

miStudio producer facts (cited, NOT imported): the slice projection is `to_layer_slice(defn, layer)`
(`miStudio/backend/src/schemas/circuit_definition.py:255`) → a schema-identical `cluster-definition/v1`
where the ONLY circuit-specific info is the name suffix ` [L{n} slice]` + `provenance.source_note`
(`parent_rung`, `partial_rendering=true`). miLLM consumes the export endpoints
`GET /circuits/{id}/export` (full) and `POST /circuits/{id}/export-slices` (`{parent, parent_rung,
parent_rung_language, slices[]}`); this feature reads the slices already embedded in the imported
document (or re-projects locally with an identical suffix/marker) — it never runs miStudio code.

## 5. API Design

Routes per FPRD §5 and `docs/mcp-contract.md` §4 `millm_circuits`; DI via `CircuitServiceDep` in
`millm/api/dependencies.py` (pattern: `get_cluster_service`); router registered in
`millm/api/routes/__init__.py::register_routes`. `ApiResponse` envelope everywhere except
`GET /api/circuits/{id}/export` (raw document — the response IS the artifact, like the cluster export).
Error codes exactly per contract §5: `CIRCUIT_NOT_FOUND` (404), `SAE_SET_INCOMPLETE` (422),
`INCOMPATIBLE_FEATURE_SPACE` (422), `UNVALIDATED_CIRCUIT` (200+envelope), `NO_ACTIVE_CIRCUIT`
(200+envelope), reused `UNKNOWN_KIND`/`PAYLOAD_TOO_LARGE` (200+envelope).

## 6. Admin UI Design

- `App.tsx`: route `/circuits`; `Sidebar.tsx`: nav entry (Waypoints icon) after Clusters.
- `pages/CircuitsPage.tsx`: list (React Query `useCircuits`) + import dialog + active banner
  (name, layers, edge count, rung badge, `serving_mode` disclosure).
- `components/circuits/CircuitCard.tsx`: rung badge (server `rung_language` verbatim), layer chips,
  edge-count pill, serveable/slice-fallback + imported badges, per-SAE warnings list, export/delete.
- `CircuitImportDialog.tsx`: tabs paste | file.
- `CircuitActivateControl.tsx`: when rung<2, a required "I understand this circuit is unvalidated
  (rung {n} — {rung_language})" checkbox that sets `acknowledge_unvalidated`; when the SAE set is
  incomplete, a slice-fallback disclosure banner ("steering a per-layer projection, not the full circuit").
- `services/circuits.ts` client mirrors the routes; types in `types/circuits.ts`.

## 7. Testing Strategy

### Unit Tests
- `tests/unit/api/test_circuit_schema.py`: valid/invalid kinds, caps (1 MB / 16 layers / 200 edges /
  20 members-per-layer), no-local-paths, hostile shapes, unknown-field survival.
- `tests/unit/api/test_circuit_schema_sync.py`: pydantic ⇄ vendored JSON schema (same pattern as
  `test_cluster_schema_sync.py`).
- `tests/unit/core/test_circuit_evidence.py`: `circuit_rung` = MIN(edges); empty ⇒ 0; `rung_language`
  values EXACT; **copy-audit: grep all runtime + UI surfaces for "causal" and assert it never co-occurs
  with a rung<2 render** (mirrors miStudio's guard).
- `tests/unit/services/test_circuit_service.py`: per-SAE compat matrix (bind/warn/block/unbound;
  serveable ⇔ all bind), name dedupe, rung<2 activation refusal without ack, export equality,
  slice-projection produces a valid `cluster-definition/v1` with the ` [L{n} slice]` marker.

### Integration Tests
- `tests/integration/test_circuit_import_workflow.py`: import → activate (full, all SAEs attached) →
  assert each member's steering applied through ITS OWN layer's SAE at authored strength; incomplete
  SAE set → `serving_mode: "slice_fallback"` via the cluster path; rung<2 refusal without ack then
  success with ack; single-active invariant across manual↔cluster↔circuit; re-export equality;
  real miStudio-exported circuit fixture round-trip.

## 8. Risks
- **Rung language drift** — any surface re-wording the rung breaks evidence honesty. Mitigation: single
  `circuit_evidence.py` + copy-audit test; UI renders the server field only.
- **Slice presented as whole circuit** — a subtle overclaim. Mitigation: partial-rendering marker
  asserted present in every fallback render; `serving_mode` always surfaced.
- **Feature 12/8 coupling** — activation delegates to two services; a partially-attached set must never
  reach the full-serve path. Mitigation: `_unbound_saes` gate before delegation; SAE_SET_INCOMPLETE test.
- **JSONB `circuit_meta` growth** bounded by the 1 MB payload cap.
