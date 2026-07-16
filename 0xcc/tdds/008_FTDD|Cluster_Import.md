# Technical Design Document: Cluster Import

## miLLM Feature 8

**Document Version:** 1.0
**Created:** July 16, 2026
**Status:** Draft
**References:** `008_FPRD|Cluster_Import.md` · `000_PADR|miLLM.md` (v1.1) · miStudio `docs/schemas/cluster-definition-v1.json`

---

## 1. Executive Summary

Cluster Import adds a validation + mapping layer in front of the existing, battle-tested profile
activation path. A `mistudio.cluster-definition/v1` document becomes a `profiles` row with
`source_kind='cluster'`; everything downstream (activation, per-request override, single-active
invariant) is reused unchanged except for one new step — λ scaling with a ±200 clamp at apply time.
Hugging Face consumption is a thin read-only layer over `huggingface_hub` primitives already in the
dependency tree.

### Key Technical Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Storage | Extend `profiles` (4 new columns, migration 007) | Single-active partial unique index cannot span tables; activation path reuse |
| λ | Raw λ=1 strengths in `steering`; scale+clamp at activation | Lossless round-trip; resolves ±300×2 vs ±200 contract conflict |
| Validation | Pydantic mirror of the vendored JSON schema + sync test | Drift-proof against the frozen v1 contract |
| Hub access | `HfApi.list_models(filter=...)` + `list_repo_files` + `hf_hub_download`, anonymous | Minimal additions; same circuit breaker as SAE downloads |
| Compatibility | Import = warn-level; activation = hard gate | Unbound imports are useful (bind later); steering with bad indices never happens |
| UI | New Clusters page filtered on `source_kind` | User decision; Profiles page untouched |

## 2. System Architecture

```
 ┌────────────┐   file/paste    ┌──────────────────┐     ┌──────────────────────┐
 │ ClustersPage│ ──────────────► │ /api/clusters/*  │ ──► │ ClusterService        │
 │ (admin-ui) │   HF browse     │ (clusters.py)    │     │  validate → map →     │
 └────────────┘ ◄────────────── └──────────────────┘     │  ProfileRepository    │
                                        │                 └─────────┬────────────┘
                                        ▼                           │ activate (λ·clamp)
                                ┌──────────────────┐       ┌────────▼────────────┐
                                │ ClusterHubService│       │ ProfileService       │
                                │ (HfApi, breaker, │       │ .activate_profile    │
                                │  TTL cache)      │       │ → SAEService.set_    │
                                └──────────────────┘       │   steering_batch     │
                                                           └─────────────────────┘
```

## 3. Database Design

```python
# millm/db/migrations/versions/007_add_cluster_columns_to_profiles.py
def upgrade() -> None:
    op.add_column("profiles", sa.Column("source_kind", sa.String(20),
                  nullable=False, server_default="manual"))
    op.add_column("profiles", sa.Column("cluster_meta", postgresql.JSONB(), nullable=True))
    op.add_column("profiles", sa.Column("intensity", sa.Float(),
                  nullable=False, server_default="1.0"))
    op.add_column("profiles", sa.Column("sensing_enabled", sa.Boolean(),
                  nullable=False, server_default=sa.false()))
    op.create_index("idx_profiles_source_kind", "profiles", ["source_kind"])

def downgrade() -> None:
    op.drop_index("idx_profiles_source_kind", table_name="profiles")
    for col in ("sensing_enabled", "intensity", "cluster_meta", "source_kind"):
        op.drop_column("profiles", col)
```

`millm/db/models/profile.py` gains the four mapped columns (defaults mirroring server defaults) and a
`is_cluster` property. `cluster_meta` stores the FULL original definition verbatim (members with
label/similarity/activation_frequency/max_activation/pinned, budget, sae/model refs, provenance,
plus import-time `warnings[]` and optional `hub_ref{repo_id, revision, path}`).

## 4. Service Design

```python
# millm/api/schemas/cluster.py — pydantic mirror of the frozen v1 contract
class ProfileMember(BaseModel):
    feature_idx: int = Field(ge=0)
    label: str | None = None
    similarity: float | None = Field(None, ge=0.0, le=1.0)
    activation_frequency: float | None = None
    max_activation: float | None = None
    strength: float = Field(ge=-300.0, le=300.0)
    sign: Literal[1, -1] = 1
    pinned: bool = False

class ClusterDefinitionV1(BaseModel):
    kind: Literal["mistudio.cluster-definition"]
    schema_version: Literal["1"]
    name: str = Field(min_length=1, max_length=120)
    narrative: str | None = Field(None, max_length=10_000)
    display_token: str | None = None
    model: DefinitionModelRef
    sae: DefinitionSAERef        # incl. no-local-paths validator on source_hint
    members: list[ProfileMember] = Field(min_length=1, max_length=20)
    budget: ProfileBudget | None = None
    provenance: DefinitionProvenance
# ClusterBundleV1: definitions ≤ 50. Caps: raw payload ≤ 1 MB checked pre-parse.
```

```python
# millm/services/cluster_service.py
class ClusterService:
    def __init__(self, profile_service: ProfileService,
                 profile_repo: ProfileRepository, sae_service: SAEService): ...

    async def import_definition(self, definition: ClusterDefinitionV1, *,
                                on_conflict: str = "rename",
                                hub_ref: dict | None = None,
                                activate: bool = False) -> ClusterImportItem:
        sae_id, bound, warnings = self._assess_compatibility(definition.sae)
        steering = {str(m.feature_idx): m.sign * m.strength for m in definition.members}
        # λ=1 basis; range warning (not error) when |v|·λ_max > 200
        name = await self._dedupe_name(definition.name, on_conflict)
        profile = await self.profile_repo.create(
            name=name, description=definition.narrative,
            model_id=definition.model.hf_id or definition.model.mistudio_model_id,
            sae_id=sae_id, layer=definition.sae.layer, steering=steering,
            source_kind="cluster",
            cluster_meta={**definition.model_dump(mode="json"),
                          "warnings": warnings, "hub_ref": hub_ref},
            intensity=(definition.budget.intensity if definition.budget else 1.0))
        ...

    def _assess_compatibility(self, ref: DefinitionSAERef) -> tuple[str | None, bool, list[str]]:
        """bind (attached SAE matches) / warn-bind (model|layer differ) /
        unbound (no SAE or n_features mismatch — activation gate is the backstop)."""

    async def activate(self, profile_id: str) -> ActivationResult:
        # bounds gate BEFORE delegation: every idx < attached sae.d_sae, else block
        # then ProfileService-style apply with λ scaling:
        #   effective = clamp(raw * profile.intensity, -200.0, 200.0)

    async def set_intensity(self, profile_id: str, intensity: float,
                            reapply: bool = True) -> dict: ...
    async def export_definition(self, profile_id: str) -> ClusterDefinitionV1:
        # rebuilt from cluster_meta — lossless
```

λ scaling touchpoints (shared with Feature 10):
- `ProfileService.activate_profile` (profile_service.py:311 area): scale+clamp before
  `set_steering_batch`.
- `InferenceService._apply_request_profile` (inference_service.py:454 area): multiply by
  `profile.intensity` before the range check; clamp instead of reject.

```python
# millm/services/cluster_hub_service.py
class ClusterHubService:
    TAG = "mistudio-cluster-definition"
    async def search(self, query=None, base_model=None, limit=30) -> list[HubRepoInfo]:
        # HfApi.list_models(filter=[TAG] + ([f"base_model:{m}"] if m else []),
        #                   search=query, limit=min(limit, 50)) — asyncio.to_thread + breaker
    async def list_definitions(self, repo_id) -> list[HubDefinitionRef]:
        # manifest.jsonl preferred; else *.cluster.json (cap 200)
    async def fetch_definition(self, repo_id, filename, revision=None) -> ClusterDefinitionV1:
        # hf_hub_download(...); enforce .cluster.json suffix + ≤1 MB; validate before return
```

## 5. API Design

Routes per FPRD §5; DI via `ClusterServiceDep`/`ClusterHubServiceDep` in `millm/api/dependencies.py`
(pattern: `get_profile_service`, dependencies.py:300); router registered in
`millm/api/routes/__init__.py::register_routes`. All responses in the `ApiResponse` envelope.

## 6. Admin UI Design

- `App.tsx`: route `/clusters`; `Sidebar.tsx`: nav entry (Boxes icon) after Profiles.
- `pages/ClustersPage.tsx`: list (React Query `useClusters`) + import dialog + active banner.
- `components/clusters/ClusterCard.tsx`: display_token chip, member chips (label tooltip),
  bound/unbound + imported badges, warnings list, narrative `<details>` markdown, budget readout,
  IntensitySlider (0..2, marks at `budget.intensity_range`), activate/export/delete.
- `ClusterImportDialog.tsx`: tabs paste | file | Hub (HubBrowser: search box, base-model toggle,
  repo → definitions list → import).
- `services/clusters.ts` client mirrors the routes; types in `types/clusters.ts`.

## 7. Testing Strategy

### Unit Tests
- `tests/unit/api/test_cluster_schema.py`: valid/invalid kinds, caps, no-local-paths, hostile shapes.
- `tests/unit/api/test_cluster_schema_sync.py`: pydantic ⇄ vendored JSON schema.
- `tests/unit/services/test_cluster_service.py`: mapping (sign fold, λ basis), compat matrix
  (bind/warn/unbound + n_features), name dedupe, clamp math incl. λ·strength > 200 warning,
  export equality.
- `tests/unit/services/test_cluster_hub_service.py`: mocked HfApi (search filter composition,
  manifest vs file listing, caps, breaker).

### Integration Tests
- `tests/integration/test_cluster_import_workflow.py`: import → activate → assert
  `sae.get_steering_values()` equals λ-clamped expectation; bundle per-item isolation; unbound refusal;
  single-active invariant across manual↔cluster; re-export equality.

## 8. Risks
- ±200 clamp semantics must match Feature 10's per-request path — shared helper `clamp_steering()`.
- HF anonymous rate limits — TTL cache + breaker (accepted).
- JSONB `cluster_meta` growth is bounded by the 1 MB payload cap.
