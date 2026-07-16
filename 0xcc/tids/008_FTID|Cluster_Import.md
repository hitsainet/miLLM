# Technical Implementation Document: Cluster Import

## miLLM Feature 8

**Document Version:** 1.0
**Created:** July 16, 2026
**Status:** Draft
**References:** `008_FPRD|Cluster_Import.md` · `008_FTDD|Cluster_Import.md`

---

## 1. File Structure

```
millm/
├── api/
│   ├── schemas/cluster.py                       (NEW — v1 contract mirror + API DTOs)
│   ├── routes/management/clusters.py            (NEW — /api/clusters router)
│   └── dependencies.py                          (MOD — ClusterServiceDep, ClusterHubServiceDep)
├── services/
│   ├── cluster_service.py                       (NEW)
│   ├── cluster_hub_service.py                   (NEW)
│   ├── profile_service.py                       (MOD — λ scale+clamp in activate_profile)
│   └── inference_service.py                     (MOD — λ in _apply_request_profile; shared clamp)
├── core/steering_range.py                       (NEW — clamp_steering(), STEERING_RANGE=200.0)
├── db/
│   ├── models/profile.py                        (MOD — 4 columns + is_cluster)
│   └── migrations/versions/007_add_cluster_columns_to_profiles.py  (NEW)
docs/schemas/cluster-definition-v1.json          (NEW — vendored, frozen)
admin-ui/src/
├── pages/ClustersPage.tsx                       (NEW)  + App.tsx/Sidebar.tsx/pages/index.ts (MOD)
├── components/clusters/{ClusterCard,ClusterImportDialog,HubBrowser,IntensitySlider}.tsx (NEW)
├── components/profiles/ImportExportButtons.tsx  (MOD — string prof_* id types)
├── services/clusters.ts, hooks/useClusters.ts, types/clusters.ts (NEW)
tests/unit/{api,services}/test_cluster_*.py      (NEW)
tests/integration/test_cluster_import_workflow.py (NEW)
```

## 2. Load-Bearing Implementation Points (verified against live code)

- **Reuse, don't fork, activation:** `ProfileService.activate_profile` (profile_service.py:277–350)
  already sequences `clear_steering → set_steering_batch → enable_steering` and `repository.set_active`.
  Insert λ scaling exactly where the steering dict is materialized (`get_steering_dict()` call site,
  ~:311): `steering = {k: clamp_steering(v * profile.intensity) for k, v in raw.items()}`.
- **Bounds gate BEFORE delegation:** `LoadedSAE.set_steering_batch` raises on idx ≥ d_sae
  (sae_wrapper.py:294-296) — a 500 from there is the failure mode to prevent. ClusterService.activate
  pre-checks `max(idx) < sae_service.get_attachment_status().d_sae` and returns a structured block.
- **Per-request path parity (Feature 10 shares this):** `_apply_request_profile`
  (inference_service.py:399–488) validates values in ±200 at :465 — change reject→clamp via the shared
  `clamp_steering()` and multiply by `profile.intensity` first. One helper, two call sites, zero drift.
- **Compat semantics mirror:** `SAEService.check_compatibility` (sae_service.py:628–702) is the
  house pattern (hard error d_in, warn layer/model). Cluster compat: hard concern = definition
  `sae.n_features` vs attached `sae.d_sae`; warn = model name / layer.
- **Hub primitives:** `SAEDownloader` already builds `HfApi` and uses `list_repo_files`
  (sae_downloader.py:399) + `snapshot_download` under `huggingface_circuit`. Add `list_models` +
  `hf_hub_download` imports in the new service, same breaker + `asyncio.to_thread` pattern.
- **Envelope:** all management routes return `ApiResponse` (`millm/api/schemas/common.py`) —
  `ApiResponse.ok(data)` / `.fail(code, message)`; mirror profiles.py route style (profiles.py:34-365).
- **Router registration:** `millm/api/routes/__init__.py::register_routes` (:17-34) — add
  `from .management import clusters` + `app.include_router(clusters.router)`.

## 3. Key Implementations

```python
# millm/core/steering_range.py
STEERING_RANGE: float = 200.0

def clamp_steering(value: float) -> float:
    """Single source of truth for the apply-time clamp (PADR v1.1: ±300×λ2 vs ±200 conflict)."""
    return max(-STEERING_RANGE, min(STEERING_RANGE, value))
```

```python
# millm/services/cluster_service.py — import mapping core
def _map_definition(self, d: ClusterDefinitionV1, sae_id, warnings, hub_ref) -> dict:
    return dict(
        name=d.name,
        description=d.narrative,
        model_id=d.model.hf_id or d.model.mistudio_model_id,
        sae_id=sae_id,                      # None ⇒ unbound
        layer=d.sae.layer,
        steering={str(m.feature_idx): float(m.sign) * m.strength for m in d.members},
        source_kind="cluster",
        intensity=(d.budget.intensity if d.budget and d.budget.intensity is not None else 1.0),
        cluster_meta={**d.model_dump(mode="json"), "warnings": warnings,
                      **({"hub_ref": hub_ref} if hub_ref else {})},
    )

def _range_warnings(self, d: ClusterDefinitionV1) -> list[str]:
    lam_max = (d.budget.intensity_range[1] if d.budget and d.budget.intensity_range else 2.0)
    hot = [m.feature_idx for m in d.members if abs(m.strength) * lam_max > STEERING_RANGE]
    return [f"members {hot} exceed ±{STEERING_RANGE:g} at λ_max={lam_max:g}; "
            f"values clamp at apply time"] if hot else []
```

```python
# millm/api/routes/management/clusters.py — import route skeleton
@router.post("/import")
async def import_clusters(payload: dict = Body(...),
                          on_conflict: str = Query("rename", pattern="^(rename|fail)$"),
                          activate: bool = Query(False),
                          service: ClusterServiceDep = None) -> ApiResponse:
    if len(json.dumps(payload)) > 1_048_576:
        return ApiResponse.fail(code="PAYLOAD_TOO_LARGE", message="Import exceeds 1 MB")
    kind = payload.get("kind")
    if kind == "mistudio.cluster-bundle":
        bundle = ClusterBundleV1.model_validate(payload)
        return ApiResponse.ok(await service.import_bundle(bundle, on_conflict=on_conflict))
    if kind == "mistudio.cluster-definition":
        item = await service.import_definition(
            ClusterDefinitionV1.model_validate(payload),
            on_conflict=on_conflict, activate=activate)
        return ApiResponse.ok(item)
    return ApiResponse.fail(code="UNKNOWN_KIND",
                            message=f"kind {kind!r} is not a supported cluster document")
```

```typescript
// admin-ui/src/services/clusters.ts — client shape
export const clustersApi = {
  list: () => request<ClusterListResponse>('/clusters'),
  import: (payload: unknown, opts?) => request('/clusters/import', {method:'POST', body: payload, ...}),
  hubSearch: (q?, baseModel?) => request(`/clusters/hub/search?...`),
  hubDefinitions: (repoId) => request(`/clusters/hub/${encodeURIComponent(repoId)}/definitions`),
  hubImport: (body) => request('/clusters/hub/import', {method:'POST', body}),
  activate: (id) => request(`/clusters/${id}/activate`, {method:'POST'}),
  deactivate: (id) => request(`/clusters/${id}/deactivate`, {method:'POST'}),
  setIntensity: (id, intensity) => request(`/clusters/${id}/intensity`, {method:'PUT', body:{intensity, reapply:true}}),
  export: (id) => request<ClusterDefinitionV1>(`/clusters/${id}/export`),
};
```

## 4. Implementation Pitfalls

1. **Do NOT bake λ into `steering`** — the dial (010) and re-export both die. λ lives in
   `profiles.intensity`; only effective values are scaled.
2. **Steering dict keys are STRINGS** in JSONB (house convention, profile.py:66); convert at the
   apply boundary exactly as `get_steering_dict()` does.
3. **`ImportExportButtons.tsx` id type** — declares `Array<{id:number}>` but backend ids are
   `prof_<hex>` strings; fix while touching the area or the Clusters page inherits the bug.
4. **`repo_id` contains a slash** — route param must be `{repo_id:path}` and the UI must
   `encodeURIComponent` it.
5. **Unbound activation refusal is a 409-style structured fail**, not an exception leak from
   `set_steering_batch`.
6. **Bundle isolation:** wrap each definition in its own try + savepoint-free repo call; aggregate
   `{imported, blocked, errors, items[]}` (miStudio import matrix precedent).
7. **Schema is FROZEN** — never "fix" the vendored JSON; the sync test exists to catch accidental
   divergence of the pydantic mirror.

## 5. Config Additions (millm/core/config.py)

```python
CLUSTER_HUB_CACHE_TTL_S: int = 300
CLUSTER_HUB_TAG: str = "mistudio-cluster-definition"
CLUSTER_INTENSITY_MIN: float = 0.5    # fallback when a definition lacks intensity_range (010 uses)
CLUSTER_INTENSITY_MAX: float = 1.5
```
