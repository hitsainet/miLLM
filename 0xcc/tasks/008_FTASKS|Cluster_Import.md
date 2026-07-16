# Task List: Cluster Import

## miLLM Feature 8

**Document Version:** 1.0
**Created:** July 16, 2026
**Status:** Not started
**References:** `008_FPRD|Cluster_Import.md` · `008_FTDD|Cluster_Import.md` · `008_FTID|Cluster_Import.md`

## Relevant Files

### Backend
- `millm/db/migrations/versions/007_add_cluster_columns_to_profiles.py` — additive migration
- `millm/db/models/profile.py` — source_kind/cluster_meta/intensity/sensing_enabled columns
- `millm/api/schemas/cluster.py` — v1 contract mirror + DTOs
- `millm/core/steering_range.py` — shared clamp helper
- `millm/services/cluster_service.py`, `millm/services/cluster_hub_service.py` — new services
- `millm/services/profile_service.py`, `millm/services/inference_service.py` — λ scale+clamp
- `millm/api/routes/management/clusters.py`, `millm/api/dependencies.py`, `millm/api/routes/__init__.py`
- `docs/schemas/cluster-definition-v1.json` — vendored frozen schema

### Frontend
- `admin-ui/src/pages/ClustersPage.tsx`, `App.tsx`, `components/layout/Sidebar.tsx`, `pages/index.ts`
- `admin-ui/src/components/clusters/*` — ClusterCard, ClusterImportDialog, HubBrowser, IntensitySlider
- `admin-ui/src/services/clusters.ts`, `hooks/useClusters.ts`, `types/clusters.ts`
- `admin-ui/src/components/profiles/ImportExportButtons.tsx` — id-type fix

### Tests
- `tests/unit/api/test_cluster_schema.py`, `test_cluster_schema_sync.py`
- `tests/unit/services/test_cluster_service.py`, `test_cluster_hub_service.py`
- `tests/integration/test_cluster_import_workflow.py`

### Notes
- Follow `008_process-task-list.md`: one sub-task at a time; full suite + commit per parent task.
- Test commands: `pytest` (backend), `npm test` in admin-ui (Vitest).

### Category Checklist Results
- Data: tasks 1.x (migration + model) ✓
- Backend/API: tasks 2.x–4.x ✓
- Frontend/UI: tasks 5.x ✓
- Business logic: tasks 2.x–3.x (mapping, compat, λ/clamp) ✓
- Integration wiring: tasks 4.x (DI, router), 5.1 (routing/nav) ✓
- Error handling & logging: 2.6, 3.4, 4.4 ✓
- Testing: every parent has paired test sub-tasks; 6.x integration ✓
- Performance & security: 2.2 caps/hostile, 3.3 breaker+cache, EC tests ✓
- Config/deploy: 4.5 (config keys); migration auto-runs on container start — no deploy change ✓
- Documentation: 7.2 (manual page) ✓

## Tasks

- [ ] 1.0 Data layer: cluster columns on profiles (covers FR-8.3; CLI-M1, CLI-M2)
  - [ ] 1.1 Write migration `007_add_cluster_columns_to_profiles.py` (upgrade+downgrade); verify round-trip locally
  - [ ] 1.2 Add columns + `is_cluster` property to `db/models/profile.py`; update `ProfileResponse` schema to expose source_kind/intensity/bound state
  - [ ] 1.3 Unit tests: model defaults, existing rows backfill as 'manual'

- [ ] 2.0 Contract + validation layer (covers FR-8.1, FR-8.6; CLI-P1, CLI-P2, CLI-P3)
  - [ ] 2.1 Vendor `docs/schemas/cluster-definition-v1.json` from miStudio (frozen copy)
  - [ ] 2.2 Implement `api/schemas/cluster.py` (ClusterDefinitionV1/BundleV1/members/budget/refs incl. no-local-paths validator; caps)
  - [ ] 2.3 Schema sync test (`test_cluster_schema_sync.py`)
  - [ ] 2.4 Hostile-payload unit tests (unknown kind, major-version mismatch, oversize, >20 members, >50 defs, path/credential content)
  - [ ] 2.5 Implement `core/steering_range.py` clamp helper + unit test
  - [ ] 2.6 Error codes: PAYLOAD_TOO_LARGE / UNKNOWN_KIND / VALIDATION_ERROR mapped to ApiResponse.fail

- [ ] 3.0 ClusterService + Hub service (covers FR-8.2, FR-8.3, FR-8.5, FR-8.7; CLI-P4, CLI-P5, CLI-M*, CLI-H*)
  - [ ] 3.1 `cluster_service.py`: _map_definition (sign fold, λ basis), _assess_compatibility (bind/warn/unbound; n_features vs d_sae), _dedupe_name, import_definition/import_bundle (per-item isolation), export_definition (lossless), set_intensity
  - [ ] 3.2 Activation gate: bounds pre-check + λ scale+clamp in ProfileService.activate_profile AND _apply_request_profile (shared helper); range warnings at import
  - [ ] 3.3 `cluster_hub_service.py`: search (list_models filter), list_definitions (manifest.jsonl → *.cluster.json fallback, cap 200), fetch_definition (hf_hub_download, suffix+size caps), TTL cache + huggingface_circuit
  - [ ] 3.4 Unit tests: mapping, compat matrix rows, dedupe, clamp math + λ·strength>200 warning, export equality, hub service w/ mocked HfApi
  - [ ] 3.5 Unbound flow: import w/o SAE → activation structured refusal

- [ ] 4.0 API routes + wiring (covers FR-8.1..8.7 API surface)
  - [ ] 4.1 `routes/management/clusters.py`: list/import/hub search/hub definitions/hub import/activate/deactivate/intensity(id + active)/export
  - [ ] 4.2 DI providers in `api/dependencies.py`; register router in `routes/__init__.py`
  - [ ] 4.3 Route tests (unit-level, service mocked): envelope shapes, query params, repo_id:path encoding
  - [ ] 4.4 Error paths: unknown id 404-style fail, unbound activation refusal, hub failures surface breaker state
  - [ ] 4.5 Config keys (CLUSTER_HUB_CACHE_TTL_S, CLUSTER_HUB_TAG, CLUSTER_INTENSITY_MIN/MAX)

- [ ] 5.0 Clusters Admin-UI page (covers FR-8.8; CLI-U1..U5)
  - [ ] 5.1 Route `/clusters` + Sidebar entry + pages barrel
  - [ ] 5.2 `services/clusters.ts` + `types/clusters.ts` + `hooks/useClusters.ts`
  - [ ] 5.3 ClustersPage list + ClusterCard (badges, warnings, narrative markdown, budget readout)
  - [ ] 5.4 ClusterImportDialog (paste/file/HF tabs) + HubBrowser
  - [ ] 5.5 IntensitySlider wired to PUT intensity (reapply on active)
  - [ ] 5.6 Fix ImportExportButtons id types (string `prof_*`)
  - [ ] 5.7 Vitest: page render, import dialog flow, activate/deactivate hooks

- [ ] 6.0 Integration verification (covers FR-8.4 end-to-end)
  - [ ] 6.1 `test_cluster_import_workflow.py`: import→activate→`get_steering_values()` equals λ-clamped expectation
  - [ ] 6.2 Bundle per-item isolation; single-active invariant manual↔cluster; re-export equality
  - [ ] 6.3 Round-trip fixture: real miStudio-exported definition file checked into tests/fixtures

- [ ] 7.0 Feature Acceptance (per instruct 008)
  - [ ] 7.1 Verify every FPRD §9 success criterion + §2 acceptance checkbox one-by-one
  - [ ] 7.2 Manual: Clusters page docs (import, HF browse, activate, intensity)
  - [ ] 7.3 Full test suite green; update CLAUDE.md Document Inventory + Current Status

## Coverage Audit
- FR-8.1..8.8 each covered by ≥1 parent task (1.0–6.0 mapped above) ✓
- Every US acceptance criterion has an implementing sub-task and a test sub-task (2.4/3.4/4.3/5.7/6.x) ✓
- Every EC (8.1–8.5) has a test (2.4 caps/hostile, 3.4 clamp+warnings, 3.5/6.x unbound+bounds, 3.3 manifest fallback) ✓
- Every TDD/TID section maps to tasks (DB→1.x, schemas→2.x, services→3.x, API→4.x, UI→5.x, tests→throughout) ✓
- Open questions: none (FPRD §13) — no spike tasks needed ✓
- Final task is Feature Acceptance ✓
