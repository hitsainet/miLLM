# Task List: Circuit Import, Slice-Fallback & Evidence Ladder

## miLLM Feature 13

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** 📋 DOCS COMPLETE 2026-07-20 — implementation not started
**References:** `013_FPRD|Circuit_Import.md` · `013_FTDD|Circuit_Import.md` · `013_FTID|Circuit_Import.md` · `docs/mcp-contract.md` (v1.1)

## Relevant Files

### Backend
- `millm/db/migrations/versions/011_add_circuits_table.py` — new `circuits` table (additive; next free after 010 on disk)
- `millm/db/models/circuit.py` — circuits model + `serveable` property
- `millm/db/repositories/circuit_repository.py` — CRUD + single-active guard
- `millm/core/circuit_evidence.py` — EvidenceRung + RUNG_LANGUAGE + circuit_rung (single vocabulary)
- `millm/api/schemas/circuit.py` — v1 circuit contract mirror + DTOs
- `millm/services/circuit_service.py` — import, per-SAE compat, activate, slice-fallback, export
- `millm/api/routes/management/circuits.py`, `millm/api/dependencies.py`, `millm/api/routes/__init__.py`
- `docs/schemas/circuit-definition-v1.json` — vendored frozen schema

### Frontend
- `admin-ui/src/pages/CircuitsPage.tsx`, `App.tsx`, `components/layout/Sidebar.tsx`, `pages/index.ts`
- `admin-ui/src/components/circuits/*` — CircuitCard, CircuitImportDialog, CircuitActivateControl
- `admin-ui/src/services/circuits.ts`, `hooks/useCircuits.ts`, `types/circuits.ts`

### Tests
- `tests/unit/api/test_circuit_schema.py`, `test_circuit_schema_sync.py`
- `tests/unit/core/test_circuit_evidence.py` (incl. copy-audit "no 'causal' below rung 2")
- `tests/unit/services/test_circuit_service.py`
- `tests/integration/test_circuit_import_workflow.py`

### Notes
- Follow `007_process-task-list.md`: one sub-task at a time; full suite + commit per parent task.
- Test commands: `pytest` (backend), `npm test` in admin-ui (Vitest).

### Category Checklist Results
- Data: tasks 1.x (migration + model + repo) ✓
- Backend/API: tasks 2.x–5.x ✓
- Frontend/UI: tasks 6.x ✓
- Business logic: tasks 3.x–4.x (rung, per-SAE compat, slice-fallback gate) ✓
- Integration wiring: tasks 5.x (DI, router), 6.1 (routing/nav) ✓
- Error handling & logging: 2.5, 4.4, 5.4 ✓
- Testing: every parent has paired test sub-tasks; 7.x integration ✓
- Performance & security: 2.4 caps/hostile, 3.3 copy-audit, EC tests ✓
- Config/deploy: 5.5 (config keys); migration auto-runs on container start — no deploy change ✓
- Documentation: 8.2 (manual page) ✓

## Tasks

- [ ] 1.0 Data layer: circuits table (covers FR-13.1 storage; CIR-P1)
  - [ ] 1.1 Write migration `011_add_circuits_table.py` (upgrade+downgrade, partial unique active index); verify round-trip locally
  - [ ] 1.2 Add `db/models/circuit.py` (columns + `serveable` property) and `db/repositories/circuit_repository.py` (CRUD + single-active guard)
  - [ ] 1.3 Unit tests: model defaults, `uq_circuits_active` enforces one active, repo CRUD

- [ ] 2.0 Contract + validation layer (covers FR-13.1, FR-13.6; CIR-P1, CIR-P2, CIR-P3, CIR-P5)
  - [ ] 2.1 Vendor `docs/schemas/circuit-definition-v1.json` from miStudio (frozen copy)
  - [ ] 2.2 Implement `api/schemas/circuit.py` (CircuitDefinitionV1/SAERef/NodeRef/Edge/Member/Budgets; reuse cluster.py refs + no-local-paths validator; `extra="allow"` for Tier-2.5 fields)
  - [ ] 2.3 Schema sync test (`test_circuit_schema_sync.py`, cloned from cluster sync test)
  - [ ] 2.4 Hostile-payload unit tests (unknown kind, major-version mismatch, oversize 1 MB, >16 layers, >200 edges, >20 members/layer, path/credential content)
  - [ ] 2.5 Error codes: PAYLOAD_TOO_LARGE / UNKNOWN_KIND / VALIDATION_ERROR mapped to ApiResponse.fail (200+envelope house style)

- [ ] 3.0 Evidence-rung vocabulary (covers FR-13.4, FR-13.5; CIR-R1, CIR-R2)
  - [ ] 3.1 Implement `core/circuit_evidence.py`: EvidenceRung enum + RUNG_LANGUAGE (verbatim), rung_language(), circuit_rung = MIN(edges) (empty ⇒ 0)
  - [ ] 3.2 Unit tests: rung values EXACT, circuit_rung MIN semantics, empty-edge ⇒ MINED
  - [ ] 3.3 **Copy-audit test**: grep runtime + UI surfaces; assert "causal" never co-occurs with a rung<2 render (mirrors miStudio guard)

- [ ] 4.0 CircuitService (covers FR-13.2, FR-13.3, FR-13.4, FR-13.5; CIR-P4, CIR-S1..S4, CIR-R3)
  - [ ] 4.1 `circuit_service.py`: import_definition (per-SAE compat bind/warn/block/unbound, serveable ⇔ all bind; dedupe name; store frozen circuit_meta + per_sae_warnings), export_definition (lossless)
  - [ ] 4.2 Activation gate: rung<2 → UNVALIDATED_CIRCUIT without ack; SAE set complete → delegate to Feature 12 `apply_circuit` (serving_mode="full"); cross-kind single-active deactivation
  - [ ] 4.3 Slice-fallback: incomplete SAE set → `to_layer_slice` per bound layer → `ClusterService.import_definition(activate=True)` (Feature 8 path, unchanged); serving_mode="slice_fallback" + bound layers; SAE_SET_INCOMPLETE when no layer binds
  - [ ] 4.4 Unit tests: per-SAE compat matrix rows, serveable logic, rung<2 refusal/ack, slice projects a valid cluster-definition/v1 with ` [L{n} slice]` marker, export equality
  - [ ] 4.5 Error paths: CIRCUIT_NOT_FOUND, SAE_SET_INCOMPLETE (with offending {feature_idx,layer,sae_id}), INCOMPATIBLE_FEATURE_SPACE

- [ ] 5.0 API routes + wiring (covers FR-13.1..13.7 API surface; matches mcp-contract §4 millm_circuits)
  - [ ] 5.1 `routes/management/circuits.py`: list (promoted/min_rung/limit/offset), active, import, activate(ack), deactivate, active/intensity, export (raw doc, no envelope)
  - [ ] 5.2 DI provider in `api/dependencies.py`; register router in `routes/__init__.py`
  - [ ] 5.3 Route tests (unit-level, service mocked): envelope shapes, query params, rung<2 gate surfaced, slice-fallback disclosure in active response
  - [ ] 5.4 Error paths: unknown id 404, UNVALIDATED_CIRCUIT 200+envelope, SAE_SET_INCOMPLETE 422
  - [ ] 5.5 Config keys (CIRCUIT_HUB_TAG, CIRCUIT_MAX_LAYERS/EDGES/MEMBERS_PER_LAYER)

- [ ] 6.0 Circuits Admin-UI page (covers FR-13.7; CIR-U1..U4)
  - [ ] 6.1 Route `/circuits` + Sidebar entry (Waypoints) + pages barrel
  - [ ] 6.2 `services/circuits.ts` + `types/circuits.ts` + `hooks/useCircuits.ts`
  - [ ] 6.3 CircuitsPage list + CircuitCard (rung badge from server rung_language, layer chips, edge count, serveable/slice badges, per-SAE warnings)
  - [ ] 6.4 CircuitImportDialog (paste/file tabs)
  - [ ] 6.5 CircuitActivateControl: unvalidated-ack checkbox when rung<2 + slice-fallback disclosure banner
  - [ ] 6.6 Vitest: page render, import dialog flow, activate with/without ack, slice disclosure

- [ ] 7.0 Integration verification (covers FR-13.2..13.5 end-to-end)
  - [ ] 7.1 `test_circuit_import_workflow.py`: import → activate (full, all SAEs attached) → each member applied through its own layer's SAE at authored strength
  - [ ] 7.2 Incomplete SAE set → slice-fallback via cluster path; serving_mode="slice_fallback"; rung<2 refusal without ack then success with ack; single-active manual↔cluster↔circuit; re-export equality
  - [ ] 7.3 Round-trip fixture: real miStudio-exported circuit definition checked into tests/fixtures

- [ ] 7.5 **Inherited from Feature 12 review (REQUIRED — do not drop)**
  - [ ] 7.5.1 **Circuit/cluster co-tenancy guard:** serving/clearing a circuit must not silently clobber an active cluster steering the same attached SAE. Detect an active cluster/profile on any target layer and refuse (409) or explicitly deactivate + warn. (F12 R2/R3 finding)
  - [ ] 7.5.2 **Cluster binds to `entries[0]`:** `cluster_service._bind_sae` / `profile_service` resolve via `state.attached_sae` (first entry) and only WARN on layer mismatch — once a multi-SAE circuit is attached this can bind a cluster to the wrong layer's SAE. Resolve via `by_layer(declared_layer)` and hard-block the mismatch. (F12 R3 architect finding — F12's "never a silent wrong-basis serve" must hold for clusters too)
  - [ ] 7.5.3 **`attach_set` side-effect parity:** it omits `SAEStatus.ATTACHED`, `create_attachment`, model auto-lock and sensing re-arm that `attach_sae` performs. At minimum take the model lock + write status so an unload can't tear out live hooks and `delete_sae`'s attached-guard works. Extract a shared `_post_attach()`. (F12 R3)
  - [ ] 7.5.4 **Composing attach/detach/serve lock:** the `SAEService` docstring claims a `_attachment_lock` that does not exist; resolve-then-apply and pre-check-then-load are check-then-act windows. Add the lock (or correct the docstring). (F12 R2/R3)
  - [ ] 7.5.5 **Hazard presentation:** `_cross_layer_hazards` is O(n²) and mostly `heuristic:co-steer-sign` (a 6/6 two-layer circuit → 36 low-signal warnings). Rank validated (rung≥2, |ES|) first and cap/aggregate the heuristic tail before rendering. (F12 R3 product)
  - [ ] 7.5.6 `/health/detailed` still reports a singular `sae_id` for an N-SAE set (metrics was fixed in F12) — add `sae_count`/`sae_ids` additively. (F12 R3)

- [ ] 8.0 Feature Acceptance (per instruct 007)
  - [ ] 8.1 Verify every FPRD §9 success criterion + §2 acceptance checkbox one-by-one
  - [ ] 8.1b **Re-verify Feature 12 §9.1/9.3/9.4 end-to-end** (two-layer circuit serves; SAE_SET_INCOMPLETE at submit AND activation; hazards surface at activation) — F12 verified these at service level only because activation lands here. (F12 R3 product finding)
  - [ ] 8.2 Manual: Circuits page docs (import, per-SAE compatibility, evidence ladder, slice-fallback, activation gate)
  - [ ] 8.3 Full test suite green; update CLAUDE.md Document Inventory + Current Status

## Coverage Audit
- FR-13.1..13.7 each covered by ≥1 parent task (1.0–7.0 mapped above) ✓
- Every US acceptance criterion has an implementing sub-task and a test sub-task (2.4/3.x/4.4/5.3/6.6/7.x) ✓
- Every EC (13.1–13.5) has a test (2.4 caps/hostile, 4.4/4.5 SAE-set + wrong-space, 3.2 empty-edge, 3.3 rephrase copy-audit) ✓
- Every TDD/TID section maps to tasks (DB→1.x, schema→2.x, rung→3.x, service→4.x, API→5.x, UI→6.x, tests→throughout) ✓
- Open questions: none (FPRD §13) — no spike tasks needed ✓
- Final task is Feature Acceptance ✓

## Review rounds (goal: 3 rounds, ≥10 findings each)
- [ ] Round 1 (multi-angle /code-review, 2 finder agents): triage + fix; watch for slice-as-whole-circuit leaks and rung rephrase.
- [ ] Round 2 (post-fix verification + fresh angles): re-verify R1 fixes hold; per-SAE gate before Feature 12 delegation; export losslessness at the boundary.
- [ ] Round 3 (/review, 4 perspectives): final sweep; copy-audit clean; single-active cross-kind invariant; fix pre-existing/latent defects too (user directive).
- Record: `.claude/context/sessions/review_feature013_R{1,2,3}_2026-07-*.md`.

## Acceptance evidence (Task 8.0)
- FPRD §9 criteria: (1) real-fixture round-trip (multi-SAE circuit → each member applied through its own
  layer's SAE at authored strengths); (2) single-SAE host → slice-fallback, serving_mode="slice_fallback",
  zero reconfiguration; (3) per-SAE compat verdicts match the cluster matrix semantics (unit rows);
  (4) copy-audit "no 'causal' below rung 2" green + 100% rung<2 activations gated; (5) raw-doc export
  equality incl. unknown-field survival + caps/hostile tests green.
- Suites: backend + frontend to be filled at close-out; builds green; manual builds with the new Circuits page.
