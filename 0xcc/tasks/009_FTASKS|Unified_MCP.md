# Task List: Unified MCP

## miLLM Feature 9

**Document Version:** 1.0
**Created:** July 16, 2026
**Status:** Implemented 2026-07-16 (miLLM 0020dce + miStudio 40f55f2) — 4.2/5.2 live E2E post-deploy; reviews + acceptance pending
**References:** `009_FPRD|Unified_MCP.md` · `009_FTDD|Unified_MCP.md` · `009_FTID|Unified_MCP.md`

## Relevant Files

### miLLM (this repo)
- `millm/api/routes/system/health.py` — ActiveProfileInfo + active_profile on detailed health
- `docs/mcp-contract.md` — normative contract
- `tests/unit/api/test_health_active_profile.py`

### miStudio (CROSS-REPO — /home/x-sean/app/miStudio/backend/src/mcp_server/)
- `config.py`, `server.py`, `tools/__init__.py` — categories/gating/wiring
- `millm_client.py`, `health_gate.py`, `tools/millm_{runtime,clusters,sensing}.py` — new modules
- `backend/tests/unit/test_mcp_millm_*.py` — client/gate/tool/topology tests
- compose + k8s env: `MILLM_API_URL`, extended `MCP_TOOL_CATEGORIES`

### Notes
- Sequenced LAST in the increment (consumes 008/010/011 endpoints).
- Cross-repo tasks (2.0–4.0) execute in the miStudio repo with its commit conventions; tracked here for
  increment completeness.

### Category Checklist Results
- Data: N/A — no schema change beyond the additive health DTO field (no migration)
- Backend/API: 1.x (miLLM field), 2.x–3.x (server) ✓
- Frontend/UI: N/A — agent surface only (FPRD §6)
- Business logic: 3.x tool semantics, XOR-source validation ✓
- Integration wiring: 2.x (config/registration), 5.x (env/deploy) ✓
- Error handling & logging: 2.3 (gate reasons), 3.4 (structured unavailable), client BackendError ✓
- Testing: 1.2, 2.4, 3.5, 4.x ✓
- Performance & security: gate TTL (2.3); auth posture documented (1.3) ✓
- Config/deploy: 5.x ✓
- Documentation: 1.1 contract doc, 3.6 SERVER_INSTRUCTIONS ✓

## Tasks

- [x] 1.0 miLLM contract deliverables (covers FR-9.3; MCP-C1..C3)
  - [x] 1.1 Write `docs/mcp-contract.md` (endpoint inventory, envelope, health-gate contract, auth posture, additive-only versioning rule)
  - [x] 1.2 Add `ActiveProfileInfo` + `active_profile` to DetailedHealthResponse; populate from `ProfileRepository.get_active()`; unit test (null + populated)
  - [x] 1.3 Document auth posture + same-segment deployment guidance in the contract doc

- [x] 2.0 [CROSS-REPO] Server plumbing (covers FR-9.1; MCP-S1, MCP-S2, MCP-S4)
  - [x] 2.1 config.py: 3 categories in VALID_CATEGORIES (opt-in) + `MILLM_API_URL`
  - [x] 2.2 `millm_client.py` with envelope unwrap → BackendError
  - [x] 2.3 `health_gate.py` (TTL 10 s; degraded=available; reasons)
  - [x] 2.4 Unit tests: client unwrap paths, gate TTL/degraded/refused
  - [x] 2.5 server.py wiring: instantiate client+gate; skip millm_* registration when URL empty (log once)

- [x] 3.0 [CROSS-REPO] miLLM tool modules (covers FR-9.2; MCP-S3, MCP-S5)
  - [x] 3.1 `tools/millm_runtime.py` (status/list/activate/set_intensity)
  - [x] 3.2 `tools/millm_clusters.py` (list/import XOR-source/hub_search/activate/export)
  - [x] 3.3 `tools/millm_sensing.py` (status/events/enable/disable)
  - [x] 3.4 Uniform structured-unavailable decorator applied to all millm_* tools
  - [x] 3.5 Tool smoke tests (mocked MiLLMClient) incl. XOR validation
  - [x] 3.6 SERVER_INSTRUCTIONS: cross-product flow paragraph

- [ ] 4.0 [CROSS-REPO] Topology verification (covers FR-9.4)
  - [x] 4.1 Matrix test: both products / miStudio-only (URL unset ⇒ categories absent) / miLLM-down (structured unavailable)
  - [ ] 4.2 Live E2E: export_cluster_definition → millm_import_cluster → millm_activate_cluster against deployed stacks (US-9.1)

- [ ] 5.0 Deployment wiring (covers rollout)
  - [x] 5.1 [CROSS-REPO] mistudio k8s/base mcp.yaml + compose: MILLM_API_URL + categories env
  - [ ] 5.2 Verify health-gate behavior on the deployed pair; record in contract doc

- [ ] 6.0 Feature Acceptance (per instruct 008)
  - [ ] 6.1 Verify FPRD §9 criteria 1–4 one-by-one (topology matrix, live flow, single-call status, contract spot-check)
  - [ ] 6.2 Full suites green in BOTH repos; update CLAUDE.md Document Inventory + Current Status

## Coverage Audit
- FR-9.1..9.4 covered (2.0/5.0, 3.0, 1.0, 4.0 respectively) ✓
- All US acceptance criteria have implementing + testing sub-tasks (US-9.1→3.2/4.2; US-9.2→3.1/3.5 +1.2;
  US-9.3→3.3/3.5; US-9.4→2.3/4.1) ✓
- ECs: 9.1 degraded→2.3/2.4; 9.2 unset URL→2.5/4.1; 9.3 mid-session outage→3.4/4.1 ✓
- TDD/TID sections all mapped; UI category justified N/A ✓
- Open questions: none — no spike tasks ✓
- Final task is Feature Acceptance ✓
