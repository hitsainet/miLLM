# Feature PRD: Unified MCP

## miLLM Feature 9

**Document Version:** 1.0
**Created:** July 16, 2026
**Status:** Draft
**Feature Priority:** Core (Increment: Cluster Runtime)
**References:** `BRD-MILLM-CLUSTERS-001.md` · `000_PPRD|miLLM.md` (v1.1, FR-9.x) · `000_PADR|miLLM.md` (v1.1) · miStudio `backend/src/mcp_server/` (cross-repo base)

---

## 1. Feature Overview

### Feature Name
Unified MCP — one agent surface across miStudio (authoring) and miLLM (serving).

### Brief Description
Evolve the existing miStudio MCP server (bearer auth, category gating, audit log, 38 tools) into the
ecosystem's single MCP endpoint by adding a miLLM HTTP client and three miLLM tool categories
(`millm_runtime`, `millm_clusters`, `millm_sensing`), gated by per-product health checks so any
deployment topology (both products / miStudio-only / miLLM-only) presents a coherent, self-describing
tool set. miLLM's deliverables are its **published API contract** (`docs/mcp-contract.md`) and one
additive health-endpoint field; the server code lands in the miStudio repo (cross-repo tasks flagged).

### Problem Statement
Agents can author, tune, validate, and export clusters through the miStudio MCP server — then hit a
wall: deploying a cluster into the serving runtime (miLLM) requires leaving the agent surface entirely.
miLLM has no MCP presence, and standing up a second server would force agents to juggle endpoints and
duplicate auth/gating infrastructure.

### Feature Goals
1. One endpoint, whole ecosystem: authoring → serving flows in a single agent session (BR-008).
2. Complete miLLM tool set: status, profiles, cluster import (file + hub), intensity, sensing (BR-009).
3. Graceful topology handling: absent product ⇒ structured "unavailable", never dead tools (BR-008).
4. Contract-first decoupling: the two release trains stay independent.

### User Value Proposition
"My agent exports a cluster from miStudio and imports/activates/dials it in miLLM in the same breath —
one MCP endpoint, and it still works sensibly when only one product is running."

### Connection to Project Objectives
Implements the BRD's "give the ecosystem ONE agent surface" objective; consumes the API surfaces built
by Features 8, 10, and 11 (hence sequenced last).

---

## 2. User Stories & Scenarios

#### US-9.1: Cross-product deploy flow
**As an** AI agent on the unified MCP endpoint
**I want to** call `export_cluster_definition` (miStudio) then `millm_import_cluster` then `millm_activate_cluster`
**So that** a tuned cluster moves from authoring to serving without leaving the session.

**Acceptance Criteria:**
- [ ] The flow completes end-to-end against live deployments of both products
- [ ] `millm_import_cluster` accepts an inline definition JSON or `{repo_id, filename}` (hub import)
- [ ] Import warnings/blocks surface in the tool result exactly as the REST API reports them

#### US-9.2: Runtime inspection & control
**As an** agent
**I want** `millm_status`, `millm_list_profiles`, `millm_activate_profile`, `millm_set_intensity`
**So that** I can inspect and steer the serving runtime.

**Acceptance Criteria:**
- [ ] `millm_status` returns health + loaded model + attached SAE + active profile {id, name, source_kind, intensity} in ONE call
- [ ] `millm_set_intensity` adjusts the active cluster's λ (documented as global)

#### US-9.3: Sensing readout
**As an** agent investigating cluster behavior
**I want** `millm_sensing_status`, `millm_get_sensing_events`, `millm_enable_sensing`, `millm_disable_sensing`
**So that** I can arm sensing and read co-activation events with token context.

**Acceptance Criteria:**
- [ ] Events return with fired members, alone/within flag, context text, and summaries

#### US-9.4: Single-product topology
**As an** operator running only miLLM (or only miStudio)
**I want** the unified server to present a coherent tool set
**So that** agents aren't confronted with dead tools.

**Acceptance Criteria:**
- [ ] Health gate polls each product (TTL ~10 s); a down product's tools return structured "<product> unavailable: <reason>"
- [ ] `/health` on the MCP server reports per-product availability

#### Edge Cases
**EC-9.1: miLLM degraded (no model loaded)** — **Trigger:** health returns `degraded`. **Behavior:**
tools remain usable (status/import work; activation reports the missing model).
**EC-9.2: MILLM_API_URL unset** — **Behavior:** millm_* categories refuse to register at startup
(mirrors the existing category-gating pattern); server runs miStudio-only.
**EC-9.3: mid-session product outage** — **Behavior:** next tool call returns the structured
unavailable result; no crash, no unregistration churn.

---

## 3. Functional Requirements

### miLLM-side deliverables (this repo)

| ID | Requirement | Priority |
|----|-------------|----------|
| MCP-C1 | Publish `docs/mcp-contract.md`: the management endpoints the unified server consumes, request/response envelopes, and error semantics | Must |
| MCP-C2 | Extend `DetailedHealthResponse` with `active_profile: {id, name, source_kind, intensity} \| null` | Must |
| MCP-C3 | Document the auth posture: management API is unauthenticated; server→miLLM traffic stays same-segment; optional bearer is future work | Must |

### Cross-repo deliverables (miStudio repo — flagged tasks)

| ID | Requirement | Priority |
|----|-------------|----------|
| MCP-S1 | Categories `millm_runtime`, `millm_clusters`, `millm_sensing` added to VALID_CATEGORIES (opt-in, not default), gated on `MILLM_API_URL` | Must |
| MCP-S2 | `MiLLMClient` unwrapping miLLM's `ApiResponse{success,data,error}` envelope | Must |
| MCP-S3 | Tool set: millm_status, millm_list_profiles, millm_activate_profile, millm_set_intensity; millm_list_clusters, millm_import_cluster (inline or hub), millm_hub_search, millm_activate_cluster, millm_export_cluster; millm_sensing_status, millm_get_sensing_events, millm_enable_sensing, millm_disable_sensing | Must |
| MCP-S4 | `HealthGate` polling `GET /api/health` per product with ~10 s TTL; down ⇒ structured unavailable results (tools stay registered) | Must |
| MCP-S5 | `SERVER_INSTRUCTIONS` documents the cross-product flow (export → import → activate → dial → sense) | Should |

---

## 4. Data Requirements
None in miLLM beyond MCP-C2 (health response shape). No migration.

## 5. API Specifications

The complete endpoint inventory consumed by the server (normative content of `docs/mcp-contract.md`):

| Tool | miLLM endpoint |
|------|----------------|
| millm_status | `GET /api/health/detailed` (incl. new active_profile) |
| millm_list_profiles | `GET /api/profiles` |
| millm_activate_profile | `POST /api/profiles/{id}/activate` |
| millm_set_intensity | `PUT /api/clusters/active/intensity` |
| millm_list_clusters | `GET /api/clusters` |
| millm_import_cluster | `POST /api/clusters/import` or `POST /api/clusters/hub/import` |
| millm_hub_search | `GET /api/clusters/hub/search` |
| millm_activate_cluster | `POST /api/clusters/{id}/activate` |
| millm_export_cluster | `GET /api/clusters/{id}/export` |
| millm_sensing_* | `GET /api/sensing/status` · `GET /api/sensing/events` · `POST /api/sensing/clusters/{id}/enable\|disable` |

Health-gate contract: `GET /api/health` → `{status: healthy|degraded|unhealthy, version, timestamp,
uptime_seconds}`; available ⇔ HTTP 200 ∧ status ≠ unhealthy (degraded = reachable, tools usable).

## 6. UI Requirements
None (agent surface). MCP server deployment note: runs in the miStudio namespace today; if co-located
with miLLM later, `k8s/base/` gains a manifest (documented, not executed this increment).

## 7. Non-Functional Requirements
- Health-gate poll adds ≤1 upstream call per 10 s per product.
- Tool latency dominated by the underlying REST call; no server-side caching of tool results.

## 8. Dependencies
- Feature 8 (cluster endpoints), Feature 10 (intensity endpoint), Feature 11 (sensing endpoints).
- miStudio MCP server codebase (`backend/src/mcp_server/` — config/category gating, client, tools, auth).

## 9. Success Criteria
1. Topology matrix: correct tool behavior in 3/3 configurations (both / miStudio-only / miLLM-only).
2. US-9.1 cross-product flow verified end-to-end on live deployments.
3. `millm_status` is one call (no tool needs two round-trips for basic state).
4. Contract doc matches the deployed API (spot-checked per endpoint).

## 10. Testing Requirements
- miLLM: health-response field test (active_profile block, incl. null when none active).
- miStudio (cross-repo): client envelope unwrap tests; health-gate TTL/degraded/unavailable tests;
  tool smoke tests against a mocked MiLLMClient; topology matrix integration test.

## 11. Rollout & Migration
Additive only. `millm_*` categories are opt-in via MCP_TOOL_CATEGORIES; existing deployments unchanged
until the operator enables them + sets MILLM_API_URL.

## 12. Out of Scope
Third-repo extraction; MCP server auth changes; miLLM-side MCP server implementation; tool coverage
beyond the four themes (e.g. model download control) — future increments.

## 13. Open Questions
None blocking. Deployment co-location (millm namespace vs mistudio namespace) is an operator choice
documented in the contract doc.

## 14. Documentation Requirements
`docs/mcp-contract.md` (normative); manual cross-reference from the Clusters page docs; miStudio-side
manual updates ride with the cross-repo tasks.

## 15. Decisions from Clarifying Questions
1. **Home: evolve the miStudio MCP server** (user decision 2026-07-16) — proven auth/gating/audit;
   alternatives (new server in miLLM, third repo) rejected per PADR v1.1.
2. Tools stay REGISTERED when a product is down (structured unavailable) rather than dynamic
   unregistration — agents can retry without tool-list churn (design default recorded).
3. `millm_*` categories are opt-in, preserving existing miStudio-only deployments (design default).
