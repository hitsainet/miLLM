# Feature PRD: Circuit Import, Slice-Fallback & Evidence Ladder

## miLLM Feature 13

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**Feature Priority:** Core (Increment: Circuit Runtime)
**References:** `BRD-MILLM-CIRCUITS-001.md` · `000_PPRD|miLLM.md` (v1.2, FR-13.x) · `000_PADR|miLLM.md` (v1.2) · `docs/mcp-contract.md` (v1.1)

---

## 1. Feature Overview

### Feature Name
Circuit Import, Slice-Fallback & Evidence Ladder — portable circuit definitions become serveable miLLM circuits, graded honestly.

### Brief Description
Import `mistudio.circuit-definition/v1` documents (multi-SAE, cross-layer, typed edges) from local JSON,
evaluate compatibility **per referenced SAE** against the attached SAE set, and register each as a
circuit that activates every member through **its own layer's SAE** at the authored per-layer budgets
under one global intensity. When not all referenced SAEs bind, activation degrades to the circuit's
per-layer `mistudio.cluster-definition/v1` **slice** — imported unchanged through the existing cluster
path (Feature 8) — never a wrong-decoder serve. Every circuit and edge carries its **EvidenceRung**
verbatim; the word "causal" never appears below rung 2, and activating a rung<2 circuit requires an
explicit acknowledgement. A new **Circuits** page in the Admin UI hosts the workflow.

### Problem Statement
miStudio now discovers cross-layer circuits, validates their edges with real causal interventions, and
grades each on an evidence ladder — then exports them as portable `circuit-definition/v1` artifacts. But
miLLM cannot run any of it: a circuit whose members live on layers L10 and L13 has no runtime, its
evidence rung has nowhere to surface, and a single-SAE host is left out entirely. The ecosystem
discovers and validates circuits it cannot serve, dial, or grade honestly at the point of live influence.

### Feature Goals
1. Executable circuit: import → serve every member through its own layer's SAE at authored budgets (BR-002, BR-004).
2. Honest per-SAE compatibility: bind / warn-bind / block / unbound, evaluated per referenced SAE (BR-003).
3. Never a mismatched serve: incomplete SAE set degrades to the per-layer cluster slice (BR-003).
4. Evidence honesty at the frontend: rung surfaced verbatim, "causal" forbidden below rung 2, rung<2 gated (BR-005).
5. Data-only safety: definitions can never execute, leak paths, or carry credentials (BR-010).

### User Value Proposition
"A circuit validated in miStudio runs in my serving stack — every member on its own layer, at the
strengths it was tuned to — and the runtime always tells me how much I can trust it. On a single-SAE
box I still get a usable per-layer projection instead of nothing."

### Connection to Project Objectives
Implements the core executable half of BRD-MILLM-CIRCUITS-001 ("make the portable circuit definition
executable") and owns the increment's evidence-rung surfacing. It sits on Feature 12 (multi-SAE serving)
and reuses Feature 8's cluster import path for slices; Features 14 (OWUI dial) and 15 (edge sensing)
build on the active circuit it establishes.

### BRD Traceability
| BR | Covered by |
|----|-----------|
| BR-002 (import + strict v1 validation) | CIR-P1, CIR-P2, US-13.1 |
| BR-003 (per-SAE compat + slice fallback) | CIR-P4, CIR-S1, CIR-S2, US-13.4 |
| BR-005 (evidence rung verbatim + rung<2 gate) | CIR-R1, CIR-R2, CIR-R3, US-13.3 |
| BR-010 (data-only safety) | CIR-P3, EC-13.1 |

---

## 2. User Stories & Scenarios

#### US-13.1: File import
**As a** researcher with a `.circuit.json` exported from miStudio
**I want to** import it into miLLM (paste or file upload)
**So that** it appears as a named circuit ready to activate, with its layers, edges, and rung shown.

**Acceptance Criteria:**
- [ ] Valid v1 definition imports and lists on the Circuits page with name, layers, edge count, rung + rung_language
- [ ] Invalid kind / schema-major mismatch is rejected with an actionable message (`UNKNOWN_KIND`)
- [ ] Name collision handled per `on_conflict` (default: rename with " (2)" suffix)
- [ ] Per-referenced-SAE compatibility verdicts recorded on the circuit row

#### US-13.2: Activate a serveable circuit
**As a** user with all referenced SAEs attached (via Feature 12)
**I want to** activate the circuit with one click
**So that** every member steers through its own layer's SAE at the authored per-layer budgets.

**Acceptance Criteria:**
- [ ] Activation applies each member through its layer's decoder; `serving_mode: "full"`
- [ ] The active circuit's name, layers, edge count, and rung are visible in circuit status
- [ ] Deactivation restores unsteered behavior; single-active invariant respected across manual/cluster/circuit
- [ ] A referenced SAE not attached ⇒ `SAE_SET_INCOMPLETE` (422) with the offending `{feature_idx, layer, sae_id}` list, then slice-fallback offered

#### US-13.3: Evidence honesty
**As a** user or agent inspecting an imported circuit
**I want** its evidence rung surfaced verbatim
**So that** I never mistake a mined circuit for a causally-validated one.

**Acceptance Criteria:**
- [ ] Circuit rung = MIN over its edges (empty edges ⇒ 0/MINED); each edge carries its own rung
- [ ] `rung_language` rendered verbatim from the ladder; the word "causal" never appears below rung 2
- [ ] Activating a rung<2 circuit without `acknowledge_unvalidated=true` refuses with `UNVALIDATED_CIRCUIT` (HTTP 200 + `success:false`)
- [ ] The acknowledgement requirement travels to MCP status and the Admin UI activation control

#### US-13.4: Slice-fallback on a single-SAE host
**As a** miLLM operator with only one SAE attached
**I want** the circuit to still steer via its per-layer slice
**So that** today's runtime is never left out, and I understand I'm steering a projection.

**Acceptance Criteria:**
- [ ] When only some referenced SAEs bind, activation consumes the circuit's per-layer `cluster-definition/v1` slice through the Feature 8 import path unchanged
- [ ] `GET /api/circuits/active` reports `serving_mode: "slice_fallback"` and the bound layer(s)
- [ ] The slice's partial-rendering marker (name suffix ` [L{n} slice]` + `provenance.source_note`) is surfaced — a slice is never presented as the whole circuit
- [ ] No member is ever steered through a mismatched SAE

#### US-13.5: Re-export
**As a** user
**I want to** export an imported circuit back to `mistudio.circuit-definition/v1`
**So that** the artifact stays mobile.

**Acceptance Criteria:**
- [ ] Export equals import semantically (members, per-layer budgets, edges, rungs, provenance preserved; unknown fields survive)

#### Edge Cases

**EC-13.1: Oversized/hostile payload** — **Trigger:** file > 1 MB, > 16 layers, > 200 edges, > 20
members/layer, non-JSON, or filesystem-path/credential content. **Behavior:** reject before parse-heavy
work (`PAYLOAD_TOO_LARGE` / validation). **Message:** specific cap or field violated.

**EC-13.2: Circuit references an SAE that is not attached** — **Trigger:** a member's layer has no
attached SAE. **Behavior:** full serving blocked (`SAE_SET_INCOMPLETE`, 422); slice-fallback offered for
bound layers. **Message:** lists offending `{feature_idx, layer, sae_id}`.

**EC-13.3: Referenced SAE attached but wrong feature space** — **Trigger:** attached SAE at that layer
has a different n_features. **Behavior:** that layer treated as unbound (`INCOMPATIBLE_FEATURE_SPACE`,
422 at activation); degrades to slice-fallback, never a wrong-basis serve.

**EC-13.4: Empty-edge circuit** — **Trigger:** members but no edges. **Behavior:** rung resolves to
0/MINED; activation requires `acknowledge_unvalidated=true`. **Message:** "associated (rung 0)".

**EC-13.5: rung_language rephrase attempt** — **Trigger:** any surface trying to word the rung itself.
**Behavior:** forbidden — all surfaces render server-supplied `rung_language`; a copy-audit test enforces
no "causal" string below rung 2.

---

## 3. Functional Requirements

### Import & Validation (FR-13.1, FR-13.6)

| ID | Requirement | Priority |
|----|-------------|----------|
| CIR-P1 | System shall parse and strictly validate `mistudio.circuit-definition/v1` payloads (kind-keyed; schema vendored + sync-tested) | Must |
| CIR-P2 | System shall enforce caps: ≤1 MB payload, ≤16 layers, ≤200 edges, ≤20 members/layer | Must |
| CIR-P3 | System shall reject definitions containing filesystem paths or credential-like content and never execute imported content | Must |
| CIR-P4 | System shall evaluate compatibility **per referenced SAE** (bind / warn-bind / block / unbound; n_features hard concern, model/layer warnings) and persist per-SAE outcomes | Must |
| CIR-P5 | Import shall reject unknown kinds and incompatible schema major versions with actionable messages | Must |

### Serving & Slice-Fallback (FR-13.2, FR-13.3)

| ID | Requirement | Priority |
|----|-------------|----------|
| CIR-S1 | A circuit shall be fully serveable **only when all referenced SAEs bind**; activation applies each member through its own layer's SAE at the authored per-layer budgets under one global λ (delegates to Feature 12) | Must |
| CIR-S2 | When the SAE set is incomplete, activation shall consume the circuit's per-layer `mistudio.cluster-definition/v1` slice through the Feature 8 cluster-import path unchanged (`serving_mode: "slice_fallback"`) — never serving a member through a mismatched SAE | Must |
| CIR-S3 | `GET /api/circuits/active` shall report `serving_mode` (`full`\|`slice_fallback`) and, in fallback, the bound layer(s); the slice's partial-rendering marker is surfaced | Must |
| CIR-S4 | Per-layer budgets/strengths shall be FROZEN as authored — never recomputed against the local SAE set | Must |

### Evidence Ladder (FR-13.4, FR-13.5)

| ID | Requirement | Priority |
|----|-------------|----------|
| CIR-R1 | System shall store and surface each circuit's and each edge's `rung` (0–3) + `rung_language` verbatim from the ladder wherever steering state is shown | Must |
| CIR-R2 | Circuit rung shall be the MIN over its edges (empty edges ⇒ 0/MINED); the word "causal" shall never appear below rung 2 (copy-audit test) | Must |
| CIR-R3 | Activating a rung<2 circuit shall require `acknowledge_unvalidated=true`; otherwise `UNVALIDATED_CIRCUIT` (HTTP 200 + `success:false`), carried to MCP + UI | Must |

### Circuits UI (FR-13.7)

| ID | Requirement | Priority |
|----|-------------|----------|
| CIR-U1 | New sidebar page "Circuits" listing imported circuits with layers, edge count, rung badge (rung_language verbatim), serveable/slice-fallback + imported badges, per-SAE warnings | Must |
| CIR-U2 | Import dialog (paste / file tabs) | Must |
| CIR-U3 | Activation control with the unvalidated-rung gate (explicit acknowledgement checkbox when rung<2) and slice-fallback disclosure | Must |
| CIR-U4 | Activate/deactivate, export download; existing Clusters/Profiles pages continue to show only their own row kinds | Must |

---

## 4. Data Requirements

Migration `011_add_circuits_table.py` (new `circuits` table — the circuit is NOT a `profiles` row; it is
a graph over layers, and slice-fallback materializes an ordinary cluster profile via the Feature 8 path):

```sql
CREATE TABLE circuits (
    id            VARCHAR(40) PRIMARY KEY,        -- 'circ_<hex>'
    name          VARCHAR(120) NOT NULL,
    model_id      VARCHAR(200) NULL,
    rung          SMALLINT NOT NULL DEFAULT 0,    -- MIN over edges (0..3)
    layers        JSONB NOT NULL,                 -- sorted list of referenced layers
    edge_count    INTEGER NOT NULL DEFAULT 0,
    circuit_meta  JSONB NOT NULL,                 -- full original definition (lossless) + per-SAE warnings + hub_ref?
    is_active     BOOLEAN NOT NULL DEFAULT FALSE, -- single-active across manual/cluster/circuit
    serving_mode  VARCHAR(16) NULL,               -- 'full'|'slice_fallback' when active
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX uq_circuits_active ON circuits (is_active) WHERE is_active;
```

`circuit_meta` shape: `{schema_version, kind, name, narrative, model, saes[{layer,mistudio_sae_id,
n_features,d_model,hook_type,source_hint}], members[...verbatim, keyed to layer...], edges[{source,
target, type, rung, rung_language, statistics?, attribution?, validation?}], budgets{per_layer...,
intensity, intensity_range}, provenance{...}, per_sae_warnings[...], hub_ref?}`.

Vendored contract: `docs/schemas/circuit-definition-v1.json` (copied from miStudio; frozen v1) +
pydantic mirror + sync test — same pattern as `cluster-definition-v1.json`.

---

## 5. API Specifications

New router `/api/circuits` (module `millm/api/routes/management/circuits.py`), `ApiResponse` envelope
(matches `docs/mcp-contract.md` §4 `millm_circuits`):

#### GET /api/circuits?promoted=&min_rung=&limit=&offset=
Slim rows: `{id, name, rung, rung_language, layers, edge_count, serveable, imported}`.

#### GET /api/circuits/active
Active circuit + attached-SAE set + rung + `serving_mode` (`full`\|`slice_fallback`, bound layers in fallback); `null` when none.

#### POST /api/circuits/import?activate=&on_conflict=&acknowledge_unvalidated=
Body: raw `mistudio.circuit-definition/v1`. Returns `{id, name, rung, rung_language, per_sae_compat[], serveable, warnings[]}`.

#### POST /api/circuits/{id}/activate?acknowledge_unvalidated=
Full serve when all SAEs bind, else slice-fallback. Refuses `UNVALIDATED_CIRCUIT` when rung<2 and not acknowledged; `SAE_SET_INCOMPLETE`/`INCOMPATIBLE_FEATURE_SPACE` drive the fallback.

#### POST /api/circuits/{id}/deactivate
#### PUT /api/circuits/active/intensity — `{intensity, reapply}` (one global λ scales all layers; used by Feature 14)
#### GET /api/circuits/{id}/export — raw circuit document (no envelope)

Hub search (`GET /api/circuits/hub/search`) and edge-sensing routes are declared in the contract but
owned by later increment features (14/15); this feature ships the import/activate/export core.

---

## 6. UI Requirements

- Route `/circuits` + Sidebar entry (icon: `Waypoints`), page `CircuitsPage.tsx`.
- `components/circuits/`: `CircuitCard` (rung badge, layer chips, edge count, serveable/slice badges,
  per-SAE warnings), `CircuitImportDialog` (paste/file tabs), `CircuitActivateControl`
  (unvalidated-ack checkbox when rung<2, slice-fallback disclosure banner).
- API client `services/circuits.ts`; React Query hooks `hooks/useCircuits.ts`; types `types/circuits.ts`.
- Rung language ALWAYS rendered from the server field — never composed client-side.

---

## 7. Non-Functional Requirements
- Import validation completes < 500 ms for a max-size definition (pure CPU).
- No new auth surface; endpoints follow the existing unauthenticated management-API posture (NFR-4.1).
- Circuit serving delegates to Feature 12; this feature adds no hot-path latency beyond activation.

## 8. Dependencies
- Feature 12 (Multi-SAE Serving) — attached-SAE set, per-layer apply, one-λ composition.
- Feature 8 (Cluster Import) — slice-fallback rides the cluster import path unchanged.
- Feature 6 (Profile Management) — single-active invariant (now spans manual/cluster/circuit).
- `docs/mcp-contract.md` v1.1 §4 `millm_circuits` + §4a rung rule + §4b slice-fallback + §5 error codes.
- miStudio-published `mistudio.circuit-definition/v1` schema (frozen) + `evidence_ladder.py` vocabulary.

## 9. Success Criteria
1. E2E: export a validated circuit from miStudio → import → activate → every member applied through its own layer's SAE at authored strengths (round-trip test).
2. On a single-SAE host, the same circuit steers via its per-layer slice, `serving_mode: "slice_fallback"`, zero reconfiguration.
3. Per-referenced-SAE compat verdicts match the cluster matrix semantics for the same inputs.
4. 0 instances of rung<2 steering labeled "causal" anywhere (copy-audit test); 100% of rung<2 activations gated behind the acknowledgement.
5. Re-export equality + all caps/hostile-payload tests pass.

## 10. Testing Requirements
- Unit: schema validation (valid/hostile/caps), circuit rung = MIN(edges), per-SAE compat matrix,
  slice-projection consumed by the cluster path, **copy-audit "no 'causal' below rung 2"**, **schema sync test**.
- Integration: import→activate (full)→per-layer steering-values assertion; incomplete-SAE→slice-fallback;
  rung<2 activation refusal without ack; single-active invariant across manual↔cluster↔circuit; export equality.
- UI: CircuitsPage list/import/activate incl. unvalidated-ack + slice disclosure (Vitest); Playwright post-deploy.

## 11. Rollout & Migration
Alembic 008 is additive (new table only); zero behavior change for existing profiles/clusters until a
circuit is imported.

## 12. Out of Scope
Circuit discovery/validation/authoring (miStudio's job); publishing to HF; modifying any frozen schema;
recomputing per-layer budgets on import (frozen as authored); the OWUI dial (Feature 14) and edge
sensing (Feature 15) beyond the active-circuit foundation they consume.

## 13. Open Questions
None blocking — increment-level questions resolved (see §15).

## 14. Documentation Requirements
Manual page: Circuits (import, per-SAE compatibility, evidence ladder, slice-fallback, activation gate);
`docs/mcp-contract.md` cross-ref (§4 `millm_circuits`, already at v1.1).

## 15. Decisions from Clarifying Questions
Recorded from the BRD round + the 2026-07-20 clarifying round:
1. **Storage:** a new `circuits` table (a circuit is a graph, not a single-layer profile); slice-fallback
   materializes a cluster profile via the Feature 8 path — no new import machinery for the fallback.
2. **Budgets frozen on import** — never recomputed against local SAEs (matches cluster-import semantics).
3. **Evidence rung verbatim** — the ladder is the single vocabulary; "causal" forbidden below rung 2,
   enforced by a copy-audit test mirroring miStudio's.
4. **Slice-fallback is never presented as the whole circuit** — partial-rendering marker always surfaced.
5. **UI:** dedicated **Circuits page**; Clusters and Profiles pages unchanged for their row kinds.
