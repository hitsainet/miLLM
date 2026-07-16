# Feature PRD: Cluster Import

## miLLM Feature 8

**Document Version:** 1.0
**Created:** July 16, 2026
**Status:** Draft
**Feature Priority:** Core (Increment: Cluster Runtime)
**References:** `BRD-MILLM-CLUSTERS-001.md` · `000_PPRD|miLLM.md` (v1.1, FR-8.x) · `000_PADR|miLLM.md` (v1.1)

---

## 1. Feature Overview

### Feature Name
Cluster Import — portable cluster definitions become miLLM steering profiles.

### Brief Description
Import `mistudio.cluster-definition/v1` documents (and `mistudio.cluster-bundle/v1` multi-definition
files) from local JSON or public Hugging Face cluster packs, evaluate compatibility against the attached
model+SAE, and materialize each definition as a cluster-typed steering profile that activates with all
member features at their tuned strengths — zero manual tuning. A new dedicated **Clusters** page in the
Admin UI hosts the workflow.

### Problem Statement
Clusters tuned and validated in miStudio are trapped there. miLLM can steer raw feature indices, but a
user moving a 19-member tuned cluster must re-enter every strength by hand, loses the cluster's name,
narrative, budget metadata and provenance, and cannot participate in the emerging Hugging Face exchange
of community-tuned cluster packs.

### Feature Goals
1. One-action fidelity: import → activate → steer exactly as authored (BR-001..004).
2. Honest compatibility: bind / warn-bind / block / unbound, never silent mis-binding (BR-002).
3. Community consumption: browse + import public HF packs anonymously (BR-006).
4. Data-only safety: definitions can never execute, leak paths, or carry credentials (BR-007).
5. Lossless round-trip: re-export equals the imported definition (BR-003).

### User Value Proposition
"A cluster someone tuned in miStudio — or published to Hugging Face — runs in my serving stack in under
five minutes, with its meaning (narrative), its tuned strengths, and its provenance intact."

### Connection to Project Objectives
Implements the core of BRD-MILLM-CLUSTERS-001 ("make the portable cluster definition executable") and is
the foundation the other three increment features build on (009 tools, 010 dial, 011 sensing).

### BRD Traceability
| BR | Covered by |
|----|-----------|
| BR-001 (import + validation) | CLI-P1, CLI-P2 |
| BR-002 (honest compatibility) | CLI-P4, US-8.4 |
| BR-003 (lossless materialization + provenance) | CLI-M1, CLI-M2, CLI-M5, CLI-H4 |
| BR-004 (activate all members at tuned strengths) | CLI-M3, CLI-M6, US-8.2 |
| BR-005 (verify what the active cluster is doing) | CLI-U1, CLI-U3, US-8.2 criteria 2 |
| BR-006 (HF browse/import, anonymous) | CLI-H1..H5 |
| BR-007 (data-only safety) | CLI-P3, EC-8.1 |

---

## 2. User Stories & Scenarios

#### US-8.1: File import
**As a** researcher with a `.cluster.json` exported from miStudio
**I want to** import it into miLLM (paste or file upload)
**So that** it appears as a named cluster profile ready to activate.

**Acceptance Criteria:**
- [ ] Valid v1 definition imports and lists on the Clusters page with name, member count, display token
- [ ] Bundle files import per-item (one bad item never poisons the rest)
- [ ] Invalid kind / schema-major mismatch is rejected with an actionable message
- [ ] Name collision handled per `on_conflict` (default: rename with " (2)" suffix)

#### US-8.2: Activate imported cluster
**As a** user with an imported cluster
**I want to** activate it with one click
**So that** all members steer together at their tuned strengths.

**Acceptance Criteria:**
- [ ] Activation applies every member (sign·strength·λ, clamped to ±200) via the standard profile path
- [ ] The active cluster's name and member count are visible in the steering state
- [ ] Deactivation restores unsteered behavior
- [ ] Activating a cluster deactivates any active manual profile (single-active invariant)

#### US-8.3: Hugging Face browse & import
**As a** miLLM operator
**I want to** browse public cluster packs on HF filtered for my loaded model
**So that** I can import community-tuned behaviors without an HF account.

**Acceptance Criteria:**
- [ ] Search lists repos tagged `mistudio-cluster-definition` (optional base-model filter)
- [ ] Selecting a repo lists its definitions (manifest.jsonl preferred; else `*.cluster.json` files)
- [ ] Import fetches one definition anonymously and records hub provenance (repo@revision/path)
- [ ] Network failures surface gracefully (circuit breaker; cached listings)

#### US-8.4: Compatibility outcomes
**As a** user importing a definition authored against a different SAE
**I want** an honest verdict
**So that** I never steer with meaningless indices.

**Acceptance Criteria:**
- [ ] n_features mismatch vs attached SAE ⇒ warn at import, **block at activation**
- [ ] Model/layer mismatch ⇒ warn-bind (imports, activates, warnings visible)
- [ ] No SAE attached ⇒ imports as unbound; activation blocked until a compatible SAE is attached
- [ ] All warnings persist on the cluster row and display in the UI

#### US-8.5: Re-export
**As a** user
**I want to** export an imported cluster back to `mistudio.cluster-definition/v1`
**So that** the artifact stays mobile.

**Acceptance Criteria:**
- [ ] Export equals import byte-semantically (members, budget, narrative, provenance preserved)

#### Edge Cases

**EC-8.1: Oversized/hostile payload** — **Trigger:** file > 1 MB, bundle > 50 defs, > 20 members, or
non-JSON. **Behavior:** reject before parse-heavy work. **Message:** specific cap violated.

**EC-8.2: Strength exceeds steering range** — **Trigger:** |sign·strength·λ_max| > 200 for any member.
**Behavior:** import succeeds with warning; apply-time clamp to ±200. **Message:** lists affected members.

**EC-8.3: Member index out of SAE bounds** — **Trigger:** feature_idx ≥ attached SAE's d_sae.
**Behavior:** activation blocked (compat check), never a 500 from `set_steering_batch`.

**EC-8.4: Definition without n_features metadata** — **Trigger:** sae ref lacks n_features.
**Behavior:** import allowed; activation-time bounds check is the backstop; warning noted.

**EC-8.5: HF repo without manifest** — **Trigger:** tagged repo with only loose `*.cluster.json`.
**Behavior:** list files directly (cap 200).

---

## 3. Functional Requirements

### Import & Validation (FR-8.1, FR-8.2, FR-8.6)

| ID | Requirement | Priority |
|----|-------------|----------|
| CLI-P1 | System shall parse and strictly validate `mistudio.cluster-definition/v1` and `mistudio.cluster-bundle/v1` payloads (kind-keyed; schema vendored + sync-tested) | Must |
| CLI-P2 | System shall enforce caps: ≤1 MB payload, ≤50 definitions/bundle, ≤20 members/definition | Must |
| CLI-P3 | System shall reject definitions containing filesystem paths or credential-like content and never execute imported content | Must |
| CLI-P4 | System shall evaluate compatibility vs the attached model+SAE (n_features hard concern, model/layer warnings) and persist per-item outcomes | Must |
| CLI-P5 | Bundle import shall be per-item isolated with an aggregate result report | Must |

### Profile Materialization (FR-8.3, FR-8.4, FR-8.7)

| ID | Requirement | Priority |
|----|-------------|----------|
| CLI-M1 | Imported definitions shall persist as `profiles` rows with `source_kind='cluster'`, steering dict = {feature_idx: sign·strength} at λ=1 basis, and `cluster_meta` holding the full original definition | Must |
| CLI-M2 | `intensity` (λ) shall initialize from `budget.intensity` (default 1.0) and be adjustable (0..2) | Must |
| CLI-M3 | Activation shall scale stored strengths by λ and clamp effective values to [-200, 200], warning when clamping engages | Must |
| CLI-M4 | Unbound imports (no compatible SAE) shall persist with `sae_id NULL` and refuse activation with a clear message | Must |
| CLI-M5 | Export shall re-emit a lossless `mistudio.cluster-definition/v1` from `cluster_meta` | Must |
| CLI-M6 | Activating any profile (cluster or manual) shall respect the existing single-active invariant | Must |

### Hugging Face Consumption (FR-8.5)

| ID | Requirement | Priority |
|----|-------------|----------|
| CLI-H1 | System shall search HF models tagged `mistudio-cluster-definition` (optional base-model narrowing, text query, limit ≤50), anonymously | Must |
| CLI-H2 | System shall list a repo's definitions via `manifest.jsonl` (fallback: `*.cluster.json` files, cap 200) | Must |
| CLI-H3 | System shall fetch single definition files (≤1 MB, `.cluster.json` only) via `hf_hub_download` and import through the same validation path | Must |
| CLI-H4 | Hub imports shall record provenance `repo_id@revision/path` on the cluster row | Must |
| CLI-H5 | Hub calls shall use the existing circuit breaker and a short-TTL listing cache | Should |

### Clusters UI (FR-8.8)

| ID | Requirement | Priority |
|----|-------------|----------|
| CLI-U1 | New sidebar page "Clusters" listing cluster-typed profiles with display token, member chips, bound/unbound + imported badges, warnings | Must |
| CLI-U2 | Import dialog with paste / file / HF-browse tabs | Must |
| CLI-U3 | Narrative rendered as collapsible markdown; budget block (B, formula id, λ) displayed | Must |
| CLI-U4 | Activate/deactivate, intensity slider (0..2, marks at the definition's intensity_range), export download | Must |
| CLI-U5 | Existing Profiles page continues to show only `source_kind='manual'` rows | Must |

---

## 4. Data Requirements

Migration `007_add_cluster_columns_to_profiles.py` (extends `profiles` — PADR v1.1 decision):

```sql
ALTER TABLE profiles ADD COLUMN source_kind VARCHAR(20) NOT NULL DEFAULT 'manual';  -- 'manual'|'cluster'
ALTER TABLE profiles ADD COLUMN cluster_meta JSONB NULL;      -- full original definition (lossless)
ALTER TABLE profiles ADD COLUMN intensity FLOAT NOT NULL DEFAULT 1.0;               -- λ
ALTER TABLE profiles ADD COLUMN sensing_enabled BOOLEAN NOT NULL DEFAULT FALSE;     -- Feature 11
CREATE INDEX idx_profiles_source_kind ON profiles (source_kind);
```

`cluster_meta` shape: `{schema_version, kind, display_token, narrative, budget{B,B_dir,G,f_eff,
formula_id,constants,intensity,intensity_range}, sae{mistudio_sae_id,layer,hook_type,n_features,d_model,
source_hint}, model{hf_id,mistudio_model_id}, provenance{created_at,exported_at,mistudio_version,
source_note,hub_ref?}, members[...verbatim...], warnings[...]}`.

Vendored contract: `docs/schemas/cluster-definition-v1.json` (copied from miStudio; frozen v1) +
pydantic mirror + sync test.

---

## 5. API Specifications

New router `/api/clusters` (module `millm/api/routes/management/clusters.py`), `ApiResponse` envelope:

#### GET /api/clusters
List cluster-typed profiles (+ active id, per-row intensity/bound/warnings).

#### POST /api/clusters/import
Body: definition or bundle (discriminated on `kind`), `?on_conflict=rename|fail`, `?activate=false`.
Returns per-item results `{name, status: imported|imported_unbound|blocked|error, profile_id?, warnings[]}`.

#### GET /api/clusters/hub/search?q=&base_model=&limit=
#### GET /api/clusters/hub/{repo_id:path}/definitions
#### POST /api/clusters/hub/import — `{repo_id, filename, revision?, activate?}`

#### POST /api/clusters/{id}/activate · POST /api/clusters/{id}/deactivate
Delegate to ProfileService with λ scaling + clamp (CLI-M3) and bounds check (EC-8.3).

#### PUT /api/clusters/{id}/intensity — `{intensity: 0..2, reapply: true}`
#### PUT /api/clusters/active/intensity — same, addressed at the active cluster (used by 009/010)

#### GET /api/clusters/{id}/export — re-emit ClusterDefinitionV1

---

## 6. UI Requirements

- Route `/clusters` + Sidebar entry (icon: `Boxes`), page `ClustersPage.tsx`.
- `components/clusters/`: `ClusterCard`, `ClusterImportDialog` (paste/file/HF tabs), `HubBrowser`,
  `IntensitySlider`, `SensingToggle` (placeholder until Feature 11).
- API client `services/clusters.ts`; React Query hooks `hooks/useClusters.ts`.
- Rides along: fix `ImportExportButtons.tsx` profile-id type (numeric vs string `prof_*`).

---

## 7. Non-Functional Requirements
- Import validation completes < 500 ms for a max-size bundle (pure CPU).
- Hub search results cached 5 min; all HF calls behind `huggingface_circuit`.
- No new auth surface; endpoints follow the existing unauthenticated management-API posture (NFR-4.1).

## 8. Dependencies
- Feature 6 (Profile Management) — activation path, repository, single-active invariant.
- Feature 3 (Feature Steering) — `set_steering_batch` semantics.
- `huggingface_hub` (already a dependency via model/SAE downloaders).
- miStudio-published interchange schema v1 (frozen).

## 9. Success Criteria
1. E2E: export from miStudio → import → activate → identical members/strengths applied (round-trip test).
2. A public tagged HF pack imports and steers in < 5 min without a token.
3. Compatibility matrix verdicts match the miStudio reference behavior for the same inputs.
4. Re-export equality test passes.
5. All caps/hostile-payload tests pass.

## 10. Testing Requirements
- Unit: schema validation (valid/hostile/caps), mapping (sign fold, λ basis), compat matrix rows,
  clamp math, hub service (mocked HfApi), schema sync test.
- Integration: import→activate→steering-values assertion; bundle per-item isolation; unbound flow;
  export equality; activation bounds block.
- UI: ClustersPage list/import/activate (Vitest); Playwright flow post-deploy.

## 11. Rollout & Migration
Alembic 007 is additive (server_default backfills existing rows as 'manual'); zero behavior change for
existing profiles until a cluster is imported.

## 12. Out of Scope
Publishing to HF (miStudio-side); marketplace commerce; multi-SAE binding; recomputing budgets on import
(frozen as authored); modifying the v1 schema.

## 13. Open Questions
None blocking — all increment-level questions resolved (see §15).

## 14. Documentation Requirements
Manual page: Clusters (import, HF browse, activate, intensity); `docs/mcp-contract.md` cross-ref
(Feature 9); update OWUI tutorial cross-ref (Feature 10).

## 15. Decisions from Clarifying Questions
Recorded from the BRD round + the 2026-07-16 AskUserQuestion round:
1. **Scope:** all four increment capabilities in; HF **consume-only** (publishing stays miStudio-side).
2. **Storage:** extend `profiles` (not a new table) — preserves the single-active invariant and reuses
   the activation path (PADR v1.1).
3. **λ handling:** stored raw at λ=1 basis; scaled at activation; clamped to ±200 (contract-conflict fix).
4. **UI:** dedicated **Clusters page** (user choice), Profiles page unchanged for manual profiles.
5. **Budgets frozen on import** — never recomputed against the local SAE (matches miStudio load semantics).
