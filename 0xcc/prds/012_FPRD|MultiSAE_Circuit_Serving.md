# Feature PRD: Multi-SAE Attach & Circuit Serving

## miLLM Feature 12

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**Feature Priority:** Core (Increment: Circuit Runtime)
**References:** `BRD-MILLM-CIRCUITS-001.md` · `000_PPRD|miLLM.md` (v1.2, FR-12.x) · `000_PADR|miLLM.md` (v1.2)

---

## 1. Feature Overview

### Feature Name
Multi-SAE Attach & Circuit Serving — relax the single-SAE constraint so miLLM serves cross-layer circuits.

### Brief Description
Generalize miLLM's hard single-attached-SAE runtime to a `{(sae_id, layer) → (LoadedSAE, hook)}`
registry: attach several SAEs at once, load only the SAEs a circuit references (referenced-only), and
serve every circuit member through ITS OWN layer's SAE decoder — one forward hook per referenced
`(sae_id, layer)`, each bound to its own decoder. Per-layer strength budgets travel inside the definition
and are applied under a single global intensity (λ). A member whose layer has no attached SAE is rejected
at submit/activation (`SAE_SET_INCOMPLETE`, 422) — never steered through a wrong-layer basis. This is the
runtime substrate the rest of the increment (013 import, 014 dial, 015 sensing) builds on.

### Problem Statement
miLLM is hard single-SAE: `AttachedSAEState` is a singleton holding exactly one
`(_attached_sae, _attached_sae_id, _attached_layer, _hook_handle)`. A circuit whose members live on
layers L10 and L13 cannot be served — the L13 members would be steered through L10's feature basis (a
meaningless wrong-decoder delta) or dropped entirely. There is no plural attachment status, no per-layer
budget application, and no guard preventing a silent wrong-basis serve. The ecosystem discovers and
validates multi-layer circuits miLLM cannot run.

### Feature Goals
1. Multi-SAE attach keyed by `(sae_id, layer)`; only referenced SAEs loaded (BR-001).
2. Every member steered through its own layer's SAE — one hook per `(sae_id, layer)` (BR-004).
3. Per-layer budgets under one global λ, reusing the validated `freq-budget/sim-alloc/per-layer@1` allocation (BR-004).
4. Honest incompleteness: a member with no attached SAE blocks at submit/activation, never a silent wrong serve (BR-003/BR-006).
5. Documented, measured VRAM envelope — fp16 attach, two-SAE close-out (BR-001).
6. Cross-layer over-steering hazards surfaced at activation, quantified where validated (BR-011).

### User Value Proposition
"A cross-layer circuit validated in miStudio runs live in miLLM — every member through its own layer's
SAE, at its tuned per-layer budget, under one dial — instead of collapsing to a single-SAE approximation."

### Connection to Project Objectives
Implements the runtime substrate of BRD-MILLM-CIRCUITS-001 ("make the portable circuit definition
executable"): the multi-SAE attach + per-layer serving that Features 013 (import/evidence), 014 (OWUI
dial) and 015 (edge sensing) all depend on.

### BRD Traceability
| BR | Covered by |
|----|-----------|
| BR-001 (multi-SAE attach, referenced-only, VRAM envelope) | MSA-A1, MSA-A2, MSA-V1, US-12.1 |
| BR-004 (every member through its own layer's SAE, per-layer budgets under one λ) | MSA-S1, MSA-S2, MSA-S3, US-12.2 |
| BR-011 (cross-layer over-steering hazards surfaced, not corrected) | MSA-H1, MSA-H2, US-12.3 |
| BR-013 (multi-SAE serving does not materially degrade the inference path) | MSA-N1, NFR-12.1 |

---

## 2. User Stories & Scenarios

#### US-12.1: Attach a circuit's SAE set
**As a** miLLM operator importing a two-layer circuit
**I want to** attach both referenced SAEs at once (only those two)
**So that** the circuit can be served without manually attaching/detaching per layer.

**Acceptance Criteria:**
- [ ] Attaching a set keyed by `(sae_id, layer)` leaves prior attachments intact (no single-active clobber)
- [ ] Only the SAEs the circuit references are loaded; unreferenced SAEs are never pulled to GPU
- [ ] Attachment status reports the FULL set (list of `{sae_id, layer, memory_mb, steering_enabled}`)
- [ ] Re-attaching the same `(sae_id, layer)` is idempotent (existing hook removed before re-install)

#### US-12.2: Serve a cross-layer circuit
**As a** user activating a serveable circuit
**I want** every member applied through its own layer's SAE at the authored per-layer strengths
**So that** the served behavior matches the validated circuit.

**Acceptance Criteria:**
- [ ] Members are grouped by `(sae_id, layer)`; each layer's SAE gets only its own members' steering dict
- [ ] Each layer's budget is applied under one global λ (all layers scale together)
- [ ] A feature on layer L is provably steered by the SAE whose `.layer == L` (never another basis)
- [ ] Deactivation clears steering on every layer's SAE and restores unsteered behavior

#### US-12.3: Cross-layer hazard visibility
**As a** user activating a multi-layer circuit
**I want** compounding/cancellation hazards surfaced at activation
**So that** I understand over-steering risk without the runtime silently altering my config.

**Acceptance Criteria:**
- [ ] Activation returns hazard warnings (compounding, cancellation) computed across layers
- [ ] Each hazard is labeled `validated:ES=…` when a validated effect size is present, else `heuristic:weight_prior=…`
- [ ] Warnings SURFACE; the steering config is never mutated by hazard detection

#### US-12.4: Incomplete SAE set
**As a** user activating a circuit whose L13 SAE is not attached
**I want** an explicit, actionable rejection listing the offenders
**So that** I never unknowingly steer L13 members through L10's basis.

**Acceptance Criteria:**
- [ ] Submit/activation returns 422 `SAE_SET_INCOMPLETE` listing `{feature_idx, layer, sae_id}` offenders
- [ ] No member from an unbound layer is ever applied (all-or-block for full serving)
- [ ] The check runs at BOTH activation and per-request submit (never a runtime wrong-basis path)

#### Edge Cases

**EC-12.1: Member index out of the layer's SAE bounds** — **Trigger:** `feature_idx ≥ that layer's SAE d_sae`.
**Behavior:** activation blocked (bounds gate before delegation), never a 500 from `set_steering_batch`.
**Message:** lists offending `{feature_idx, layer}`.

**EC-12.2: Two members reference the same layer via different SAE ids** — **Trigger:** ambiguous
`(sae_id, layer)` for one layer. **Behavior:** reject at import/activation; a layer resolves to exactly
one attached SAE. **Message:** names the conflicting sae_ids.

**EC-12.3: VRAM would exceed the envelope** — **Trigger:** referenced set estimate (fp16) pushes past the
documented envelope. **Behavior:** attach proceeds but records a VRAM warning on the attachment status;
the per-layer slice fallback (Feature 013) is the operator's escape hatch. **Message:** measured vs envelope.

**EC-12.4: λ pushes an effective strength past ±200** — **Trigger:** `|budget·λ_max| > 200` for a member.
**Behavior:** warn at activation; clamp at apply time (shared `clamp_steering`, reused from Feature 8).

**EC-12.5: All members on one layer (degenerate circuit)** — **Trigger:** single referenced SAE.
**Behavior:** serves correctly through the single-layer path (multi-SAE registry with one entry).

---

## 3. Functional Requirements

### Multi-SAE Attachment (FR-12.1, FR-12.5, FR-12.7)

| ID | Requirement | Priority |
|----|-------------|----------|
| MSA-A1 | The attachment state SHALL be a registry keyed by `(sae_id, layer)` mapping to `(LoadedSAE, hook_handle)`; attaching one entry SHALL NOT clear others | Must |
| MSA-A2 | Only the SAEs referenced by an imported circuit SHALL be loaded (referenced-only); unreferenced SAEs never touch GPU | Must |
| MSA-A3 | Re-attaching an existing `(sae_id, layer)` SHALL remove the prior hook before re-installing (no orphaned hooks) | Must |
| MSA-V1 | Steering weight tensors SHALL be attached in fp16 (SAELoader `target_dtype`); attachment status SHALL report per-SAE and total memory_mb | Must |
| MSA-A4 | Attachment status SHALL be plural: `AttachmentStatus` becomes a set/list of per-SAE entries, surfaced to API/MCP/Admin UI | Must |

### Circuit Serving (FR-12.2, FR-12.3, FR-12.4)

| ID | Requirement | Priority |
|----|-------------|----------|
| MSA-S1 | A member on layer L SHALL be steered ONLY by the SAE whose `.layer == L`; members SHALL be grouped by `(sae_id, layer)` and each SAE receives only its own members' steering dict (mirrors miStudio `_register_steering_hooks`) | Must |
| MSA-S2 | Each layer's per-layer budget SHALL be applied under a single global λ∈[0,2] using the validated `freq-budget/sim-alloc/per-layer@1` allocation (γ=0 ⇒ B=B_dir); joint cross-layer calibration is deferred | Must |
| MSA-S3 | Effective per-member strength SHALL be `clamp_steering(budget·sign·λ)` (shared ±200 clamp, reused from Feature 8); a bounds pre-check SHALL run before delegation to `set_steering_batch` | Must |
| MSA-S4 | A member whose layer has no attached SAE SHALL be rejected at submit AND activation with 422 `SAE_SET_INCOMPLETE`, listing `{feature_idx, layer, sae_id}` offenders — never a silent wrong-basis serve | Must |
| MSA-S5 | Deactivation SHALL clear steering on every attached SAE participating in the circuit and restore unsteered behavior | Must |

### Cross-Layer Hazards (FR-12.6)

| ID | Requirement | Priority |
|----|-------------|----------|
| MSA-H1 | Activation SHALL compute cross-layer over-steering hazards (compounding, cancellation) across the circuit's layers and return them as warnings | Must |
| MSA-H2 | Each hazard SHALL be labeled `validated:ES=…` when a validated effect size is present in the definition, else `heuristic:weight_prior=…`; hazard detection SHALL NEVER mutate the steering config | Must |

---

## 4. Data Requirements

**No new database table.** Attachment state is in-memory (the `AttachedSAEState` singleton, generalized to
a registry) exactly as today — it is process/GPU state, not persisted. The circuit definition itself is
stored by Feature 013 (as a `profiles`/circuit row); Feature 012 owns only the runtime attach/serve path.

The registry entry shape (in-memory):
`{(sae_id: str, layer: int): AttachedEntry{ sae: LoadedSAE, hook_handle, memory_mb, steering_enabled }}`.

Circuit serving reads the per-layer budgets from the imported circuit-definition (Feature 013). Members
carry `{feature_idx, layer, sae_id, budget (B_dir), sign}`; the per-layer allocation and λ are applied at
serve time, not persisted separately.

---

## 5. API Specifications

Extends the existing SAE/attachment surface; all responses in the `ApiResponse` envelope. New/changed
routes (circuit-scoped import/activate live in Feature 013; Feature 012 owns the attach + serve mechanics):

#### GET /api/sae/attachments
Plural attachment status: list of `{sae_id, layer, memory_mb, steering_enabled, monitoring_enabled}` +
total_memory_mb + VRAM warning (if any). (Supersedes the singular status shape additively.)

#### POST /api/sae/attach-set — `{ sae_layers: [{sae_id, layer}], dtype: "fp16" }`
Attach a referenced set (referenced-only loading); idempotent per `(sae_id, layer)`; returns the plural
status. Records a VRAM warning when the fp16 estimate exceeds the envelope (EC-12.3).

#### POST /api/sae/detach — `{ sae_id, layer }` (or full clear)
Remove one `(sae_id, layer)` hook or the whole set.

#### Serving hooks (internal, consumed by Feature 013 activation):
`SAEService.set_circuit_steering(members, intensity)` — group by `(sae_id, layer)`, bounds-gate,
`SAE_SET_INCOMPLETE` on any unbound layer, `clamp_steering(budget·sign·λ)` per member, hazards computed
and returned (never applied).

---

## 6. UI Requirements

- **Circuits** tab (shared with Feature 013): attachment status shows the plural SAE set — one chip per
  `(sae_id, layer)` with memory_mb, a total, and any VRAM warning badge.
- Activation surface (Feature 013 UI) shows the cross-layer hazard warnings with their `validated:`/
  `heuristic:` labels; `SAE_SET_INCOMPLETE` renders the offender list.
- No standalone page in this feature — the substrate surfaces through the Circuits tab that Feature 013 adds.

---

## 7. Non-Functional Requirements
- **NFR-12.1 (BR-013):** Attaching additional SAEs SHALL NOT materially degrade the OpenAI-compatible
  inference path — per-layer hooks add O(n_layers) constant-cost residual adds; no second forward pass.
- fp16 attach keeps per-SAE steering-tensor VRAM ~64 MB (measured, Gemma-2-2B d_in=2048/d_sae=8192).
- No new auth surface (unchanged v1.0 management-API posture).

## 8. Dependencies
- Feature 2 (SAE Management) — download/load/attach lifecycle, `SAELoader.load(..., dtype)`.
- Feature 3 (Feature Steering) — `LoadedSAE.apply_steering` / `set_steering_batch` semantics (unchanged).
- Feature 8 (Cluster Import) — shared `clamp_steering` helper; per-layer slice fallback is Feature 013.
- Feature 13 (Circuit Import) — provides the circuit definition + per-layer budgets that serving consumes.

## 9. Success Criteria
1. A validated two-layer circuit serves with every member applied through its own layer's SAE at the
   authored per-layer strengths (E2E round-trip assert against per-layer `get_steering_values()`).
2. Attaching two Gemma-2-2B SAEs' steering tensors in fp16 measures ≤128 MB (within the <200 MB envelope).
3. A member on an unbound layer yields 422 `SAE_SET_INCOMPLETE` at submit AND activation — never a wrong serve.
4. Cross-layer hazards surface at activation, correctly `validated:`/`heuristic:`-labeled, config unmutated.
5. Multi-SAE serving adds no user-perceivable latency vs single-SAE baseline (NFR-12.1).

## 10. Testing Requirements
- Unit: registry set/clear/idempotent-reattach; group-by-`(sae_id,layer)` mapping; bounds gate;
  `SAE_SET_INCOMPLETE` offender list; clamp math; hazard labeling (`validated:`/`heuristic:`); memory estimate.
- Integration: attach-set → activate circuit → per-layer `get_steering_values()` equals λ-clamped
  expectation on each layer; incomplete-set block; detach clears per-layer; single-layer degenerate case.
- Perf: latency delta single-SAE vs two-SAE serve (env-gated, GPU host).

## 11. Rollout & Migration
No migration — attachment state is in-memory. The plural attachment-status shape is additive; the singular
status field is preserved (first entry) for existing clients until Feature 013 UI lands. Zero behavior
change until a circuit's SAE set is attached.

## 12. Out of Scope
Circuit import/validation/evidence ladder (Feature 013); per-layer slice fallback (Feature 013); the OWUI
dial (Feature 014); edge sensing (Feature 015); joint cross-layer budget calibration (deferred, BRD
§future_considerations); SAE eviction policy when a new circuit exceeds VRAM (deferred — slice fallback is
the escape hatch); recomputing budgets on import (frozen as authored).

## 13. Open Questions
None blocking. The multi-SAE VRAM envelope is measured (§9, task 1.0); eviction policy is explicitly
deferred to a follow-on (the slice fallback covers the over-envelope case for now).

## 14. Documentation Requirements
Manual: multi-SAE attachment + circuit serving mechanics; `docs/mcp-contract.md` cross-ref (circuit
category, Feature 013/015); attachment-status shape change note.

## 15. Decisions from Clarifying Questions
Recorded from the BRD round + the 2026-07-20 lock:
1. **Full multi-SAE now** — attach several SAEs at once; a feature on layer L is ALWAYS steered by the
   SAE whose `.layer == L` (group hooks by `(sae_id, layer)`, mirroring miStudio `_register_steering_hooks`).
2. **fp16 attach** — steering weights cast to fp16 (SAELoader already casts to `target_dtype`); VRAM is
   dtype-conditional, not a blocker for 2–3 layer circuits (measured: 128 MB fp16 / 256 MB fp32, two SAEs).
3. **Per-layer budgets under ONE global λ** — reuse the validated `freq-budget/sim-alloc/per-layer@1`
   allocation; joint cross-layer calibration explicitly deferred (γ=0 ⇒ B=B_dir).
4. **Incompleteness is a hard block** — `SAE_SET_INCOMPLETE` (422) at submit/activation, never a silent
   wrong-basis serve.
5. **Hazards surface, never correct** — quantified from validated ES where present, else `heuristic`;
   detection, not auto-correction (mirrors miStudio hazards-v2).
