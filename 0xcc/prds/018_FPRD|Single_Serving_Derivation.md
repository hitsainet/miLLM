# Feature PRD: Single Circuit-Serving Derivation

## miLLM Feature 18

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**Feature Priority:** Secondary (Increment: Circuit Consolidation)
**References:** `BRD-MILLM-CIRCUITS-002.md` (BR-002) · `000_PPRD|miLLM.md` (v1.3, FR-18.x) · `000_PADR|miLLM.md` (v1.3) · `docs/mcp-contract.md` (v1.2)

---

## 1. Feature Overview

### Feature Name

Single Circuit-Serving Derivation — one `CircuitSteeringEngine` that answers "what does serving this circuit mean", consumed by every caller that serves one.

### Brief Description

Serving a circuit today is derived independently in four places: `_serve_full` at activation (`circuit_service.py:424`), `set_intensity` at the management dial (`:799`), the per-request OWUI dial (`inference_service.py:955`), and the echo-side predicate that decides whether a rung header may be emitted (`:806-822`). Each independently flattens the definition into serving members, resolves which layers participate, and decides whether the circuit is serving at all. They agree today only because three separate comments keep being obeyed and because `CircuitService._serving_members` was made a `@staticmethod` so a fourth caller could reach it unbound. This feature extracts the derivation into a `CircuitSteeringEngine` constructed honestly from the attachment registry alone — the one thing the derivation actually needs — so the four call sites consume a result rather than each computing one. Because the engine needs only the registry, the `SAEService.__new__(SAEService)` bypass at `inference_service.py:743` (which hand-sets `_sae_state` and leaves every other field absent, so any future field access becomes a swallowed `AttributeError` and a silently unsteered response still carrying a rung header) becomes unnecessary and is retired with it. The engine also computes a circuit's CLAIM SET — the layers its serving members reach — which Feature 19's contention model consumes, so activation and contention agree by construction rather than by two implementations that happen to match.

### Problem Statement

Feature 14's two worst defects were both instances of one class: a derivation drifting from its twin. **F14-R1-01** (CRITICAL) — the dial de-scaled live steering values by `circuit.intensity`, the DB column, while `_serve_full` applies `definition.budget.intensity`, the document field; a circuit whose document declares a non-1.0 budget intensity was scaled by the wrong factor on every dialled request, silently, with no error, and no division could invert the ±200 clamp that had already discarded the overflow. **F14-R2-01** (CRITICAL) — R1's own fix hardened the per-layer restore loop and left its *input* incomplete: the snapshot filtered on `circuit.layers` (the DB column) while the apply drove off the definition's member layers, so any layer in one and not the other was dialled and never restored — a per-request override leaking permanently into global state, which is the precise class of bug R1's fix existed to prevent, one level up. F14-R2-02 then found the same shape a third time (three surfaces each re-deriving "is this circuit steering?"), and its fix was explicitly recorded as *"Fixing the class, not the instance"* — but it fixed only the *is-it-serving* predicate, leaving the *what-does-serving-mean* derivation still quadrupled. Every one of these is a wrong answer produced by correct code, because the code was correct about a different source of truth than its twin. No test can durably close this: the tests pin the agreement, and the next edit to either side breaks it again.

### Feature Goals

1. Reduce circuit-serving derivations from four to exactly one, verified by a test that fails if a second appears (BR-002, FR-18.1).
2. Retire the `SAEService.__new__` constructor bypass and the failure mode it creates — a swallowed `AttributeError` yielding a silently unsteered response that still carries a rung header (BR-002, FR-18.2).
3. Compute the claim set from the same derivation that serves, so Feature 19's contention refusals describe exactly the layers activation will touch (FR-18.3).
4. Preserve `_serving_members`' flattening rules and the canonical sign rule byte-for-byte, proven by characterization tests written BEFORE any code moves (BR-002, locked decision 3).
5. Ship behaviour-preserving: the existing backend and frontend suites stay green throughout, and are the floor rather than the target (BRD constraints).

### User Value Proposition

"Changing how a circuit is served means changing one thing, not finding four copies that must agree — and I find out at test time if someone adds a fifth."

### Connection to Project Objectives

Directly serves the increment's first business objective: *"Eliminate the class of defect that produced 11 criticals across 001 by making the invariants unrepresentable rather than test-guarded."*

### BRD Traceability

| BR | Requirement (abbrev.) | Coverage IDs |
|----|-----------------------|--------------|
| BR-002 | Exactly ONE serving derivation consumed by activation, intensity and the per-request dial; no caller bypasses a constructor to reach it | ENG-D1, ENG-D2, ENG-D3, ENG-C1, ENG-C2, ENG-C3, ENG-K1 |
| BR-005 | No capability accepted as shipped without a test that FAILS when its wiring is cut | ENG-V1, ENG-V2, ENG-V3 |
| BR-011 | Layer-exclusive claims for concurrent serving (F19 consumes this feature's claim set) | ENG-K1, ENG-K2, ENG-K3 |

## 2. User Stories & Scenarios

#### US-18.1: One place to change serving
**As a** maintainer
**I want to** change how a circuit's members are flattened, scaled or applied in exactly one place
**So that** a change cannot land on one caller and miss three.

**Acceptance Criteria:**
- [ ] Activation, `set_intensity`, the per-request dial and the echo predicate all obtain serving members and claim layers from `CircuitSteeringEngine`, never by flattening a definition themselves
- [ ] A test enumerates the derivation call sites and FAILS if a second flattening implementation appears
- [ ] No behaviour change: the full existing suite is green before, during and after the extraction

#### US-18.2: An honestly constructed engine
**As a** maintainer
**I want** the object that applies circuit steering to be constructible by calling its constructor
**So that** a missing field is a startup error, not a swallowed `AttributeError` and a lie in a response header.

**Acceptance Criteria:**
- [ ] `CircuitSteeringEngine` takes only the attachment registry; no repository, cache dir, emitter or downloader
- [ ] `SAEService.__new__(SAEService)` no longer appears anywhere in the codebase; a test asserts its absence
- [ ] The per-request dial path constructs the engine normally and needs no request-scoped DI

#### US-18.3: Activation and contention agree by construction
**As an** operator activating a second circuit
**I want** the contention check to reason about exactly the layers activation will steer
**So that** a refusal is never wrong in either direction — no false refusal, no unclaimed layer silently overwritten.

**Acceptance Criteria:**
- [ ] `claim_set(definition)` returns the layers reached by the SAME member list `serving_members(definition)` returns
- [ ] A test asserts claim set == the distinct layers of the serving members, on definitions exercising every flattening rule
- [ ] The claim set is derived from the definition's members, never from the `circuits.layers` DB column (the F14-R2-01 source)

#### Edge Cases

**EC-18.1: A member contributing from both sources** — **Trigger:** a `cluster_ref` member carries frozen `expanded_members` AND its own `feature`. **Behavior:** BOTH contribute, exactly as `circuit_service.py:644-646` does today. Taking only one silently drops authored members from the intervention; this rule is preserved verbatim and pinned by characterization test before the move.

**EC-18.2: A duplicate `(layer, feature_idx)` after flattening** — **Trigger:** the same key arrives from `expanded_members` and from `feature`, or twice within `expanded_members`. **Behavior:** collapsed at flatten time, first occurrence wins (`:648-651`). The serving path rejects a repeated key outright with `duplicate_member` (`sae_service.py:562-572`), so the dedupe is load-bearing, not cosmetic — losing it turns a valid circuit into a 422.

**EC-18.3: A negative authored strength** — **Trigger:** a member authored at a negative budget. **Behavior:** the canonical sign rule holds — a NEGATIVE budget is already directional and MUST NOT be multiplied by `sign`, which would double-negate a suppression into an amplification (`sae_service.py:66-76`). The engine does not reimplement `_directional_budget`; it delegates to the single existing function, which is shared with `cluster_service` so circuits and clusters steer identically for the same authored member.

**EC-18.4: Definition intensity vs DB column** — **Trigger:** a circuit whose document declares `budget.intensity` different from the `circuit.intensity` DB column. **Behavior:** the engine exposes ONE resolution (`definition.budget.intensity if definition.budget else circuit.intensity`, per `:421-423`) and every caller uses it. This is the F14-R1-01 divisor bug expressed as a single function, so the two fields can no longer be confused by a caller.

**EC-18.5: A layer in the DB column but not in the members (or vice versa)** — **Behavior:** the claim set is the members' layers, full stop. The `circuits.layers` column is display/query metadata and is never a serving input. This is F14-R2-01's root cause removed rather than test-guarded.

**EC-18.6: No serving members after flattening** — **Behavior:** the engine returns an empty member list and an empty claim set; callers treat that as "not serving" uniformly. Activation's empty-member path already clears and disables every attached layer (`sae_service.py:543-548`) rather than leaving a previous circuit armed; the dial no-ops and the echo predicate suppresses the rung header. All three must reach the same verdict from the same call.

**EC-18.7: No attached SAE on any claimed layer** — **Behavior:** unchanged from today — the dial no-ops and the rung header is suppressed, because an echoed header with nothing steering is an evidence-grade overclaim (F14-R2-02). The engine reports claimed-but-unattached layers distinctly from claimed-and-attached, so the caller decides; the engine never decides silently.

**EC-18.9: A circuit serving in slice-fallback** — **Trigger:** not every referenced SAE is bound, so `_serve_slices` (`circuit_service.py:435`) renders per-layer cluster slices through the UNCHANGED Feature 8 import path rather than through `set_circuit_steering`. **Behavior:** the engine covers FULL serves only; a slice serve remains the cluster path's, keeps its own λ on the slice's cluster profile, and `set_intensity` continues to report that the dial was recorded but not applied (`:805-813`). The engine must not absorb the slice path — a slice is an ordinary `cluster-definition/v1` by design, and pulling it into circuit serving would erase the distinction the fallback exists to preserve.

**EC-18.8: A concurrent detach between derive and apply** — **Behavior:** unchanged — `set_circuit_steering` holds `_ATTACHMENT_LOCK` across resolve→apply (`sae_service.py:509-513`). The engine does not widen, narrow or re-implement that lock; a derivation computed outside it is advisory until applied under it.

## 3. Functional Requirements

### The Single Derivation (FR-18.1)

| ID | Requirement | Priority |
|----|-------------|----------|
| ENG-D1 | A `CircuitSteeringEngine` shall be the sole implementation of: flatten definition→serving members, resolve serving intensity, compute claim set, apply steering | Must |
| ENG-D2 | Flattening shall preserve today's rules EXACTLY: both-sources collection from `expanded_members` AND the member's own `feature`; dedupe on `(layer, feature_idx)`, first wins (EC-18.1, EC-18.2) | Must |
| ENG-D3 | Serving intensity resolution shall be a single function returning `definition.budget.intensity if definition.budget else circuit.intensity` (EC-18.4) | Must |
| ENG-D4 | `CircuitService._serving_members` and `InferenceService._circuit_serving_members` shall be removed, not merely delegated — a delegating shim is a second call site that can grow a body | Must |
| ENG-D5 | The canonical sign rule shall be delegated to the existing `_directional_budget`, never reimplemented (EC-18.3) | Must |

### Honest Construction (FR-18.2)

| ID | Requirement | Priority |
|----|-------------|----------|
| ENG-C1 | The engine's constructor shall take only the attachment registry (`AttachedSAEState`), defaulting to the process singleton; no repository, cache dir, emitter, downloader or loader | Must |
| ENG-C2 | `SAEService.__new__(SAEService)` shall be removed; `_sae_service_for_dial` shall be deleted along with it | Must |
| ENG-C3 | No construction path shall leave a field unset that any reachable method reads; the engine shall have no `Optional` field whose `None` case is unhandled | Must |
| ENG-C4 | `SAEService.set_circuit_steering` shall remain the applying implementation, called by the engine — this feature moves the DERIVATION, not the apply | Must |

### Claim Sets (FR-18.3)

| ID | Requirement | Priority |
|----|-------------|----------|
| ENG-K1 | `claim_set(definition)` shall return the distinct layers of `serving_members(definition)`, computed from the same member list, never from `circuits.layers` (EC-18.5) | Must |
| ENG-K2 | The claim set shall distinguish claimed-and-attached from claimed-but-unattached layers, leaving the policy decision to the caller (EC-18.7) | Must |
| ENG-K3 | The claim set shall be exposed as a stable public method suitable for Feature 19's contention check, documented as that feature's input | Must |

### Behaviour Preservation & Reachability (BR-002, BR-005)

| ID | Requirement | Priority |
|----|-------------|----------|
| ENG-V1 | Characterization tests shall pin current flattening, intensity-resolution and claim-layer behaviour BEFORE any code moves | Must |
| ENG-V2 | A reachability test shall FAIL when the engine's wiring into each of the four call sites is cut — asserting invocation, not merely that a method exists | Must |
| ENG-V3 | A single-derivation guard test shall FAIL if a second flattening implementation is introduced | Must |
| ENG-V4 | Mutation testing shall be applied to the engine module; survivors shall be pinned or explicitly recorded | Must |

## 4. Data Requirements

None. No schema change, no migration. The engine is pure derivation over an already-parsed `CircuitDefinitionV1` plus the in-memory attachment registry. The `circuits.layers` column is unchanged on disk and remains valid for display and query — this feature only stops it being read as a *serving* input (EC-18.5). Feature 19 introduces the `circuit_layer_claims` table that persists what this feature computes; F18 owns the computation, F19 owns its materialisation.

## 5. API Specifications

No new endpoints and no changes to request or response shapes. `PUT /api/circuits/active/intensity`, `POST /api/circuits/{id}/activate`, `POST /api/circuits/{id}/deactivate` and the per-request `steering_intensity` dial all keep their exact current contracts, including `reapplied`, `warnings`, `hazards`, `applied_per_layer`, and the `X-miLLM-Circuit-Rung` header semantics. `docs/mcp-contract.md` needs no surface change for this feature; the v1.2 bump belongs to Feature 20. **This is a pure refactor at the API boundary — any response-shape delta is a defect, not a feature, and the integration suite is the gate.**

## 6. UI Requirements

None (internal). No admin-ui component changes. The frontend suite is a regression gate only: it must stay green, proving no response shape the UI reads has shifted.

## 7. Non-Functional Requirements

Zero measurable latency delta on the chat hot path. The per-request dial currently constructs a half-initialised `SAEService` per dialled request; constructing a `CircuitSteeringEngine` bound to the singleton registry is strictly cheaper (no downloader, loader or hooker instantiation) and must not be worse. The engine holds no request state and no lock of its own; `_ATTACHMENT_LOCK` ownership stays inside `set_circuit_steering` exactly as today (EC-18.8). Thread-safety posture is unchanged: derivation is pure and re-entrant, apply is serialised by the existing lock.

## 8. Dependencies

- **Feature 17** (request-scoped sensing context) — F18 lands on the settled context; sequenced strictly after, per BRD execution_order steps 3→4.
- **Feature 12** (multi-SAE serving) — provides `SAEService.set_circuit_steering`, `AttachedSAEState`, the `(sae_id, layer)` registry and the ±200 clamp gate. The engine calls it; it is not rewritten.
- **Feature 13** (circuit import/activation) — owns `CircuitService`, `CircuitDefinitionV1`, `circuit_meta` storage and the evidence gate.
- **Feature 14** (circuit dial) — owns the per-request dial path and the rung/λ echo predicates; its two criticals are this feature's evidence base.
- **Consumed by Feature 19** (concurrent serving) — takes `claim_set` as the contention model's input. F19's design of record (`0xcc/docs/circuit-contention-model.md`) is already settled, so F18 writes to a known consumer.
- **Consumed by Feature 20** (MCP circuit surface) — written against settled code, per the BRD's locked sequencing.

## 9. Success Criteria

1. Circuit-serving derivation count is exactly 1, down from 4 (baseline verified at `circuit_service.py:424`, `:799`, `inference_service.py:955`, `:806-822`); a guard test fails if it rises.
2. `SAEService.__new__` appears nowhere in the codebase; a test asserts its absence and the dial constructs its engine normally.
3. `claim_set(definition)` equals the distinct layers of `serving_members(definition)` on every characterization fixture, including definitions exercising EC-18.1 and EC-18.2.
4. Flattening behaviour is byte-identical to the pre-refactor implementation: characterization tests written before the move pass unchanged after it.
5. The canonical sign rule holds — a negative authored strength is served without `sign` multiplication, asserted directly rather than inferred from an applied value.
6. Full existing suite green at every commit of the extraction (backend and frontend), with a reachability test per call site that fails when its wiring is cut.
7. Mutation testing run against the engine module; every survivor either pinned by a new test or recorded with a rationale.

## 10. Testing Requirements

- **Unit:** characterization of flattening (both-sources EC-18.1, dedupe EC-18.2, empty EC-18.6, ordering), intensity resolution (EC-18.4 document-vs-column), claim set == member layers (EC-18.5), sign rule preservation (EC-18.3), engine constructibility with no arguments beyond the registry, claimed-but-unattached reporting (EC-18.7).
- **Integration:** activation → `set_intensity` → per-request dial → restore, asserting all four reach identical serving members and identical applied values for one definition; the F14-R1-01 regression (authored 150 at λ=2, dial to 1.0, expect 150 not 100); the F14-R2-01 regression (a layer in the definition but absent from the DB column is claimed, dialled AND restored); rung-header suppression parity when nothing is steering.
- **Reachability (BR-005):** four tests, one per call site, each FAILING when the engine call is removed — invocation asserted, not existence.
- **Mutation:** the engine module, with survivors pinned or recorded.
- **E2E (post-deploy):** activate a two-layer circuit, dial it via OWUI, confirm applied values and restore against the management API's report of the same operation.

## 11. Rollout & Migration

Behaviour-preserving refactor; no migration, no flag, no data change, no deploy step. Rollout is a normal deploy. The rollback is a revert — safe at any commit because every commit keeps the suite green (ENG-V1 discipline). Because no schema or contract changes, no downgrade path is needed, in deliberate contrast to Feature 19, whose first concurrent activation is a one-way door.

## 12. Out of Scope

Any change to steering math, the ±200 clamp gate, the hazard model, the evidence rung or its language; `set_circuit_steering`'s internals (the engine calls it unchanged); the cluster steering path beyond continuing to share `_directional_budget`; the `circuit_layer_claims` table and the contention refusal itself (Feature 19); the MCP surface (Feature 20); the `circuits.layers` column's continued use for display and query.

## 13. Open Questions

None blocking. Three upstream documentation defects were found during authoring and are recorded in §15 for correction by their owners rather than propagated here.

## 14. Documentation Requirements

A short architecture note in the manual's circuits section stating that circuit serving has one derivation and naming it, so the next contributor looking for "where does serving happen" finds one answer. `docs/mcp-contract.md` needs no change. The PADR needs a new decision entry — see §15 item 3.

## 15. Decisions from Clarifying Questions

1. **The engine takes the attachment registry, not an `SAEService`** (2026-07-20). Taking a service would preserve the constructor problem in a new location — the dial would still need one, and the cheapest way to get one is still the `__new__` bypass. Taking the registry is what makes retiring the bypass a consequence of the design rather than a separate discipline. `set_circuit_steering` stays on `SAEService`; the engine calls it via a normally-constructed service where one exists, and against the registry where one does not.

2. **`_serving_members` is removed, not kept as a delegating shim** (2026-07-20). A shim is a second call site with a body that can grow one, and the existing `@staticmethod`-by-contract comment is direct evidence that the unbound-call arrangement needed a written promise to stay safe. Deleting both `CircuitService._serving_members` and `InferenceService._circuit_serving_members` makes the guarantee structural. Their docstrings' RULES survive verbatim in the engine — the flattening contract is preserved, the call sites are not.

3. **Feature 18 has no PADR decision of record and one is added** (2026-07-20). It is the only Circuit Consolidation feature without one — verified by reading PADR v1.3 §10, whose Circuit Consolidation group covers F16, F17 and F19 but not F18. A `#### Single serving derivation vs four coordinated call sites` entry is authored as task 6.3, covering the engine's shape, the `__new__` retirement, and the fact that the canonical sign rule — which lives in miLLM only as `_directional_budget`'s docstring at `sae_service.py:66-76`, not in any architecture record — becomes normative text.

4. **Mutation testing is anchored to BRD locked decision (3) and RSK-001, not to BR-005** (2026-07-20). BR-005 is the *reachability* requirement; the mutation-testing FR is FR-17.6 and belongs to Feature 17. This feature carries a mutation task because the increment's verification tier applies to it, and because the engine is a behaviour-preserving move where a surviving mutant is the sharpest available evidence that a preserved rule is untested. It is recorded as satisfying the locked decision, and BR-005 is cited only for the reachability tests (ENG-V2).

5. **The claim set ships in F18 even though its consumer ships in F19** (2026-07-20). FR-18.3 requires activation and contention to agree *by construction*; that is only achievable if the contention input is computed by the serving derivation. Shipping it later would mean F19 writes a second layer-derivation against a first one — the exact pattern this feature exists to eliminate. It is dead code for one feature's duration, which is the correct trade.

6. **Four derivations, not three** (2026-07-20). The BRD and PPRD name three (`circuit_service.py:424`, `:799`, `inference_service.py:955`). Live-code verification found a fourth: `_steering_circuit_uncached` (`inference_service.py:806-822`) independently derives serving members AND the participating-layer set to decide whether a rung header may be emitted. It is in scope — leaving it would preserve the F14-R2-02 defect class on the evidence surface specifically, which is the worst place to leave it. The BRD's success metric ("baseline: 3") is understated and is recorded as such.
