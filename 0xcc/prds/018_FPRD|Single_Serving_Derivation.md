# Feature PRD: Single Circuit-Serving Derivation

## miLLM Feature 18

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**Feature Priority:** Secondary (Increment: Circuit Consolidation)
**References:** `BRD-MILLM-CIRCUITS-002.md` (BR-002) · `000_PPRD|miLLM.md` (v1.3, FR-18.x) · `000_PADR|miLLM.md` (v1.3)

---

## 1. Feature Overview

### Feature Name
Single Circuit-Serving Derivation — one implementation of "serve this circuit", not three.

### Brief Description
Serving a circuit is derived independently in three places:

| Site | Location |
|---|---|
| Activation (`_serve_full`) | `circuit_service.py:424` |
| Intensity change (`set_intensity`) | `circuit_service.py:799` |
| Per-request dial | `inference_service.py:955` |

Each parses the stored definition, flattens members, dumps edges and calls `set_circuit_steering`.
They agree today only because three separate implementations happen to match — and **Feature 14's two
worst defects were both consequences of them not matching**: R1-01 (the dial de-scaled from the DB
column while activation applied the document's intensity, so every member was scaled by the wrong
factor) and R2-01 (the dial's snapshot was keyed off `circuit.layers` while its apply was keyed off the
definition's members, leaking a per-request override permanently into global state).

This feature extracts a `CircuitSteeringEngine` as the one derivation, consumed by all three. It also
retires the `SAEService.__new__` bypass at `inference_service.py:743`, which the dial uses to reach
`set_circuit_steering` — a half-constructed service whose failure mode is a swallowed `AttributeError`
and a **silently unsteered response still carrying a rung header**.

### Business Requirement Traceability

| BR | Coverage |
|----|----------|
| BR-002 — serving SHALL have exactly ONE derivation; no caller SHALL bypass a constructor to reach it | FR-18.1, FR-18.2, FR-18.3 |

---

## 2. User Stories & Scenarios

**US-18.1 — A change lands once.**
As a maintainer changing how a circuit is served, I want to change one thing, so that I cannot leave
two of three call sites on the old behaviour.

*Acceptance:* `set_circuit_steering` has exactly one caller; the three sites route through the engine.

**US-18.2 — The dial cannot fail silently.**
As an operator, I want a dial that cannot serve an unsteered response while advertising a rung, so that
the header and the output agree.

*Acceptance:* no code path constructs a service by bypassing `__init__`; a failure in the serving path
surfaces rather than degrading to "unsteered but claimed".

**US-18.3 — Contention agrees with activation.**
As the author of Feature 19, I want a circuit's claim set computed by the same code that serves it, so
that the contention check and the serve cannot disagree about which layers a circuit touches.

*Acceptance:* the claim set is produced by the engine, not re-derived.

### Edge Cases

- **EC-18.1: Slice-fallback.** `_serve_slices` steers through the cluster profile path, not
  `set_circuit_steering`. The engine covers full serves; slice serves remain the cluster path's.
- **EC-18.2: Dial with no repository.** The engine must be constructible with only the attachment
  registry — that is the legitimate need the `__new__` bypass was meeting badly.
- **EC-18.3: Duplicate `(layer, feature_idx)`.** `_serving_members` collapses duplicates because the
  serving path rejects a repeated key; the engine MUST preserve that exactly.
- **EC-18.4: Negative strength.** The canonical sign rule survives unchanged: a negative strength is
  already directional, and multiplying by `sign` double-negates suppression into amplification
  (`_directional_budget`, `sae_service.py:66`).
- **EC-18.5: Both-sources members.** A `cluster_ref` contributes its frozen `expanded_members` AND its
  own `feature` when both are present; taking one silently drops authored members.

---

## 3. Functional Requirements

- **FR-18.1:** Serving a circuit SHALL have exactly one implementation, consumed by activation
  (`circuit_service.py:424`), intensity changes (`:799`) and the per-request dial
  (`inference_service.py:955`).
- **FR-18.2:** No caller SHALL construct a service by bypassing its constructor in order to reach
  steering; a half-constructed service whose failure mode is a swallowed `AttributeError` and a
  silently unsteered response SHALL NOT be reachable.
- **FR-18.3:** A circuit's claim set — the layers its serving members reach — SHALL be computed by that
  same derivation, so activation and Feature 19's contention check agree by construction.

---

## 4. Data Requirements

None. This is a pure refactor: no schema, no config, no persisted shape changes.

---

## 5. API Specifications

No endpoint changes. `set_intensity`'s response shape is unchanged by this feature (Feature 16 owns its
`reapplied` correction).

---

## 6. UI Requirements

None.

---

## 7. Non-Functional Requirements

- **NFR-18.1:** Behaviour-preserving. The existing backend suite is the floor, not the target.
- **NFR-18.2:** The engine SHALL be constructible without a repository or a DB session, so the dial
  does not pull request-scoped DI into the inference hot path (the real requirement behind EC-18.2).

---

## 8. Dependencies

- **Feature 17** — lands on the settled request-scoped context; sequenced after it.
- Feature 12 (`set_circuit_steering`, the attachment registry), Feature 13 (activation), Feature 14 (the dial).
- **Feature 19 consumes FR-18.3.** The claim set must exist before contention can be enforced.

---

## 9. Success Criteria

1. Exactly one call site for `set_circuit_steering` (baseline: three).
2. No `__new__` constructor bypass anywhere in the serving path (baseline: one, `inference_service.py:743`).
3. The claim set is produced by the engine and consumed by activation and contention.
4. Full backend suite green throughout, with no acceptance criterion regressing.
5. The F14 R1-01 and R2-01 regression tests still pass — they pin the behaviours the duplication broke.

---

## 10. Testing Requirements

- Characterization: pin the current serving behaviour of all three sites BEFORE extraction.
- Unit: the engine's flattening equals `_serving_members` exactly, including dedupe (EC-18.3),
  both-sources collection (EC-18.5) and the sign rule (EC-18.4).
- Unit: the engine constructs without a repository (EC-18.2).
- **Mutation (BR-005):** removing the engine's dedupe, or its sign handling, MUST fail a test.

---

## 11. Rollout & Migration

No migration. Internal refactor with no external surface change.

---

## 12. Out of Scope

- Changing what serving DOES — strengths, clamping, hazards and budgets are unchanged.
- Slice-fallback serving (EC-18.1), which belongs to the cluster path.
- The `reapplied` correction (Feature 16).

---

## 13. Open Questions

1. Should the engine own hazard computation (`_cross_layer_hazards`), or only the serve? Leaning
   **only the serve** — hazards are already surfaced at activation and the dial currently discards
   them, which Feature 14 R3 recorded as a separate gap.

---

## 14. Documentation Requirements

- PADR: already records the decision (v1.3).
- No user-facing documentation change.

---

## 15. Decisions from Clarifying Questions

- **Extract an engine rather than have the dial call `CircuitService`**: the dial must not acquire a
  repository or a DB session, which is the legitimate need the `__new__` bypass met badly.
- **The engine computes the claim set**: Feature 19 needs it, and a second implementation would
  reintroduce exactly the drift this feature removes.
