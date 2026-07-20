# Feature PRD: Request-Scoped Sensing Context

## miLLM Feature 17

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**Feature Priority:** Secondary (Increment: Circuit Consolidation)
**References:** `BRD-MILLM-CIRCUITS-002.md` (BR-001) · `000_PPRD|miLLM.md` (v1.3, FR-17.x) · `000_PADR|miLLM.md` (v1.3) · `docs/circuit-contention-model.md` (§4)

---

## 1. Feature Overview

### Feature Name
Request-Scoped Sensing Context — one owner for the position counter, the fire rings, and the event budget.

### Brief Description
Edge sensing today distributes three pieces of per-request state across N independently-armed
`LoadedSAE` objects: each SAE advances its own `_edge_token_offset`, all of them write into one shared
`EdgeFireRing`, and each enforces its own copy of the per-request event cap. This feature replaces that
arrangement with a single `SensingRequestContext`, created at request start and passed to each
participating SAE, which owns (a) the absolute token position counter, (b) the fire rings — **one per
`(request, circuit)`**, never one shared ring — and (c) the per-request event budget, attributed per
circuit. The edge machinery moves out of `sae_wrapper.py` into `millm/ml/edge_sensing.py`, where it can
be exercised without constructing a `LoadedSAE`. No user-visible behaviour changes.

### Problem Statement
Three of Feature 15's eight criticals across three review rounds share exactly one root cause: **N
per-SAE counters must agree on an absolute coordinate that no component owns, and the shared ring's
lifetime is managed by whoever remembers to call it.** The code is correct today only because three
separate comments keep being obeyed, and each review round found a new way to violate them — an early
return that skipped the offset advance (R1-03), a hook that pruned a sibling's fires (R1-01), and a
prune declared request-level and never wired (R2-01, then repeated one level up in R3). The invariants
are maintained by convention and pinned by tests; they are not enforced by construction. Feature 19
then makes the shared ring actively dangerous: it lets two circuits serve at once, and `edge_key` is
not unique across circuits.

### Feature Goals
1. Retire the N per-SAE position counters in favour of one request-scoped counter (BR-001, FR-17.1).
2. Give each `(request, circuit)` pair its own ring, so cross-circuit edge-key collision cannot
   fabricate an observation (FR-17.2).
3. Attribute the event budget per circuit, so one busy circuit cannot starve another's observations
   (FR-17.3).
4. Move ring lifetime — creation, pruning, release — into the context, not into whichever hook runs
   last (FR-17.4).
5. Extract the edge machinery into `millm/ml/edge_sensing.py`, testable without a `LoadedSAE` (FR-17.5).
6. Prove behaviour preservation: characterization tests green BEFORE any code moves; mutation testing
   applied after (FR-17.6).

### User Value Proposition
"The edge-sensing invariants that took eight criticals across three review rounds to get right become
impossible to violate rather than guarded by comments — so the next feature to touch circuits does not
pay the same tax."

### Connection to Project Objectives
Delivers the lead item of BRD-MILLM-CIRCUITS-002's *Structural consolidation* theme, and is the
prerequisite the BRD names for Feature 18's single serving derivation. It is also the precondition for
Feature 19: the contention model (`docs/circuit-contention-model.md` §4) specifies the context's
required N-circuit shape and states that designing it around one circuit and generalising later
"would repeat the exact mistake this increment exists to correct."

### BRD Traceability

| BR | Requirement (abbrev.) | Coverage IDs |
|----|-----------------------|--------------|
| BR-001 | Position, ring and budget owned by ONE request-scoped context; per-SAE counter divergence, prune races and budget skew structurally impossible rather than test-guarded | CTX-C1, CTX-C2, CTX-R1, CTX-R2, CTX-B1, CTX-B2, CTX-L1, CTX-L2 |
| BR-006 | Truncation identifies WHICH layer shed data rather than marking a whole request truncated | CTX-B3, CTX-B4 |
| BR-011 (enabling) | Concurrent circuits — the context must be built for N circuits from the outset | CTX-R1, CTX-R2, CTX-B2 |
| (constraint) | Refactors behaviour-preserving and PROVABLY so; existing suites are the floor | CTX-V1, CTX-V2, CTX-V3 |

---

## 2. User Stories & Scenarios

#### US-17.1: A maintainer changes the edge matcher without breaking coordinates
**As a** maintainer touching edge sensing
**I want** the absolute position to come from one place
**So that** I cannot introduce a divergence between two layers by adding an early return.

**Acceptance Criteria:**
- [ ] Exactly ONE position counter exists per request; no `LoadedSAE` carries `_edge_token_offset`
- [ ] Adding a new early-return path to the sensing pass cannot desynchronise layers, because the
      counter is advanced by the context at the pass boundary, not by each return path
- [ ] A test fails if a per-SAE position counter is reintroduced

#### US-17.2: Two circuits sense concurrently without fabricating observations
**As an** operator serving two circuits (Feature 19)
**I want** each circuit's edges matched only against its own fires
**So that** no recorded observation describes an edge that fired in neither circuit.

**Acceptance Criteria:**
- [ ] Each `(request, circuit)` pair has its own ring; rings are never shared across circuits
- [ ] Two circuits containing the SAME `edge_key` produce independent observations; circuit A's
      upstream fire never matches circuit B's downstream fire
- [ ] A test constructs exactly that collision and asserts zero cross-circuit matches

#### US-17.3: A busy circuit does not silence a quiet one
**As an** operator serving two circuits
**I want** the event budget attributed per circuit
**So that** a circuit whose edges fire constantly cannot exhaust another circuit's observations.

**Acceptance Criteria:**
- [ ] The budget is per request, attributed per circuit; exhausting circuit A's share leaves circuit
      B's intact
- [ ] Reaching a budget never stops upstream-fire recording for any layer (the R2-03/R3-02 rule)
- [ ] `truncated` identifies which circuit and which layer shed, not the whole request (BR-006)

#### US-17.4: The edge machinery is testable on its own
**As a** test author
**I want** the ring and matcher to live in their own module
**So that** I can exercise them without a `LoadedSAE` stub that drifts from the real class.

**Acceptance Criteria:**
- [ ] `millm/ml/edge_sensing.py` holds `EdgeSpec`, `CircuitSensingConfig`, `SensedEdge`,
      `EdgeFireRing`, `SensingRequestContext` and the matcher; importable without `sae_wrapper`
- [ ] Tests construct the matcher directly — no hand-written six-attribute stub (the R3 harness
      blind spot that hid both R1's and R2's criticals)

#### Edge Cases
**EC-17.1: A layer is suppressed for an entire request** — **Trigger:** one SAE's `suppressed()` is true
every pass (embeddings pass, or a layer whose hook never fires). **Behavior:** the context still advances
the shared position for that pass, and the suppressed layer still reports progress, so ring pruning is
not stalled by a layer that never reports. (Live-code defect today: both early-return paths in
`_sense_edges` return BEFORE `note_layer_progress`, so `_progress` can stay below two entries and the
ring never prunes — see §15.6.)
**EC-17.2: A request begins with zero armed circuits** — **Behavior:** no context is created; collect
returns empty; no ring is allocated. Parity with today's `("", [], False)` guard.
**EC-17.3: A circuit is disarmed mid-request** — **Behavior:** the context retains the begin-time circuit
identity snapshot; observations already in its ring are attributed to the circuit that was armed when
the boundary opened, never to a circuit armed later (preserves R2-04 / R3-04).
**EC-17.4: Two circuits share a layer under `allow_layer_overlap`** (Feature 19) — **Behavior:** one
context, one position counter, TWO rings. The shared layer's hook writes each circuit's fires into that
circuit's own ring. Composed-layer marking is Feature 19's concern, not this feature's.
**EC-17.5: A hung generation thread wakes after the request closed** — **Behavior:** the context is
released at close; a late write finds no open context and is dropped with a log, never landing in the
next request's ring (preserves R3-06's post-hang disarm).
**EC-17.6: A pass raises inside the matcher** — **Behavior:** the exception is logged and never reaches
the forward pass, AND the position still advances, so a failing pass cannot desynchronise layers
(preserves R2-10's `TestSensingFailuresAreNotSilent`).
**EC-17.7: 200-edge circuit at the contract maximum** — **Behavior:** per-edge retention and the
saturation shed behave exactly as today; the extraction must not regress the R3 `bisect` match path.

---

## 3. Functional Requirements

### The Context (FR-17.1, FR-17.4)

| ID | Requirement | Priority |
|----|-------------|----------|
| CTX-C1 | A `SensingRequestContext` shall be created once per sensed request and passed to each participating SAE; it shall own the absolute token position counter. The per-SAE `_edge_token_offset` fields shall be removed | Must |
| CTX-C2 | The context shall advance position once per forward pass at the pass boundary, so no early-return path within the sensing body can skip it | Must |
| CTX-C3 | The context shall own phase (`prefill`→`decode`) transition, for the same reason | Must |
| CTX-L1 | Ring lifetime — creation, pruning and release — shall be owned by the context; no hook shall be required to know sibling state in order for pruning to occur | Must |
| CTX-L2 | The context shall be released at request close; a write arriving after close shall be dropped and logged, never applied to a subsequent request | Must |
| CTX-L3 | The context shall snapshot circuit identity at boundary-open; a mid-request re-arm shall not re-attribute observations | Must |

### Rings (FR-17.2)

| ID | Requirement | Priority |
|----|-------------|----------|
| CTX-R1 | The context shall hold ONE `EdgeFireRing` per `(request, circuit)` pair — never one ring shared across circuits | Must |
| CTX-R2 | An `edge_key` present in two active circuits shall produce independent observations per circuit; a fire recorded for circuit A shall never be matchable by circuit B's downstream fire | Must |
| CTX-R3 | Ring pruning shall remain bounded by construction — pruning to the slowest layer's progress, without any caller external to the context | Must |
| CTX-R4 | Per-edge fire retention (`_MAX_FIRES_PER_EDGE`) and the `bisect` match path shall be preserved with their measured latency characteristics | Must |

### Budget (FR-17.3, BR-006)

| ID | Requirement | Priority |
|----|-------------|----------|
| CTX-B1 | The event budget shall be per request, held by the context — not N independent per-SAE caps | Must |
| CTX-B2 | The budget shall be attributed per circuit, so exhausting one circuit's share leaves another's intact | Must |
| CTX-B3 | Reaching a budget shall suppress only the downstream event append; upstream fire recording shall continue for every layer (the R2-03 / R3-02 starvation rule) | Must |
| CTX-B4 | Truncation shall identify which layer (and circuit) shed, rather than OR-ing across layers and stamping every row; `truncated_layers` shall appear in the status payload (BR-006) | Must |

### Extraction (FR-17.5)

| ID | Requirement | Priority |
|----|-------------|----------|
| CTX-E1 | `EdgeSpec`, `CircuitSensingConfig`, `SensedEdge`, `EdgeFireRing`, `SensingRequestContext` and the matcher shall live in `millm/ml/edge_sensing.py` | Must |
| CTX-E2 | The module shall be importable and exercisable without constructing a `LoadedSAE` | Must |
| CTX-E3 | `sae_wrapper.py` shall retain only the thin hook-facing entry point; the 13 `_edge_*` instance fields shall reduce to the context reference plus what genuinely belongs to the SAE | Must |
| CTX-E4 | Re-exports shall preserve existing import sites, or all import sites shall be updated in the same change; no import shall break | Must |
| CTX-E5 | R2's superseded prune methods (`prune_ring`, `safe_prune_boundary`, `prune_between_passes`), which have zero production callers, shall be deleted rather than carried across (§15.5) | Must |

### Verification (FR-17.6)

| ID | Requirement | Priority |
|----|-------------|----------|
| CTX-V1 | Characterization tests pinning current matcher behaviour shall be written and GREEN BEFORE any code moves | Must |
| CTX-V2 | The same characterization suite shall be green after the move, unmodified — a change to a characterization test is a behaviour change and must be justified in the record | Must |
| CTX-V3 | Mutation testing shall be applied to the resulting module; surviving mutants on load-bearing lines shall be pinned or explicitly recorded | Must |
| CTX-V4 | Existing suites (backend ≥1597, frontend ≥272) shall remain green throughout; no Feature 15 acceptance criterion shall regress | Must |

---

## 4. Data Requirements

**No schema change.** This feature is behaviour-preserving at the persistence boundary.

One payload addition, from BR-006: the edge-sensing status response gains `truncated_layers` (a list of
layer numbers that shed data), and `circuit_edge_sensing_events.truncated` becomes per-row-accurate
rather than a request-wide OR. Both are additive; no migration is required because the column already
exists and only its fill rule changes.

Contract note: `docs/mcp-contract.md` moves to v1.2 with no breaking change — `truncated_layers` is an
added field on an existing status payload.

## 5. API Specifications

**No new endpoints.** `GET /api/circuits/sensing/status` gains `truncated_layers: int[]`. The
`millm_circuit_sensing_status` MCP tool surfaces the same field verbatim (Feature 16 owns the
registration; this feature only adds the field to the payload it already returns).

Everything else — routes, envelopes, error codes, WS event names and payload shapes — is unchanged, and
that is a requirement, not an accident: a behaviour-preserving refactor that changes an API surface has
not preserved behaviour.

## 6. UI Requirements

**None.** No UI tab. The only visible consequence is that the edge-sensing status strip may render
`truncated_layers` where it previously showed a request-wide truncation flag; that is a Feature 16/15
surface and is out of scope here beyond not regressing it.

## 7. Non-Functional Requirements

- **Latency parity is the acceptance bar, not "fast enough".** F15's matcher was measured three times
  (1430 ms → 78.5 ms → 39.19 ms → sub-ms) and each measurement missed the path the previous fix had
  changed. The extracted module must be benchmarked on the SAME shapes as F15's final measurements:
  saturated 4096-token pass, 200-edge circuit, and the cross-layer ordering where the upstream layer
  records an entire prefill before the downstream layer matches ascending.
- Per-pass allocation must not grow: one context per request, N rings per request where N is the number
  of armed circuits (1 today, ≥2 under Feature 19).
- Un-armed cost remains one boolean check in the hook.
- The context must be safe under the serial request queue; sensing remains serial-only (F15 EDGE-S1).

## 8. Dependencies

- **Feature 15** — the machinery being restructured, and the source of every invariant that must survive.
- **Feature 19 / `docs/circuit-contention-model.md` §4** — supplies the required N-circuit shape. Per the
  BRD, the contention model is DESIGN-FIRST: it is settled (the document exists) and constrains this
  feature, but Feature 19's implementation lands after. This feature must build for N circuits without
  waiting for N circuits to be servable.
- **Feature 12** — the multi-SAE attachment registry the context is threaded through.
- Feature 18 depends on THIS feature ("lands on the settled context"), so this feature must land first.

## 9. Success Criteria

1. Exactly ONE position counter exists per request (BRD metric baseline: N per-SAE counters); a test
   fails if a per-SAE counter is reintroduced.
2. One ring per `(request, circuit)`; a constructed cross-circuit `edge_key` collision produces zero
   fabricated observations.
3. Budget attributed per circuit; a saturating circuit demonstrably does not reduce another circuit's
   recorded observations.
4. Ring pruning occurs with no caller outside the context, including when a layer is suppressed for an
   entire request (EC-17.1 — the live defect in §15.6 is fixed).
5. Characterization suite written first, green before AND after the move, unmodified.
6. Mutation testing run on `edge_sensing.py`; every surviving mutant on a load-bearing line either
   pinned by a new test or recorded with a reason.
7. Backend ≥1597 / frontend ≥272 green; every Feature 15 §9 criterion re-verified, none regressed.
8. `sae_wrapper.py` no longer carries the edge machinery; the 13 `_edge_*` fields are gone from
   `LoadedSAE` except the context reference.

## 10. Testing Requirements

- **Characterization (FIRST, before any move):** the current matcher's observable behaviour — ordering,
  lag window, same-position non-match, newest-antecedent selection, non-destructive read, per-edge
  retention eviction, saturation shedding, cap-does-not-starve-siblings, offset advance on every return
  path, progress-based pruning. These are written against the CURRENT code and must pass unchanged
  against the moved code.
- **Unit:** context lifecycle (create/advance/close, double-close, write-after-close); per-circuit ring
  isolation incl. the shared-`edge_key` collision; per-circuit budget attribution; per-layer truncation
  attribution; module importable without `LoadedSAE`.
- **Integration:** full arm→generate→collect→persist flow byte-identical to Feature 15's; two-circuit
  concurrent sensing (behind Feature 19's flag, or with a direct two-context construction if the flag
  is not yet live); suppressed-layer pruning; post-hang release.
- **Mutation:** applied to `edge_sensing.py` after the move, following R3's practice — break a
  load-bearing line, run the suite, revert. R3 found four unpinned lines in one pass that two rounds of
  reading missed.
- **Benchmark:** the three F15 latency shapes, asserted against the per-layer-denominated threshold
  (BR-013, owned by another feature in this increment — this feature must not regress the measurement).

## 11. Rollout & Migration

No migration, no config change, no flag. The feature is complete when the old structures are gone —
a partial state in which both the context and the per-SAE counters exist is explicitly not a shippable
increment, because it doubles the state the invariants must hold across.

## 12. Out of Scope

Concurrent circuit ACTIVATION (Feature 19 — this feature builds the shape, does not lift the
invariant); the single serving derivation (Feature 18); the steering epoch (BR-003); MCP registration
(Feature 16); alone-vs-within per-event classification (BR-007); the per-layer overhead threshold
(BR-013). Any behaviour change to the matcher's semantics — if the characterization tests reveal
current behaviour is wrong, that is a finding to record, not a change to make inside this feature.

## 13. Open Questions

None blocking. The N-circuit shape is settled by `docs/circuit-contention-model.md` §4; the ring
decision is settled by PADR v1.3 ("One ring per (request, circuit) vs one ring per request").

## 14. Documentation Requirements

Internal only — no manual change (no user-facing surface). The module docstring for
`millm/ml/edge_sensing.py` shall record WHY the context owns these three things, citing the three
review rounds, so the next maintainer does not re-derive the per-SAE arrangement. `docs/mcp-contract.md`
v1.2 records the `truncated_layers` field addition.

## 15. Decisions from Clarifying Questions

1. **One ring per `(request, circuit)`, never one shared ring.** `edge_key` is synthesised as
   `{up_idx}@{up_layer}->{down_idx}@{down_layer}` and is NOT unique across circuits — two circuits can
   legitimately contain the same edge. A shared ring would let circuit A's upstream fire match circuit
   B's downstream fire and record an observation of an edge that fired in **neither**. A fabricated
   observation on an evidence surface is categorically worse than a missed one, and Feature 19 makes
   this reachable in practice.
2. **Built for N circuits from the outset**, per BRD assumption and contention-model §4. Designing for
   one and generalising later is the exact mistake this increment exists to correct.
3. **Characterization-first is a gate, not a guideline** (FR-17.6). This is the most defect-dense code
   in the arc: 8 criticals in 3 rounds, every round finding a regression in the previous round's fix.
4. **Behaviour-preserving means the characterization tests do not change.** A modified characterization
   test is a behaviour change requiring explicit justification in the review record.
5. **R2's prune methods are deleted, not moved.** `prune_ring`, `safe_prune_boundary` and
   `prune_between_passes` (`circuit_sensing_service.py:526/538/550`) have **zero production callers** —
   verified live. R3 superseded them with `note_layer_progress` but never removed them, so the codebase
   currently carries a dead second pruning design alongside the live one. Carrying it into the new
   module would preserve the ambiguity the context exists to eliminate.
6. **The suppressed-layer pruning gap is in scope.** Both early-return paths in `_sense_edges`
   (`sae_wrapper.py:1069-1075` and `:1080-1096`) advance the offset and `return` **before**
   `note_layer_progress` is reached — it lives in the `finally` of the block below them. A layer
   suppressed for an entire request therefore never reports progress, `_progress` can stay below the
   two entries `note_layer_progress` requires, and the ring never prunes. This is R3's own fix carrying
   R1's shape: a correctness step placed where a return path can skip it. The context fixes it by
   construction — position and progress advance together at the pass boundary, above any early return.
