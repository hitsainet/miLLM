# Task List: Single Circuit-Serving Derivation

## miLLM Feature 18

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** ✅ IMPLEMENTED 2026-07-21 — 3 review rounds pending
**References:** `018_FPRD|Single_Serving_Derivation.md` · `018_FTDD|Single_Serving_Derivation.md` · `018_FTID|Single_Serving_Derivation.md` · `BRD-MILLM-CIRCUITS-002.md` (BR-002)

## Relevant Files
- `millm/ml/circuit_steering.py` — NEW; `ServingPlan` + `CircuitSteeringEngine`, the one derivation
- `millm/services/circuit_service.py` — `_serving_members` (:623-662) deleted; `_serve_full` (:415-433) and `set_intensity` (:798-803) consume the engine
- `millm/services/inference_service.py` — `_circuit_serving_members` (:720-731) and `_sae_service_for_dial` (:733-745) deleted; dial (:890-985) and `_steering_circuit_uncached` (:806-822) consume the engine
- `millm/services/sae_service.py` — `for_registry` classmethod added; `set_circuit_steering` (:481-513) and `_directional_budget` (:66-76) UNCHANGED
- `tests/unit/ml/test_circuit_steering_engine.py` — NEW; characterization, claim set, honest construction
- `tests/unit/services/test_circuit_service.py`, `test_circuit_dial.py` — repointed at the engine
- `tests/unit/services/test_circuit_steering.py` — must need NO change (the apply is untouched)
- `tests/integration/test_single_serving_derivation.py` — NEW; four-way identity, F14 regressions, reachability
- `0xcc/adrs/000_PADR|miLLM.md` — new §10 decision entry (F18 has none today)

### Notes
- Depends on **Feature 17** (request-scoped context) and is sequenced strictly AFTER it, per BRD `execution_order` steps 3→4 — F18 lands on the settled context. Also depends on F12 (`set_circuit_steering`, `AttachedSAEState`), F13 (`CircuitService`, `CircuitDefinitionV1`) and F14 (the dial and echo paths).
- **Four derivations, not three.** The BRD/PPRD baseline of 3 is understated; `inference_service.py:806-822` is a fourth, on the evidence surface. Verified by `grep -n "_serving_members\|_circuit_serving_members\|set_circuit_steering"`.
- No migration, no config key, no contract change. Any API response-shape delta is a defect, not a feature.
- Test commands: `pytest` (backend), `npm test` in admin-ui (Vitest).

### Category Checklist Results
- Data: none — no schema change; `circuits.layers` keeps its column and loses only its serving read (1.4) ✓
- Backend/API: 3.x (call-site rewiring), 4.x (contract-parity tests); no new endpoints ✓
- Frontend/UI: none — internal feature; frontend suite is a regression gate only (5.2) ✓
- Business logic: 2.x (flattening, intensity, claim set, sign rule delegation) ✓
- Integration wiring: 3.x (four call sites), 2.5 (`for_registry` replacing the `__new__` bypass) ✓
- Error handling & logging: 3.5 (dial `except Exception` no longer masks a construction defect), 4.4 (offender/422 parity) ✓
- Testing: characterization BEFORE the move (1.x), paired per parent, 4.x integration, 5.x reachability + mutation ✓
- Performance & security: 4.5 (zero hot-path latency delta; engine strictly cheaper than the bypassed service) ✓
- Config/deploy: none — no key, no flag, no migration; rollback is a revert (1.5) ✓
- Documentation: 6.x (manual note + PADR decision of record) ✓

## Tasks

- [x] 1.0 Behaviour preservation harness — BEFORE any code moves (covers ENG-V1; BR-002)
  - [x] 1.1 Record the baseline: full backend + frontend suite green, counts captured, at the pre-refactor commit
  - [x] 1.2 Characterization tests against the LIVE `CircuitService._serving_members` (:623-662): both-sources (EC-18.1), dedupe first-wins (EC-18.2), empty (EC-18.6), definition order, per-layer `sae_id` resolution — written and passing against the OLD code
  - [x] 1.3 Characterization of intensity resolution against live `_serve_full` (:421-423): document field wins over the DB column when both are present and differ (EC-18.4)
  - [x] 1.4 Characterization of today's participating-layer sets at all four sites, asserting they currently agree — the pre-move witness that the move preserves agreement rather than creating it
  - [x] 1.5 Confirm rollback safety: no migration, no config key, no contract change; every commit independently revertible

- [x] 2.0 The engine (covers FR-18.1, FR-18.3; ENG-D1..D5, ENG-C1, ENG-C3, ENG-K1..K3)
  - [x] 2.1 `millm/ml/circuit_steering.py`: `ServingPlan` frozen dataclass (members, intensity, claimed_layers, attached_layers, `unattached_layers`, `is_serveable`)
  - [x] 2.2 `CircuitSteeringEngine.__init__(state: AttachedSAEState | None = None)` — one defaulted arg; no repository, cache dir, emitter, downloader or loader (ENG-C1)
  - [x] 2.3 `serving_members` — the verbatim move of `:639-662`; RULES docstring carried over, `@staticmethod`-by-contract paragraph dropped (it documented a workaround being retired)
  - [x] 2.4 `serving_intensity` (ENG-D3) and `claim_set` defined AS the layers of `serving_members` (ENG-K1), plus `plan_for` deriving both from ONE member list
  - [x] 2.5 `SAEService.for_registry` classmethod — total construction, replacing the `__new__` bypass; `apply()` routes through it when no DI service is supplied (ENG-C2)
  - [x] 2.6 Verify the sign rule is DELEGATED: engine carries `budget`/`sign` untouched, `_directional_budget` keeps exactly its two current call sites (ENG-D5, EC-18.3)
  - [x] 2.7 Unit tests: repoint 1.2/1.3 characterizations at the engine and confirm they pass UNCHANGED; add claim-set identity, attached/unattached split (EC-18.7), and "no reachable method reads an unset field" (ENG-C3)

- [x] 3.0 Rewire the four call sites (covers FR-18.1, FR-18.2; ENG-D4, ENG-C2)
  - [x] 3.1 `_serve_full` (:415-433) → `plan_for` + `apply`; `bound_layers` from `plan.claimed_layers`, not `definition.layers()`
  - [x] 3.2 `set_intensity` (:798-803) → `plan_for(intensity=…)` + `apply`; `reapplied`/`warnings` shapes unchanged
  - [x] 3.3 The per-request dial (:890-985) → `plan_for`; **snapshot derived from `plan.claimed_layers`**, the same field the apply drives (closes F14-R2-01 structurally)
  - [x] 3.4 The echo predicate `_steering_circuit_uncached` (:806-822) → `plan.is_serveable`; `_STEERING_CIRCUIT_MEMO` wrapper (:798-804) left exactly as it is
  - [x] 3.5 DELETE `CircuitService._serving_members`, `InferenceService._circuit_serving_members` and `_sae_service_for_dial` — no shims (ENG-D4); confirm `SAEService.__new__` appears nowhere
  - [x] 3.6 Confirm `sae_service.py` diff is ADDITIVE only (`for_registry`); `set_circuit_steering`, `_set_circuit_steering_locked` and `_directional_budget` untouched (ENG-C4)

- [ ] 4.0 Integration & contract parity (covers ENG-V1; BR-002)
  - [x] 4.1 `test_single_serving_derivation.py`: one definition through activation, `set_intensity`, dial and echo — identical serving members AND identical applied per-layer values from all four
  - [x] 4.2 F14-R1-01 regression: authored 150 at λ=2, dial to 1.0, assert **150 not 100**; plus a circuit whose document intensity differs from the DB column
  - [x] 4.3 F14-R2-01 regression: a member layer absent from the `circuits.layers` column is claimed, dialled AND restored
  - [ ] 4.4 Offender/422 parity: duplicate `(layer, feature_idx)` and `SAE_SET_INCOMPLETE` produce byte-identical responses pre- and post-refactor
  - [ ] 4.5 Zero hot-path latency delta; rung-header suppression parity when nothing is steering (EC-18.7)
  - [ ] 4.6 Slice-fallback boundary (EC-18.9): a partially-bound circuit still routes through `_serve_slices` and the cluster path, the engine is NOT involved, and `set_intensity` still reports recorded-but-not-applied

- [ ] 5.0 Reachability & mutation (covers ENG-V2..V4; BR-005, BRD locked decision 3 / RSK-001)
  - [x] 5.1 Four reachability tests — one per call site — each FAILING when its engine wiring is cut; invocation asserted, never existence (the F15 `TestRingPruningIsWired` anti-pattern is explicitly excluded)
  - [x] 5.2 Single-derivation guard: a test that FAILS if a second flattening implementation appears; frontend suite green as a regression gate
  - [x] 5.3 Mutation testing on `millm/ml/circuit_steering.py`; every survivor pinned by a new test or recorded with a rationale
  - [x] 5.4 Assert the derivation count is exactly 1 and `SAEService.__new__` is absent from the tree

- [ ] 6.0 Documentation (covers FPRD §14)
  - [ ] 6.1 Manual: a short circuits-architecture note naming the one derivation, so "where does serving happen" has one answer
  - [ ] 6.2 Update the PPRD Feature 18 block if the four-vs-three baseline is corrected upstream (see Upstream defects below)
  - [x] 6.3 **New PADR §10 decision entry** — `#### Single serving derivation vs four coordinated call sites`: engine shape, `__new__` retirement, and the canonical sign rule promoted to normative architecture text (it lives today only in `_directional_budget`'s docstring)

- [ ] 7.0 Feature Acceptance (per instruct 008)
  - [ ] 7.1 Verify FPRD §9 criteria 1–7 + all US/EC boxes one-by-one
  - [ ] 7.2 Confirm behaviour preservation: backend + frontend suites green at EVERY commit of the extraction, not only at the end
  - [ ] 7.3 Full suite green; update CLAUDE.md Document Inventory + Current Status; confirm `docs/mcp-contract.md` needs no change

## Coverage Audit
- FR-18.1→2.x/3.x/5.2/5.4; FR-18.2→2.2/2.5/3.5/5.4; FR-18.3→2.4/2.7/4.1 ✓
- US-18.1→3.x/5.2; US-18.2→2.2/2.5/3.5; US-18.3→2.4/2.7 — implementing + testing sub-tasks each ✓
- EC-18.1→1.2/2.3/2.7; EC-18.2→1.2/2.3/4.4; EC-18.3→2.6; EC-18.4→1.3/2.4/4.2; EC-18.5→2.4/3.3/4.3; EC-18.6→1.2/2.7; EC-18.7→2.7/4.5; EC-18.8→3.6; EC-18.9→4.6 ✓
- BRD: BR-002→ENG-D1..D5/C1..C4/K1; BR-005→5.1/5.2; BR-011→2.4/2.7 (claim set as F19's input) ✓
- TDD/TID sections all mapped (engine→2.x, rewiring→3.x, what-doesn't-change→3.6, testing→4.x/5.x) ✓
- Open questions: none — no spike tasks ✓
- Final task is Feature Acceptance ✓

## Review rounds (goal: 3 rounds, ≥10 findings each)
- [ ] Round 1 (multi-angle /code-review, 2 finder agents): ≥10 findings — fix criticals, document deferrals.
      Watch for: a fifth derivation reintroduced by a "convenience" helper; the dial's snapshot re-deriving
      its layer set instead of reading `plan.claimed_layers` (F14-R2-01's exact shape); `serving_intensity`
      called with the DB column by one caller and the document field by another (F14-R1-01); an engine-side
      `sign` multiplication double-negating suppression; `for_registry` leaving a field unset and merely
      relocating the swallowed AttributeError; the echo predicate (`:806-822`) missed entirely because the
      BRD said three call sites; dedupe dropped as "redundant" (it 422s, it does not double-apply).
- [ ] Round 2 (post-fix verification + fresh angles): ≥10 findings — verify R1 fixes hold; hunt regressions.
      Watch for: R1's own fixes drifting — the F14 premise held twelve rounds for twelve, and F14-R2-01 was
      R1's fix hardening a loop while leaving its input incomplete. Specifically: a fix that repoints one
      caller and leaves a sibling; a characterization test rewritten to match new behaviour rather than
      pinning old; `set_circuit_steering` edited to accommodate the engine (the premise-violating change);
      the ContextVar memo folded into the engine, giving a process singleton request state.
- [ ] Round 3 (/review, 4 perspectives): ≥10 findings — fix, pin mutation survivors.
      Watch for: reachability tests that assert existence rather than invocation (the F15
      `TestRingPruningIsWired` anti-pattern); mutation survivors in the flattening dedupe and the claim-set
      identity; `claim_set` correct but unexercised until F19 and therefore under-tested.
- [ ] Record: `.claude/context/sessions/review_feature018_R{1,2,3}_2026-07-*.md`.
- Directive: fix latent/pre-existing defects surfaced during review, not only regressions.

## Acceptance evidence

### FPRD §9 criteria, verified one-by-one
*(completed at feature close; each criterion gets a verdict and evidence cell)*

| # | Criterion | Verdict | Evidence |
|---|-----------|---------|----------|
| 1 | Derivation count exactly 1, down from 4; guard test fails if it rises | — | |
| 2 | `SAEService.__new__` absent; dial constructs its engine normally | — | |
| 3 | `claim_set` == distinct layers of `serving_members` on every fixture | — | |
| 4 | Flattening byte-identical: pre-move characterizations pass unchanged | — | |
| 5 | Sign rule holds — negative strength served without `sign` multiplication | — | |
| 6 | Suite green at every commit; reachability test per call site fails when wiring is cut | — | |
| 7 | Mutation run on the engine; survivors pinned or recorded | — | |

### Upstream defects found while authoring — recorded, not propagated
1. **The BRD/PPRD baseline of three derivations is understated.** There are four: `inference_service.py:806-822` (`_steering_circuit_uncached`) independently flattens members and derives a participating-layer set to gate the rung header. The BRD's success metric reads "baseline: 3". F18 treats it as four; the metric should be corrected by its owner.
2. **BR-005 is the reachability requirement, not the mutation-testing one.** Mutation testing anchors to BRD locked decision (3) / RSK-001 and lands as FR-17.6 (Feature 17). Task 5.3 is anchored accordingly; 5.1/5.2 cite BR-005.
3. **The PPRD Feature-Requirements Matrix rows 16–20 are off by one column** — Feature 18's ✅ sits under FR-17.x. The prose blocks are authoritative and correct (F18 ↔ FR-18.x); the matrix needs a fix in the PPRD rather than propagation here.
4. **Feature 18 has no PADR decision of record** — the only Circuit Consolidation feature without one (v1.3 §10 covers F16, F17, F19). Task 6.3 authors it. Note also that miLLM's PADR uses named `####` decisions under §10, NOT miStudio's IDL-N numbering; cross-references must use increment + decision name.
5. **The canonical sign rule exists in miLLM only as a docstring** (`sae_service.py:66-76`), not in any architecture record, despite being load-bearing for both circuits and clusters. Task 6.3 promotes it to normative text.

---

## Implementation record (2026-07-21)

**Suite 1855 → 1873** green / 1 skipped. Frontend 272. CI green.

Structural claims, verified rather than asserted:

| Claim | Check |
|---|---|
| exactly ONE serving derivation | `grep -rn "CircuitMember("` → `circuit_steering.py` only |
| four call sites consume the plan | activation, operator dial, per-request dial, echo predicate |
| no shims | `_serving_members`, `_circuit_serving_members`, `_sae_service_for_dial` all absent |
| the `__new__` bypass is gone | `for_registry` verified TOTAL against `__init__`'s assignments |

**Mutation sweep on the engine: 12 mutations, ZERO survivors** — including the
two that would silently corrupt steering (a truthiness check on a `0.0`
authored intensity, and pre-applying `sign` in the flattening).

### What the refactor broke, and what caught it

Eleven dial/epoch failures, twice, from a missing `SAEService` import at the
rewired scope. **Both times the dial's broad `except Exception` turned a
NameError into a soft `circuit_dial_apply_failed` warning** — a wiring bug
degrading silently into a no-op, exactly as task 3.5 anticipated. I spent three
wrong hypotheses (singleton identity, `_inference_service`, `detach_sae`) before
instrumenting the actual call, which showed the cause in one line.

### The reachability tests were wrong first

The first version asserted `CircuitSteeringEngine` appears in
`__code__.co_names` — a source grep wearing a reachability costume. Cutting the
dial's wiring while leaving its local import in place kept the name and the test
GREEN. Rewritten to spy on `plan_for` and drive the real async paths:
`cut_dial` went **0 → 7 failures**.

### The single-derivation guard found two more sites on its first run

Both legitimate and DIFFERENT: the schema's per-layer cap counter, and the
slice-fallback projection which dedupes on `feature_idx` alone because a cluster
is keyed by index while serving keys on `(layer, feature_idx)`. Named in the
test so nobody folds them into the engine.
