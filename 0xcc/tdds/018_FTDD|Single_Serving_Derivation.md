# Technical Design Document: Single Circuit-Serving Derivation

## miLLM Feature 18

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**References:** `018_FPRD|Single_Serving_Derivation.md` · `012_FTDD|MultiSAE_Circuit_Serving.md` · `014_FTDD|Circuit_Dial.md` · `0xcc/docs/circuit-contention-model.md`

---

## 1. Executive Summary

This is an EXTRACTION, not a rewrite: every rule the new module enforces already exists and is already correct in at least one of the four call sites, and the design's entire value is that afterwards there is one place holding it instead of four places that must agree. `CircuitSteeringEngine` lives in `millm/ml/circuit_steering.py` — deliberately in `ml/`, not `services/`, because it depends on the attachment registry and the definition schema and on nothing else, and a module that cannot import a repository cannot grow a hidden second source of truth. It is constructed with one argument, an `AttachedSAEState`, defaulting to the process singleton, so the per-request dial can build one on the hot path without touching request-scoped DI — which is precisely the pressure that produced the `SAEService.__new__` bypass, and removing the pressure is what retires the bypass. The engine exposes four operations: `serving_members(definition)` (the flattening, moved verbatim from `CircuitService._serving_members` including its both-sources and dedupe rules), `serving_intensity(circuit, definition)` (the single answer to the document-vs-column question that produced F14-R1-01), `claim_set(definition)` (the layers those members reach, split into claimed-and-attached vs claimed-but-unattached), and `apply(...)` (which calls the UNCHANGED `SAEService.set_circuit_steering` under its existing `_ATTACHMENT_LOCK`). Activation, `set_intensity`, the per-request dial and the rung-echo predicate become consumers of a `ServingPlan` value rather than four computations of one — and because `claim_set` is derived from the same member list `apply` serves, Feature 19's contention model cannot disagree with activation without a type error rather than a silent wrong answer.

### Key Technical Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Module home | `millm/ml/circuit_steering.py`, not `services/` | Depends only on the registry + definition schema; a module that cannot import a repository cannot grow a second source of truth |
| Construction | `CircuitSteeringEngine(state: AttachedSAEState = AttachedSAEState())` — one arg, defaulted | Removes the pressure that produced `SAEService.__new__`; the dial builds one honestly on the hot path |
| Return shape | ONE `ServingPlan` dataclass carrying members + intensity + claim set | Callers that need two of the three cannot fetch them from different derivations at different times |
| Flattening | MOVED verbatim from `circuit_service.py:639-662`, both-sources + dedupe intact | The serving path 422s on a repeated key; the rules are load-bearing, not stylistic (EC-18.2) |
| Sign rule | DELEGATED to `sae_service._directional_budget`, never reimplemented | Negative budget is already directional; a second copy is exactly the drift this feature exists to end (EC-18.3) |
| Intensity | ONE `serving_intensity()` — `definition.budget.intensity if definition.budget else circuit.intensity` | F14-R1-01 was a caller picking the other field; a single function makes the choice unavailable |
| Claim set | Distinct layers OF THE SERVING MEMBERS, never `circuits.layers` | F14-R2-01's root cause: snapshot keyed off the DB column while apply drove off the members (EC-18.5) |
| Attachment split | `claimed_attached` vs `claimed_unattached` returned separately | The engine reports; the caller decides. Activation slice-falls-back, the dial no-ops, the echo suppresses — three policies, one fact |
| Apply | `SAEService.set_circuit_steering` UNCHANGED, called by the engine | Moves the derivation, not the apply; the ±200 clamp, hazards, offender collection and `_ATTACHMENT_LOCK` all stay put (ENG-C4) |
| Old call sites | DELETED, not shimmed | A delegating shim is a call site with a body that can grow one; `_serving_members`' `@staticmethod`-by-contract comment is evidence the arrangement needed a written promise |
| Verification | Characterization tests BEFORE the move; reachability tests per call site; mutation on the module | A behaviour-preserving move is only provable if the behaviour was pinned first (ENG-V1) |

## 2. System Architecture

```
   BEFORE — four derivations, agreeing by discipline
 ┌────────────────────┐  flatten+intensity  ┌──────────────────────┐
 │ CircuitService     │────────────────────►│ SAEService           │
 │  _serve_full :424  │                     │  set_circuit_steering│
 │  set_intensity :799│────────────────────►│  (the real apply)    │
 └────────────────────┘                     └──────────────────────┘
 ┌────────────────────┐  flatten (unbound)          ▲
 │ InferenceService   │──┐ CircuitService.          │
 │  dial        :955  │  │  _serving_members  ──────┘ via SAEService.__new__
 │  echo pred :806-822│──┘ (@staticmethod by contract)   (half-built, :743)
 └────────────────────┘

   AFTER — one derivation, agreeing by construction
 ┌────────────────────┐                  ┌──────────────────────────────┐
 │ CircuitService     │                  │ CircuitSteeringEngine        │
 │  _serve_full       │─── plan_for() ──►│  serving_members()  ← MOVED  │
 │  set_intensity     │                  │  serving_intensity() ← EC-4  │
 └────────────────────┘                  │  claim_set()        → F19    │
 ┌────────────────────┐                  │  apply()  ──┐                │
 │ InferenceService   │─── plan_for() ──►│             │                │
 │  dial              │                  │  ctor: (AttachedSAEState)    │
 │  echo predicate    │                  └─────────────┼────────────────┘
 └────────────────────┘                                ▼
                                          ┌──────────────────────────────┐
                                          │ SAEService.set_circuit_       │
                                          │ steering  — UNCHANGED         │
                                          │ _ATTACHMENT_LOCK, ±200 clamp, │
                                          │ hazards, offender collection  │
                                          └──────────────────────────────┘
```

## 3. The Engine (`millm/ml/circuit_steering.py`)

The module holds one dataclass and one class. `ServingPlan` is the unit of agreement: a caller that has a plan cannot have obtained its members from one derivation and its layers from another, which is the shape of both F14 criticals.

```python
# millm/ml/circuit_steering.py — declarations
@dataclass(frozen=True)
class ServingPlan:
    members: list[CircuitMember]        # flattened, deduped, in definition order
    intensity: float                    # the ONE resolution (EC-18.4)
    claimed_layers: frozenset[int]      # distinct layers of `members` — F19's input
    attached_layers: frozenset[int]     # subset actually backed by an attached SAE
    @property
    def unattached_layers(self) -> frozenset[int]: ...   # claimed - attached
    @property
    def is_serveable(self) -> bool: ...                  # members and attached_layers non-empty


class CircuitSteeringEngine:
    def __init__(self, state: "AttachedSAEState" | None = None) -> None: ...
    def serving_members(self, definition) -> list[CircuitMember]: ...   # MOVED verbatim
    def serving_intensity(self, circuit, definition) -> float: ...      # ONE resolution
    def claim_set(self, definition) -> frozenset[int]: ...              # layers of the members
    def plan_for(self, circuit, definition, *, intensity=None) -> ServingPlan: ...
    def apply(self, plan, definition, *, sae_service=None) -> "CircuitSteeringResult": ...
```

`serving_members` is the moved body of `circuit_service.py:639-662` and nothing else changes in it: both `m.expanded_members` AND `m.feature` contribute when both are present, the `(layer, feature_idx)` dedupe collapses repeats first-wins, and `sae_id` is resolved per member layer via `definition.sae_for_layer`. The docstring's RULES move with the body; the `@staticmethod`-by-contract paragraph does not, because it documented a workaround that no longer exists — an instance method on an honestly-constructed engine needs no purity promise.

`claim_set` is defined as `frozenset(m.layer for m in self.serving_members(definition))` and is specified that way rather than as an equivalent-but-separate walk, so the identity FR-18.3 requires holds by construction and not by two implementations agreeing. `plan_for` calls it through the same member list it stores, so a `ServingPlan` is internally consistent by the time any caller sees it.

`apply` takes an optional `sae_service`: `CircuitService` passes its own DI-constructed instance, and the per-request dial passes none, in which case the engine constructs a minimal applier bound to the registry it already holds. This is the seam that retires `SAEService.__new__` — the dial needs an apply, not a service, and the engine is the thing that knows the difference. `set_circuit_steering` itself is untouched, so the ±200 clamp, the duplicate-member offender collection, the hazard quantification and the `_ATTACHMENT_LOCK` composing guard all keep their current behaviour and their current tests.

## 4. Call-Site Rewiring

```python
# circuit_service.py — _serve_full (was :419-426)
plan = self._engine.plan_for(circuit, definition)
outcome = self._engine.apply(plan, definition, sae_service=self._sae_service)
return {"serving_mode": "full", "bound_layers": sorted(plan.claimed_layers), ...}

# circuit_service.py — set_intensity (was :798-803)
plan = self._engine.plan_for(circuit, definition, intensity=float(intensity))
self._engine.apply(plan, definition, sae_service=self._sae_service)

# inference_service.py — the dial (was :899-959; _sae_service_for_dial DELETED)
plan = _ENGINE.plan_for(circuit, definition, intensity=lam)
if not plan.is_serveable:
    return None                                  # same no-op rules, one source
saved = self._snapshot_layers(plan.claimed_layers)   # SAME set the apply drives (EC-18.5)
outcome = _ENGINE.apply(plan, definition)

# inference_service.py — the echo predicate (was :806-822)
plan = _ENGINE.plan_for(circuit, definition)
return circuit if plan.is_serveable else None    # header suppressed on the same fact
```

The dial's snapshot deriving from `plan.claimed_layers` is the structural close of F14-R2-01: the saved set and the applied set are now the same object's field, so they cannot be filtered from different sources. `bound_layers` in the activation response is derived from the plan rather than from `definition.layers()`, which removes the last place the response can describe a layer set the apply did not touch. The `_STEERING_CIRCUIT_MEMO` ContextVar memoisation (`inference_service.py:53-57`, `:800-804`) is preserved exactly — it caches the predicate's *result*, not a derivation, so it is orthogonal to this change and must not be disturbed.

## 5. What Deliberately Does Not Change

`SAEService.set_circuit_steering` and `_set_circuit_steering_locked` keep every line, including the resolution cache keyed by `(sae_id, layer)` rather than layer alone, the fail-closed offender collection, the substitution warnings for a declared-but-unattached SAE, the empty-members clear-and-disable path, and the heuristic-hazard tail cap. `_directional_budget` keeps its home in `sae_service.py` and gains no second caller-side copy — the engine's members carry `budget` and `sign` unmodified and the apply applies the rule, exactly as today. The `circuits.layers` column keeps its writes and its display and query uses; only its read as a *serving* input disappears. `_ATTACHMENT_LOCK` keeps its scope: the engine derives outside it and applies inside it, which is what today's code does, and widening it to cover derivation would turn a pure function into a contended one for no benefit (EC-18.8).

## 6. Testing Strategy

### Unit
- `tests/unit/ml/test_circuit_steering_engine.py`: flattening characterization — both-sources (EC-18.1), dedupe first-wins (EC-18.2), empty (EC-18.6), definition order preserved, `sae_id` per-layer resolution; `serving_intensity` document-vs-column (EC-18.4) with **an assertion that the document field wins when both are present and differ**; `claim_set` == distinct member layers on every fixture; `attached`/`unattached` split (EC-18.7); constructor takes one defaulted arg and **no reachable method reads an unset field**.
- `tests/unit/services/test_circuit_steering.py` (existing): unchanged and green — it is the behaviour-preservation witness for the apply path.

### Integration
- `tests/integration/test_single_serving_derivation.py`: one definition through activation, `set_intensity`, the dial and the echo predicate, asserting **identical serving members and identical applied per-layer values from all four**; the F14-R1-01 regression (authored 150 at λ=2, dial to 1.0, expect **150 not 100**); the F14-R2-01 regression (a member layer absent from the DB column is claimed, dialled AND restored); rung-header suppression parity when nothing is steering.
- Reachability (BR-005): four tests, each cutting one call site's engine wiring and asserting the suite **fails** — invocation, not existence.

### E2E (post-deploy)
Activate a two-layer circuit, dial it through OWUI, and reconcile applied values and post-request restore against `PUT /api/circuits/active/intensity` reporting the same operation.

## 7. Risks

- The extraction is mechanical but wide — four call sites, two of them on the chat hot path. Mitigated by characterization-before-move (ENG-V1) and by keeping the apply untouched, so a regression can only be in the derivation, which is pure and cheap to test. Residual: a caller that reads a `ServingPlan` field the old code computed slightly differently; the four-way identity integration test is the specific guard.
- The `claim_set` ships with no consumer until Feature 19, so it is unexercised in production for one feature's duration — a classic place for a defect to hide. Mitigated by unit-testing it against the identity FR-18.3 states rather than against an expected literal, and by F19 taking it as its declared input rather than re-deriving.
- Deleting `_serving_members` and `_circuit_serving_members` is an API break for any test that calls them directly. That is intended and is the point of ENG-D4, but it means the diff touches more test files than the feature's logic warrants — accepted, and the tests are updated to call the engine rather than being deleted.
- The dial's engine is a module-level singleton bound to the `AttachedSAEState` singleton. If a future test needs to inject a different registry it must construct its own engine rather than patching the module global; documented in the module docstring so the next contributor does not discover it by writing a flaky test.
