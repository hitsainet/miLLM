# Technical Design Document: Single Circuit-Serving Derivation

## miLLM Feature 18

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**References:** `018_FPRD|Single_Serving_Derivation.md` · `000_PADR|miLLM.md` (v1.3) · `BRD-MILLM-CIRCUITS-002.md` (BR-002)

---

## 1. Executive Summary

Three call sites independently derive "serve this circuit". They agree by coincidence, and twice they
did not: F14 R1-01 (wrong intensity source) and F14 R2-01 (snapshot keyed off a different source than
the apply). This feature makes the agreement structural.

`CircuitSteeringEngine` takes only the attachment registry, exposes `serve(definition, intensity)` and
`claim_set(definition)`, and becomes the single caller of `set_circuit_steering`.

### Key Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Shape | A collaborator class, not a mixin or free functions | It needs one dependency (the registry) and is used from two services; a class makes that dependency explicit and constructible honestly. |
| Dependency | Attachment registry only | The dial must not acquire a repository or DB session — that is the real requirement the `__new__` bypass was meeting badly. |
| Ownership of flattening | Engine absorbs `_serving_members` | It is already `@staticmethod` "by contract" (`circuit_service.py:624-627`) precisely because two callers needed it; the engine is where that contract belongs. |
| Claim set | Engine computes it | Feature 19 needs it; a second implementation would reintroduce the drift being removed. |
| Hazards | NOT moved in this feature | Already surfaced at activation; the dial discarding them is a separate recorded gap (F14 R3). |

---

## 2. System Architecture

```
                       CircuitSteeringEngine(registry)
                        |          |            |
          serve()  <----+          |            +----> claim_set()
                                   |                       |
      +----------------------------+---------------+       |
      |                            |               |       v
CircuitService._serve_full   set_intensity   InferenceService dial   Feature 19 contention
   (:424 today)                (:799 today)        (:955 today)
```

After extraction, `set_circuit_steering` has exactly one caller: the engine.

---

## 3. The Engine (`millm/services/circuit_steering_engine.py`, NEW)

```python
# millm/services/circuit_steering_engine.py
class CircuitSteeringEngine:
    """The ONE derivation of "serve this circuit".

    Constructed with the attachment registry and nothing else, so the
    per-request dial can use it without pulling request-scoped DI into the
    inference hot path — the legitimate need that
    ``InferenceService._sae_service_for_dial`` (inference_service.py:743) was
    meeting by calling ``SAEService.__new__`` and hand-setting one field.
    """

    def __init__(self, state: "AttachedSAEState") -> None:
        self._state = state

    @staticmethod
    def serving_members(definition) -> list["CircuitMember"]:
        """Flatten to the Feature 12 serving shape.

        Absorbed verbatim from CircuitService._serving_members. Two rules are
        load-bearing and MUST NOT be simplified:
          * a cluster_ref contributes its frozen expanded_members AND its own
            feature when both are present — taking one silently drops authored
            members from the intervention;
          * duplicates on (layer, feature_idx) are collapsed, because the
            serving path rejects a repeated key outright.
        """

    def claim_set(self, definition) -> set[int]:
        """Layers this circuit's serving members reach (Feature 19)."""
        return {m.layer for m in self.serving_members(definition)}

    def serve(self, definition, intensity: float) -> "CircuitSteeringResult":
        """Apply every member through ITS OWN layer's SAE at `intensity`."""
```

---

## 4. Call-Site Migration

| Site | Today | After |
|---|---|---|
| `_serve_full` | parses, flattens, dumps edges, calls `set_circuit_steering` (`circuit_service.py:424`) | `engine.serve(definition, intensity)` |
| `set_intensity` | same again (`:799`) | `engine.serve(definition, intensity)` |
| Per-request dial | same again via the `__new__` bypass (`inference_service.py:955`) | `engine.serve(definition, lam)` |

`InferenceService._sae_service_for_dial` (`:743`) is **deleted**, along with the `SAEService.__new__`
call it exists to make. `InferenceService._circuit_serving_members` becomes a thin delegate to the
engine or is deleted with its callers updated.

---

## 5. Preserving the sign rule

`_directional_budget` (`sae_service.py:66`) implements the canonical rule: a **negative strength is
already directional**, so multiplying by `sign` double-negates suppression into amplification. It is
applied at `sae_service.py:632` inside `set_circuit_steering` and at `:751` for hazards. The engine does
**not** re-implement it — it calls `set_circuit_steering`, which owns it. This is stated explicitly
because "consolidating serving" reads like an invitation to move the math, and moving it is how the
double-negation bug returns.

---

## 6. Testing Strategy

| Level | Coverage |
|---|---|
| Characterization | Pin all three sites' current behaviour BEFORE extraction; they must be byte-identical after |
| Unit — flattening | Engine output equals `_serving_members` for: plain members, cluster_ref with expanded only, cluster_ref with BOTH (EC-18.5), duplicate keys (EC-18.3), negative strengths (EC-18.4) |
| Unit — construction | Constructible with only the registry; no repository, no session (EC-18.2) |
| Unit — claim set | Matches the layers `serve` actually touches, for full and single-layer circuits |
| Regression | F14's R1-01 and R2-01 tests still pass — they pin what the duplication broke |
| Structural | `set_circuit_steering` has exactly ONE caller; no `__new__` bypass in the serving path |
| **Mutation** | Removing dedupe, or the both-sources collection, MUST fail a test |

---

## 7. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| A behaviour-preserving refactor changes behaviour | High | Characterization tests green before the move; the F14 regression tests are the specific tripwires |
| The sign rule gets "tidied" during consolidation | High | §5 states it explicitly; a mutation test on `_directional_budget` |
| Slice-fallback accidentally routed through the engine | Medium | EC-18.1; slice serving stays on the cluster path and a test asserts it |
| The engine grows a repository later, re-creating the dial's problem | Medium | Constructor takes the registry only; a test asserts the dial constructs it without DI |
