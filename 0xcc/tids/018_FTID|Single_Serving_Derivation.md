# Technical Implementation Document: Single Circuit-Serving Derivation

## miLLM Feature 18

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**References:** `018_FTDD|Single_Serving_Derivation.md` · `018_FPRD|Single_Serving_Derivation.md`

---

## 1. File Structure

```
millm/services/circuit_steering_engine.py   # NEW — the one derivation
millm/services/circuit_service.py           # 2 call sites -> engine; _serving_members moves out
millm/services/inference_service.py         # 1 call site -> engine; _sae_service_for_dial DELETED
tests/unit/services/test_circuit_steering_engine.py        # NEW
tests/unit/services/test_serving_characterization.py       # NEW (written FIRST)
```

---

## 2. Load-Bearing Implementation Points (verified against live code)

| Point | Location | Why it matters |
|---|---|---|
| Derivation #1 | `circuit_service.py:424` | `_serve_full` — `outcome = self._sae_service.set_circuit_steering(...)` |
| Derivation #2 | `circuit_service.py:799` | `set_intensity` — the same call again |
| Derivation #3 | `inference_service.py:955` | The dial — `self._sae_service_for_dial().set_circuit_steering(...)` |
| The bypass | `inference_service.py:743` | `svc = SAEService.__new__(SAEService)` then `svc._sae_state = AttachedSAEState()`. Its docstring justifies it: "a repository-free instance is sufficient — and avoids pulling request-scoped DI into the inference hot path." That NEED is legitimate; the mechanism is not. |
| `_serving_members` | `circuit_service.py:624`, `@staticmethod` "STATIC BY CONTRACT" (`:626-627`) | Already static *because* two callers needed it — evidence the engine is where it belongs |
| Sign rule | `_directional_budget`, `sae_service.py:66`; applied `:632` and `:751` | A negative strength is already directional; multiplying by `sign` double-negates |
| Registry | `AttachedSAEState`, `sae_service.py` | The engine's only dependency |

---

## 3. Key Implementations

### 3.1 Order of operations

1. Write characterization tests against all three sites and get them green.
2. Create the engine, absorbing `_serving_members` verbatim.
3. Migrate `_serve_full`, then `set_intensity`, then the dial — one at a time, suite green between each.
4. Delete `_sae_service_for_dial` and the `__new__` call.
5. Add the structural tests (one caller; no bypass).

### 3.2 What NOT to move

`set_circuit_steering` keeps the sign rule, the clamping, the per-layer resolution and the hazard
computation. The engine is a *caller*, not a reimplementation. "Consolidating serving" reads like an
invitation to move that math; moving it is how the double-negation bug returns.

### 3.3 The claim set

`claim_set(definition)` derives from `serving_members`, not from `circuit.layers` (the DB column). F14
R2-01 was exactly this confusion: the column and the definition's member layers can disagree, and the
snapshot keyed off the column while the apply keyed off the members.

---

## 4. Implementation Pitfalls

1. **Do not use `circuit.layers` for the claim set.** It is a cached DB column; the definition's members
   are the truth. This is the F14 R2-01 bug in a new location.
2. **Do not simplify `_serving_members`.** Both-sources collection and dedupe are each load-bearing and
   each look redundant in isolation.
3. **Do not move `_directional_budget`.** See §3.2.
4. **Do not route slice-fallback through the engine.** `_serve_slices` steers through the cluster
   profile path (EC-18.1).
5. **Delete the bypass, do not wrap it.** Leaving `_sae_service_for_dial` as a thin shim keeps the
   half-constructed-service pattern reachable and the structural test would pass anyway.
6. **Migrate one site at a time.** Three simultaneous migrations make a behaviour change impossible to
   attribute — the exact difficulty that made F14's defects expensive.

---

## 5. Config Additions

None.

---

## 6. Divergences from the BRD's assumptions

- The BRD describes the bypass as reaching "a half-constructed service"; verified — `__new__` skips
  `__init__` entirely and only `_sae_state` is set, so `repository`, `_emitter` and every other field
  are absent. It works today because `set_circuit_steering`'s call graph happens not to touch them,
  which is a runtime coincidence re-verified by nobody.
- The BRD says "three derivations". Confirmed exactly three call sites of `set_circuit_steering`
  outside its definition. `_serving_members` additionally has two callers inside `circuit_service.py`
  (`:413`, `:733`) plus the dial's delegate, so flattening is derived in more places than serving is.
