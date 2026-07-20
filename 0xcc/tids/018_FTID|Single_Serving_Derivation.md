# Technical Implementation Document: Single Circuit-Serving Derivation

## miLLM Feature 18

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**References:** `018_FPRD|Single_Serving_Derivation.md` · `018_FTDD|Single_Serving_Derivation.md` · `0xcc/reviews/review_feature014_circuit_dial_2026-07-20.md` · `..._R2_2026-07-20.md`

---

## 1. File Structure

```
millm/
├── ml/circuit_steering.py               (NEW — ServingPlan + CircuitSteeringEngine; the ONE derivation)
├── services/circuit_service.py          (MOD — _serving_members DELETED :623-662; _serve_full :415-433 and
│                                          set_intensity :798-803 consume plan_for/apply; engine held on self)
├── services/inference_service.py        (MOD — _circuit_serving_members DELETED :720-731;
│                                          _sae_service_for_dial DELETED :733-745; dial :890-985 and
│                                          _steering_circuit_uncached :806-822 consume plan_for)
├── services/sae_service.py              (UNCHANGED — set_circuit_steering :481-513 and _directional_budget
│                                          :66-76 are called, never edited)
tests/unit/ml/test_circuit_steering_engine.py          (NEW — characterization + claim set + construction)
tests/unit/services/test_circuit_service.py            (MOD — calls the engine, not _serving_members)
tests/unit/services/test_circuit_dial.py               (MOD — same)
tests/integration/test_single_serving_derivation.py    (NEW — four-way identity + F14 regressions + reachability)
```

## 2. Load-Bearing Implementation Points (verified against live code)

- **There are FOUR derivations, not three.** The BRD and PPRD name `circuit_service.py:424`, `:799` and `inference_service.py:955`. `grep -n "_serving_members\|set_circuit_steering\|_circuit_serving_members"` also returns `inference_service.py:806-822` (`_steering_circuit_uncached`), which independently calls `_circuit_serving_members` (`:813`), builds `member_layers = {m.layer for m in members}` (`:819`) and tests it against `AttachedSAEState().entries()` (`:820`) to decide whether a rung header may be emitted. That is a full serving derivation on the EVIDENCE surface. It is in scope. Leaving it out would preserve F14-R2-02's defect class exactly where an overclaim is worst.

- **`_serving_members` is `circuit_service.py:623-662` and moves verbatim.** The body to preserve: `sources = list(m.expanded_members or [])` then `if m.feature is not None: sources.append(m.feature)` (`:644-646`) — BOTH, not either; `key = (m.layer, feat.feature_idx)` with `if key in seen: continue` (`:648-650`) — first wins; `sae_id` from `definition.sae_for_layer(m.layer)` (`:642-643`). Its docstring states the reason the dedupe is load-bearing: *"the serving path rejects a repeated key outright"* — confirmed at `sae_service.py:562-572`, which appends a `duplicate_member` offender and, via the fail-closed offender collection, turns a valid circuit into a 422. **Do NOT "simplify" the dedupe.**

- **The `@staticmethod`-by-contract comment records the workaround being retired.** `circuit_service.py:626-631` explains it: *"the inference dial calls this to flatten members by exactly the same rules activation uses. R2 flagged that calling it unbound relied on an unwritten purity promise — one future `self.` reference would turn every dialled request into an AttributeError."* The mirror is `inference_service.py:730`: `# _serving_members is a @staticmethod by contract — see its docstring.` Both disappear. The RULES paragraph (`:633-637`) moves to the engine; the purity-promise paragraph does not, because an instance method on an honestly-constructed engine needs no promise.

- **`_sae_service_for_dial` is `inference_service.py:733-745` and is the bypass.** `svc = SAEService.__new__(SAEService)` (`:743`) then `svc._sae_state = AttachedSAEState()` (`:744`) — and nothing else. Compare `SAEService.__init__` (`:348-382`), which also sets `self.repository`, `self.emitter`, `self._inference_service`, `self._downloader`, `self._loader`, `self._hooker`, `self._active_downloads` and `self._cancelled_downloads`. Every one of those is absent on the bypassed instance. The docstring's claim — *"`set_circuit_steering` touches nothing but the singleton registry and the attached SAEs"* — is true **today**, and is a promise about a method the author of the next `set_circuit_steering` edit will not read. The failure mode is specific: the dial wraps the call in `except Exception` (`:960-967`), so an `AttributeError` from a newly-read field is caught, logged as `circuit_dial_apply_failed`, restored, and the request proceeds **unsteered while `active_circuit_rung()` still emits `X-miLLM-Circuit-Rung`** — a response advertising evidence for an intervention that did not happen.

- **The intensity resolution that produced F14-R1-01 is `circuit_service.py:421-423`** — `definition.budget.intensity if definition.budget else circuit.intensity`. The dial divided by `circuit.intensity` instead. The live code now carries the post-mortem inline at `inference_service.py:937-947`, including *"`_serve_full` applies `definition.budget.intensity`, which is a DIFFERENT field from `circuit.intensity` (the DB column)"*. Move that expression into `serving_intensity()` and delete both readers.

- **The F14-R2-01 comment marks the snapshot line to rewrite.** `inference_service.py:890-895` reads: *"R2: derive the participating layers from the DEFINITION, the same source the apply below uses. Keying the snapshot on circuit.layers (the DB column) while applying to the definition's member layers let any layer present in one and not the other be dialled but never restored… The two must not be allowed to drift."* Today `member_layers` (`:904`) and the apply's `members` (`:899`) come from two calls that happen to match. After the move both come from one `ServingPlan`, so the comment's final sentence becomes a type invariant rather than an instruction.

- **The canonical sign rule is `sae_service.py:66-76` and must not be copied.** `_directional_budget(budget, sign)` returns `b if b < 0 else float(sign) * b`, with the docstring: *"A NEGATIVE budget is already directional — the `sign` field is redundant there and must NOT be multiplied in (doing so double-negates a suppression into an amplification)."* It is applied once, at `sae_service.py:632` (`raw = _directional_budget(m.budget, m.sign) * intensity`) and in the hazard sign derivation at `:751-752`. It is shared with `cluster_service`. **The engine passes `budget` and `sign` through untouched and never computes an effective strength itself** — if the engine ever multiplies, there are two sign rules and this feature has caused the class of bug it exists to remove.

- **`set_circuit_steering` owns the lock and keeps it.** `sae_service.py:509-513` acquires `_ATTACHMENT_LOCK` and delegates to `_set_circuit_steering_locked`, with the comment *"Hold the composing lock across resolve→apply: a concurrent detach/attach between the two would otherwise write steering into a just-detached SAE."* The engine derives OUTSIDE the lock (pure, cheap) and applies INSIDE it (via the unchanged method). Do not widen the lock to cover derivation and do not add a second lock.

- **The echo memo is orthogonal — preserve it.** `_STEERING_CIRCUIT_MEMO` is a ContextVar (`inference_service.py:53-57`) read and set in `_steering_circuit` (`:798-804`) around `_steering_circuit_uncached`. It caches the predicate's *result* per request, not a derivation. Rewiring the uncached body to `plan_for` must leave the memo wrapper exactly as it is; folding the memo into the engine would give the engine request state, which it must not have.

- **Existing tests that call the moved symbols.** `tests/unit/services/test_circuit_service.py`, `test_circuit_dial.py`, `test_circuit_steering.py` and `tests/integration/test_circuit_dial_workflow.py` are the files to expect breakage in. `test_circuit_steering.py` exercises `set_circuit_steering` directly and should need NO change — if it does, the apply was edited and the feature's premise is violated.

## 3. Key Implementations

```python
# millm/ml/circuit_steering.py — the moved flattening (rules preserved verbatim)
def serving_members(self, definition: "CircuitDefinitionV1") -> list["CircuitMember"]:
    """Flatten the circuit's members into the Feature 12 serving shape.

    A ``cluster_ref`` contributes its frozen ``expanded_members`` AND its own
    ``feature`` when both are present — taking only one silently dropped
    authored members from the intervention. Duplicates on a
    ``(layer, feature_idx)`` are collapsed because the serving path rejects
    a repeated key outright.
    """
    out: list[CircuitMember] = []
    seen: set[tuple[int, int]] = set()
    for m in definition.members:
        ref = definition.sae_for_layer(m.layer)
        sae_id = ref.mistudio_sae_id if ref else None
        sources = list(m.expanded_members or [])
        if m.feature is not None:
            sources.append(m.feature)              # BOTH sources — never either/or
        for feat in sources:
            key = (m.layer, feat.feature_idx)
            if key in seen:
                continue                            # first wins; 422 if this leaks through
            seen.add(key)
            out.append(CircuitMember(
                feature_idx=feat.feature_idx, layer=m.layer,
                budget=feat.strength,               # raw; sign rule applied at APPLY time only
                sign=feat.sign, sae_id=sae_id, label=feat.label,
            ))
    return out
```

```python
# millm/ml/circuit_steering.py — the two derivations that produced F14's criticals
def serving_intensity(self, circuit, definition) -> float:
    """The ONE answer to document-vs-column (EC-18.4 / F14-R1-01).

    ``definition.budget.intensity`` is the DOCUMENT's field; ``circuit.intensity``
    is the DB dial column. They differ routinely — set_intensity writes only the
    column. Callers must not choose between them.
    """
    if definition.budget is not None:
        return float(definition.budget.intensity)
    return float(circuit.intensity)

def claim_set(self, definition) -> frozenset[int]:
    """The layers this circuit's SERVING MEMBERS reach — F19's contention input.

    Defined as the layers of `serving_members`, not an equivalent separate walk,
    so activation and contention agree by construction (FR-18.3). NEVER
    `circuit.layers` (the DB column) — that mismatch was F14-R2-01.
    """
    return frozenset(m.layer for m in self.serving_members(definition))

def plan_for(self, circuit, definition, *, intensity: float | None = None) -> ServingPlan:
    members = self.serving_members(definition)
    claimed = frozenset(m.layer for m in members)          # same list — cannot drift
    attached = frozenset(e.layer for e in self._state.entries() if e.layer in claimed)
    lam = self.serving_intensity(circuit, definition) if intensity is None else float(intensity)
    return ServingPlan(members=members, intensity=lam,
                       claimed_layers=claimed, attached_layers=attached)
```

```python
# millm/ml/circuit_steering.py — apply; the seam that retires SAEService.__new__
def apply(self, plan: ServingPlan, definition, *, sae_service=None):
    """Apply through the UNCHANGED SAEService.set_circuit_steering.

    `sae_service` is passed by CircuitService (DI-constructed). The per-request
    dial passes none: the engine already holds the registry, which is the only
    thing the apply needs, so no half-built service is required.
    """
    from millm.services.sae_service import SAEService
    svc = sae_service if sae_service is not None else SAEService.for_registry(self._state)
    return svc.set_circuit_steering(                 # lock, clamp, hazards all unchanged
        plan.members, plan.intensity,
        edges=[e.model_dump(mode="json") for e in definition.edges],
    )
```

```python
# millm/services/sae_service.py — the honest replacement for the __new__ bypass
@classmethod
def for_registry(cls, state: "AttachedSAEState") -> "SAEService":
    """A steering-only SAEService: every field set, none left absent.

    Replaces `SAEService.__new__(SAEService)` at inference_service.py:743, whose
    unset fields made any future field read an AttributeError swallowed by the
    dial's `except Exception` — a silently unsteered response still carrying a
    rung header. Construction here is total: a field added to __init__ that this
    path cannot supply is a startup failure, not a runtime lie.
    """
    svc = cls(repository=None, cache_dir=None, emitter=None, inference_service=None)
    svc._sae_state = state
    return svc
```

## 4. Implementation Pitfalls

1. **Characterize BEFORE moving.** ENG-V1 is sequenced first for a reason: a behaviour-preserving move is only provable if the behaviour was pinned while the old code still ran. Write `tests/unit/ml/test_circuit_steering_engine.py` against `CircuitService._serving_members` first, watch it pass, then repoint it at the engine. A test written after the move pins the new behaviour, whatever it is.
2. **Both sources, always.** `expanded_members` AND `feature` (EC-18.1). The tempting `sources = m.expanded_members or [m.feature]` is wrong and silently drops authored members from a live intervention.
3. **Dedupe first-wins, and it is load-bearing.** `sae_service.py:562-572` 422s on a repeated `(layer, feature_idx)` with `duplicate_member`. Dropping the dedupe does not produce a double-applied member; it produces a rejected circuit (EC-18.2).
4. **Never multiply by `sign` in the engine.** The rule lives once, at `sae_service.py:66-76`, and is applied once, at `:632`. A negative budget is already directional; multiplying double-negates suppression into amplification (EC-18.3). The engine carries `budget` and `sign` through untouched.
5. **Claim set from members, never from `circuits.layers`.** The DB column keeps its display and query uses and loses its serving read (EC-18.5). This is the structural close of F14-R2-01 — if any new code path reads `circuit.layers` to decide what to steer, snapshot or restore, the defect is back.
6. **The dial's snapshot must come from `plan.claimed_layers`.** Not from a second `{m.layer for m in members}` comprehension, even one that would compute the same set. The whole point is that the saved set and the applied set are the same object's field.
7. **Delete both old flatteners; do not shim.** `CircuitService._serving_members` AND `InferenceService._circuit_serving_members` (ENG-D4). A delegating shim is a call site with a body, and bodies grow.
8. **Do not touch `set_circuit_steering`.** If `tests/unit/services/test_circuit_steering.py` needs edits, the apply was changed and the feature's behaviour-preservation premise is broken. Stop and reconsider rather than updating the test.
9. **Leave the ContextVar memo alone.** `_STEERING_CIRCUIT_MEMO` caches the predicate result per request (`inference_service.py:53-57`, `:798-804`). The engine must stay stateless; moving the memo into it gives a process-singleton request state, which is a cross-request leak.
10. **`for_registry` must be total.** If `SAEService.__init__` cannot accept `None` for the repository and cache dir today, widen those parameters explicitly rather than reintroducing a partial construction by another name (ENG-C3). The test that matters asserts no reachable method reads an unset field — not that the object was created.
11. **Reachability tests must cut the wiring.** Per BR-005 and the PADR's reachability decision: the test must FAIL when the engine call is removed. Feature 15 shipped a `TestRingPruningIsWired` that asserted an entry point existed while nothing called it — the precise anti-pattern excluded here.
12. **Four call sites, not three.** The echo predicate at `inference_service.py:806-822` is the one most easily missed and the one where a stale derivation does evidence-grade harm.

## 5. Config Additions

None. This feature adds no configuration key, no environment variable and no feature flag. `CIRCUIT_INTENSITY_MIN`/`CIRCUIT_INTENSITY_MAX` (`millm/core/config.py:92-93`) keep their current meaning and their current clamp site inside `_set_circuit_steering_locked` (`sae_service.py:525-535`) — the engine does not pre-clamp λ, because a second clamp is a second derivation.
