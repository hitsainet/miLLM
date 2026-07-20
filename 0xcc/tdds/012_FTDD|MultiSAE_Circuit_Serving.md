# Technical Design Document: Multi-SAE Attach & Circuit Serving

## miLLM Feature 12

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**References:** `012_FPRD|MultiSAE_Circuit_Serving.md` · `000_PADR|miLLM.md` (v1.2) · miStudio steering reference (`steering_service.py::_register_steering_hooks`)

---

## 1. Executive Summary

Feature 12 generalizes miLLM's single-attached-SAE runtime into a `{(sae_id, layer) → (LoadedSAE,
hook_handle)}` registry, then serves a circuit by grouping members per `(sae_id, layer)` and applying each
layer's own decoder — exactly the multi-hook pattern miStudio validated in `_register_steering_hooks`. No
new decoder math is introduced: `LoadedSAE.apply_steering` already does
`modified = original + Σ strength·W_dec[idx,:]`, matching miStudio. What changes is **hook multiplicity**
(one hook per referenced SAE/layer, each bound to its own SAE) and **per-layer binding** (a member on
layer L never touches another layer's basis). Attachment state stays in-memory (no DB table). The one hard
new invariant is honest incompleteness: any member whose layer has no attached SAE blocks with
`SAE_SET_INCOMPLETE` (422) at both submit and activation.

### Key Technical Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Attachment state | Generalize `AttachedSAEState` singleton → registry keyed by `(sae_id, layer)` | Multi-SAE requires plural state; still process/GPU state, not persisted |
| Hook multiplicity | One `SAEHooker.install` per referenced `(sae_id, layer)`, each bound to its own SAE | A layer-L member must steer through the L-decoder; mirrors miStudio `_register_steering_hooks` |
| Decoder math | Reuse `apply_steering` unchanged | Already `original + Σ strength·W_dec[idx,:]` — identical to miStudio; only multiplicity/binding change |
| dtype | Attach steering weights in fp16 (`SAELoader.load(..., dtype=fp16)`) | Measured 128 MB (fp16) vs 256 MB (fp32) for two Gemma-2-2B SAEs — fp16 fits the <200 MB envelope |
| Incompleteness | Hard block `SAE_SET_INCOMPLETE` at submit + activation | Never serve a member through a wrong-layer decoder (BR-006/RSK-006) |
| Budgets | Per-layer `freq-budget/sim-alloc/per-layer@1` under one λ; γ=0 ⇒ B=B_dir | Validated per-layer allocation reused; joint calibration deferred (BRD) |
| Hazards | Compute at activation, label `validated:ES=…`/`heuristic:weight_prior=…`, return only | Detection-not-correction (miStudio hazards-v2); config never mutated |
| Storage | No new table — attachment is in-memory | Attachment is GPU/process state; the definition is stored by Feature 013 |

## 2. System Architecture

```
 ┌──────────────┐  attach-set / serve  ┌────────────────────┐    ┌──────────────────────────┐
 │ Circuits tab │ ───────────────────► │ /api/sae/attach-*  │──► │ SAEService (generalized)  │
 │ (admin-ui)   │  plural status       │ /api/sae/attachments│   │  registry{(sae_id,layer)} │
 └──────────────┘ ◄─────────────────── └────────────────────┘    └───────────┬──────────────┘
                                                                              │ per referenced (sae,layer)
                                                                  ┌───────────▼──────────────┐
                                                                  │ SAEHooker.install(model,  │
                                                                  │   layer, sae) → handle    │  (one per layer)
                                                                  └───────────┬──────────────┘
   set_circuit_steering(members, λ):                                          │
     group members by (sae_id, layer)                             ┌───────────▼──────────────┐
     bounds-gate + SAE_SET_INCOMPLETE                             │ LoadedSAE.apply_steering  │
     each SAE.set_steering_batch(its members)                     │  orig + Σ str·W_dec[idx]  │
     hazards (validated/heuristic) → returned                     └───────────────────────────┘
```

## 3. Database Design

**No schema change.** Attachment state is the in-memory `AttachedSAEState` singleton, generalized from four
scalar fields to a dict. There is no persisted attachment table (there never was — attachment is
process/GPU state re-established on restart by re-attach). The circuit definition and its per-layer budgets
are persisted by Feature 013 (a circuit/profile row); Feature 012 reads them at serve time.

Generalized singleton (in-memory only):

```python
# millm/services/sae_service.py — AttachedSAEState generalized (was 4 scalar fields, ~lines 82–160)
@dataclass
class AttachedEntry:
    sae: LoadedSAE
    sae_id: str
    layer: int
    hook_handle: Any
    steering_enabled: bool = False
    monitoring_enabled: bool = False

class AttachedSAEState:
    # registry keyed by (sae_id, layer) → AttachedEntry (replaces _attached_sae/_id/_layer/_hook_handle)
    _entries: dict[tuple[str, int], AttachedEntry]
    def set(self, sae, sae_id, layer, hook_handle) -> None: ...   # per-key; removes prior hook on that key
    def clear(self, sae_id=None, layer=None) -> None: ...         # one key, or all
    def get(self, sae_id, layer) -> AttachedEntry | None: ...
    def by_layer(self, layer) -> AttachedEntry | None: ...        # the (unique) SAE serving layer L
    def entries(self) -> list[AttachedEntry]: ...                  # plural status source
```

Backward compatibility: the singular `AttachmentStatus` (FPRD §3 MSA-A4) becomes a plural
`AttachmentStatusSet{entries: list[...], total_memory_mb, vram_warning}`; the legacy singular fields are
derived from `entries[0]` for existing clients until Feature 013 UI supersedes them.

## 4. Service Design

```python
# millm/services/sae_service.py — serving core (new method; groups by (sae_id, layer))
class SAEService:
    def set_circuit_steering(self, members: list[CircuitMember],
                             intensity: float) -> CircuitSteeringResult:
        """Apply a circuit: every member through ITS OWN layer's SAE.

        1. Resolve each member's layer → attached SAE via state.by_layer(layer).
        2. Any member whose layer has no attached SAE → collect as offender.
           If offenders: raise SAESetIncompleteError({feature_idx, layer, sae_id}...) (422).
        3. Group members by (sae_id, layer). Bounds pre-check: max(idx) < that SAE.d_sae,
           else block (never a 500 from set_steering_batch).
        4. Per layer: effective = clamp_steering(budget * sign * intensity); B=B_dir (γ=0).
           sae.set_steering_batch(effective_dict); sae.enable_steering(True).
        5. Compute cross-layer hazards (compounding/cancellation); label validated:/heuristic:.
        6. Return CircuitSteeringResult(applied_per_layer, hazards, clamp_warnings) — NEVER mutate config.
        """
```

Grouping mirrors miStudio's `_register_steering_hooks`: one hook per `(sae_id, layer)`, each hook bound to
its own `LoadedSAE`, each SAE receiving only the members whose `layer` matches. `SAEHooker.install` is
already single-SAE/single-layer (`install(model, layer, sae) -> handle`) — it is called once per referenced
layer; no change to the hooker itself beyond being invoked N times.

```python
# millm/services/sae_service.py — hazards (detection only)
def _cross_layer_hazards(self, members, intensity) -> list[Hazard]:
    """Compounding: same-sign members across layers push the residual the same way (super-additive
    risk). Cancellation: opposite-sign cross-layer members. Quantified from the definition's
    validated effect size where present (label 'validated:ES=<es>'); else the weight-prior heuristic
    (label 'heuristic:weight_prior=<w>'). Returns warnings; the caller surfaces them and applies nothing."""
```

Attach path (referenced-only, fp16):

```python
# millm/services/sae_service.py — attach a referenced set
async def attach_set(self, sae_layers: list[tuple[str, int]]) -> AttachmentStatusSet:
    for sae_id, layer in sae_layers:                       # referenced-only: exactly what the circuit needs
        sae = self._loader.load(path, device="cuda", dtype=torch.float16)   # fp16 steering weights
        handle = self._hooker.install(self._model, layer, sae)             # one hook per (sae_id, layer)
        self._state.set(sae, sae_id, layer, handle)
    total = sum(e.sae.estimate_memory_mb() for e in self._state.entries())
    return AttachmentStatusSet(entries=..., total_memory_mb=total,
                               vram_warning=(total > VRAM_ENVELOPE_MB))       # EC-12.3
```

## 5. API Design

Routes per FPRD §5 on the existing SAE router (`millm/api/routes/management/sae.py`); DI via the existing
`SAEServiceDep` (`millm/api/dependencies.py:234`). All responses in the `ApiResponse` envelope.
`SAESetIncompleteError` maps to `ApiResponse.fail(code="SAE_SET_INCOMPLETE", ...)` with the offender list
in `details` (422). Circuit activation (which calls `set_circuit_steering`) is the Feature 013 route;
Feature 012 owns the service method + the attach/attachments routes.

## 6. Admin UI Design

- **Circuits** tab (created by Feature 013; Feature 012 contributes the attachment panel): one chip per
  `(sae_id, layer)` with memory_mb, a total-memory readout, and a VRAM-warning badge when over envelope.
- Attachment status consumed via a plural `useAttachments` hook; the singular attachment display in the
  existing SAE panel keeps working (derives from `entries[0]`).
- `SAE_SET_INCOMPLETE` and hazard warnings render on the Feature 013 activation surface (offender list +
  `validated:`/`heuristic:` labels).

## 7. Testing Strategy

### Unit Tests
- `tests/unit/services/test_attached_state_registry.py`: set/clear/idempotent-reattach per key;
  `by_layer` uniqueness; `entries()` plural; orphaned-hook removal on re-attach.
- `tests/unit/services/test_circuit_steering.py`: group-by-`(sae_id, layer)`; bounds gate;
  `SAE_SET_INCOMPLETE` offender list; `clamp_steering(budget·sign·λ)`; γ=0 ⇒ B=B_dir; hazard labeling
  (`validated:ES=…` vs `heuristic:weight_prior=…`); config-not-mutated assertion.
- `tests/unit/services/test_attach_set.py`: referenced-only loading; fp16 dtype passed to loader;
  total memory + VRAM warning threshold (EC-12.3).

### Integration Tests
- `tests/integration/test_multisae_serving.py`: attach two SAEs → serve circuit → per-layer
  `get_steering_values()` equals λ-clamped expectation on EACH layer; incomplete-set 422; detach clears
  per-layer; single-layer degenerate case; two-member-same-layer conflict rejected (EC-12.2).

### Perf (env-gated, GPU host)
- Latency delta single-SAE vs two-SAE serve (NFR-12.1); peak-VRAM harness (two Gemma-2-2B SAEs, fp16).

## 8. Risks
- **Wrong-basis serve (RSK-006):** the `by_layer` resolution + `SAE_SET_INCOMPLETE` gate at submit AND
  activation is the mitigation — no member ever reaches `set_steering_batch` on a mismatched SAE.
- **VRAM (RSK-001):** fp16 attach measured 128 MB for two SAEs (within <200 MB); over-envelope records a
  warning and defers to the Feature 013 slice fallback rather than failing hard.
- **Hook multiplicity leaks:** N hooks means N removals — `clear()` iterates all entries; re-attach removes
  the prior hook on that key (reuses the existing orphaned-hook-removal guard in `set()`).
- **Latency (RSK-003/BR-013):** per-layer hooks are O(n_layers) constant residual adds, no second pass;
  measured in the perf harness.
