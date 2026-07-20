# Technical Implementation Document: Multi-SAE Attach & Circuit Serving

## miLLM Feature 12

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**References:** `012_FPRD|MultiSAE_Circuit_Serving.md` · `012_FTDD|MultiSAE_Circuit_Serving.md`

---

## 1. File Structure

```
millm/
├── services/
│   └── sae_service.py                         (MOD — AttachedSAEState → registry; attach_set;
│                                                       set_circuit_steering; _cross_layer_hazards;
│                                                       plural AttachmentStatus)
├── ml/
│   ├── sae_hooker.py                           (UNCHANGED — install() called once per (sae_id, layer))
│   ├── sae_wrapper.py                          (UNCHANGED — apply_steering already matches miStudio)
│   └── sae_loader.py                           (UNCHANGED — load(..., dtype=fp16) already supported)
├── core/steering_range.py                      (REUSE — clamp_steering(), STEERING_RANGE=200.0, Feature 8)
├── api/
│   ├── schemas/sae.py                          (MOD — AttachmentStatusSet, AttachedEntry DTOs)
│   ├── routes/management/sae.py                (MOD — /attachments, /attach-set, /detach)
│   └── schemas/circuit.py                      (NEW-partial — CircuitMember DTO shared with Feature 013)
├── core/errors.py                              (MOD — SAESetIncompleteError)
admin-ui/src/
├── components/circuits/AttachmentPanel.tsx     (NEW — plural (sae_id,layer) chips + VRAM badge)
├── hooks/useAttachments.ts, services/sae.ts    (MOD — plural attachment shape)
tests/unit/services/test_attached_state_registry.py, test_circuit_steering.py, test_attach_set.py (NEW)
tests/integration/test_multisae_serving.py      (NEW)
```

## 2. Load-Bearing Implementation Points (verified against live code)

- **`AttachedSAEState` is the thing that generalizes** (sae_service.py:72–160). Today it holds four scalar
  fields — `_attached_sae`, `_attached_sae_id`, `_attached_layer`, `_hook_handle` — with `set()` and
  `clear()`. Replace them with `_entries: dict[(sae_id, layer) → AttachedEntry]`. `set()` already removes
  a prior hook before overwriting (lines 134–147, "orphaned_hook_removed_before_overwrite") — keep that
  guard PER KEY. `clear()` (lines 149–160) becomes per-key-or-all. Add `by_layer(layer)` — the unique SAE
  serving layer L — this is what circuit serving resolves each member through.
- **`AttachmentStatus` dataclass** (sae_service.py:39–51) is singular; it becomes a plural
  `AttachmentStatusSet` (list of per-SAE entries + total_memory_mb + vram_warning). Preserve the singular
  fields derived from `entries[0]` so existing SAE-panel clients don't break.
- **`SAEHooker.install(model, layer, sae) -> handle`** (sae_hooker.py:48–92) is ALREADY per-SAE/per-layer,
  each hook closing over its own `sae` via `_create_hook_fn(sae)` (line 79/117). Multi-SAE = call it once
  per referenced `(sae_id, layer)`; NO change to the hooker. `_get_layer` (line 202) resolves the layer
  module; `_create_hook_fn` binds the SAE. Each installed hook is independent.
- **`LoadedSAE.apply_steering`** (sae_wrapper.py:232–274) already does
  `modified = original + Σ strength·W_dec[idx,:]` (delta pre-built in `_rebuild_steering_delta`,
  lines 406–436) — IDENTICAL to miStudio. Per-layer binding is automatic: each SAE has its own
  `_steering_values`/`_steering_delta`, so calling `set_steering_batch` on layer L's SAE with only L's
  members is the whole mechanism. No decoder-math change.
- **`set_steering_batch` bounds check** (sae_wrapper.py:347–363) raises `ValueError` on `idx ≥ d_sae`. A
  raw call is a 500 risk — pre-check `max(idx) < state.by_layer(layer).sae.d_sae` in `set_circuit_steering`
  and return a structured block (reuses the Feature 8 bounds-gate precedent).
- **`SAELoader.load(path, device, dtype=None)`** (sae_loader.py:87–154) casts all four tensors to
  `target_dtype` (lines 127–132). Pass `dtype=torch.float16` at attach → fp16 steering weights → the
  measured 64 MB/SAE. `estimate_memory_mb` (sae_wrapper.py:770 / sae_config.py:280) reports per-SAE.
- **`clamp_steering` already exists** (`core/steering_range.py`, Feature 8) — reuse it for
  `clamp_steering(budget·sign·λ)`; do NOT re-implement the ±200 clamp.
- **DI unchanged:** `SAEServiceDep` (dependencies.py:234) already injects `SAEService`; the new methods ride
  on the existing service. Routes extend `routes/management/sae.py` (existing SAE router).

## 3. Key Implementations

```python
# millm/services/sae_service.py — generalized state (registry)
@dataclass
class AttachedEntry:
    sae: LoadedSAE
    sae_id: str
    layer: int
    hook_handle: Any
    steering_enabled: bool = False
    monitoring_enabled: bool = False

class AttachedSAEState:  # singleton, thread-safe (keep the existing _lock)
    _entries: dict[tuple[str, int], AttachedEntry]  # replaces the 4 scalar fields

    def set(self, sae, sae_id, layer, hook_handle) -> None:
        key = (sae_id, layer)
        with self._lock:
            prev = self._entries.get(key)
            if prev is not None and prev.hook_handle is not None:
                try: prev.hook_handle.remove()   # keep the orphaned-hook guard, PER KEY
                except Exception as e: logger.warning("error_removing_orphaned_hook", error=str(e))
            self._entries[key] = AttachedEntry(sae, sae_id, layer, hook_handle)

    def by_layer(self, layer: int) -> Optional[AttachedEntry]:
        matches = [e for e in self._entries.values() if e.layer == layer]
        return matches[0] if len(matches) == 1 else None   # ambiguity → None → EC-12.2 reject

    def clear(self, sae_id=None, layer=None) -> None:
        with self._lock:
            keys = list(self._entries) if sae_id is None else [(sae_id, layer)]
            for k in keys:
                e = self._entries.pop(k, None)
                if e and e.hook_handle is not None:
                    try: e.hook_handle.remove()
                    except Exception as ex: logger.warning("error_removing_hook", error=str(ex))
```

```python
# millm/services/sae_service.py — circuit serving core
def set_circuit_steering(self, members: list[CircuitMember], intensity: float) -> CircuitSteeringResult:
    # 1. Resolve + collect offenders (member layer has no unique attached SAE)
    offenders = [
        {"feature_idx": m.feature_idx, "layer": m.layer, "sae_id": m.sae_id}
        for m in members if self._state.by_layer(m.layer) is None
    ]
    if offenders:
        raise SAESetIncompleteError(offenders)          # → 422 SAE_SET_INCOMPLETE

    # 2. Group by (sae_id, layer); bounds-gate; apply per layer under one λ
    per_layer: dict[int, dict[int, float]] = {}
    clamp_warnings: list[str] = []
    for m in members:
        entry = self._state.by_layer(m.layer)
        if not (0 <= m.feature_idx < entry.sae.d_sae):
            raise SAESetIncompleteError([{"feature_idx": m.feature_idx, "layer": m.layer,
                                          "reason": "index_out_of_bounds"}])  # EC-12.1, not a 500
        eff = clamp_steering(m.budget * m.sign * intensity)   # B=B_dir (γ=0); shared ±200 clamp
        if abs(m.budget * m.sign * intensity) > STEERING_RANGE:
            clamp_warnings.append(f"feature {m.feature_idx}@L{m.layer} clamped to ±{STEERING_RANGE:g}")
        per_layer.setdefault(m.layer, {})[m.feature_idx] = eff

    for layer, steering in per_layer.items():
        sae = self._state.by_layer(layer).sae
        sae.set_steering_batch(steering)                 # only this layer's members
        sae.enable_steering(True)

    hazards = self._cross_layer_hazards(members, intensity)   # returned, NEVER applied
    return CircuitSteeringResult(applied_per_layer=per_layer, hazards=hazards,
                                 clamp_warnings=clamp_warnings)
```

```python
# millm/services/sae_service.py — hazard labeling (detection only)
def _cross_layer_hazards(self, members, intensity) -> list[dict]:
    out = []
    for haz in detect_cross_layer(members, intensity):   # compounding | cancellation
        es = haz.validated_effect_size                   # from the circuit definition, if present
        label = f"validated:ES={es:.3g}" if es is not None else \
                f"heuristic:weight_prior={haz.weight_prior:.3g}"
        out.append({"kind": haz.kind, "layers": haz.layers, "label": label,
                    "message": haz.message})             # SURFACE; caller applies nothing
    return out
```

```python
# millm/core/errors.py
class SAESetIncompleteError(Exception):
    """A circuit member's layer has no (unique) attached SAE — never serve through a wrong basis."""
    def __init__(self, offenders: list[dict]):
        self.offenders = offenders
        super().__init__(f"SAE set incomplete: {len(offenders)} member(s) have no attached SAE for their layer")
```

## 4. Implementation Pitfalls

1. **One hook per `(sae_id, layer)` — bind each to ITS OWN sae.** `_create_hook_fn(sae)` closes over the
   passed SAE; calling `install` in a loop is correct only if each iteration passes the layer's own SAE.
   Do NOT share one hook across layers.
2. **`by_layer` must be UNIQUE.** Two attached SAEs claiming the same layer (or a member with an ambiguous
   `(sae_id, layer)`) → reject (EC-12.2). A layer resolves to exactly one SAE, else `SAE_SET_INCOMPLETE`.
3. **Bounds-gate BEFORE `set_steering_batch`** — `set_steering_batch` raises `ValueError` on `idx ≥ d_sae`
   (sae_wrapper.py:358); a raw call surfaces as a 500. Pre-check against THAT layer's `d_sae` (not layer 0's).
4. **fp16 at attach, not later.** Pass `dtype=torch.float16` into `SAELoader.load`; `apply_steering` already
   re-casts the delta to the hidden-states dtype on first call (sae_wrapper.py:256–260) — the fp16 weights
   are the VRAM win, don't cast back up.
5. **Hazards NEVER mutate config.** `_cross_layer_hazards` returns warnings only; the steering dicts are
   already applied before hazards are computed and are not revisited (mirrors miStudio hazards-v2).
6. **Preserve the singular attachment shape.** Existing SAE-panel clients read the old `AttachmentStatus`
   fields; derive them from `entries[0]` so nothing breaks before Feature 013 UI supersedes them.
7. **`clear()` removes N hooks.** Detaching the whole set iterates every entry; a partial detach removes
   only that key's hook. Do not leave a stale entry after its hook is removed.
8. **γ=0 ⇒ B=B_dir.** Do not re-derive per-layer budgets — they are frozen as authored in the definition;
   serving only applies `clamp_steering(B_dir·sign·λ)`.

## 5. Config Additions (millm/core/config.py)

```python
MULTISAE_VRAM_ENVELOPE_MB: int = 200      # documented close-out envelope (fp16 two-SAE = 128 MB, within)
MULTISAE_ATTACH_DTYPE: str = "float16"    # steering-weight attach dtype (measured 64 MB/SAE)
CIRCUIT_INTENSITY_MIN: float = 0.0        # global λ floor (off)
CIRCUIT_INTENSITY_MAX: float = 2.0        # global λ ceiling (shared with Feature 14 dial)
```
