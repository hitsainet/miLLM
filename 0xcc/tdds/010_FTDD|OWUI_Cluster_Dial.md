# Technical Design Document: OWUI Cluster Dial

## miLLM Feature 10

**Document Version:** 1.0
**Created:** July 16, 2026
**Status:** Draft
**References:** `010_FPRD|OWUI_Cluster_Dial.md` · `008_FTDD|Cluster_Import.md` (λ/clamp semantics)

---

## 1. Executive Summary

The dial rides the machinery the per-request `profile` override already proved: request field →
serial routing → apply inside the queue semaphore → restore in `finally`. The change generalizes
`_apply_request_profile` into `_apply_request_steering(profile_name, intensity)` and adds symbolic-λ
resolution. The OWUI side is deliberately dumb: a Filter that injects one field.

### Key Technical Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Transport | Per-request body field (vendor extension) | `extra="ignore"` both directions; concurrency-safe |
| Resolution | Symbolic λ resolved server-side from active cluster's intensity_range | Plugin stays range-ignorant |
| Apply | Generalized `_apply_request_steering`; λ=0 ⇒ enable_steering(False) for the request | One code path for profile+dial |
| Routing | Field presence forces serial (extends has_profile) | CBM shares global SAE state |
| Plugin | Single-file Filter Function, inlet-only | No restore logic needed client-side |

## 2. System Architecture

```
 OWUI chat ──► Filter.inlet: body["steering_intensity"]="max" ──► POST /v1/chat/completions
                                                                        │
                                              ┌─────────────────────────▼──────────────────────┐
                                              │ InferenceService (serial queue semaphore)      │
                                              │  _resolve_intensity("max") → λ=range[1]        │
                                              │  _apply_request_steering(profile?, λ):         │
                                              │    base = named profile | active profile | live │
                                              │    λ==0 → save + enable_steering(False)        │
                                              │    else → save + set_steering_batch(           │
                                              │              {i: clamp(base_i·λ)})             │
                                              │  … generate …                                   │
                                              │  finally: _restore_request_profile(saved)      │
                                              └────────────────────────────────────────────────┘
```

## 3. Request Schema Change

```python
# millm/api/schemas/openai.py — beside `profile` (:60-61)
steering_intensity: Optional[Union[float, Literal["off", "min", "max"]]] = Field(
    default=None,
    description="miLLM extension: per-request cluster intensity λ (0..2) or off|min|max "
                "resolved against the active cluster's intensity range.")

@field_validator("steering_intensity")
def _validate_intensity(cls, v):
    if isinstance(v, (int, float)) and not (0.0 <= float(v) <= 2.0):
        raise ValueError("steering_intensity must be within [0, 2]")
    return v
```

## 4. Inference Service Design

```python
# millm/services/inference_service.py
def _resolve_intensity(self, raw, active_profile) -> float:
    if raw is None: return None
    if isinstance(raw, (int, float)): return float(raw)
    rng = ((active_profile.cluster_meta or {}).get("budget") or {}).get("intensity_range") \
          if active_profile else None
    lo, hi = (rng or (settings.CLUSTER_INTENSITY_MIN, settings.CLUSTER_INTENSITY_MAX))
    return {"off": 0.0, "min": float(lo), "max": float(hi)}[raw]

async def _apply_request_steering(self, profile_name: str | None,
                                  intensity_raw) -> SavedSteering | None:
    """Generalizes _apply_request_profile (:399). Base = named profile (repo.get_by_name)
    if given, else the ACTIVE profile row (raw λ=1 values), else live values.
    Saves current {values, enabled}; λ==0 → enable_steering(False);
    else set_steering_batch({i: clamp_steering(base_i * λ)}). Returns saved state."""
```
- Call sites: both generation paths where `_apply_request_profile` runs today (non-streaming :831-900
  block, streaming :1096-1175 block); the existing `_restore_request_profile` in `finally` is reused
  verbatim (same saved-state shape).
- Composition rule (DIAL-A7): `profile` selects the base; `steering_intensity` scales it; a request
  with only the dial scales the active/live base.
- Routing: `_use_cbm_for_request(..., has_profile=bool(request.profile) or
  request.steering_intensity is not None)` (:354 condition).
- Header echo: set `X-miLLM-Steering-Intensity: {λ:.3f}` next to the existing `X-miLLM-Backend`
  header (chat.py:78 area).

## 5. OWUI Function Design

```python
# integrations/openwebui/millm_dial_filter.py
"""miLLM Cluster Dial — Open WebUI Filter Function.
Install: Admin → Functions → Import; enable per model or globally."""
from pydantic import BaseModel
from typing import Literal, Optional

class Filter:
    class Valves(BaseModel):
        enabled: bool = True
    class UserValves(BaseModel):
        dial: Literal["default", "off", "min", "max"] = "default"

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        if not self.valves.enabled:
            return body
        dial = (__user__ or {}).get("valves", self.UserValves()).dial
        if dial != "default":
            body["steering_intensity"] = dial
        return body
```
No `outlet`: restoration is server-side per request. The file carries a header comment documenting the
miLLM version requirement and the EC-10.4 rollout property.

## 6. API Design
FPRD §5. No management routes added (global endpoint is Feature 8's).

## 7. Testing Strategy

### Unit (`tests/unit/services/test_request_intensity.py`, `tests/unit/api/test_openai_schemas.py` ext)
- Schema: numeric bounds, symbolic values, rejection messages.
- `_resolve_intensity`: range present / absent (config fallback) / no active profile.
- `_apply_request_steering`: λ=0 disable path; scaling+clamp parity with Feature 8 (shared helper);
  composition with `profile`; saved/restored shape.

### Integration (`tests/integration/api/test_chat_completions.py` ext)
- Field on streaming + non-streaming; serial routing asserted (backend header); global steering values
  byte-identical before/after; no-active-cluster no-op with logged notice; header echo.

### E2E (post-deploy)
- Scripted identical-prompt off/min/max comparison; OWUI plugin manual verification per manual steps.

## 8. Risks
- Filter API surface differences across OWUI versions → plugin kept to the stable Filter/inlet/Valves
  core; version note in the file header.
- Users conflating global vs per-request dials → header echo + manual section distinguish them.
