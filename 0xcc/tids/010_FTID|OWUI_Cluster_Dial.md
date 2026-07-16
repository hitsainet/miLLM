# Technical Implementation Document: OWUI Cluster Dial

## miLLM Feature 10

**Document Version:** 1.0
**Created:** July 16, 2026
**Status:** Draft
**References:** `010_FPRD|OWUI_Cluster_Dial.md` · `010_FTDD|OWUI_Cluster_Dial.md`

---

## 1. File Structure

```
millm/
├── api/schemas/openai.py                (MOD — steering_intensity field + validator)
├── api/routes/openai/chat.py            (MOD — header echo; pass field through)
├── services/inference_service.py        (MOD — _resolve_intensity, _apply_request_steering,
│                                         routing condition, call-site swaps)
integrations/openwebui/millm_dial_filter.py   (NEW — single-file OWUI Function)
manual/docs/tutorials/open-webui.md      (MOD — Function install + dial section)
tests/unit/services/test_request_intensity.py (NEW)
tests/unit/api/test_openai_schemas.py    (MOD — field cases)
tests/integration/api/test_chat_completions.py (MOD — dial cases)
```

## 2. Load-Bearing Implementation Points (verified)

- **The proven template is `_apply_request_profile`** (inference_service.py:399-488): applied inside
  the request-queue semaphore, saves `{values, enabled}` (:475-478), restores via
  `_restore_request_profile` in `finally` (:898, :1175). The dial generalizes THIS function — do not
  build a parallel mechanism.
- **Range check at :465 becomes clamp** — replace the reject with `clamp_steering()` from Feature 8's
  `millm/core/steering_range.py` (single source of truth; Feature 8 lands first).
- **Serial routing condition** at `_use_cbm_for_request` (:354-358): extend `has_profile` to include
  the dial field. CBM must never see a dialed request (global SAE state).
- **Schema tolerance**: `ChatCompletionRequest.model_config = {"extra": "ignore"}` (openai.py:63) is
  what makes rollout safe — do not tighten it.
- **Header pattern**: `X-miLLM-Backend` is set in chat.py:78-101 for both stream/non-stream — mirror
  for `X-miLLM-Steering-Intensity` (only when the field was present).
- **Active-profile base**: `ProfileRepository.get_active()` (profile_repository.py:115) returns the
  raw λ=1 steering dict; scale by resolved λ, NOT by `profile.intensity` (the request λ overrides the
  stored λ for this request — dial semantics).

## 3. Key Implementations

```python
# inference_service.py — call-site swap (both paths)
saved = None
try:
    if request.profile or request.steering_intensity is not None:
        saved = await self._apply_request_steering(
            request.profile, request.steering_intensity)
    ... generate ...
finally:
    if saved is not None:
        self._restore_request_profile(saved)
```

```python
# _apply_request_steering core (inside the semaphore)
active = await self._profile_repo.get_active() if profile_name is None else None
base_profile = (await self._profile_repo.get_by_name(profile_name)) if profile_name else active
lam = self._resolve_intensity(intensity_raw, base_profile or active)
sae = self._sae_service.get_attached_sae()
if sae is None:
    logger.info("steering_intensity ignored: no SAE attached")
    return None
saved = SavedSteering(values=sae.get_steering_values(), enabled=sae.is_steering_enabled)
if base_profile is None and lam is None:
    return None                                # nothing to do
base = (base_profile.get_steering_dict() if base_profile
        else {i: v for i, v in saved.values.items()})   # live values as λ=1 base
if lam == 0.0:
    sae.enable_steering(False)
    self._echo_intensity = 0.0
    return saved
lam = 1.0 if lam is None else lam
sae.set_steering_batch({i: clamp_steering(v * lam) for i, v in base.items()})
sae.enable_steering(True)
self._echo_intensity = lam
return saved
```

## 4. Implementation Pitfalls

1. **Request λ OVERRIDES stored λ** — do not multiply both (`intensity` column is the global dial's
   state; the request dial is absolute). EC test pins this.
2. **λ=None ≠ λ=1.0**: None means "field absent — leave live steering untouched"; 1.0 means "apply the
   base at unit intensity". The no-op path must not save/restore needlessly (perf + log noise).
3. **No-active-cluster + dial**: manual active profile scales fine; NO profile at all with only a dial
   → treat live values as base (or no-op if steering disabled) — never 500, never enable steering that
   wasn't configured.
4. **Restore must run on stream cancellation** — the existing `finally` placement already covers
   client disconnects mid-stream; keep the swap inside the same try/finally, don't move it.
5. **Do not import OWUI types in the plugin** beyond pydantic — the file must run inside OWUI's
   sandbox with no miLLM dependencies.
6. **Header echo on streaming**: headers go out before the body — compute λ at apply time and stash it
   (e.g. contextvar/request-scoped attr) before `StreamingResponse` is constructed.

## 5. Config
Uses Feature 8's `CLUSTER_INTENSITY_MIN/MAX` fallbacks; nothing new.
