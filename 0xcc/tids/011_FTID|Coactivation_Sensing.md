# Technical Implementation Document: Co-Activation Sensing

## miLLM Feature 11

**Document Version:** 1.0
**Created:** July 16, 2026
**Status:** Draft
**References:** `011_FPRD|Coactivation_Sensing.md` · `011_FTDD|Coactivation_Sensing.md`

---

## 1. File Structure

```
millm/
├── ml/sae_wrapper.py                    (MOD — SensingConfig/SensedHit, arm/disarm/_sense/begin/collect)
├── ml/sae_hooker.py                     (MOD — one armed branch in hook_fn)
├── services/sensing_service.py          (NEW)
├── services/inference_service.py        (MOD — begin/collect + _notify_sensing, IdCaptureStoppingCriteria,
│                                          routing condition)
├── services/profile_service.py          (MOD — arm/disarm on activate/deactivate)
├── services/sae_service.py              (MOD — disarm on detach)
├── db/models/sensing_event.py           (NEW)
├── db/repositories/sensing_repository.py (NEW)
├── db/migrations/versions/008_create_sensing_events_table.py (NEW)
├── api/schemas/sensing.py               (NEW)
├── api/routes/management/sensing.py     (NEW)  + dependencies.py, routes/__init__.py (MOD)
├── sockets/progress.py                  (MOD — emit_sensing_event)
├── core/config.py                       (MOD — SENSING_* keys)
admin-ui/src/components/clusters/sensing/{SensingPanel,SensingEventDetail}.tsx (NEW)
admin-ui/src/services/sensing.ts, hooks/useSensing.ts (NEW)
tests/unit/{ml,services,db}/test_sensing*.py (NEW)
tests/integration/test_sensing_workflow.py (NEW)
```

## 2. Load-Bearing Implementation Points (verified against live code)

- **Hook insertion point** — `sae_hooker.py::hook_fn` :159-165 currently guards
  `if sae.is_monitoring_enabled: ... sae._capture_activations(sae.encode(x))`. Sensing branch is a
  SIBLING (evaluate even when monitoring is off), placed BEFORE `apply_steering` (:168) so positions
  reflect the pre-steer residual read.
- **Do NOT reuse `_capture_activations`** (sae_wrapper.py:466-479): it compacts columns positionally to
  `_monitored_features` (captured column i = monitored list order, NOT feature i) and only the last
  pass survives (`_last_feature_acts_per_item` overwritten per pass). Both properties are fatal for
  sensing. The member-only `_sense` encode avoids both.
- **`suppressed()`** (sae_wrapper.py:449) — `_sense` must early-return when `self._suppressed` (embeddings
  passes already run under it; keeps the buffer clean).
- **Request boundaries live in the serial queue** — the same semaphore-guarded blocks where
  `_apply_request_profile` runs (inference_service.py:831-900 non-stream, :1096-1175 stream).
  `begin_sensing_request` right after apply; `collect` in the same `finally` region as restore/notify.
- **Flush placement** — beside `_notify_monitoring` calls (:857 non-stream, :1096 stream path). Note
  `_notify_monitoring` is sync; `_notify_sensing` is async (DB) — call with await in the async paths.
- **Streaming token ids** — piggyback the stopping-criteria pattern
  (`_make_event_stopping_criteria`, :59): criteria are invoked EVERY generation step with the full
  `input_ids` tensor; storing the reference is zero-copy and survives early stop. Non-streaming:
  `outputs[0]` (:860 area). Prefill: `inputs.input_ids`.
- **Routing** — `_use_cbm_for_request` (:314-366): add
  `or (settings.SENSING_FORCE_SERIAL and self._sensing_service.is_armed)` beside the has_profile
  condition (:354). Non-forced CBM: skip begin ⇒ collect returns empty ⇒ unsensed (SEN-S1).
- **WS emitter pattern** — `emit_activation_update` (sockets/progress.py:493-549): captured
  `_main_loop` + `run_coroutine_threadsafe`; copy exactly, event name `sensing:event`.
- **Migration numbering** — 008 (007 is Feature 8's). Both features touch migrations; keep order.

## 3. Key Implementations

```python
# sae_wrapper.py — _sense core
def _sense(self, hidden_states: torch.Tensor) -> None:
    if self._suppressed or self._sensing is None or self._sensing_done:
        return
    x = hidden_states[0] if hidden_states.dim() == 3 else hidden_states   # (seq, d_in)
    acts = torch.relu(x.to(self._W_enc_m.dtype) @ self._W_enc_m + self._b_enc_m)  # (seq, m)
    fired = acts > self._sensing.thresholds                                # (seq, m)
    counts = fired.sum(dim=-1)                                             # (seq,)
    hot = (counts >= self._sensing.min_k).nonzero(as_tuple=True)[0]
    if hot.numel():
        self._append_hits(hot, acts, fired, counts)     # debounce vs buffer tail; cap→_sensing_done
    self._sensing_token_offset += x.shape[0]
    if self._sensing_phase == "prefill":
        self._sensing_phase = "decode"
```

```python
# sensing_service.py — config build (thresholds from cluster_meta)
def _build_config(self, profile: Profile) -> SensingConfig:
    meta = profile.cluster_meta or {}
    members = meta.get("members", [])
    overrides = meta.get("sensing", {})
    eps = float(overrides.get("epsilon", settings.SENSING_EPSILON))
    floor = float(overrides.get("theta_floor", settings.SENSING_THETA_FLOOR))
    idxs, thetas, missing = [], [], 0
    for m in members:
        idxs.append(int(m["feature_idx"]))
        mx = m.get("max_activation")
        if mx is None: missing += 1
        thetas.append(max(floor, eps * mx) if mx is not None else floor)
    mode = "floor_only" if missing == len(members) else "epsilon_max"
    min_k = int(overrides.get("min_k", max(2, math.ceil(0.3 * len(idxs)))))
    k = min(int(overrides.get("context_tokens", settings.SENSING_CONTEXT_TOKENS)), 64)
    return SensingConfig(profile.id, idxs, torch.tensor(thetas), mode, min_k, k)
```

```python
# sensing_service.py — context decode (off hot path)
def _context(self, full_ids: torch.Tensor, hit: SensedHit, k: int, tokenizer):
    if k == 0 or full_ids is None:
        return None, None
    lo = max(0, hit.pos_start - k)
    hi = min(full_ids.shape[-1], hit.pos_end + 1 + k)
    ids = full_ids[0, lo:hi].tolist() if full_ids.dim() == 2 else full_ids[lo:hi].tolist()
    return tokenizer.decode(ids, skip_special_tokens=True), ids
```

```python
# inference_service.py — IdCaptureStoppingCriteria (streaming ids, zero-copy)
class IdCaptureStoppingCriteria(StoppingCriteria):
    def __init__(self): self.latest_ids = None
    def __call__(self, input_ids, scores, **kw) -> bool:
        self.latest_ids = input_ids      # reference only; read post-generation
        return False
```

```python
# repositories/sensing_repository.py — retention core
async def prune(self, profile_id: str, cap: int, max_age_days: int) -> int:
    # age prune + keep newest `cap` via
    # DELETE WHERE profile_id=:p AND id NOT IN (
    #   SELECT id FROM sensing_events WHERE profile_id=:p ORDER BY created_at DESC LIMIT :cap)
```

## 4. Implementation Pitfalls

1. **Buffer hygiene** — `begin_sensing_request` MUST reset buffer/offset/phase/done; a missed begin on
   an unsensed path must yield an empty collect, never stale hits from a prior request.
2. **dtype/device** — `W_enc_m` cache inherits the SAE's device/dtype; cast `x` (hidden states may be
   fp16/bf16 while SAE weights differ) exactly as `encode()` does.
3. **Debounce across passes** — a span can continue across pass boundaries during decode (position
   p then p+1 in the next pass); merge with the buffer TAIL, not only within-pass.
4. **`ambient_fired_count`** — only when monitoring is enabled AND un-compacted
   (`sae._monitored_features is None`); read from the monitoring capture for the LAST position only if
   the event includes it — otherwise NULL. Never estimate.
5. **Speculative rejected drafts** — offset accounting counts verification-pass positions; some sensed
   positions may correspond to later-discarded tokens. Accepted + documented; do NOT try to reconcile
   against accepted-token counts in v1.
6. **WS payload excludes context_text** (user content; size) — UI fetches detail via REST.
7. **CASCADE test** — deleting a cluster profile must remove its events (FK) — pin with a test.
8. **Arm state vs column** — `profiles.sensing_enabled` is persistent intent; ARMED is runtime state
   (active cluster + enabled + SAE attached). Status endpoint reports both distinctly.

## 5. Config Additions (millm/core/config.py)

```python
SENSING_EPSILON: float = 0.1
SENSING_THETA_FLOOR: float = 0.0
SENSING_CONTEXT_TOKENS: int = 16          # ±K; hard max 64
SENSING_MAX_EVENTS_PER_REQUEST: int = 20
SENSING_MAX_EVENTS_PER_CLUSTER: int = 1000
SENSING_MAX_AGE_DAYS: int = 7
SENSING_FORCE_SERIAL: bool = True
SENSING_MAX_OVERHEAD_MS: float = 5.0
```
