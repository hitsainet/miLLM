# Technical Implementation Document: Steering Epoch

## miLLM Feature 16

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**References:** `016_FTDD|Steering_Epoch.md` · `016_FPRD|Steering_Epoch.md`

---

## 1. File Structure

```
millm/services/sae_service.py            # AttachedSAEState: the epoch field, property, bump method
millm/services/circuit_service.py        # 3 bump sites + truthful `reapplied`
millm/services/profile_service.py        # 2 bump sites
millm/services/inference_service.py      # capture at save (2 shapes), compare at restore
tests/unit/services/test_steering_epoch.py          # NEW
tests/integration/test_steering_epoch_workflow.py   # NEW
```

No new modules, no migration, no config key.

---

## 2. Load-Bearing Implementation Points (verified against live code)

| Point | Location | Why it matters |
|---|---|---|
| `_ATTACHMENT_LOCK` | `sae_service.py:52` | Module-level `threading.RLock`. Reentrant, so a bump from a caller already holding it is safe. |
| Locking discipline | `sae_service.py:340` | Docstring: attach/serve paths "hold the process-wide `_ATTACHMENT_LOCK` so a check-then-act" is atomic. The bump joins that discipline. |
| `set_circuit_steering` | `sae_service.py:481`, takes the lock at `:512` | The main circuit write path. |
| `clear_circuit_steering` | `sae_service.py:686` | Clearing is authoritative too — a request must not restore over a deliberate clear. |
| `attach_set` / `detach_sae` | `sae_service.py:1438` / `:1647` | Attachment changes invalidate a snapshot taken against the old set (EC-16.4). |
| `CircuitService.activate` / `deactivate` / `set_intensity` | `circuit_service.py:253` / `:664` / `:758` | |
| `reapplied` | `circuit_service.py:796` (init `False`), `:804` (set `True`), `:816` (returned) | The falsehood this feature corrects. |
| `ProfileService.activate_profile` / `deactivate_profile` | `profile_service.py:357` / `:479` | The Feature 10 half of the same window. |
| Circuit saved shape | `inference_service.py:919-920` (values/enabled per layer), returned `:935` and `:993` | |
| Circuit apply-failure rollback | `inference_service.py:966` | Restores its own snapshot — same epoch, must still proceed (EC-16.6). |
| Profile saved shape | `inference_service.py:1193-1194`, `:1241` | Two construction sites; both need the epoch. |

---

## 3. Key Implementations

### 3.1 The registry field

Add to `AttachedSAEState.__init__` alongside `_entries`. Do **not** reset it in `clear()` — an epoch
that goes backwards defeats the comparison. It resets only with the process, which is also when all
attachments are cleared.

### 3.2 Capturing at save

Read the epoch **once**, at the moment the snapshot is taken, in the same block that builds
`saved_layers` / `values`. Reading it later (e.g. when returning) opens a smaller version of the very
window this feature closes.

### 3.3 Comparing at restore

`_restore_request_profile` handles both shapes. The guard goes at the **top**, before either branch, so
a shape added later inherits it by default rather than by remembering.

### 3.4 Truthful `reapplied`

`set_intensity` must capture the epoch its own write produced and compare at response-build time —
not simply read "is anything newer", which would report `superseded` for its own bump.

---

## 4. Implementation Pitfalls

1. **Do not reset the epoch in `clear()` or `disarm()`.** A counter that restarts can collide with a
   saved value and silently permit a stale restore.
2. **Do not bump in `set_steering_batch`.** That is the low-level write used *by* the authoritative
   paths; bumping there would double-count and, worse, bump during a per-request apply — making every
   request supersede itself.
3. **A missing `epoch` key means proceed**, not skip. Old saved state must behave exactly as today.
4. **The apply-failure rollback must not be skipped** (EC-16.6). It is same-epoch by construction, so
   no special case is needed — but a test must pin it, because "skip when superseded" reads like it
   should apply there too.
5. **Nine writers, not three.** Attach/detach and `clear_circuit_steering` are easy to miss; the
   enumeration test exists because a new writer without a bump is the realistic regression.
6. **`reapplied` is not the only claim.** If the profile path gains a similar affirmative field, it
   inherits the same obligation.

---

## 5. Config Additions

None.

---

## 6. Divergences from the BRD's assumptions

- BR-003 describes "the request queue" as the coordination point. The implementation uses the
  **attachment lock**, not the queue — the queue serialises *requests*, while the writes needing
  coordination are admin mutations that never enter it. The epoch is read and written under
  `_ATTACHMENT_LOCK`, which every authoritative writer already holds.
- The BRD says `set_intensity` returns `"reapplied": true`; verified at `circuit_service.py:816`, with
  the flag set at `:804`. The fix is a conjunction at the return site, not a rewrite of the branch.
