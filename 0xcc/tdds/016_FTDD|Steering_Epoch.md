# Technical Design Document: Steering Epoch

## miLLM Feature 16

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**References:** `016_FPRD|Steering_Epoch.md` · `000_PADR|miLLM.md` (v1.3) · `BRD-MILLM-CIRCUITS-002.md` (BR-003)

---

## 1. Executive Summary

A per-request steering override is a save/apply/restore sandwich. The restore is unconditional, so any
authoritative write landing between save and restore is overwritten — silently, and while the API that
accepted it reports success.

The fix is one integer. `AttachedSAEState` gains a monotonic `steering_epoch`, bumped under
`_ATTACHMENT_LOCK` by every authoritative writer. The saved-state dict carries the epoch observed at
save time; restore compares and **skips when superseded**.

### Key Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Coordination mechanism | Monotonic epoch on the registry | One integer, no blocking, no layering inversion. Extending the request semaphore to admin mutations turns a management call into a 503 behind a long generation and risks deadlock (PADR v1.3). |
| Granularity | Global, not per-layer | Per-layer counters yield a half-old/half-new state harder to reason about than either outcome. Skipping globally always leaves the NEWER authoritative state in place. |
| Conflict resolution | Last authoritative writer wins | An operator's explicit act outranks a transient per-request dial. |
| Scope | Circuit AND profile paths together | The window is identical on both; fixing one leaves the bug one file away — the exact propagation pattern seen across the 001 increment. |
| Where the bump lives | Inside the existing lock | The writers already hold `_ATTACHMENT_LOCK`; a bump outside it could interleave between check and act. |

---

## 2. System Architecture

```
operator                     request in flight
   |                                |
   |                        save(values, epoch=N)
   |  set_intensity  ------------>  |
   |    bump -> epoch=N+1           |
   |                          generate...
   |                                |
   |                        restore: epoch N != N+1  -> SKIP + log
   v                                v
 operator's value stays live   request returns
```

Under EC-16.1 (no concurrent write) the epoch is unchanged and the restore proceeds exactly as today.

---

## 3. Registry Change (`millm/services/sae_service.py`)

```python
# millm/services/sae_service.py — AttachedSAEState
class AttachedSAEState:
    def __init__(self) -> None:
        ...
        self._steering_epoch: int = 0

    @property
    def steering_epoch(self) -> int:
        """Monotonic counter of authoritative steering writes.

        Read by a per-request override at save time and compared at restore.
        A per-request restore that finds it advanced SKIPS — the operator's
        newer write wins over a snapshot taken before it existed.
        """
        with _ATTACHMENT_LOCK:
            return self._steering_epoch

    def bump_steering_epoch(self, reason: str) -> int:
        """Record an authoritative write. MUST be called under, or by a caller
        that already holds, _ATTACHMENT_LOCK — a bump outside the lock can
        interleave between another writer's check and act."""
        with _ATTACHMENT_LOCK:
            self._steering_epoch += 1
            logger.debug("steering_epoch_bumped",
                         epoch=self._steering_epoch, reason=reason)
            return self._steering_epoch
```

### Call sites (verified live)

| Writer | Location |
|---|---|
| `CircuitService.activate` | `circuit_service.py:253` |
| `CircuitService.deactivate` | `circuit_service.py:664` |
| `CircuitService.set_intensity` | `circuit_service.py:758` |
| `ProfileService.activate_profile` | `profile_service.py:357` |
| `ProfileService.deactivate_profile` | `profile_service.py:479` |
| `SAEService.set_circuit_steering` | `sae_service.py:481` |
| `SAEService.clear_circuit_steering` | `sae_service.py:686` |
| `SAEService.attach_set` | `sae_service.py:1438` |
| `SAEService.detach_sae` | `sae_service.py:1647` |

Attach/detach are included deliberately: an SAE swapped mid-request must not receive a snapshot taken
against its predecessor (EC-16.4).

---

## 4. Request Lifecycle (`millm/services/inference_service.py`)

Both saved shapes gain `epoch`:

```python
# circuit path — inference_service.py:935
return {"circuit": True, "epoch": state.steering_epoch, "layers": saved_layers}

# profile path — inference_service.py:1193 / :1241
return {"values": sae.get_steering_values(), "enabled": True,
        "epoch": state.steering_epoch}
```

Restore gains one guard, applied to BOTH branches of `_restore_request_profile`:

```python
# millm/services/inference_service.py — _restore_request_profile
saved_epoch = saved.get("epoch")
current = AttachedSAEState().steering_epoch
if saved_epoch is not None and saved_epoch != current:
    # An authoritative writer landed between our save and now. Restoring
    # would overwrite it with a snapshot taken before it existed.
    logger.info("request_restore_skipped_superseded",
                saved_epoch=saved_epoch, current_epoch=current,
                request_id=..., path="circuit" if saved.get("circuit") else "profile")
    return
```

**The apply-failure rollback is exempt by construction** (EC-16.6): `_apply_request_circuit_steering`'s
`except` at `inference_service.py:966` restores a snapshot taken microseconds earlier within the same
epoch, so the comparison passes and the rollback proceeds.

---

## 5. Truthful `reapplied` (`millm/services/circuit_service.py`)

`set_intensity` computes `reapplied` at `:796-816`. It becomes truthful by capturing the epoch it
produced and confirming it is still current when the response is built:

```python
# millm/services/circuit_service.py — set_intensity
applied_epoch = self._sae_service.state.bump_steering_epoch("set_intensity")
...
still_current = AttachedSAEState().steering_epoch == applied_epoch
return {..., "reapplied": reapplied and still_current,
        "superseded": (not still_current) or None}
```

---

## 6. Testing Strategy

| Level | Coverage |
|---|---|
| Unit — registry | Epoch starts at 0, increments monotonically, is read under the lock |
| Unit — writers | Each of the nine call sites bumps exactly once per authoritative write |
| Unit — restore | Skips when advanced; proceeds when unchanged; handles a missing `epoch` key (older saved state) by proceeding |
| Unit — both shapes | Circuit and profile saved dicts both carry and honour the epoch |
| Unit — rollback | The apply-failure path still restores (EC-16.6) |
| Integration | A simulated in-flight request whose restore is superseded leaves the operator's value live; `reapplied` reflects reality |
| **Mutation** | Deleting the epoch comparison MUST fail a test (BR-005) |

---

## 7. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| A writer is missed, so its window stays open | High | Enumerate all nine call sites in a test that asserts each bumps; a new writer without a bump is the realistic regression |
| The global epoch skips restores unnecessarily (EC-16.2) | Low | Skipping leaves the newer authoritative state live — the safe direction. Logged so the frequency is observable |
| A stale saved dict lacking `epoch` | Low | Treated as "proceed", preserving today's behaviour exactly |
| Bump placed outside the lock | Medium | The registry method takes the lock itself; a bump from a caller not holding it is still correct because `RLock` is reentrant |
