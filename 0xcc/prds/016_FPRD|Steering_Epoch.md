# Feature PRD: Steering Epoch

## miLLM Feature 16

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**Feature Priority:** Secondary (Increment: Circuit Consolidation)
**References:** `BRD-MILLM-CIRCUITS-002.md` (BR-003) · `000_PPRD|miLLM.md` (v1.3, FR-16.x) · `000_PADR|miLLM.md` (v1.3)

---

## 1. Feature Overview

### Feature Name
Steering Epoch — an operator's change to live steering is never silently reverted by an in-flight request.

### Brief Description
Per-request steering overrides save the current steering state, apply their own, and restore the saved
state when the request finishes. That restore is **unconditional**. If an operator changes live
steering *while a request is generating*, the request's restore writes the old values back over the
operator's change. The operator sees a successful API response and a runtime that quietly ignored it —
and in the circuit case `set_intensity` returns `"reapplied": true` (`circuit_service.py:816`), an
affirmative statement that the change is live when it is not.

This feature adds a monotonic `steering_epoch` to `AttachedSAEState`, bumped under the process-wide
`_ATTACHMENT_LOCK` (`sae_service.py:52`) by every authoritative writer. A per-request override captures
the epoch when it saves and **skips its restore when the epoch has advanced** — the later
authoritative writer wins.

### Business Requirement Traceability

| BR | Coverage |
|----|----------|
| BR-003 — an operator action SHALL NOT be silently reverted by an in-flight request; the later authoritative writer wins; any API response reporting re-application SHALL be truthful | FR-16.1, FR-16.2, FR-16.3, FR-16.4 |

### Why this is first in the increment
It is the smallest item in BRD-002, independent of every other theme, and the only one that corrects a
**falsehood an operator can observe today**. It delivers value before the large refactors land and
carries no dependency on the request-scoped context (F17) or the single derivation (F18).

---

## 2. User Stories & Scenarios

**US-16.1 — The operator's change survives.**
As an operator, when I raise a circuit's intensity while a long generation is running, I want my change
to be in effect afterwards, so I do not have to guess whether it took and re-apply it.

*Acceptance:* with a request in flight, `PUT /api/circuits/active/intensity` to a new λ; when the
request completes, live steering reflects the new λ, not the pre-request snapshot.

**US-16.2 — The API tells the truth.**
As an agent or a script, I want a response that says the change was applied to mean it was applied, so
I can trust the result without re-reading state.

*Acceptance:* `set_intensity` never returns `"reapplied": true` for a value subsequently reverted.

**US-16.3 — Supersession is observable.**
As an operator debugging surprising steering, I want the logs to show when a restore was skipped, so
"my change vanished" and "my change won" are distinguishable after the fact.

*Acceptance:* a skipped restore logs both epochs and the request id at INFO or above.

### Edge Cases

- **EC-16.1: No concurrent mutation.** Epoch unchanged at restore → restore proceeds exactly as today.
  This is the overwhelmingly common path and MUST be behaviour-identical.
- **EC-16.2: Mutation on an unrelated layer.** The epoch is global, so an unrelated write still advances
  it and the restore is skipped. Accepted deliberately: skipping leaves the newer authoritative state
  in place, which is the safe direction. Per-layer epochs were rejected (PADR v1.3) as producing a
  half-old/half-new state harder to reason about than either.
- **EC-16.3: Two requests in flight.** The serial queue means one override at a time; the second saves
  after the first restores, so epochs are naturally ordered.
- **EC-16.4: Detach/attach mid-request.** Attachment changes bump the epoch, so the restore is skipped
  rather than writing values onto an SAE swapped underneath it.
- **EC-16.5: Profile path.** The identical window exists on the Feature 10 cluster/profile path
  (`inference_service.py:1193`, `:1241`) and is covered by the same field in the same change.
- **EC-16.6: Apply-failure rollback.** `_apply_request_circuit_steering`'s failure path
  (`inference_service.py:966`) restores its own snapshot immediately; that restore is within the same
  epoch and MUST still proceed.

---

## 3. Functional Requirements

- **FR-16.1:** `AttachedSAEState` SHALL carry a monotonic `steering_epoch`, bumped under
  `_ATTACHMENT_LOCK` by every authoritative writer: `CircuitService.activate` (`:253`), `deactivate`
  (`:664`), `set_intensity` (`:758`); `ProfileService.activate_profile` (`:357`), `deactivate_profile`
  (`:479`); `SAEService.set_circuit_steering` (`:481`), `clear_circuit_steering` (`:686`), `attach_set`
  (`:1438`), `detach_sae` (`:1647`).
- **FR-16.2:** A per-request steering override SHALL capture the epoch at save time, store it in the
  saved-state dict, and SHALL SKIP the restore when the current epoch differs.
- **FR-16.3:** A skipped restore SHALL be logged with the saved epoch, the current epoch and the request
  id, so supersession is observable rather than silent.
- **FR-16.4:** `PUT /api/circuits/active/intensity` SHALL NOT report `"reapplied": true` for a change an
  in-flight request reverted. The same guarantee SHALL apply to the Feature 10 profile path.

---

## 4. Data Requirements

No schema change. The epoch is in-memory state on the `AttachedSAEState` singleton, consistent with the
rest of the attachment registry, and resets on restart where attachments are cleared anyway.

Both saved-state shapes gain one key:

```
{"circuit": True, "epoch": 7, "layers": [...]}        # circuit path  (inference_service.py:935)
{"values": {...}, "enabled": True, "epoch": 7}        # profile path  (inference_service.py:1193)
```

---

## 5. API Specifications

No new endpoints. One response-shape correction: `set_intensity`'s `reapplied`
(`circuit_service.py:796-816`) becomes truthful, reporting `false` with a `superseded` note when a
concurrent restore would have undone the change.

---

## 6. UI Requirements

None. The defect is invisible in the UI precisely because the API reported success.

---

## 7. Non-Functional Requirements

- **NFR-16.1:** The comparison SHALL add no measurable latency — one integer read under a lock already held.
- **NFR-16.2:** Behaviour SHALL be identical when no concurrent mutation occurs (EC-16.1).

---

## 8. Dependencies

Feature 12 (`AttachedSAEState`), Feature 14 (circuit dial), Feature 10 (profile dial). **No dependency
on Features 17–20** — this lands first and alone.

---

## 9. Success Criteria

1. With a request in flight, an operator's `set_intensity` survives its completion.
2. `reapplied: true` is never returned for a reverted change.
3. A skipped restore appears in the logs with both epochs.
4. Both the circuit and Feature 10 profile paths are covered by the same mechanism.
5. No regression: the backend suite stays green with restore behaviour unchanged under EC-16.1.

---

## 10. Testing Requirements

- Unit: each authoritative writer bumps the epoch; restore skipped when advanced; restore proceeds when
  unchanged; both saved shapes carry it; the apply-failure rollback still restores (EC-16.6).
- Integration: a simulated in-flight request whose restore is superseded leaves the operator's value live.
- **Mutation (BR-005):** removing the epoch comparison MUST fail a test.

---

## 11. Rollout & Migration

No migration, no config flag. The change corrects a falsehood; there is no deployment in which the old
behaviour is preferable.

---

## 12. Out of Scope

- Per-layer epochs (rejected, PADR v1.3).
- Blocking admin mutations behind the request queue (rejected: turns management calls into 503s behind
  long generations, inverts layering, deadlock risk).
- Notifying a client mid-generation that its steering was superseded — the hook is already installed.

---

## 13. Open Questions

None. The mechanism was settled during Feature 14 Round 3, which evaluated three candidates; rationale
in `0xcc/reviews/review_feature014_circuit_dial_R3_2026-07-20.md`.

---

## 14. Documentation Requirements

- `docs/mcp-contract.md`: `reapplied` is authoritative.
- OpenAI-API reference: one sentence that an operator's concurrent change wins.

---

## 15. Decisions from Clarifying Questions

- **Last authoritative writer wins** over strict per-request isolation: an operator's explicit act
  outranks a transient per-request dial.
- **Global epoch, not per-layer:** simpler, and the skip direction is always the safe one.
- **Both paths in one change:** the profile path has the identical window; fixing one leaves the other.
