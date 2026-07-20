# Task List: Steering Epoch

## miLLM Feature 16

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** 📋 DOCS COMPLETE 2026-07-20 — implementation not started
**References:** `016_FPRD|Steering_Epoch.md` · `016_FTDD|Steering_Epoch.md` · `016_FTID|Steering_Epoch.md`

## Relevant Files
- `millm/services/sae_service.py` — `AttachedSAEState` epoch field/property/bump; 4 writer sites
- `millm/services/circuit_service.py` — 3 writer sites + truthful `reapplied`
- `millm/services/profile_service.py` — 2 writer sites
- `millm/services/inference_service.py` — capture at save (both shapes), compare at restore
- `tests/unit/services/test_steering_epoch.py` — NEW
- `tests/integration/test_steering_epoch_workflow.py` — NEW

## Tasks

- [ ] 1.0 Registry (covers FR-16.1)
  - [ ] 1.1 `_steering_epoch` on `AttachedSAEState`, `steering_epoch` property and `bump_steering_epoch(reason)`, both under `_ATTACHMENT_LOCK`
  - [ ] 1.2 Confirm it is NOT reset by `clear()` — a backwards-going epoch permits a stale restore
  - [ ] 1.3 Unit tests: starts at 0, monotonic, survives `clear()`

- [ ] 2.0 Authoritative writers (covers FR-16.1)
  - [ ] 2.1 Bump in `set_circuit_steering` (`:481`), `clear_circuit_steering` (`:686`), `attach_set` (`:1438`), `detach_sae` (`:1647`)
  - [ ] 2.2 Bump in `CircuitService.activate` (`:253`), `deactivate` (`:664`), `set_intensity` (`:758`)
  - [ ] 2.3 Bump in `ProfileService.activate_profile` (`:357`), `deactivate_profile` (`:479`)
  - [ ] 2.4 **Enumeration test**: assert each of the nine sites bumps exactly once — a new writer without a bump is the realistic regression
  - [ ] 2.5 Assert `set_steering_batch` does NOT bump (it is the low-level write used by the paths above; bumping there makes every request supersede itself)

- [ ] 3.0 Capture and compare (covers FR-16.2, FR-16.3)
  - [ ] 3.1 Capture at save in the circuit shape (`inference_service.py:935`, `:993`)
  - [ ] 3.2 Capture at save in BOTH profile construction sites (`:1193`, `:1241`)
  - [ ] 3.3 Guard at the TOP of `_restore_request_profile`, before either branch, so a later shape inherits it
  - [ ] 3.4 Log `request_restore_skipped_superseded` with both epochs, request id and path
  - [ ] 3.5 A missing `epoch` key proceeds (preserves today's behaviour for stale state)
  - [ ] 3.6 Test EC-16.6: the apply-failure rollback (`:966`) still restores

- [ ] 4.0 Truthful `reapplied` (covers FR-16.4)
  - [ ] 4.1 `set_intensity` captures the epoch its own write produced and compares at response build (`circuit_service.py:796-816`)
  - [ ] 4.2 Must not report `superseded` for its own bump
  - [ ] 4.3 Audit the profile path for equivalent affirmative claims

- [ ] 5.0 Verification (covers all)
  - [ ] 5.1 Integration: in-flight request superseded → operator's value live afterwards
  - [ ] 5.2 **Mutation (BR-005)**: delete the epoch comparison → a test MUST fail
  - [ ] 5.3 Full suite green; EC-16.1 behaviour byte-identical

- [ ] 6.0 Feature Acceptance
  - [ ] 6.1 Verify FPRD §9 criteria 1–5 and all US/EC boxes one-by-one
  - [ ] 6.2 Docs: `mcp-contract.md` (`reapplied` authoritative) + OpenAI-API reference sentence
  - [ ] 6.3 Update CLAUDE.md + PPRD Feature 16 status

## Coverage Audit
| FR | Tasks |
|---|---|
| FR-16.1 | 1.1, 2.1–2.5 |
| FR-16.2 | 3.1, 3.2, 3.3, 3.5, 3.6 |
| FR-16.3 | 3.4 |
| FR-16.4 | 4.1, 4.2, 4.3 |

## Review rounds (goal: 3 rounds, ≥10 findings each)
- [ ] Round 1 (multi-angle `/code-review`): ≥10 findings. **Watch:** a writer missed so its window stays open; the bump placed outside the lock; `set_steering_batch` bumping and making every request supersede itself; the epoch reset somewhere it goes backwards; the rollback path wrongly skipped.
- [ ] Round 2 (attack Round 1's fixes + fresh angles): ≥10 findings. **Watch:** the guard added to only one branch of `_restore_request_profile`; `reapplied` reporting superseded for its own bump; a saved dict built at one site and not the other; the enumeration test asserting existence rather than behaviour.
- [ ] Round 3 (`/review`, 4 perspectives): ≥10 findings. **Watch:** log noise if EC-16.2 skips are frequent in practice; whether skipping is ever the WRONG direction; a concurrency window between reading the epoch and taking the lock; mutation coverage genuinely failing when the comparison is cut.

## Acceptance evidence
_(to be completed at 6.0)_
