# Feature 16 — Steering Epoch: Review Round 2

**Date:** 2026-07-20
**Scope:** commit `67edfe9` (R1's fixes) + `d87b97b` (the feature)
**Method:** adversarial pass, mission = *find bugs in Round 1's own fixes*
**Findings:** 17 · **11 fixed** · remainder deferred

**Thirteen for thirteen.** And this round's headline is the sharpest yet: R1's flagship fix was not
merely buggy — it was **structurally incapable of working**, and deleting its entire wiring left all
1621 tests green.

---

## The critical three

### F16-R2-01 — CRITICAL: R1's revert ledger could never fire for the case it was built for
R1 added `note_restore_reverted` / `was_reverted` so `set_intensity` could detect an in-flight restore
having overwritten an operator. The two epochs **do not correspond, by construction**:

- the restore records only when `saved_epoch == current_epoch` — i.e. only when *nothing bumped*;
- `set_intensity`'s `applied_epoch` is always *post-bump*.

An operator bump forces the restore to SKIP (so it never records); a restore that proceeds requires no
operator bump (so there is nothing to report). Mutually exclusive. Verified by execution:
`snapshot=0 applied=1 → restore SKIPPED → was_reverted(1) = False`, always.

Worse, it produced **false positives on ordinary traffic**: every uncontended request's restore noted
its epoch, nothing ever removed it, and a later operator write landing on a poisoned epoch was told it
had been superseded when it was live. R1 turned "always says true" into "says false on healthy idle
traffic" — arguably worse, because it trains operators to ignore the warning.

**Fix: removed the mechanism entirely, not patched.** Once the guard works, an in-flight restore
*cannot* revert an operator: their bump advances the epoch, the restore sees the mismatch and skips.
The only way a write stops being live is another **authoritative** write — which the plain epoch
comparison already detects. R1 built a ledger to solve a problem the guard had already solved.
**Test:** `test_an_in_flight_restore_cannot_revert_an_operator_write` + `test_no_ledger_remains`.

### F16-R2-02 — CRITICAL: the ledger's own tests could not see it
Deleting `note_restore_reverted(current_epoch)` left **1621 passed**. The tests called the ledger
primitives directly and never exercised the restore path, and the `set_intensity` stub read the epoch
*without bumping* — the fixture-agrees-by-construction pattern. R1's commit message even claims the
replacement "pins set_intensity's return value"; it pinned it against a stub that misrepresented
production.

### F16-R2-03 — CRITICAL: `applied_epoch` named whoever wrote last, not us
`circuit_service.py` re-read `steering_epoch` *after* the steering call returned, so a second operator
landing in that gap made `still_current` compare equal and report `reapplied: true` for a value
already superseded.

Fixing this exposed a second bug in my own fix: I first stored the epoch on `SAEService` — but that is
constructed **per request** via `Depends`, so `_last_write_epoch` was 0 for every caller while the
shared counter had advanced, making *every clean write* report `superseded`. The epoch now rides the
**outcome object**, which already travels back to every caller.

A third layer surfaced from that: the outcome is a `MagicMock` in several fixtures, so
`getattr(outcome, "applied_epoch", None) or ...` took a truthy sentinel that never equalled the real
counter. Now type-checked (`isinstance(int)`, excluding `bool`), degrading to a live read.

---

## Also fixed

| # | Finding | Fix |
|---|---------|-----|
| 04 | `_reverted_epochs` was **unbounded in the steady state** — the prune filtered by `> epoch - 256`, and requests don't bump, so at a low epoch it kept everything (2000 entries at epoch 0). The bound test bumped before every note, the one condition under which it held | moot — mechanism removed |
| 05 | Pruning **dropped still-queryable epochs**, so truthfulness silently degraded to "not reverted" with time — fail-open on a guarantee | moot — mechanism removed |
| 06 | `operator_clear_steering`'s bump was **unpinned**: deleting it left 1621 green. R1 fixed six routes and pinned five | `test_operator_clear_steering_bumps` |
| 07 | `request_id` threading was **unpinned**: removing it from a call site left 1621 green, because the log test injected a dict that already had the key — it tested rendering, never supply. My first replacement counted occurrences, which was still too weak | now asserts **every** `_apply_request_steering` call site supplies it |
| 09 | `clear_circuit_steering` bumped **outside the lock**, the identical race its sibling's comment says must be prevented | bump moved inside `_ATTACHMENT_LOCK` |
| 10 | The internal `clear_circuit_steering()` inside `_set_circuit_steering_locked` **double-bumped** one logical action *and* bumped unconditionally — defeating the dial's `authoritative=False` through a back door | `authoritative=False` on the internal call |
| 11 | `set_intensity` set `reapplied = True` with **no try/except**, though the DB write had already committed — a raise left persisted λ diverging from live steering with the caller told nothing | wrapped; warns and reports the divergence |
| 13 | `if cleared:` used a **proxy for the wrong predicate** in both directions | narrowed with the lock fix; residual noted |

---

## Deferred

| # | Finding | Why |
|---|---------|-----|
| A | **`/v1/completions` never applies request steering at all** — `create_text_completion` has no `_apply_request_steering` call, so `steering_intensity` is silently ignored on that endpoint | A genuine pre-existing gap, and NOT an epoch bug: the path has no snapshot, no epoch and no restore. Fixing it means adding the whole per-request steering flow to a third endpoint — its own change, recorded as debt |
| B | `cluster_service.py:254` still returns the untruthful `reapplied` | R1 deferred it as "a one-line application of the same ledger"; that premise is now void since the ledger is gone. The correct fix is the guard-based one, still a small change, still outside F16's stated scope |
| C | `superseded` + `reapplied` cannot express slice-fallback vs never-attempted | Both are `false`; only the free-text warning distinguishes them |
| D | The λ=0 fast path has no epoch-advancing call, so `authoritative` doesn't cover it | Correct today (it relies on the guard); a future writer added there inherits no protection |

---

## Verification

- `pytest tests/unit tests/integration` → **1627 passed, 1 skipped**
- `test_steering_epoch.py` 24 → **30**

**Mutation sweep — all five fail as they must:**

| Mutation | Result |
|---|---|
| Revert FR-16.4 to the unconditional claim | fails |
| Delete the `operator_clear_steering` bump | fails |
| Restore the late epoch capture | fails |
| Internal clear bumps again (defeats `authoritative=False`) | fails |
| Remove the guard | fails (5 tests) |

**Round 2 outcome:** R1's flagship mechanism was structurally impossible and had to be deleted rather
than repaired — and fixing the finding it was meant to address exposed two further defects in my own
fix (per-request service state, then a mock sentinel). The lesson is the increment's own thesis
restated: a mechanism added to compensate for a broken guard is harder to reason about than the guard,
and it hid behind tests that modelled it rather than production.
