# Feature 16 — Steering Epoch: Review Round 1

**Date:** 2026-07-20
**Scope:** commit `d87b97b`
**Method:** two concurrent reviewers — (a) correctness/concurrency of the mechanism, (b) API/observability/test-integrity/requirement coverage
**Findings:** 28 (14 + 14) · **13 fixed** · remainder deferred

**Verdict on arrival: the feature did not deliver its headline guarantee**, despite 12 passing tests
and a mutation control that genuinely passed. Both reviewers reached that conclusion independently,
by different routes.

---

## The critical four

### F16-R1-01 — CRITICAL: the epoch was captured at RETURN, after the apply
`inference_service.py:994` — the snapshot is built at ~915, `set_circuit_steering` fires at ~956, and
the epoch was read at ~994. The saved epoch therefore always equalled the POST-apply epoch, so any
operator write landing during the apply window was **absorbed**: the restore compared equal, proceeded,
and reverted them. That is US-16.1 — the primary user story — still broken.

The FTID forbids this by name (§3.2: *"Reading it later opens a smaller version of the very window this
feature closes"*). I wrote that sentence and then implemented the thing it prohibits.

**Fix:** capture at snapshot time, in the same block that builds `saved_layers`.
**Test + mutation:** `TestCaptureHappensAtSnapshotTime`; restoring the late read fails it.

### F16-R1-02 — CRITICAL: the per-request apply bumped its own epoch
`sae_service.py:551` — the dial calls `set_circuit_steering`, which bumped unconditionally. FTID
pitfall 2 forbids exactly this ("every request supersedes itself"). The late read in R1-01 was
MASKING it, which is why both had to be fixed together: capturing early alone would have made every
request skip its own restore.

**Fix:** `set_circuit_steering(..., authoritative: bool = True)`; the dial passes `False`.
**Test:** `test_a_per_request_apply_does_NOT_bump`.

### F16-R1-03 — CRITICAL: `superseded` never reached the client
`api/schemas/circuit.py:290` — `CircuitIntensityResponse` declared `reapplied` and `warnings` but not
`superseded`, and inherits no `extra="allow"`, so Pydantic **silently dropped it**. The service
computed the flag correctly and the API threw it away — leaving a client unable to distinguish
"superseded by an operator" from "slice-fallback, never applied", the two cases the field exists to
separate.

**Fix:** declared on the model, and changed from `(...) or None` (tri-state `True|None`, never `False`)
to a plain `bool`.

### F16-R1-04 — CRITICAL: `reapplied` could not detect the case it exists for
`circuit_service.py:824` — `still_current = steering_epoch == applied_epoch` can never catch an
in-flight restore, because the restore does NOT bump (correctly — bumping there is R1-02). So after a
restore reverted an operator's intensity, the API still returned `reapplied: true`. FR-16.4 promised
precisely the opposite.

**Fix:** the restore, when it PROCEEDS, records the epoch it wrote over (`note_restore_reverted`);
`set_intensity` asks `was_reverted(applied_epoch)`. A bounded ledger (<=256 entries) is the only way a
non-bumping writer can report what it did.

---

## Also fixed

| # | Finding | Fix |
|---|---------|-----|
| 05 | **Six operator-facing routes never bumped** — `set_steering`, `set_steering_batch`, `enable_steering`, `disable_steering`, `clear_steering`. These are the exact actor in the feature's own narrative, and the original nine-writer enumeration missed them entirely | bumped in the four `SAEService` methods, covering all six routes at once |
| 06 | `set_intensity` **double-bumped** (the steering call already bumped); `activate` triple-bumped. Any snapshot taken between two bumps of one logical action saw a spurious mismatch | bump only when nothing else did |
| 07 | `clear_circuit_steering` bumped **even on a total no-op**, so every empty clear superseded all in-flight restores with no compensating write | bump only when something was cleared |
| 08 | The enumeration test **grepped source text**, so all nine writers could be commented out or wrapped in `if False:` and it stayed green — the `TestRingPruningIsWired` anti-pattern this project has shipped before | `TestWritersBumpBehaviourally` calls the writers and observes the counter |
| 09 | **FR-16.4 had ZERO tests.** Reverting it to the original defect left 1609/1609 green | `TestSetIntensityReturnIsTruthful` pins the RETURN VALUE, not the primitives — my first attempt tested the ledger and MUT-A still passed, the same trap one level down |
| 10 | FR-16.3's **request id was missing** from the skip log, and unavailable — the function took only the saved dict | threaded `request_id` through both apply functions and the two generation call sites |
| 11 | The skip log did not name **which layers were left dialled**, though skipping means they are not restored | `layers_left_dialled` added |
| 12 | `test_the_comparison_precedes_the_branch_demultiplex` anchored on a **comment string** and matched the log kwarg rather than the guard; a pure rename passed it, and a real defect raised `ValueError` instead of asserting | superseded by behavioural coverage |
| 13 | FPRD section 14 documentation entirely unwritten | `mcp-contract.md` 4a-ter (`reapplied` is authoritative, both false-cases distinguished) + the OpenAI-API reference sentence |

---

## Deferred

| # | Finding | Why |
|---|---------|-----|
| A | **`cluster_service.py:254` has the identical untruthful `reapplied`** with no epoch capture | Real, and outside F16's stated scope (FR-16.4 names the PROFILE path, which was fixed). Tracked debt; a one-line application of the same ledger |
| B | Skipping leaves the request's transient values live on layers the operator never touched — a BLEND, not a clean hand-off | The design's accepted cost (EC-16.2), now at least OBSERVABLE: the skip log names the stranded layers. A full fix means selectively restoring non-contended layers — F17/F18 territory |
| C | No operator-visible epoch or supersession history | Genuine observability gap; belongs with F20's status surface |
| D | EC-16.3 (two requests in flight) rests on an unverified claim about the serial queue | Correct today; needs a concurrency harness |
| E | `_reverted_epochs` reset semantics if the singleton is re-created | Bounded by count; the reset hazard is the pre-existing `_instance = None` test pattern |

---

## Verification

- `pytest tests/unit tests/integration` -> **1621 passed, 1 skipped** (was 1609)
- `test_steering_epoch.py` 12 -> **24**

**Mutation controls — three that previously SURVIVED now fail:**

| Mutation | Before R1 | After R1 |
|---|---|---|
| Revert FR-16.4 to the original lie | 1609 passed (survived) | **fails** |
| Comment out the operator-route bumps | 12 passed (survived) | **fails** |
| Restore the late epoch capture | (untested) | **fails** |
| Guard -> `if False` | 3 failed | still fails |

**Round 1 outcome:** four criticals, all meaning the feature's two headline user stories were unmet on
the circuit path while the suite was green. The original mutation control was real but tested the one
mechanism already well covered — the guard's PRESENCE — not its INPUTS. The capture site and the
`reapplied` predicate are where the correctness actually lived.
