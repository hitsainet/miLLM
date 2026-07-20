# Feature 16 — Steering Epoch: Review Round 3

**Date:** 2026-07-20
**Scope:** commits `d87b97b` (feature) + `67edfe9` (R1) + `45f241b` (R2)
**Method:** two concurrent perspectives — Architect (attacking R2's fixes) and QA/Product (a **48-mutation sweep**)
**Findings:** 26 (13 + 13) · **9 fixed** · remainder deferred

**Fourteen for fourteen** — and this round is qualitatively worse than the previous two. R2's
*central argument* was unsound, not merely its implementation.

---

## The critical finding

### F16-R3-01 — CRITICAL: R2 deleted the ledger on a premise that does not hold
R2's stated justification for removing `note_restore_reverted`/`was_reverted` was:

> *"once the guard works, an in-flight restore CANNOT revert an operator."*

**That is false on the apply-failure rollback path.** It called
`_restore_request_profile({"circuit": True, "layers": saved_layers})` with **no `epoch` key**, and the
guard reads `saved_epoch is not None and ...` — so the rollback *always proceeded*. Executed: with an
operator write landing at epoch 7 during a failing apply, the guard evaluates `False` and the restore
overwrites the operator, while `set_intensity` had already returned `reapplied: true`.

R2 justified the exemption as "a snapshot from microseconds ago within the same epoch." But
`set_circuit_steering` can raise arbitrarily late, so the window is not microseconds — it is however
long the apply took before failing.

The deeper error was expressing a *deliberate exemption* as an *absence*: the guard could not
distinguish "old saved state" from "the rollback opting out", so a well-meaning edit either way broke
one of them. **Fix:** the rollback now carries its epoch like every other caller and is exempt from
nothing.

---

## Also fixed

| # | Finding | Fix |
|---|---------|-----|
| 02 | **`activate()` double-bumps** — `_serve_full` bumps, then `activate` bumps again. One activation advanced the epoch 0→2, and any request whose snapshot landed *between* the two saw a spurious mismatch, skipped its restore and stranded its transient λ in global state permanently. This is R1's own finding 06, fixed for `set_intensity` and never applied to the structurally identical case | `_serve_full` passes `authoritative=False`; `activate` owns the bump |
| 03 | **`deactivate()` double-bumps** the same way via `clear_circuit_steering` | `authoritative=False` |
| 06 | **`applied_epoch: int = 0` collides with a real epoch.** At a fresh boot the counter *is* 0, so a default-constructed result compares equal and reports a false "still current" | defaults to `-1`; an unset field can never alias a genuine value |
| 07 | `_serve_full` **discarded `applied_epoch`** from the outcome, so the activate path had nothing to compare and could not report supersession at all | carried through |
| — | **The root cause behind six survivors** | see below |

### The root cause: the guard was tested against dicts the tests wrote
Every restore-guard test hand-builds `{"epoch": 0, ...}` as a literal. **Not one test asserted
`saved["epoch"]` coming out of a real apply.** So the guard was exhaustively tested while the
production code that *populates* what it reads had no coverage at all.

That single cause explains six of the eleven surviving mutations at once — dropping the epoch from the
circuit dict, the profile dict, the λ=0 return, and three `request_id` sites.

**Fix:** `TestTheEpochSurvivesARealApply` — four tests that call the real apply and assert the key
survives, including a full round-trip (real apply → operator write → real restore, no hand-built dict
anywhere).

**Mutation control:** dropping the epoch key previously left **1627 green**; it now **fails 4 tests**.

---

## The mutation sweep

**48 mutations run, 11 survived.** All twelve `bump_steering_epoch` call sites are now genuinely
well-pinned — R1 and R2 fixed that thoroughly. The surviving defects had all moved **one layer down**,
into the data the guard reads.

A methodology note worth keeping: the reviewer's *first* driver reported a survivor that an isolated
re-run contradicted, caused by a shared backup path in the harness itself. Rebuilt with per-spec
backups and a post-restore hash check. **A mutation harness needs its own integrity check** — otherwise
it manufactures both false survivors and, more dangerously, false "caught" results.

---

## Documentation corrected

`docs/mcp-contract.md` §4a-ter still documented `superseded: true` as *"reverted by a per-request
restore"* — a cause **R2's own fix made unreachable** — and told operators to "re-issue it" against a
phantom. Rewritten: `superseded` now means another **authoritative** write landed after yours, and the
doc states explicitly that a per-request dial can no longer be the cause.

---

## Deferred — including one design decision for the product owner

| # | Finding | Why deferred |
|---|---------|--------------|
| **A** | **The global scalar epoch is the wrong abstraction.** Both reviewers concluded this independently. It conflates "did anything change?" with "did *my* thing get overwritten?", and every round has added a compensator (R1 a ledger; R2 an outcome-carried epoch, a type guard and an `authoritative` flag). The concrete failure mode is **false skips**: any writer supersedes *every* in-flight restore, including on disjoint layers. **F19's concurrent circuits make this structural** — two circuits on separate layers would continuously strand each other's restores. Recommendation: per-layer versioning (`dict[layer, int]`) plus a process nonce, so the skip is precise instead of global | **A design amendment, not a fix** — it rewrites F16's core and the FTDD. Raised for an explicit product decision rather than taken unilaterally |
| B | `/v1/completions` never applies request steering; `extra="ignore"` drops `steering_intensity` at *parse* time, while the manual recommends that endpoint for base-model steering experiments | A real pre-existing gap, not an epoch bug. Adding the whole per-request steering flow to a third endpoint is its own change |
| C | R2's `isinstance(int)` fix and the `authoritative=False` internal-clear fix are both **unpinned** (mutations survive) | Recorded; the highest-value tests were spent on the six-survivor root cause first |
| D | No startup reset for the epoch; a snapshot at epoch 0 pre-restart aliases epoch 0 post-restart | Subsumed by (A)'s process nonce |
| E | `cluster_service.py:254` still returns an untruthful `reapplied` | Deferred by R1 as "one line of the ledger", then re-deferred by R2 after the ledger was deleted — surviving two rounds on mutually contradictory grounds. Now explicitly tracked |

---

## Verification

- `pytest tests/unit tests/integration` → **1631 passed, 1 skipped** (was 1627)
- `test_steering_epoch.py` 30 → **34**
- Mutation: dropping the epoch key fails 4 tests (previously survived the whole suite)

**Round 3 outcome:** one critical — R2's deletion premise was false on a live path — plus two
double-bumps of the class R1 fixed once and missed twice, an unsafe default that aliased a real
epoch, and a documentation section describing a deleted mechanism. The recurring lesson across all
three rounds is that each round's tests validated the mechanism it had just built while the layer
beneath it went uncovered.
