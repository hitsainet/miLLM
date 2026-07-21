# Feature 018 — Review Round 2 (2026-07-21)

**Suite:** 1904 → **1922** green / 1 skipped. CI green.
**9 findings, 9 fixed.** Every fix negative-controlled. Final sweep: 8 mutations, 0 survivors.

## Round 2 broke round 1, as it has every time

**Seven of the nine are defects in R1 fixes.**

| # | Finding | Origin |
|---|---|---|
| **01** | **R1-12's NaN sentinel could reach the apply** — `_serve_full` read `plan.intensity` without checking `has_intensity` | R1-12 |
| **04** | **and NaN resolves to the CEILING, not to NaN.** `max(0.0, min(2.0, nan)) == 2.0` — a member authored at 150 would have served at λ=2 → clamped 200. Not "nonsense invisibly": MAXIMUM AGGRESSION invisibly | R1-12 |
| 02 | R1-13's new `ValueError` traced end to end — the schema bounds `steering_intensity` to [0,2], so it is defence in depth, not a user-triggerable 500 | R1-13 |
| 03 | R1-08's "snapshot" is a tuple of LIVE references — correct not to copy (GPU tensors), so the safety argument is the dial's immediate copy, now asserted with a no-await guard | R1-08 |
| 05 | **the R1-09 placement test could not fail** — moving the construction back inside the try keeps the asserted substring | R1-09 |
| 06 | `member_layers` left dead by R1-08 — the second time an R1 fix orphaned something | R1-08 |
| **07** | **`claimed_entries`' filter was load-bearing and unprotected** — removing it passed the whole suite, and unfiltered the dial would save/dial/restore another tenant's layer | R1-08 |
| 08 | R1-06's widened regex was wrong in BOTH directions — missed `myField`/`SAE_STATE`/`_cache2`, hallucinated from comments and comparisons. Replaced with an AST walk | R1-06 |
| 09 | `for_registry`'s docstring claimed a "value vs missing attribute" distinction that does not exist — both raise AttributeError | R1-05 |

## The one that matters most

**R2-04.** The codebase already knew. `_resolve_circuit_intensity` has rejected
non-finite intensities since F14 R3, with a comment naming the ceiling
behaviour explicitly. R1-12 introduced NaN into the sibling path **twelve lines
away** with no such guard, and my own R2-01 commit message then described the
consequence incorrectly as silent nonsense rather than silent maximum
aggression.

Reading the neighbouring function would have prevented both the defect and the
mischaracterisation.

## Test quality

**R2-05 is the sixth test in this increment that could not fail.** Its own
docstring cited the lesson it violated. The others: a `caplog` capture this
structlog codebase never reaches, an underscore-only regex, an aliased shim, an
index comparison against a string that also appears in a comment, and a
mutation aimed at an anchor that does not exist.

Every one was caught by the negative control and by nothing else.

## Closing findings (R2-16 … R2-20)

**R2-16/17.** The realistic four-way identity, and the unchanged `_serve_full`
response shape. Together these pin that consolidating four derivations into one
engine did not move the observable output.

**R2-18.** R2-03 established that `claimed_entries` holds LIVE registry entries,
so a detach mid-request leaves stale handles in the plan. What makes that safe
is the restore path, and no F18 test asserted it. The restore does not trust the
captured handle: it re-resolves through `state.get(sae_id, layer)` and skips
what has gone. Without that, a restore writes steering values into a DETACHED
SAE — reviving an intervention on a layer the operator deliberately released.

Also pinned: each layer restores independently. A failing layer must not abort
the loop, or the survivors stay permanently dialled — the per-request override
leaking into global state that restore exists to prevent.

**R2-19.** `_steering_circuit_uncached` now asks `plan.is_serveable`; its
verdict must follow CURRENT attachment. Asserted by driving the real method
across an attach/detach transition. A verdict surviving a detach puts a rung
header on a response describing an intervention that is no longer running.

**R2-20 — totality by use, not by name-matching.** R2-08's AST guard compares
the field NAMES `for_registry` assigns against `__init__`'s. Neither it nor the
registry-identity test exercises the instance, so a field set to the WRONG VALUE
passes both. The R1-A defect (`_repository` for `repository`) was a NAMING
error, in the AST test's reach once its regex was fixed; a value error is not.
`_sae_state = None` satisfies name parity exactly and fails the moment the dial
runs. Fixed by driving the real `set_circuit_steering` on a real
`for_registry()` instance.

Negative controls: `_sae_state = None` is caught **only** by the new use-test;
dropping `svc.repository` is caught by both. Tree confirmed clean after restore.

## Round verdict

**20 findings, 20 fixed. Suite 1904 → 1956 green. CI green.**

Seven of the first nine findings were collisions with R1's own fixes — R1-08
orphaning `attached_layers()` and leaving `member_layers` dead, R1-12's
`UNSET_INTENSITY` resolving through `max(0.0, min(2.0, nan))` to the CEILING,
R1-05's underscore error surviving under a docstring asserting it was fixed.
The pattern holds across every round of this increment: **the most productive
target in round N is round N-1's fix.**

The standing lesson, now sixfold: a test is not evidence until a mutation has
made it fail.
