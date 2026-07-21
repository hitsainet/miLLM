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
