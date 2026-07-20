# Feature 017 — Review Round 3 (2026-07-20)

**Suite:** 1791 → **1835** green / 1 skipped.
**20 findings, 20 fixed.** Every fix negative-controlled.

## The pattern changed

R2's headline was "sixteen of twenty were R1 fixes colliding." R3 is different,
and the difference is worth naming: **an independent reviewer mutated all
eighteen of R2's fixes and every one failed at least one test.** R2's fixes are
the best-protected code in the feature.

R3's real finding is one level up: **fixes from earlier rounds that were never
pinned at all**, invisible to a 1796-test suite.

| # | Finding | Kind |
|---|---|---|
| 01 | R2-12's stall **through a different door** — a layer going dark AFTER begin left pruning waiting forever (512 vs 8 retained fires) | R2 fix broken |
| 02 | the reclaim released only `_armed_saes`, so a swapped-out SAE kept a dead context and self-bound a private ring | R2 fix broken |
| 07 | `requests_truncated` counted per DRAIN — one request drained 3× read 1→2→3 | R2 fix broken |
| 11 | **R1's overhead fix was still broken**: it zeroed AFTER summing, so the stale value was counted once — the exact number its comment says it prevents | R1 fix wrong |
| 12 | `_last_request_member_fires` had the identical bug **one line down** | missed by the round that fixed its neighbour |
| 13 | `EventBudget.truncated_layers()` had **zero production readers** — two rounds kept two sources in agreement, one was never consulted | declared-but-unwired, 5th instance |
| 08/09/10 | three earlier fixes with **no test at all**, each surviving its mutation | unpinned fixes |
| 18 | **no route test asserted any of the five fields** rounds 1–3 added | the F16 R1 mode, one layer up |

Others: 03 pause-reason transitions · 04 cap-ordering honesty · 05 the budget's
own truncation API was an untested safety net · 06 the merge sort was
unprotected · 14 an inconsistent arming ran in a half-state · 15 `context_tokens`
per-layer collapse (recorded as F19 debt, not reachable today) · 16 the event
tiebreak is behaviourally inert (recorded, deliberately not tested) · 17
strictly-before pinned through the real path · 19 the broadcast's privacy
RESULT, including a leak smuggled through `summary` · 20 the whole capability,
end to end.

## What this round says about writing tests

**Four of my regression tests initially passed against the mutation they were
written for.** Each time the fixture agreed with the code by construction:

- `_member_stats`: the `max()` branch runs only when both values are real
  positives, so a None fixture takes a different branch. Observable **only**
  when a SMALLER value arrives after a larger one. I reasoned about the branch
  twice and got it wrong twice; running both versions against every ordering
  settled it in seconds.
- overhead: a test giving every layer a clean begin passes, because
  `_reset_edge_buffer` already cleared the field. Only a layer ABSENT from begin
  reaches the drain-time path.
- the merge sort: `collect_edges` walks layers in sorted order, so putting late
  positions on the higher layer left the output already ordered.

That is the R1-12 anti-pattern — a fixture that cannot fail — committed by me
four more times, sixty findings after I first flagged it. **The negative control
is not a formality; it is the only thing that distinguishes a test from a
comment.**

Also: one edit reported success while matching a stale anchor and changing
nothing. The guard "worked" until it was executed.

## Deliberately not tested

- The `events.sort` tiebreak is **behaviourally inert** — verified across three
  shapes. Kept for readability; a test on it would pass for the wrong reason.
- The drain-time overhead zeroing is now belt-and-braces after the `begun` gate.
- Neither strictly-before guard is individually pinnable because neither is
  individually load-bearing (measured: each alone 0 failures, both 5). The
  invariant is pinned through the production path instead.
