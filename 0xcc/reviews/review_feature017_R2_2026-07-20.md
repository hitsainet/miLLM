# Feature 017 — Review Round 2 (2026-07-20)

**Suite:** 1739 → **1791** green / 1 skipped.
**20 findings, 20 fixed.** Every fix negative-controlled.

## The headline: round 2 broke round 1

**Sixteen of the twenty findings are round-1 fixes colliding — with the original
code, with each other, or with round-2 fixes made minutes earlier.** Each round-1
fix was correct in isolation and tested in isolation. The interaction surface is
where all of this lived.

| # | Finding | Collision |
|---|---|---|
| 01 | R1's concurrency guard **deadlocked sensing permanently** — one hung request refused every later one, forever, and disarm+re-arm didn't clear it | R1-03 vs recovery |
| 02 | R1-06's "clear the stale reason" **erased R1-02's `layer_unavailable`** — a half-dark circuit looked healthy | R1-06 vs R1-02 |
| 03 | R1's budget fix **starved quiet layers for a whole request** — a sibling's spending latched them dark | R1-05 vs the per-SAE latch |
| 04/05 | an all-dark begin **orphaned its context** (costing the next request) and **inflated `requests_sensed`** | R1-02 vs R1-03 vs R1-06 |
| 12 | R2-10 **waited forever for a layer R1-02 says can be dark** | R2-10 vs R1-02 |
| 14 | R2-02's own fix made skip reasons **lag one request behind** | R2-02 vs itself |
| 16 | `_request_context_tokens` kept the order-dependence **R2-08 fixed one expression away** | R2-08 vs its sibling field |
| 18 | counters reset asymmetrically, so a healthy re-arm **read as the wiring-failure signature** R1-06 defined | R1-06 vs disarm |

Others: 06 stale truncation named a recovered layer · 07 the solo-context
fallback bypassed the per-circuit budget · 08 the cap came from
`_armed_layers[0]` · 09 `cap=0` armed a dead circuit · 10 a single-layer circuit
never pruned (512 vs 8 retained fires) · 11 throttling was invisible · 13 a rare
truncation could be superseded before anyone polled · 15 a reclaimed request's
observations were discarded silently · 17 context capture could go dark with the
suite green · 19 `self._ctx = None` was load-bearing and untested · 20 truncation
accounting pinned against double-count and carry-over.

## What found them

**Execution and mutation, not reading.** Every confirmed finding came from
running the code or breaking a line. Three had *survived* earlier mutation
sweeps because the mutation was mis-aimed or the mechanism was already dead —
a surviving mutation is only a finding once you have confirmed it applied.

**Attacking a fix minutes after shipping it works.** R2-12 and R2-14 were found
by immediately attacking R2-10 and R2-02.

## Deliberately not changed

- The saturation warning fires once per request. Checked as a possible
  log-flood regression from R1's fix to the opposite defect; once per request is
  right for an operator-actionable condition.
- `bisect_left` vs `bisect_right` and the `-inf` sentinel survive as individual
  mutants. Documented inert singles with an in-process pair-mutation control.

## Contract

`docs/mcp-contract.md` → **v1.3**: `requests_sensed`, `requests_truncated`,
`ws_throttled`. Each exists because two very different situations produced
identical readings. Additive-only.
