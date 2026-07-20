# Feature 017 — Review Round 1 (2026-07-20)

**Suite:** 1695 → **1739** green / 1 skipped. Frontend 272.
**20 findings, 20 fixed.** Every fix negative-controlled: the mutation that
reverts it was run and the pinning test was watched to fail.

## Criticals

| # | Finding | Verified by |
|---|---|---|
| 01 | **`EventBudget` entirely dead** — `try_spend` had zero production suppliers, so the per-circuit cap never applied | cap 3 → **9 events**, `spent` 0 |
| 02 | **A dark layer reported COMPLETE** — a layer absent at `begin_request` was skipped, and `truncated_layers: []` claims every layer reported | half the circuit blind, status clean |
| 08 | **A reverted O(log n) fix** — `bisect_left` became a linear walk during the "pure move"; the docstring still described the bisect | 7.38ms → 0.55ms, **13x** |
| 14 | **`ws_dropped` counted throttling as loss** — a healthy 20-event flush reported 15 dropped | measured |
| 16 | **`record()` had no test caller** — the WS privacy guarantee's only coverage was a source grep for a substring | `include_context=True` mutation now caught |

## The rest

03 concurrency guard (`MAX_CONCURRENT_REQUESTS>1` fabricated attribution) ·
04 stale `truncated_layers` survived disarm (`layers: []` + `truncated: [13]`) ·
05 budget refusals dropped events without flagging truncation *(introduced by
the 01 fix)* · 06 every skip now names itself + `requests_sensed` distinguishes
quiet traffic from sensing that never ran · 07 a shed truncated without telling
the budget · 09 truncation attributed to `spec.down_layer`, naming an unarmed
layer · 10 the growth baseline's `short*8` term never bound · 11 `total <= 3`
passed when sensing was dead · 12 `to_device` tested a CPU→CPU no-op · 13 a
source grep guarding the wrong function after task 3.1 moved the matcher ·
15 the ambient count was order-dependent (3 or 9 from the same state) ·
17/18 batch-of-one and mid-request migration had no coverage · 19 an exception
test with a `try/except` escape hatch · 20 the speculative-decoding rationale
lived only in a comment.

## What this round says about the method

**Five of the twenty were mine, introduced during F17 itself** — 01, 05, 08,
and the two acceptance claims below. The extraction was described as a pure
move and was not: it dropped a bisect and left a mechanism unwired.

**Reading did not find any of the criticals.** Each came from executing the code
or mutating a line. 08 in particular had *survived my own earlier mutation
sweep* — `bisect_left → bisect_right` was recorded as a harmless survivor, and
it was harmless precisely because the bisect was no longer being called.

**Two acceptance claims were wrong and are corrected in the repo:** FPRD
criterion 3 ("budget attributed per circuit") was recorded as met on three green
tests that exercised an unwired class, and criterion 8 was partially wrong.

**Fixes introduce defects.** Finding 05 exists only because the 01 fix created
it: wiring the budget made refusals drop events *without* flagging truncation —
silent-dark reintroduced by the fix for silent-dark. Round 2 must attack these
fixes specifically.

**Process:** editing the tree while mutation-testing subagents ran in it cost
three silent reverts of uncommitted work. Commit first, or stage outside the
repo. Recorded to memory.

Two reported findings did **not** reproduce (an `EventBudget` off-by-one and a
`cap=0` bypass) and were left alone rather than "fixed" — verified directly:
cap=3 allows exactly 3, cap=0 refuses the first spend.
