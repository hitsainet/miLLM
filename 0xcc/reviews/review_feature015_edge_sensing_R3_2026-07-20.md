# Feature 15 — Circuit Edge Sensing: Review Round 3

**Date:** 2026-07-20
**Scope:** commits `27427b2..73c0b8e` (implementation + R1 + R2)
**Method:** two concurrent perspectives — Architect (owning R2's deferred items) and QA/Test +
Product. The QA reviewer ran **14 mutation experiments** (break a load-bearing line, run the suite,
revert) to find what the tests do not actually pin.
**Findings:** 27 · **14 fixed** · remainder deferred

**Five for five.** And the headline finding is the most uncomfortable one of the increment.

---

## I repeated R1's exact error, one level up

R2-01's entire finding was: *"R1 declared pruning request-level and never wired the call — and even
wrote a test asserting nobody calls it, pinning the dead state."*

R2's fix added `prune_ring`, `safe_prune_boundary` and `prune_between_passes` to the service — and
**never wired them either.** Grep returns zero production callers. I then wrote
`TestRingPruningIsWired`, a name that asserts the opposite of the truth: it checks the entry point
*exists* and computes the right boundary, never that anyone calls it. **Same bug, third round, third
shape, now with a test named after it.**

Both R3 reviewers found this independently.

**Why it kept happening:** both designs required a caller that knows *global* progress — R1 put that
knowledge in the hook (which can't have it), R2 put it in the service (which is never on the per-pass
path). The third design removes the requirement: **the ring tracks each layer's progress itself** via
`note_layer_progress(layer, through)`, called from `_sense_edges`'s `finally`, and prunes to the
`min()` across layers. No hook needs to know about siblings — which is exactly what made the previous
two designs unwireable. Bounded by construction rather than by a caller remembering.
**Tests:** `TestRingPrunesItself`, including one asserting `note_layer_progress` appears in
`_sense_edges` — pinning the *wiring*, not the entry point.

---

## Other criticals fixed

### F15-R3-02 — the per-request cap starved sibling layers
R2-03 fixed *shedding* to keep recording upstream fires because "siblings depend on it." **The cap has
the identical shape and was not fixed.** `_edge_done` caused an early `return` from the whole pass, so
a layer hitting `max_events_per_request` stopped feeding the shared ring for the rest of the request,
silently blinding every uncapped sibling. Proven by probe: layer 10 saturates on edge A, layer 13 then
sees zero firings for edge B and reports `truncated=False`.
**Fix:** the cap now suppresses only the downstream append (`continue`, not `return`), exactly as
`shed` does. **Test:** `TestCapDoesNotStarveSiblings`.

### F15-R3-03 — R2's reverse scan was O(n) on the *normal* path
R2-02 replaced a forward scan with a backward one plus an early `break`. The `break` is correct; the
`continue` is the hole. Hooks run in layer order, so the upstream layer records its **entire** prefill
before the downstream layer matches ascending — meaning every match walks the whole tail via
`continue` before reaching the window check. Measured: **39.19 ms at 4096 tokens for ONE edge**
(contract allows 200), against a 5 ms budget.

R2's benchmark missed it for the same reason R1's did: all three `TestMatchDownIsBounded` cases query
`down_pos` at or beyond the newest fire, where `break` hits on iteration one. **R2 measured the path
it didn't change — the precise error R2's own write-up attributes to R1.**
**Fix:** `bisect` to the insertion point, then scan backward from there.

### F15-R3-04 — `_request_circuit_id` survived both `collect_edges` and `disarm`
R2-04 added the identity snapshot but no teardown, so a drain arriving after a disarm attributed rows
to a circuit no longer armed — R2-04 narrowed the mis-attribution window without closing it.
**Fix:** `close_request()`, called from `_notify_circuit_sensing`'s `finally` and on disarm.

---

## Test-integrity findings — the round's most valuable output

The QA reviewer's mutation run found **four load-bearing lines that no test caught**:

| Mutation | Was it caught? | Now |
|---|---|---|
| WS payload leaks prompt text (`include_context=True`) | **NO** — 135/135 green | `TestWebSocketPayloadCarriesNoPromptText`, negative-control verified |
| `record()` reads live `circuit_id` (reverts the R2-04 critical) | **NO** — 123/123 green | pinned by the identity tests |
| Retention keeps oldest, deletes every new event | **NO** | pinned |
| `_edge_done` guard removed | latency test only | `TestCapDoesNotStarveSiblings` |

The privacy one deserves emphasis: **R1 recorded "Privacy holds" under *verified clean* — verified by
reading, not by pinning.** One word would have broken the manual's entire `### Privacy` promise
undetectably. That is the difference between a review that inspects and a review that tests.

### The harness blind spot
`FakeSAE` grafted `_match_edges` onto a hand-written six-attribute stub, so **37 detection tests never
ran `_sense_edges`** — and *both* R1's and R2's criticals lived in `_sense_edges`, not `_match_edges`.
The stub made them unrepresentable. Asked where the author's blind spots became the tests' blind
spots, the reviewer named this precisely, and it is correct: I wrote both.
**Fix:** `FakeSAE` now derives its edge state from a **real `LoadedSAE`**, so a field added to the real
class arrives automatically. This immediately caught `_edge_ambient_fired` diverging.

---

## Also fixed

| # | Finding | Fix |
|---|---------|-----|
| 05 | **EDGE-R2 `ambient_fired_count` was never implemented.** Column, migration, model and API field all shipped; nothing ever wrote it — a permanently-NULL field the API advertised. Neither prior review recorded it | accumulated per pass, summed across layers, written on every row |
| 06 | No post-hang disarm for F15. F11 has one citing 011 R1; because F15's ring is **shared**, a woken hung thread corrupts *every* layer's coordinates rather than one buffer | disarm + `close_request` in the same block |
| 07 | `sae_service.detach_sae` had no F15 branch, so detaching one SAE of an armed circuit left the service believing it was armed on a layer whose SAE was gone | mirrors the F11 call |
| 08 | Speculative decoding disables sensing entirely but invisibly — an operator sees `armed: true` and zero events | `paused_reason` on the service and the status schema |

---

## Deferred — with designs recorded

| # | Item | Why deferred |
|---|------|--------------|
| A | **Request-scoped budget + position counter owned by a `SensingRequestContext`.** Three of the eight criticals across R1–R3 share one root cause: N per-SAE counters must agree on an absolute coordinate no single component owns. A context created at `begin_request` owning the counter, ring and budget makes offset divergence, prune races and budget skew *unrepresentable* rather than test-guarded | The right fix, and too large to land inside a review round. **This is the top follow-on item.** |
| B | **Move the edge machinery out of `sae_wrapper.py`** into `millm/ml/edge_sensing.py`. 145 of 1316 lines are F15; `LoadedSAE` now carries 11 `_edge_*` fields beside F11's, and R1-03 was caused precisely by two independently-advanced counters in one class | Pairs naturally with (A) |
| C | `truncated` is OR'd across layers and stamped on every row, so one saturated layer marks a whole request's edges truncated. Design settled: per-row attribution plus `truncated_layers` in the status payload | Part of (A)'s budget rework |
| D | Two concurrently-served circuits cannot both sense; `arm_for_circuit` silently disarms the first | Single-active-circuit is F13's invariant today; revisit if that changes |
| E | F14's dial changes activations and therefore fire rates, but thresholds are frozen at arm time — turning the dial silently re-calibrates sensitivity | Genuine cross-feature interaction; needs a product decision (re-arm on dial? warn?) |
| F | FTDD §96 specifies the downstream **"pops"** the matching ring entry; the implementation reads non-destructively, so one upstream fire can father many events. The non-consuming read is arguably better evidence | **Amend the FTDD at acceptance** rather than the code |
| G | `_edge_thresholds_cpu` dead; `_member_stats` `else` arm still order-observable; unsensable list uncapped in the UI (~200 entries under slice-fallback pushes everything off-screen, inverting its purpose); `truncated` not signalled in the list view | Cosmetic/low |

---

## Verification

- `pytest tests/unit tests/integration` → **1588 passed, 1 skipped** · frontend **255 passed**
- `test_edge_sensing.py` 53 → **62** · `test_circuit_sensing_service.py` 33 → **37**
- Negative control on the privacy mutation: flipping `include_context` now **fails**

**Round 3 outcome:** three criticals, all regressions in R2's work, plus an unimplemented requirement
neither earlier round caught and four unpinned load-bearing lines found by mutation rather than by
reading. The most useful lesson is methodological: **two rounds of careful reading missed what
fourteen mutations found in one pass.** Reading confirms what the code says; mutation confirms what
the tests would notice.
