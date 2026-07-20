# Feature 15 — Circuit Edge Sensing: Review Round 1

**Date:** 2026-07-20
**Scope:** commits `27427b2..94b9347` (persistence, detection core, service, wiring, API/WS, frontend, docs)
**Method:** two concurrent reviewers — detection core, and lifecycle/API/persistence/frontend
**Findings:** 37 (17 + 20) · **16 fixed** · remainder deferred or already clean

Both reviewers independently identified the same structural cause: **F15 mirrored Feature 11's
*structure* but not its three rounds of *lifecycle hardening*.** In several cases the F11 fix sits in
a code comment a few lines from the F15 gap.

---

## The critical four — all in the detection core, all invisible to the suite

### F15-R1-01 — CRITICAL: cross-layer sensing went silently dark on any real traffic
`sae_wrapper.py` — `_match_edges` called `ring.prune_before(abs_pos)` **inside the per-position
loop**. My own comment claimed "fires older than the window can never match again" — true within one
SAE's walk, and false across SAEs. The upstream layer's hook walks an entire prefill and prunes the
ring down to `last_pos - max_lag` **before the downstream layer's hook ever runs**, destroying
exactly the fires the downstream needs.

Reviewer's proof: upstream fires at pos 2, downstream at pos 4, lag window 3, one ordinary sibling
member firing at 6–11 → **0 edges detected instead of 1**. The entire point of the feature —
cross-layer detection — fails, while status still reports `armed`.

**Fix:** pruning is now a **request-level** operation. A hook cannot know whether a sibling still
needs a fire, so no hook may prune; `prune_before` carries a docstring saying so. The ring bounds
its own growth by count (`_MAX_FIRES_PER_EDGE = 512`, dropping oldest) instead.
**Tests:** `TestUpstreamNoiseDoesNotDestroyCrossLayerDetection` — including a test asserting no hook
calls `prune_before`, so re-introducing the call fails the build. **Negative control run:** restoring
the old line makes 2 of the 4 new tests fail; reverting makes 36 pass.

*Why the suite missed it:* every fixture used single-layer edges and quiet rows, and `_match_edges`
`continue`s on a row where nothing fired — so the prune never executed in any test.

### F15-R1-02 — CRITICAL: 1430 ms inside the forward hook against a 5 ms budget
The positions × edges Python loop did a scalar tensor read per edge per position. Measured at the
contract's 200-edge maximum: 45 ms @128 tok, 178 @512, 715 @2048, **1430 @4096 — 286× budget.**
`CIRCUIT_SENSING_MAX_OVERHEAD_MS` only *logged* after the fact; nothing shed load.

**Fix, in two parts.** (a) Vectorise: find fired positions **per column** once with `nonzero()`, then
iterate only over actual fires — cost scales with fire count, not `seq_len × edge_count`. Upstream
sorts before downstream at equal positions, so a same-position co-fire still correctly does not match.
(b) **Shed load before building the event list.** The cap bounds *output*, but the cost of *finding*
fires is paid first, so a miscalibrated threshold still cost 189 ms even though only 20 events
survived. A pass with more than `max(cap × 8, 2048)` fires is skipped, flagged `truncated`, and
logged — when a pass is that saturated the thresholds are wrong and the observations are noise.

**Measured after:** saturated 4096-token pass **0.9 ms** (was 1430); realistic 3.4–4.2 ms, inside
budget. Pinned by `TestSaturationLoadShedding`, including a latency assertion.

### F15-R1-03 — CRITICAL: per-SAE offsets diverged, corrupting the shared ring's key
`_edge_token_offset` advances in `finally`, but the early returns (`_suppressed`, `not _edge_began`,
`_W_enc_e is None`) and the batched-pass guard return *before* the `try`. One suppressed pass on one
SAE left its offset behind its siblings'. Because the ring is keyed on **absolute position** and
**shared**, every subsequent match used mutually shifted coordinates — fabricating matches and losing
real ones. Feature 11 tolerates the same drift because its buffer is self-contained; F15 cannot.
**Fix:** advance offset and phase on *every* pass, then decide whether to sense.
**Tests:** `TestPositionOffsetsStayInSync`, including two SAEs staying aligned when one is suppressed.

### F15-R1-04 — CRITICAL: an out-of-range column aborted the entire pass
`_assemble` hands every layer the whole spec, so a width mismatch between two layers' member slices
yields a column beyond the narrower slice. The resulting `IndexError` was swallowed by the broad
`except`, abandoning **all** positions and **all** edges for that pass — including upstream recording.
`arm_edge_sensing` validated `member_indices` against `d_sae` but never the columns.
**Fix:** validate `up_col`/`down_col` against the slice width at arm time, where it is a clean error.

---

## Also fixed

| # | Finding | Fix |
|---|---------|-----|
| 05 | **Arming never disarmed the prior SAE set**, leaking GPU hooks permanently and making several other findings reachable in practice | `arm_for_circuit` disarms first; `_armed_saes` remembers exactly what was armed so a later disarm reaches *those* SAEs, not whatever happens to be attached |
| 06 | `sensable_edges` counted only `_armed_layers[0]`'s upstream specs, so a circuit whose edges flow from a higher layer (L13→L20) reported **0 while sensing perfectly** — an operator would read that as "sensing is broken" | count distinct `edge_key` across all layers; same fix in the arm-time log |
| 07 | `_member_stats` used `setdefault`, so a `max_activation=None` from `expanded_members` could mask the member's own real value — declaring a perfectly sensable edge **unsensable depending on iteration order** | keep the largest usable value per key |
| 08 | The WS throttle dropped **entire flushes** within 100 ms, losing a whole request's events. An F15 invention: F11 only caps count-per-flush and never discards on timing | flush-level time throttle removed; per-flush count cap retained |
| 09 | An emit failure was swallowed without incrementing `ws_dropped`, so status showed events recorded, `ws_dropped=0`, and the UI showed nothing — **the discrepancy was unobservable** | everything undelivered is counted, whatever the cause |
| 10 | `collect_sensed_edges` drained without an open boundary (F11 returns `("", [], False)`), so a stray pass could surface stale edges under an empty `request_id` | F11 parity guard |
| 11 | `_edge_overhead_ms` was zeroed only at `begin`, so a layer that missed `begin` re-contributed stale overhead to the next request, inflating the number driving the warning | zeroed at collect |
| 12 | Warn flags (`_edge_batch_warned`, `_edge_saturation_warned`) were never reset, so after one warning a later independent violation went unlogged for the SAE's lifetime | reset per request |
| 13 | **Pre-existing (F11):** `_sensing_batch_warned` has the identical defect | fixed there too, per the standing "address pre-existing issues" directive |
| 14–16 | Dead `_edge_thresholds_cpu`, `threshold_mode` degeneracy, `_context` end-bound guard | recorded; see deferred |

---

## Verified clean (worth recording — these were specific suspicions)

- **Migration 012 is correct.** Head chain linear and single-headed; downgrade drops all three indexes, then the table, then the column, correctly ordered. Model exported from `db/models/__init__.py`.
- **`_CircuitSnapshot` defined after `_toggle` is legal** — the name resolves at call time from module globals. Not a defect.
- **Privacy holds.** `record()` emits `to_dict(include_context=False)` and that is the only WS path; the emitter passes payloads through untouched. Context reaches a client only via `GET /events/{id}`.
- **`edge_rung_language` is rendered verbatim everywhere**, rung<2 carries an explicit unvalidated badge in both list and detail, and `RUNG_VARIANT` maps colour only — never language.
- **`DELETE` without `circuit_id` clearing everything is intended**, matches F11, and the UI gates it behind a confirm naming "ALL".

---

## Deferred

| # | Finding | Why |
|---|---------|-----|
| A | The event cap is per-SAE but conceptually per-request; an N-layer circuit can emit up to N×cap, and a capped layer stops recording *upstream* fires, asymmetrically blinding uncapped siblings | Real, and the correct fix is a request-scoped budget owned by the service — the same restructuring as the shared-position counter below. R2 candidate |
| B | Absolute position derives from N independent per-SAE counters rather than one request-scoped counter | R1-03 makes the current scheme correct, but a single counter would make the class of bug unrepresentable. Structural; deserves its own change |
| C | Lifecycle gaps: SAE detach/attach has no F15 branch; the streaming early-exit drain and post-hang disarm exist for F11 three lines from where F15 needs them; `_notify_circuit_sensing` doesn't snapshot circuit identity (a mid-request re-arm writes circuit A's observations under circuit B's id) | **Highest-value deferred item** — finding C's last clause produces *confidently wrong data*, not just lost sensing. Scheduled first for R2 |
| D | `_edge_thresholds_cpu` is dead code; F15 computes no normalised severity score | Either consume it or delete it — R2 |
| E | `threshold_mode` can't express "mixed" | Cosmetic vs F11's per-member intent |
| F | `_context` guards `start >= total` but not `end` | No crash (`hi` is clamped); span silently truncates |
| G | `CircuitSummary` has no `sensing_enabled`, so the UI derives per-card state from the global status route; no `PUT /config` analogue for `max_token_lag` | Contract gaps recorded at implementation time |

---

## Verification

- `pytest tests/unit tests/integration` → **1560 passed, 1 skipped**
- `test_edge_sensing.py` 32 → **45**
- Negative control run for R1-01: the new tests fail against the old code, pass against the fix

**Round 1 outcome:** four criticals, every one of them a silent-failure mode — dark detection, a
286× latency blowout, corrupted shared coordinates, and a swallowed abort. All four were invisible to
a green 71-test suite because the fixtures used single-layer edges, short sequences, and quiet rows.
