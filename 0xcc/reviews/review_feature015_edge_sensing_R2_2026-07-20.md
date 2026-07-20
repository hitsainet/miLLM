# Feature 15 — Circuit Edge Sensing: Review Round 2

**Date:** 2026-07-20
**Scope:** commit `69bedf6` (R1's fixes) + the implementation beneath it
**Method:** adversarial pass with the explicit mission *find bugs in Round 1's own fixes*
**Findings:** 16 · **11 fixed** · 5 deferred

**Four features running, four for four.** Three of R1's four "critical fixes" introduced new
defects, and R1's own success metrics concealed them.

---

## The R1 fixes that broke something

### F15-R2-01 — CRITICAL: R1 moved pruning to "request-level" and never added the call
R1-01 declared that only `CircuitSensingService` may prune, since a hook cannot know whether a
sibling still needs a fire. It then **never wired the call**. `grep prune_before` returned only the
definition, its comment, and tests — including one I wrote asserting *nobody* calls it, which
**permanently pinned the dead state**. The ring bounded only by count, and `_fires` is a dict keyed
by `edge_key` with no eviction, so a 200-edge circuit retained 200×512 tuples until `clear()`.

**Fix:** `prune_ring`, `safe_prune_boundary` and `prune_between_passes` on the service. The boundary
is the **minimum** `_edge_token_offset` across armed layers — pruning above that would discard a fire
a lagging sibling still needs.
**Tests:** `TestRingPruningIsWired`, including one asserting the boundary is the slowest layer.

### F15-R2-02 — CRITICAL: R1 traded a 286× latency blowout for a 15× one
R1-02's switch to count-based bounding made `match_down` linear-scan up to 512 retained fires **per
downstream fire** — measured at 19.2 µs/call, **78.5 ms on a 4096-token pass against a 5 ms budget.**
R1's "0.9 ms saturated" measurement only exercised the *shed* path, which returns before any matching
happens, so the benchmark never touched the new hot spot.

**Fix:** `fires` is ascending, so scan **backward and break at the window edge** — the first hit is
already the newest antecedent and everything earlier is out of window.
**Tests:** `TestMatchDownIsBounded`, with a latency assertion against a full ring.

### F15-R2-03 — CRITICAL: load shedding starved sibling layers
R1-02's shedding returned **before recording any upstream fires.** Shedding is decided per-SAE
per-pass, so a saturated *upstream* layer recorded nothing into the shared ring while a quiet
*downstream* sibling — which did not shed — reported `truncated=False` and detected **zero edges it
should have detected.** The truncation flag landed on the layer that shed, not the layer that lost
data: the operator saw a clean, empty result. **This is exactly the silently-dark mode R1-01 existed
to eliminate, reintroduced through the R1-02 fix.**

**Fix:** shedding now skips only the expensive *downstream matching*. Upstream recording is a dict
append, it is what siblings depend on, and it is kept.
**Test:** `TestShedStillFeedsSiblings` — a saturated upstream layer must still feed a quiet sibling.

### F15-R2-04 — CRITICAL: circuit identity was read at drain time (R1 deferred item C)
`record()` read `self._circuit_id` when draining, so a re-arm between `begin_request` and the flush
persisted **circuit A's observations under circuit B's id** — confidently wrong data, not merely lost
sensing. `request_id` was already snapshotted per-SAE; identity was not.
**Fix:** `begin_request` snapshots `_request_circuit_id` (and the context-token count) when the
boundary opens; `record()` uses that.
**Test:** `TestRequestIdentityIsSnapshotted`.

---

## Also fixed

| # | Finding | Fix |
|---|---------|-----|
| 05 | `build_configs` mutated `self._max_token_lag` **before arming could fail**, so a circuit that never armed still changed the reported lag — and the next `EdgeFireRing` was built from it | the lag is committed only by a successful arm |
| 06 | `disarm` cleared every field **except** `_max_token_lag`, so a circuit with no override silently inherited the previous circuit's window | reset to the configured default |
| 07 | Arm-time column validation checked only the upper bound, so `-2` passed and the matcher's `0 <= col` guard silently skipped that half — the edge reported armed and sensable and simply **never fired**. R1-04 replaced a loud `IndexError` with silent non-detection | `-1 <= col < width`; `-1` is the legitimate "not my half" sentinel |
| 08 | `_emit` kept the **first** 5 payloads, and `collect_edges` sorts by `down_pos` — so a live panel always showed a request's *earliest* edges and never its most recent, the opposite of the ring's own "recent history is what matters" policy | keep the newest; everything undelivered still counted |
| 09 | `should_emit`/`_last_ws_emit_ts`/`_WS_MIN_INTERVAL_S` were left write-only dead code after R1-08 removed the flush throttle, while F11's identically-named fields are live — two services silently differing behind the same names | documented as a compatibility no-op with the reason recorded |
| 10 | **Process finding.** While fixing R2-03 I introduced a `NameError` (`shed` computed in `_match_edges`, referenced in `_sense_edges`). The broad `except` swallowed it and turned a hard crash into **silent non-detection**; the ring tests stayed green and only an end-to-end assertion caught it | `TestSensingFailuresAreNotSilent` — a raising pass must log, and must still advance the offset so it cannot desynchronise siblings |
| 11 | **`millm/sockets/progress.py` was structurally corrupt.** My earlier regex insert appended the emitter method *after* the module-level singleton, creating an orphaned second class body and a duplicate `progress_emitter`. `emit_circuit_sensing_event` was therefore **not a method of `ProgressEmitter` at all** — every WS broadcast would have failed at runtime, and no test covered it | file repaired; `emit_sensing_event`, `emit_circuit_sensing_event` and `create_socket_io` all verified present on the singleton |

R2-11 deserves emphasis: the feature's entire live-broadcast path was dead, and the only reason it
surfaced is that a new test called the method through the real emitter instead of a mock.

---

## Deferred

| # | Finding | Why |
|---|---------|-----|
| A | The event cap is per-SAE but conceptually per-request; a capped layer stops recording upstream fires, blinding uncapped siblings. R1-01's shared-ring design gave the cap cross-layer blast radius | Needs a request-scoped budget owned by the service — the same restructuring as the shared position counter. **Leads R3.** |
| B | `truncated` is OR'd across layers and stamped on every row, so one saturated layer marks every edge of the request truncated | Same restructuring as (A) |
| C | The ring's `_max_lag` and `config.max_token_lag` are independent values that agree only via a side effect | R2-05 made the side effect reliable; the coupling should still be made explicit |
| D | No post-hang disarm branch for F15 (F11 has one, with a comment citing 011 R1). A hung thread writes into the next request's ring — and because the ring is *shared*, it corrupts every layer | R3 |
| E | `_edge_thresholds_cpu` still dead; `_member_stats` max-vs-first-non-None | R3 |

---

## Verification

- `pytest tests/unit tests/integration` → **1575 passed, 1 skipped**
- `test_edge_sensing.py` 45 → **53** · `test_circuit_sensing_service.py` 26 → **33**

**Round 2 outcome:** four criticals, all of them regressions in R1's own work, and one dead
broadcast path that no test had ever exercised. The recurring shape is unchanged from earlier
features: a fix that is correct in isolation but wrong about the system around it — pruning declared
request-level without a caller, a benchmark that measured the path it didn't change, and a load-shed
that protected latency by starving the very sibling coordination the previous fix established.
