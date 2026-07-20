# Task List: Request-Scoped Sensing Context

## miLLM Feature 17

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**Feature Priority:** Secondary (Increment: Circuit Consolidation)
**References:** `017_FPRD|Request_Scoped_Context.md` · `017_FTDD|Request_Scoped_Context.md` · `017_FTID|Request_Scoped_Context.md` · `BRD-MILLM-CIRCUITS-002.md` (BR-001) · `docs/circuit-contention-model.md` (§4)

## Relevant Files
- `millm/ml/edge_sensing.py` — NEW: EdgeSpec, CircuitSensingConfig, SensedEdge, EdgeFireRing, EventBudget, the context, the matcher
- `millm/ml/sae_wrapper.py` — MOD: 145 lines of edge machinery and 13 `_edge_*` fields removed; thin `sense_edges` + `bind_context` remain
- `millm/ml/sae_hooker.py` — MOD: call site only (:181-183)
- `millm/services/circuit_sensing_service.py` — MOD: context creation; DELETE `prune_ring`/`safe_prune_boundary`/`prune_between_passes` (:526/:538/:550)
- `millm/services/inference_service.py` — MOD: begin returns a context (:1493), notify takes it (:1524), six call sites
- `millm/api/schemas/circuit_sensing.py` — MOD: `truncated_layers` on the status schema
- `tests/unit/ml/test_edge_sensing_characterization.py` — NEW, written FIRST, never edited after
- `tests/unit/ml/test_sensing_request_context.py`, `tests/unit/ml/test_edge_sensing_ring_isolation.py` — NEW
- `tests/unit/ml/test_edge_sensing.py`, `tests/unit/services/test_circuit_sensing_service.py` — MOD
- `tests/integration/test_circuit_edge_sensing_workflow.py` — UNCHANGED (the preservation proof)

### Notes
- Depends on Feature 15 (the machinery) and on `docs/circuit-contention-model.md` §4 (the required
  N-circuit shape). Feature 18 depends on THIS feature, so this lands before it. Feature 19 implements
  the concurrency this feature is shaped for.
- **This is the most defect-dense code in the arc**: 8 criticals across 3 rounds, and every round found
  a critical regression in the previous round's fix. Task 1.0 is a gate, not a preliminary.
- Test commands: `pytest tests/unit tests/integration` (backend, floor 1597), `npm test` in `admin-ui`
  (floor 272).

### Category Checklist Results
- Data: no schema change; payload-only `truncated_layers` (4.1) ✓
- Backend/API: 4.x ✓ · Frontend/UI: none (no UI surface) — status strip untouched beyond the field ✓
- Business logic: 2.x (context/ring/budget), 3.x (extraction) ✓
- Integration wiring: 3.4 (inference), 3.5 (service) ✓
- Error handling & logging: 2.4 (write-after-close), 2.5 (never break generation) ✓
- Testing: 1.0 gate, paired throughout, 5.0 mutation + benchmark ✓
- Performance & security: 5.3 benchmark (three F15 shapes); no new data surface ✓
- Config/deploy: none by design (FTID §5) ✓
- Documentation: 6.2 module docstring + contract v1.2 ✓

## Tasks

- [x] **1.0 Characterization gate — COMPLETE 2026-07-20, green BEFORE any code moved (FR-17.6, CTX-V1)**

  **Gate results.** 24 characterization tests (`test_edge_sensing_characterization.py`) + 4 parity
  baselines (`test_edge_sensing_baseline.py`), all green against the pre-extraction code.

  **The gate paid for itself immediately — it found a live defect (task 1.3's stated purpose).**
  The saturated-4096-token baseline measured **549 ms**, not the ~1 ms the F15 R1 load-shedding
  work claimed. Root cause: shedding bounded the DOWNSTREAM matching but left the UPSTREAM half
  unbounded, and the upstream half is per-edge. At the contract's 200-edge maximum a shed pass
  built ~260k events — 544 ms at 200 edges vs 5 ms at one. R2 was right that siblings depend on
  upstream recording; it simply had to be bounded too. Fixed by capping fired positions per COLUMN
  (`_EDGE_SHED_POSITIONS_PER_COL = 64`), keeping the newest since `match_down` reports the nearest
  antecedent. **544 ms → 24.7 ms**, and the cost now tracks distinct columns rather than edge count.

  **Parity targets for the extraction** (a 3x regression fails the suite):

  | Shape | Measured | Assertion |
  |---|---|---|
  | Saturated 4096-token, 200 edges | 24.7 ms | < 50 ms |
  | Realistic 4096-token, 200 edges | ~10 ms | < 60 ms |
  | Typical 512-token | ~3 ms | < 30 ms |
  | Growth 512 → 4096 | sublinear | < 8x |

  **Suite baselines:** backend **1659** passed / 1 skipped; frontend 272.

  From here on, an edit to 1.1/1.2 is a behaviour change requiring justification in the review
  record (CTX-V2).
  - [ ] 1.1 `tests/unit/ml/test_edge_sensing_characterization.py` against the CURRENT code: strict up→down ordering; lag boundary at exactly L and L+1; same-position co-fire does NOT match; newest-antecedent selection; non-destructive read (one upstream fathers several events); `_MAX_FIRES_PER_EDGE` evicts oldest; prefill→decode phase flips exactly once
  - [ ] 1.2 Characterize the eight fixed criticals by behaviour, one test each: R1-01 cross-layer survives a noisy upstream; R1-02/R2-02/R3-03 latency shapes; R1-03 offset advances on EVERY return path (suppressed, batched, raising); R1-04 out-of-range column is a clean arm-time error; R2-03 shed still feeds siblings; R3-02 cap still feeds siblings; R2-04/R3-04 identity snapshotted and released
  - [ ] 1.3 Run against current code — **all green**. Record the count in the task list. A failure here is a live defect found before the refactor, not a test bug
  - [ ] 1.4 Record the baseline: backend/frontend suite counts, and the three benchmark measurements (saturated 4096-token pass, 200-edge circuit, cross-layer ordering) — these are the parity targets
  - [ ] 1.5 **Gate:** do not start 2.0 until 1.1–1.4 are done. From here on, an edit to 1.1/1.2 is a behaviour change requiring justification in the review record (CTX-V2)

- [ ] 2.0 The context (covers FR-17.1, FR-17.3, FR-17.4; CTX-C1..C3, CTX-B1..B4, CTX-L1..L3)
  - [ ] 2.1 `millm/ml/edge_sensing.py` skeleton: module docstring recording WHY the context owns position/rings/budget, citing the three rounds (CTX-E1, no `sae_wrapper` import)
  - [ ] 2.2 `SensingRequestContext` (name resolved vs Feature 11's at `inference_service.py:110` — decide and record): `position`, `phase`, `circuit_ids` frozenset snapshot, `advance(layer, seq)` advancing position+phase AND calling `note_layer_progress` on every ring unconditionally (CTX-C1..C3, closes FPRD §15.6 / EC-17.1)
  - [ ] 2.3 `rings: dict[circuit_id → EdgeFireRing]` — **one per (request, circuit)** — with `ring(circuit_id)`; docstring carrying the fabrication rationale verbatim (CTX-R1, CTX-R2)
  - [ ] 2.4 `close()` + write-after-close: `advance` returns -1 and logs; a late write never lands in the next request (CTX-L2, EC-17.5)
  - [ ] 2.5 `EventBudget` with `try_spend(circuit_id, layer)` per-circuit attribution and `truncated_layers(circuit_id)`; False means the CALLER continues, never returns (CTX-B1..B4, EC-17.3)
  - [ ] 2.6 Unit tests `test_sensing_request_context.py`: advance/close/double-close/write-after-close; progress reported on suppressed passes (§15.6 regression pin); budget isolation across two circuits; `try_spend` False never suppresses upstream recording; `truncated_layers` names only the shedding layer
  - [ ] 2.7 Unit tests `test_edge_sensing_ring_isolation.py`: two circuits with the SAME `edge_key`, A fires upstream, B fires downstream in window → **zero** cross matches; negative control against a single shared ring documented and run once by hand (CTX-R2, US-17.2)

- [ ] 3.0 Extraction (covers FR-17.5; CTX-E1..E5)
  - [ ] 3.1 Move `EdgeSpec` (:51), `CircuitSensingConfig` (:78), `SensedEdge` (:98), `EdgeFireRing` (:123 incl. `record_up`, `match_down`, `prune_before`, `note_layer_progress`, `clear`, `_MAX_FIRES_PER_EDGE`) into the new module **with their docstrings intact** (FTID pitfall 5); hoist `import bisect` to module scope
  - [ ] 3.2 Move the matcher (`_match_edges` :1137) and the sensing body (`_sense_edges` :1050) into module-level functions taking the context; delete the triplicated offset advance (:1072/:1093/:1123) in favour of one `ctx.advance()` above the guards
  - [ ] 3.3 `LoadedSAE`: thin `sense_edges` wrapper + `bind_context`; `arm_edge_sensing` (:938) drops the `ring` parameter; retain the `d_sae` bounds check (:947-957) and R2-07's `-1 <= col < width` verbatim; keep `is_edge_sensing_armed` a plain boolean (EDGE-S3)
  - [ ] 3.4 Reduce the 13 `_edge_*` fields to the four the FTDD §4 table retains; **delete `_edge_thresholds_cpu`** (dead since R1-14, re-recorded R2-E/R3-G) rather than moving it (CTX-E3)
  - [ ] 3.5 **Delete** `prune_ring` (:526), `safe_prune_boundary` (:538), `prune_between_passes` (:550) from `circuit_sensing_service.py` and their tests (`test_circuit_sensing_service.py:412/:423/:437`) — zero production callers, R2's superseded design (CTX-E5)
  - [ ] 3.6 Absorb `begin_edge_sensing_request`'s caller-clears-the-ring convention (:1028 docstring) into context construction; no participant clears anything
  - [ ] 3.7 Verify `to_device` (:1316) against the split — edge weight caches stay on the SAE; a cache on the wrong device is a silent wrong answer, not a crash
  - [ ] 3.8 `sae_hooker.py:181-183` call-site update; retarget `test_edge_sensing.py` imports and drop the hand-written stub fixtures in favour of the real classes (the R3 harness blind spot)

- [ ] 4.0 Wiring + payload (covers CTX-B4, CTX-L1..L3)
  - [ ] 4.1 `circuit_sensing_service.py`: `begin_request` builds the context over the armed circuit **set** (F19-ready) and binds it to each SAE; `collect_edges` drains once and returns per-circuit `truncated_layers` instead of a request-wide boolean
  - [ ] 4.2 `inference_service.py`: `_circuit_sensing_begin` (:1493) returns the context; `_notify_circuit_sensing` (:1524) takes it; `close_request` moves onto the context — **all six call sites together** (:1857/:2025/:2349, :1938/:2301/:2415) (FTID pitfall 11)
  - [ ] 4.3 `truncated_layers` on the status schema + route response (BR-006); `docs/mcp-contract.md` → v1.2, additive only
  - [ ] 4.4 Route/schema tests for the new field; assert no other field, envelope, error code or WS payload changed

- [ ] 5.0 Verification (covers FR-17.6; CTX-V1..V4)
  - [ ] 5.1 **Characterization suite green, unmodified** — diff `test_edge_sensing_characterization.py` against its 1.x commit and confirm zero changes (CTX-V2). Any diff is a finding
  - [ ] 5.2 **Mutation testing on `edge_sensing.py`** (R3's practice: break, run, revert, record). Minimum set: strict-before in `match_down`; window comparison; `bisect` insertion point; the unconditional `advance`; `note_layer_progress` inside it; `try_spend` boundary; **ring lookup key mutated to a constant — this MUST fail, and if it does not, 2.7 is not pinning what it claims**. Every survivor pinned or recorded with a reason (CTX-V3)
  - [ ] 5.3 Benchmark the three F15 shapes against 1.4's baseline: saturated 4096-token pass, 200-edge circuit, cross-layer ordering (upstream records a full prefill before downstream matches ascending). All three, because each F15 benchmark measured a path its own fix had not changed
  - [ ] 5.4 `test_circuit_edge_sensing_workflow.py` passes **unchanged** — the outside-boundary preservation proof
  - [ ] 5.5 Full suites: backend ≥1597, frontend ≥272 (CTX-V4)

- [ ] 6.0 Feature Acceptance (per instruct 008)
  - [ ] 6.1 Verify FPRD §9 criteria 1–8 and all US/EC boxes one-by-one; re-verify every Feature 15 §9 criterion, none regressed
  - [ ] 6.2 Module docstring records the three-round history; `docs/mcp-contract.md` v1.2 confirmed; CLAUDE.md Document Inventory + Current Status updated
  - [ ] 6.3 Confirm the structural metrics: exactly 1 position counter per request; 0 `_edge_token_offset` fields on `LoadedSAE`; 0 production-callerless prune methods; `sae_wrapper.py` line count reduced by ~145

## Coverage Audit
- FR-17.1→2.2/3.2/6.3; FR-17.2→2.3/2.7/5.2; FR-17.3→2.5/2.6; FR-17.4→2.2/2.4/3.6; FR-17.5→3.x; FR-17.6→1.0/5.1/5.2 ✓
- US-17.1→2.2/2.6/6.3; US-17.2→2.3/2.7; US-17.3→2.5/2.6; US-17.4→3.1/3.8 — implementing + testing sub-tasks each ✓
- EC-17.1→2.2/2.6; EC-17.2→4.1; EC-17.3→2.5/2.6; EC-17.4→2.3/2.7; EC-17.5→2.4; EC-17.6→1.2/3.2; EC-17.7→5.3 ✓
- BRD: BR-001→2.0/3.0 (all CTX-C/R/B/L ids); BR-006→2.5/4.3; BR-011 (enabling)→2.3/2.7 ✓
- CTX-E1..E5→3.1/3.3/3.4/3.5/3.8; CTX-V1..V4→1.0/5.x ✓
- TDD/TID sections mapped (context→2.x, extraction→3.x, wiring→4.x, verification→5.x) ✓
- Open questions: none — no spike tasks ✓
- Final task is Feature Acceptance ✓

## Review rounds (goal: 3 rounds, ≥10 findings each)

- [ ] **Round 1** (multi-angle `/code-review`, 2 finder agents): ≥10 findings — fix criticals, document deferrals.
      **Watch items, drawn from the F15 review history:**
      1. A return path added in the move that skips `ctx.advance()` — R1-03's exact shape, and the live
         `note_layer_progress` gap proves the shape recurs even after being fixed once.
      2. A ring shared where it must be per-circuit — grep every `EdgeFireRing(` construction and every
         `ring(` lookup; a lookup keyed by anything but `circuit_id` fabricates observations.
      3. `try_spend`/shed returning where it must continue — the R2-03/R3-02 starvation bug through a
         third door.
      4. A mechanism declared and never wired. **Three rounds in a row produced one** (R1 pruning, R2
         pruning again, and R3's own `TestRingPruningIsWired` asserting an entry point exists rather
         than that anyone calls it). For every new method: grep for a PRODUCTION caller, and require
         the test to fail when the wiring is cut, not when the method is deleted.
      5. A latency fix benchmarked on the path it did not change — R1, R2 and R3 each did this once.
      6. A docstring lost in the move, particularly `prune_before` and `note_layer_progress`, which
         carry the reasons two earlier designs failed.
      7. Dead code laundered into the new module (`_edge_thresholds_cpu`, R2's prune trio).
      8. A characterization test edited to make the refactor pass.
      9. `_reset_edge_buffer`-style state that resets some fields and not others (R2-06: `_max_token_lag`
         survived a disarm and leaked into the next circuit).
      10. A broad `except` converting a real error into silent non-detection (R2-10 — a `NameError`
          swallowed into a green suite).
- [ ] **Round 2** (post-fix verification + fresh angles): ≥10 findings — verify R1's fixes hold; hunt
      regressions **in R1's own work specifically**, since twelve of twelve rounds across this arc found
      one. Re-run the mutation set after R1's fixes, not just the suite.
- [ ] **Round 3** (`/review`, 4 perspectives incl. QA/mutation): ≥10 findings — fix, pin mutation
      survivors. Explicitly ask: does any test assert that a thing EXISTS rather than that it is CALLED?
- [ ] Record: `.claude/context/sessions/review_feature017_R{1,2,3}_2026-07-*.md`
- Directive: fix latent/pre-existing defects surfaced during review, not only regressions.

## Acceptance evidence
*[To be completed at acceptance — FPRD §9 criteria 1–8 verified one-by-one, with the characterization
diff, the mutation record, the three benchmark comparisons against the 1.4 baseline, and the structural
metrics from 6.3.]*
