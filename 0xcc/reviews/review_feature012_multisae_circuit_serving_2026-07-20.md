# Review — Feature 12 (Multi-SAE Attach & Circuit Serving)

Scope: `git diff abde92b..HEAD` (F12 commits e456adb, 8c29f88, 6b01a80, f738e07).
Goal: 3 rounds, ≥10 findings per assessment (≥20/round). Fix bugs + resilience/reliability/UX.

## Round 1 (2026-07-20) — 3 parallel finder agents (correctness+concurrency / back-compat+resilience / UX+frontend+tests)

**36 candidate findings.** Consolidated + triaged below (dedup'd). H=high, M=med, L=low.

### Confirmed & FIXED in R1
1. **[H] set_circuit_steering merges, not replaces** — `set_steering_batch` unions into `_steering_values`, so a prior circuit/cluster/manual steering on a layer leaks into a new serve. → clear each participating layer's SAE before applying.
2. **[H] double `by_layer` TOCTOU + None deref** — apply loop re-resolves `by_layer(layer).sae` (2nd call, no lock, no None guard). → resolve each layer→entry ONCE in step 1, reuse in apply; guard None.
3. **[H] intensity λ never bounds-checked** — negative λ inverts the whole circuit; huge λ saturates. → clamp to `[CIRCUIT_INTENSITY_MIN, CIRCUIT_INTENSITY_MAX]` at entry.
4. **[H] detach_sae multi-layer cleans only own_entries[0]** — a sae_id on 2 layers: 2nd SAE stays armed/resident. → loop ALL own_entries for clear_steering/monitoring/disarm/to_cpu.
5. **[H] detach_sae unconditionally unlocks the model** while other circuit SAEs remain hooked. → gate the auto-unlock on `not is_attached` (registry empty).
6. **[H] attach_set: no rollback + no VRAM pre-check + skips DB/lock/sensing side-effects** — mid-set failure leaks GPU weights & leaves DB/lock inconsistent. → per-item try/except frees the just-loaded SAE + rolls back this call's entries on failure; add free-VRAM pre-check; create attachment rows + lock model per attached key.
7. **[H] attach_sae single-attach guard trips on any registry entry** — after attach_set, a normal single attach of a different layer is wrongly "already attached". → make the guard key-aware (`get(sae_id,layer)`), not `is_attached`.
8. **[M] unlocked reads** in `by_layer`/`get`/`_first`/`entries` comprehensions → dict-changed-size race under concurrent clear. → snapshot under `_lock`.
9. **[M] duplicate (layer, feature_idx) members silently last-write-wins** → detect & reject (SAE_SET_INCOMPLETE reason=duplicate_member).
10. **[L] memory_freed_mb `sum(...) or fallback`** truthiness → real 0 falls through. → explicit `if own_entries`.
11. **[L] SAESetIncompleteError message** says "no attached SAE" even for ambiguous/out-of-bounds → per-offender reason already partially present; broaden message.
12. **[M] AttachmentPanel total `?? 0`** shows false "0 MB" when null → render "—".
13. **[M] AttachmentPanel error leaks raw error.message** → friendly message.

### Consumer back-compat (single-first reads) — TRIAGED
The singular properties returning `entries[0]` mean health/cluster/monitoring/sensing under-report or mis-bind when >1 SAE is attached. **Decision:** F12 ships multi-SAE ATTACH + circuit SERVING; the cluster/monitoring/sensing binding-by-layer belongs to F13 (which owns circuit activation + slice-fallback) and is tracked there. For F12 we (a) fix the health `/detailed` + `/metrics` to report the plural count (cheap, avoids silent under-report), and (b) leave a documented note that single-SAE consumers bind to the first entry until F13. Not silently ignored — see F13 tasks.

### Deferred to R2/R3 or later (valid, not R1-critical)
- Frontend attach/detach ACTIONS + Circuits tab wiring (panel is currently read-only/orphaned) — **UX gap, fix in R2**.
- FTASKS 4.2/4.3 drift (plural detach route claimed, only singular exists) — **correct the record or add route in R2**.
- Hazard key uses (layer, feature_idx) without sae_id (cross-SAE index collision) — fix in R2.
- clear_circuit_steering wipes whole layer (co-tenant cluster) — fix in R2 (track applied feature set).
- Missing tests: partial-failure rollback, non-first detach, 3+-member hazard/fan-out, concurrency, intensity bounds, route vram boundary/null — add across R2.

## Round 2 (2026-07-20) — 2 parallel finders (verify R1 fixes / fresh angles + deferred)

**21 candidate findings** (all 6 R1 fixes verified correct; the hypothesized `resolved[layer]` KeyError was traced and DISPROVEN). Two genuinely-critical NEW bugs the first round missed.

### Confirmed & FIXED in R2
1. **[H] CircuitMember double-negation** — `budget * sign * intensity` violated the canonical sign rule (a negative budget is already directional; `cluster_service` fixed this exact bug). A `budget=-100, sign=-1` suppression served as `+100·λ`. → added `_directional_budget()` (shared rule), applied in both `set_circuit_steering` and `_cross_layer_hazards`.
2. **[H] Free-VRAM pre-check defeated by NULL `file_size_bytes`** — projection collapsed to 0, letting a huge attach OOM. → estimate from dims (2·d_in·d_sae·bytes), `max(by_dims, by_file)`, +10% headroom.
3. **[M] intensity==0 left steering "enabled" with zeros** — /metrics counted N active features while hazards showed none. → λ=0 (or all-zero) clears + disables the layer.
4. **[M] MIN>MAX misconfig silently pinned every serve to `lo`** → detect inverted bounds, fall back to [0,2] with a warning.
5. **[M] Orphaned UI** — `AttachmentPanel` was rendered nowhere → mounted on `SAEPage` (in a Card, below the single-SAE card).
6. **[L→M] member.sae_id ignored → wrong-basis on ambiguous layer** — a member naming its SAE now resolves via `get(sae_id, layer)` first (disambiguates two-SAE-on-one-layer AND prevents a silent wrong-SAE serve).
7. **[M] FTASKS 4.2/4.3 route drift** — claimed `/api/sae/*` (singular) + a nonexistent plural detach → corrected to `/api/saes/*` + noted the multi-SAE-aware singular detach; clarified detach-one-vs-all is covered at the registry/service level.

### Assessed & intentionally NOT changed (with rationale)
- **[H] Circuit/cluster co-tenancy clobber** (serving a circuit clears a co-located active cluster's steering) — REAL, but the cross-subsystem ownership guard belongs to **F13 circuit activation** (which owns deactivating a conflicting cluster + surfacing the warning). F12 has no activation caller yet. **Recorded as an explicit F13 requirement** — not silently dropped.
- **[H] resolve/apply detach race** (no composing `_attachment_lock`) — the individual registry ops are locked; a full cross-call mutex is a broader change touching attach/detach/serve. F12 serving has no live concurrent caller yet (activation is F13). **Recorded for F13**; the docstring's stale `_attachment_lock` claim noted.
- **[L] cross-SAE hazard key collision** — DISPROVEN (keys include both layers; duplicates rejected).
- **[L] bf16 envelope** — bf16 and fp16 are both 2 bytes; no drift. `attach_bytes` now derived from `torch.finfo(dtype)`.
- **[L] dead fields** (steering_apply_count/monitoring_enabled unrendered) — kept in the API for diagnostics; panel render deferred (cosmetic).

+7 tests (sign rule ×2, λ=0 disable, sae_id disambiguation, +R1 adjustments). Backend 1002 passed / 1 skipped; admin-ui 217 passed; tsc clean.

## Round 3 (2026-07-20) — /review 4 perspectives (product + architect / QA + test engineer)

**23 findings, 3 of them REPRODUCED live bugs that survived both prior rounds.**

### Confirmed & FIXED in R3
1. **[CRITICAL] R2's sae_id fix was defeated by a layer-keyed cache** — `resolved.get(m.layer)` short-circuited before the sae_id branch, so with two SAEs on one layer the FIRST member's SAE captured every later member on that layer (**reproduced**: sae-b's member applied to sae-a's decoder). R2's test passed only because it used a single member. → cache keyed by `(sae_id, layer)`; grouping/apply keyed by the RESOLVED entry `(entry.sae_id, entry.layer)`; `applied_per_layer` still reported per-layer for the caller contract. Regression test added (2 members, 2 SAEs, 1 layer).
2. **[HIGH] Empty member list was a silent no-op** leaving the PREVIOUS circuit armed and firing while returning success (**reproduced**). → empty members now clears + disables every attached layer (explicit OFF).
3. **[HIGH] A member naming an unattached sae_id silently fell back** to whatever SAE is on that layer — a wrong-basis serve with no signal. → the substitution is recorded and surfaced in the result (`clamp_warnings`) + logged.
4. **[HIGH] `attach_set` service body had ZERO direct tests** — the R2 rollback + free-VRAM gate were entirely unverified (route tests mock attach_set away). → new `test_attach_set_service.py`: 11 tests covering happy path, idempotent skip, dedup, pre-validation (incompatible/missing → nothing loaded), hook-install failure rollback, load failure rollback, pre-existing attachment preserved, VRAM refusal before loading, NULL file_size_bytes still gated, sufficient-VRAM proceed.
5. **[MED] Step 2 re-iterated the raw `members` list**, discarding step 1's dedup/bounds guarantees (a trap if duplicate ever downgrades to a warning). → step 2 now iterates the `validated` list carrying each member's resolved entry.
6. **[MED] SAEPage rendered the panel unconditionally** → duplicate/empty display for the single- and zero-SAE cases (contradicting its own comment). → `AttachmentPanel` gained `minCount` (returns null below threshold, before loading/error branches); SAEPage passes `minCount={1}`. 2 tests added.
7. **[LOW] `is_attached`/`count` read `_entries` unlocked**, inconsistent with every other accessor. → both now take the lock (verified no reentrancy).

### Assessed & deliberately NOT changed (recorded, not dropped)
- **[HIGH] Cluster activation binds to `entries[0]` with only a WARNING on layer mismatch** — the singular back-compat property means cluster serving can pick the wrong SAE once a circuit set is attached, i.e. F12's "never a silent wrong-basis serve" holds for circuits but not for clusters. This is a **pre-existing cluster-path behavior** that F12's registry makes reachable. Correct home is **F13** (which owns circuit/cluster activation + the co-tenancy ownership guard) — recorded there as a REQUIRED task, not deferred silently.
- **[HIGH] `attach_set` omits attach_sae's side effects** (DB `SAEStatus.ATTACHED`, `create_attachment`, model auto-lock, sensing re-arm). Real inconsistency. Attachment for circuits is in-memory by FPRD §4 design, but the model-lock omission is a genuine risk. → **F13 task** (activation is where a circuit's attachment becomes user-visible/persistent).
- **[MED] `set_circuit_steering` has no route/MCP caller in F12** — activation is F13 by FPRD §12 design. FTASKS 7.1 amended to state §9.1/9.3/9.4 are **service-level verified**, user-observable verification deferred to F13 acceptance (rather than claiming them verified).
- **[MED] Hazards are O(n²) and mostly `heuristic:co-steer-sign`** — a 6/6 two-layer circuit emits 36 low-signal warnings. Contract is correct; **presentation/ranking is F13's** (it renders them). Recorded.
- **[MED] No composing attach/detach/serve lock** (the docstring's `_attachment_lock` never existed) — recorded for F13; the stale docstring claim should be corrected there.
- **[LOW]** Shared `clamp_intensity` helper (cluster vs circuit envelopes), `/health/detailed` singular `sae_id`, dead `steering_apply_count`/`monitoring_enabled` render, detach drain `except: pass` breadth, concurrency stress test — all recorded as follow-ups.

+16 tests (3 R3 regressions + 11 attach_set service + 2 panel minCount). **Backend 1016 passed / 1 skipped; admin-ui 219 passed; tsc clean.**

## Outcome
3 rounds · **80 findings surfaced** (36 + 21 + 23) · **27 fixed** · zero regressions · every deferral has a named owner (F13) rather than being dropped. The three reproduced serving bugs (layer-keyed cache, empty-members no-op, silent SAE substitution) were only findable because each round attacked the *previous round's fixes* — R2's own fix contained R3's critical bug.
