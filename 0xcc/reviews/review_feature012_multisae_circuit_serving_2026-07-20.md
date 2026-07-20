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

Round 2: pending. Round 3: pending.
