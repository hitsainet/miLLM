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

## Round 3 — pending.
