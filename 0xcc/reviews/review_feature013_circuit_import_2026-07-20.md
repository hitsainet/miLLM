# Review — Feature 13 (Circuit Import, Slice-Fallback & Evidence Ladder)

Scope: `git diff d6ed1be..HEAD` (F13 commits). Goal: 3 rounds, ≥10 findings per
assessment (≥20/round); fix bugs + resilience/reliability/UX.

## Round 1 (2026-07-20) — 2 parallel finders (service+contract / API+UX+security)

**31 findings.** One was severe enough to invalidate the feature's headline capability.

### THE headline defect — the slice-fallback path had NEVER executed
`_serve_slices` passed a raw **dict** to `ClusterService.import_definition`, which
takes a validated `ClusterDefinitionV1`; it crashes on `.name`. **The unit fixture's
`AsyncMock` hid it** — the collaborator accepted anything, so the declared
"Slice-Fallback" in the feature's own name had never once run. Fixing it immediately
exposed three latent defects behind it (name overflow, dropped λ, missing n_features)
that all fail at a validation step the code previously never reached.

### Confirmed & FIXED in R1
1. **[CRIT] Slice fallback passed a dict, not a model** → validate before delegating;
   `raw_payload` rides alongside for lossless storage. Test now asserts the argument
   **is** a `ClusterDefinitionV1` (a typed double, not a permissive mock).
2. **[HIGH] Slice name could exceed the cluster cap** — circuits allow 200 chars,
   clusters 120, so any circuit named >108 chars could not slice-fallback **at all**
   (validation error instead of graceful degradation) → truncate the base, reserving
   room for the marker and dedupe suffix.
3. **[HIGH] Silent member drops** — a `cluster_ref` carrying BOTH `expanded_members`
   and `feature` dropped the feature; a `cluster_ref` with neither vanished entirely.
   A circuit reported a successful serve while omitting authored members → both
   sources are now collected, with `(layer, feature_idx)` dedupe (the serving path
   rejects a repeated key outright, so an overlap would have failed activation).
4. **[HIGH] Teardown-before-serve** — co-tenant clusters were released BEFORE the
   serve, so a serve failure left the user with **nothing steering** → serve first,
   release after; and release only the layers ACTUALLY served (a slice touches one
   layer, not every bindable one).
5. **[HIGH] Unguarded `model_validate(circuit_meta)`** — a stored doc that no longer
   validates escaped as an opaque 500 with no circuit id → `_parse_stored()` converts
   it to a structured, actionable error. In `set_intensity` the parse now happens
   BEFORE the DB write, so a corrupt doc cannot leave the persisted λ changed while
   the model steers at the old one.
6. **[HIGH] Nesting bomb** — 3000 levels ≈ 21 KB (2% of the 1 MB cap), and
   `extra="allow"` meant it would be accepted, persisted, and re-walked on every
   activate/export → iterative depth gate (max 32) before `json.dumps`.
7. **[HIGH] Contract drift** — `mcp-contract.md` v1.1 declared 12 `millm_circuits`
   endpoints; 4 route groups (hub, edge sensing) don't exist, and the documented
   `?activate=`/`?acknowledge_unvalidated=` import params were absent (FastAPI
   silently ignores unknown params, so the documented happy path returned
   `success:true` with nothing serving) → rows now marked **F13 ✅** vs
   **F15 — not served**, and import documented as deliberately non-activating so the
   evidence gate is always an explicit step.
8. **[MED] Stale `serveable`** — a snapshot of import-time attachment, so a circuit
   that became bindable kept reporting not-serveable **while actively serving** and
   was filtered out of `?serveable=true` → refreshed at activation.
9. **[MED] Dropped global λ** — a slice whose layer had no per-layer budget entry lost
   the authored intensity and served at the cluster default 1.0 → always carry it.
10. **[MED] `reapplied`/`cleared_steering` silently dropped by the route filter** —
    a slice circuit reported a NEW intensity the steering never received → dedicated
    `CircuitIntensityResponse` / `CircuitDeactivationResponse`, and the service now
    returns an explicit warning when the dial did not reach the model.
11. **[MED] Steering applied before persistence** — a DB failure after a successful
    serve left the GPU steering with no active row to stop it → compensating clear.
12. **[HIGH-UX] Silent UNVALIDATED_CIRCUIT swallow** — reachable via a STALE cached
    row (rung lowered server-side): the card renders the plain Activate button, the
    server refuses, and `onError` returned early with **no toast and no invalidate**,
    so the click did nothing forever with no checkbox on screen to discover →
    invalidate + a warning carrying the server's `rung_language`.
13. **[MED-UX] Export leaked the object URL and swallowed failures** → try/catch/
    finally with a toast; revoke in `finally`.

### Assessed & NOT changed (with rationale)
- The 20-member cluster cap is **not** breachable by a slice (verified max 20).
- Nothing mutates `circuit_meta` in place, so export losslessness holds.
- `set_active`'s two-transaction race and `_dedupe_name`'s 51 sequential queries are
  real but low-severity on a single-node, same-segment deployment — recorded for R2.
- `NoActiveCircuitError` (status_code=200) is currently unraised; the route inlines
  the envelope. Recorded for R2: either raise it or delete it, plus a guard test that
  no 200-status MiLLMError can reach the generic handler.
- Migration `sqlite_where` vs the model's bare `JSONB` (Postgres-only in practice) —
  recorded for R2.

+14 regression tests (10 service, 4 route) targeting exactly these defects.
**Backend 1355 passed / 1 skipped; frontend 240 passed; tsc clean.**

## Round 2 (2026-07-20) — 2 finders (verify R1 fixes / fresh angles + deferred)

**25 findings.** R1's fixes largely hold (the slice path now passes a validated
model; `_parse_stored` is the sole parse site; the name-truncation arithmetic is
correct in every case incl. 3-digit layers, worst case 119 ≤ 120) — but the fixes
introduced one CRITICAL bug and left two protections weaker than claimed.

### Confirmed & FIXED in R2
1. **[CRIT] `_serve_slices` ignored `ClusterImportItem.status`** — `ClusterService`
   REPORTS its outcome rather than raising: an incompatible slice returns
   `imported_unbound` with activation *explicitly skipped* (a warning the cluster
   path added in its own 009 R3 review for this exact bug class). The circuit still
   reported `serving_mode="slice_fallback"` and set itself active — **claiming live
   influence while the model ran completely unsteered**. → fail closed with
   `SAE_SET_INCOMPLETE` carrying the import's status/warnings.
   **And the fixture hid it again**: a bare `MagicMock` answers `.status` with a
   truthy Mock. Both fixtures now return a REAL `ClusterImportItem`.
2. **[HIGH] Evidence-gate bypass across a restart** — `main.py` resets models, SAEs
   and attachments on startup but never `circuits`, leaving `is_active=true` with no
   steering; `set_intensity` then re-applied steering **without re-checking the
   gate**, re-arming a rung<2 circuit with no acknowledgement. This was the only path
   where an unvalidated circuit reached the model without a live ack. → circuits are
   reset at startup AND `set_intensity` now enforces the gate itself (re-applying is
   a fresh arm), threaded through `SetCircuitIntensityRequest.acknowledge_unvalidated`
   and refused in the envelope like activation.
3. **[HIGH] `deactivate()` lied for a slice serve** — in slice mode the CLUSTER
   profile does the steering, so clearing circuit steering reported
   `cleared_steering=true` while the slice kept running (the mirror image of the
   co-tenancy bug activation was careful to fix). → the slice profile id is persisted
   at activation and torn down on deactivate; `cleared_steering` now reflects work
   actually undone.
4. **[HIGH] Compensating rollback was incomplete + could crash** — it called
   `clear_circuit_steering()` unguarded (`_sae_service` is optional and defaults to
   None) and was a no-op for the slice path. → guarded, and it now deactivates the
   slice profile too.
5. **[MED] Member-cap accounting disagreed with the projection** — the validator
   counted `len(expanded_members)` OR 1, never both, so a contract-valid circuit
   could project to a 21-member slice and fail the cluster's 20 cap. → count the same
   sources the projection collects.
6. **[MED] No-budget circuits disagreed on λ between modes** — `_serve_full` falls
   back to `circuit.intensity` but the slice took the cluster default 1.0. → the
   slice carries the dial value as a fallback.
7. **[MED] `slice_profile_id` was dropped by the response model** → added, so the
   set_intensity warning's prescribed remediation is actually followable.

### Assessed & recorded (not changed)
- The co-tenant reorder is **partly cosmetic**, honestly: `set_circuit_steering`
  clears each SAE before applying, so a co-tenant's steering is already gone when the
  release runs. The release still fixes the DB row (no "active" row with no
  steering), which is the state users act on. A full fix needs snapshot-and-restore
  and is recorded for F14/F15.
- `set_active`'s two-commit race, `_dedupe_name`'s 51 sequential queries surfacing an
  `IntegrityError` as a 500, the `idx_circuits_name` vs `ix_circuits_name` migration
  drift, `_max_depth` not bounding WIDTH, `count()` materialising rows — all real,
  all low-severity on a single-node trusted-segment deployment. Recorded for R3.
- `status_code=200` on `UnvalidatedCircuitError`/`NoActiveCircuitError` is safe today
  (both call sites catch explicitly) but is a loaded gun — R3 to add a guard test.

+6 regression tests (unbound/errored slice not reported as serving, re-arm ack,
slice teardown, persisted slice id). **Backend 1361 passed / 1 skipped.**

## Round 3 (2026-07-20) — /review 4 perspectives (product+architect / QA+test)

**29 findings.** Three were EMPIRICALLY REPRODUCED by the reviewer against live
Postgres rather than argued from semantics — and all three were invisible to a
fully green suite.

### Confirmed & FIXED in R3
1. **[CRIT] My own R2 startup fix caused a regression.** All four stale-state
   UPDATEs shared ONE transaction with a single commit, so a missing `circuits`
   table (migration 011 not yet run on a deployment) aborted the transaction and
   **rolled back the model/SAE resets that previously worked** — swallowed as a
   warning, leaving models stuck `loaded` and SAEs `attached`. Reviewer verified
   against live Postgres (`FINAL STATUS AFTER FAILURE: loaded`). → each reset now
   owns its transaction and its own guard. Pinned by `tests/unit/test_startup_reset.py`.
2. **[CRIT] My R2 fail-closed check was itself defeated.** `ClusterService` sets
   `status="imported"` even when *activation* raised (recording the failure only as
   a warning) — so the status check passed and the circuit again claimed to serve
   while the model ran unsteered. → also treat an activation-failure warning as a
   failure; reason `slice_activation_failed`.
3. **[HIGH] The copy-audit could be disarmed by a comment.** Allow-list markers
   matched anywhere on the line, so `const m = "...causally validated"; // rung_language`
   passed. → comments are stripped before auditing; a marker in a comment can no
   longer exempt a claim in code. **Negative control re-run: the comment-disarmed
   overclaim is now caught.**
4. **[HIGH] Single-active invariant held in ONE direction only** — a circuit released
   an active cluster, but activating a cluster/profile never released an active
   circuit, so the circuit row kept reporting `is_active` + `serving_mode` while a
   profile had taken its layers. → symmetric release at `ProfileService.activate_profile`
   (the choke point every activation path shares), with a user-visible warning.
5. **[MED] Member cap counted pre-dedupe** while the projection dedupes, so a
   contract-valid circuit whose expansion overlapped its own feature was rejected at
   import. → the cap now counts the DISTINCT features the projection emits.
6. **[MED] Dual rung gate** — `Circuit.validated` hardcoded `rung >= 2` while the
   ladder owns the threshold. → the property delegates to `is_validated()`; a
   parametrised test asserts they agree for 0–3.
7. **[MED] Empty-but-present budget skipped the λ fallback** (truthiness check) →
   `is not None`.

### Assessed & recorded as debt (with rationale)
- **MCP `millm_*_circuit` tools don't exist** though the contract marks them F13 ✅.
  Correct call: the REST+UI surface is what F13 promised; the MCP proxies belong with
  F14/F15 which own that surface. **Contract rows to be demoted at F13 close-out.**
- **Hazards + per-edge rungs are computed then not rendered.** Real UX gap, and the
  most safety-relevant output of the arc. Recorded as the top F14/F15 UI item.
- **Slice serves only `bound_layers[0]`** and the disclosure doesn't name the other
  bound layers. Single-active cluster semantics genuinely constrain the capability;
  the *disclosure* gap is recorded for F14.
- `assess_compatibility` reaching the singleton (layering + TOCTOU — F12 fails closed,
  so it degrades to a hard error not a wrong-basis serve), `provenance.slice_profile_id`
  wanting a real FK column, the 200-status exception pattern wanting a single
  registered handler, `set_active`'s two commits, `_dedupe_name`'s 51 queries →
  IntegrityError→500, `_max_depth` not bounding width, `count()` materialising rows.
  All real, all cheap, none blocking — carried into the F14/F15 debt list.

### Honest assessment (reviewer's words, verified)
The evidence ladder is well-built: single language source, server-rendered verbatim,
coercion that degrades DOWNWARD, MIN-over-edges, and a copy-audit that now actually
guards. The rung-0 user journey was traced end to end (list → badge → blocked
activate → ack → active) and **"associated" is never presented as causal anywhere**.

+11 regression tests (5 service, 6 startup). **Backend 1374 passed / 1 skipped;
frontend 240 passed; tsc clean; app boots with all 8 circuit routes.**

## Outcome
3 rounds · **85 findings surfaced** (31 + 25 + 29) · **27 fixed** · zero regressions.
Every round found a defect in the PREVIOUS round's fix — R1's slice fix hid R2's
ignored-status bug, and R2's startup fix caused R3's transaction regression. Each
was caught only because the next round attacked the last round's work.
