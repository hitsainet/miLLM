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

## Round 2 — pending.
## Round 3 — pending.
