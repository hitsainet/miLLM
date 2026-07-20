# Task List: Concurrent Circuit Serving

## miLLM Feature 19

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**References:** `019_FPRD|Concurrent_Circuit_Serving.md` · `019_FTDD|Concurrent_Circuit_Serving.md` · `019_FTID|Concurrent_Circuit_Serving.md` · `docs/circuit-contention-model.md` (design of record) · `BRD-MILLM-CIRCUITS-002.md` (BR-011, RSK-007, RSK-008)

## Relevant Files
- `millm/db/models/circuit.py` — DROP the `uq_circuits_active` Index from `__table_args__` (:96-102)
- `millm/db/models/circuit_layer_claim.py`, `db/repositories/circuit_claim_repository.py` — claim row + repo
- `millm/db/repositories/circuit_repository.py` — add `list_active()`; `get_active()` raises on 2 rows (:83)
- `millm/db/migrations/versions/013_add_circuit_layer_claims.py` — drop index, add table, TESTED downgrade
- `millm/services/circuit_claim_registry.py` — LayerClaim/ContentionVerdict/assess/claim/release/reconcile
- `millm/services/circuit_service.py` — claim phase in `activate`; owner-scoped `deactivate` + rollback
- `millm/services/sae_service.py` — `AttachedSAEState` owner map; `apply_owner`/`release_owner`/`_rebuild_layer`
- `millm/services/profile_service.py` — `_release_active_circuit` iterates `list_active()` (:330-350)
- `millm/services/inference_service.py` — `_steering_circuit` → `_steering_circuits`; rung suppression (:780-850)
- `millm/api/schemas/circuit.py`, `api/routes/management/circuits.py` — `allow_layer_overlap`, `/claims`, active LIST
- `millm/core/errors.py` — `CircuitLayerContentionError` + `ERROR_CLASSES` (:320-353)
- `millm/core/config.py` — `CIRCUIT_ALLOW_CONCURRENT`
- `admin-ui/src/components/circuits/contention/*`, `CircuitCard.tsx`
- `tests/unit/services/test_circuit_{claim_registry,contention_gate,rung_suppression}.py`, `test_sae_owner_provenance.py`
- `tests/integration/test_concurrent_circuit_serving.py`, `test_circuit_claim_race.py`, `test_migration_013.py`

### Notes
- Depends on Feature 18 (single serving derivation supplies the claim set) and Feature 17 (request-scoped
  context owns per-circuit provenance). Execute after both — per the BRD execution order, step 5.
- Migration numbering: **013** (`012_add_circuit_edge_sensing.py` is the current disk tail).
- The design of record is `0xcc/docs/circuit-contention-model.md` — its decisions are settled; do not re-open.
- Test commands: `pytest` (backend), `npm test` in admin-ui (Vitest).

### Category Checklist Results
- Data: 1.x (claim table + model + repo + migration + downgrade) ✓
- Backend/API: 4.x ✓
- Frontend/UI: 5.x ✓
- Business logic: 2.x (registry/assess/claim/release), 3.x (activate gate, owner provenance, rung suppression) ✓
- Integration wiring: 3.x (circuit_service/sae_service/profile_service/inference_service), 4.2 (DI/router) ✓
- Error handling & logging: 4.4 (`CIRCUIT_LAYER_CONTENTION`), 3.5 (override logged), 2.5 (reconciliation logs) ✓
- Testing: paired throughout + 6.x integration (incl. downgrade RUN, race, header omission) ✓
- Performance & security: one indexed query per activation; no hot-path cost (7.1) ✓
- Config/deploy: 3.7 `CIRCUIT_ALLOW_CONCURRENT`; migration auto-runs; flag-gated rollout (7.3) ✓
- Documentation: 7.2 manual contention section ✓

## Tasks

- [ ] 1.0 Claim persistence + migration (covers FR-19.1, FR-19.6; CLAIM-C3, CLAIM-M1..M3)
  - [ ] 1.1 `db/models/circuit_layer_claim.py` (circuit_id FK CASCADE, layer, claimed_at, released_at, composed, `steering_keys` JSONVariant) + remove the `uq_circuits_active` Index from `circuit.py.__table_args__`
  - [ ] 1.2 Migration `013_add_circuit_layer_claims.py`: drop `uq_circuits_active`, create the table + `uq_circuit_layer_claim_live` partial unique index with BOTH `postgresql_where` AND `sqlite_where`, backfill the (at most one) currently-active circuit's claims
  - [ ] 1.3 Downgrade: deactivate all but the most recently activated circuit FIRST, then drop the table, then recreate `uq_circuits_active` (order is load-bearing — the index creation fails while 2 rows are active)
  - [ ] 1.4 `circuit_claim_repository.py` (`live_claims`/`claim`/`release`/`mark_composed`/`reconcile`) + `circuit_repository.list_active()`
  - [ ] 1.5 Repo unit tests: CASCADE on circuit delete, partial-index uniqueness on SQLite AND (if available) Postgres, `release` touches only the caller's rows

- [ ] 2.0 Claim registry (covers FR-19.1, FR-19.2, FR-19.4; CLAIM-C1..C5, CLAIM-R2, CLAIM-K1..K2)
  - [ ] 2.1 `LayerClaim` / `ContentionVerdict` dataclasses; `has_contention` vs `has_collision` kept structurally distinct (one is overridable, one is never)
  - [ ] 2.2 `assess()` — self-excluding (EC-19.3), collision computed per HOLDER, contended layers + incumbents named
  - [ ] 2.3 `claim()` — INSERT; catch the unique-index `IntegrityError` and convert to a contention refusal (EC-19.7 race)
  - [ ] 2.4 `release()` — `released_at = now()` for this circuit_id only; returns released layers; `mark_composed()` flips incumbent rows too
  - [ ] 2.5 `reconcile()` — drop orphan claims (EC-19.4) and, with the flag false, demote to the most recently activated circuit (EC-19.5); logs every demotion; runs unconditionally at startup
  - [ ] 2.6 Unit tests: disjoint/overlap/same-key predicates, self-exclusion, composed rows bypass the exclusive index while exclusive ones do not, reconciliation both branches

- [ ] 3.0 Serving integration (covers FR-19.1, FR-19.3, FR-19.5; CLAIM-O1..O5, CLAIM-D1..D4)
  - [ ] 3.1 `AttachedSAEState` owner map + `apply_owner`/`release_owner`/`_rebuild_layer` (full recompute from owners; RAISES on a colliding owner map)
  - [ ] 3.2 Route `_set_circuit_steering_locked`'s apply loop (`sae_service.py:652-659`) through `apply_owner` — replacing the clear-before-write that wipes co-tenants
  - [ ] 3.3 Claim phase in `circuit_service.activate`: after the serving derivation, before the serve; collision check FIRST and unconditional, then contention vs `allow_layer_overlap`; claim set = `served_layers` (EC-19.1)
  - [ ] 3.4 Owner-scoped `deactivate` and activation rollback — release this circuit's claims and keys only, never the global `clear_circuit_steering()`
  - [ ] 3.5 Override path: `composed_layers` + warning in the response, `allowed_layer_overlap` echo, explicit `logger.warning` naming both circuits and the layers (CLAIM-O4)
  - [ ] 3.6 `profile_service._release_active_circuit` iterates `list_active()` — a profile taking layers must release EVERY circuit holding one (and must not start raising in a best-effort block)
  - [ ] 3.7 `CIRCUIT_ALLOW_CONCURRENT` config key (default `false` for ONE release, BR-011a). **Flag-off REFUSES LOUDLY**, naming configuration as the reason — it MUST NOT fall back to the silent single-active disarm this feature replaces; that silent fallback IS the bug (CLAIM-M4)
  - [ ] 3.8 Test the flag-off refusal explicitly: a second activation with the flag false produces a refusal naming configuration, NOT a silent disarm of the incumbent
  - [ ] 3.9 Record the dated flip commitment in the BRD/PPRD; an unflipped flag makes a shipped capability unreachable, which is the defect class this increment exists to eliminate
  - [ ] 3.8 Unit tests: co-tenant survival on release, `_rebuild_layer` collision raise, gate ordering, claim-set = served layers, flag-off parity

- [ ] 4.0 API + rung suppression (covers FR-19.2, FR-19.3; CLAIM-R1, CLAIM-R3, CLAIM-O2, CLAIM-O3)
  - [ ] 4.1 `_steering_circuit` → `_steering_circuits` (plural, contextvar memo PRESERVED); `active_circuit_rung` returns None for `len != 1` AND for a single circuit on a composed layer
  - [ ] 4.2 `allow_layer_overlap` on BOTH the Query param (`circuits.py:208`) and the body schema (`schemas/circuit.py:281`); `GET /api/circuits/active` returns a LIST with `?single=true` compatibility; `GET /api/circuits/claims`
  - [ ] 4.3 **Informed refusal (BR-011 binding condition)**: the `CIRCUIT_LAYER_CONTENTION` payload carries `measured_hazard` — the close-out result AND its "one model, one fixture" caveat — plus `override_param` and `rung_header_suppressed_if_overridden`. A refusal that states only the fact of contention FAILS this task
  - [ ] 4.4 **Loud override**: every `allow_layer_overlap` use is echoed in the response (`composed_layers`), logged at WARNING with both circuit ids, and surfaced in the UI — mirroring `acknowledged_unvalidated` at `circuit_service.py:349-351`
  - [ ] 4.5 Test that an override is UNREACHABLE without first receiving the informed refusal, so the measurement cannot be bypassed by a client that guesses the parameter
  - [ ] 4.3 Route tests: refusal envelope (200 + `success:false`, incumbent named by NAME), atomicity (no claim row, no steering, incumbent untouched), override response shape, claims listing
  - [ ] 4.4 `core/errors.py`: `CircuitLayerContentionError` (code `CIRCUIT_LAYER_CONTENTION`, `status_code = 200`) + `ERROR_CLASSES` registration, following `UnvalidatedCircuitError` (:289-300)

- [ ] 5.0 Circuits-page contention UI (covers FPRD §6)
  - [ ] 5.1 `ClaimsStrip` (layer → claimant, composed badged) fed by `GET /api/circuits/claims`
  - [ ] 5.2 `ContentionDialog` — names the incumbent, lists contended layers, offers "Deactivate '{incumbent}'" / "Compose anyway"; on a same-key collision shows the colliding pairs with both strengths and offers NO compose action
  - [ ] 5.3 `CircuitCard`: claimed-layer chips; a composed circuit shows the composed badge INSTEAD of its rung badge, never both
  - [ ] 5.4 Vitest: dialog renders the incumbent name, compose action absent on collision, composed card shows no rung badge

- [ ] 6.0 Integration verification (covers FR-19.1..19.6 end-to-end)
  - [ ] 6.1 Two disjoint circuits serve simultaneously; both steer; neither clears the other; `GET /active` lists both
  - [ ] 6.2 Contention refusal end-to-end: envelope, incumbent named, atomic (nothing applied, incumbent untouched)
  - [ ] 6.3 Override: `composed_layers` reported AND **`X-miLLM-Circuit-Rung` absent from a real composed response** on BOTH the streaming and non-streaming paths
  - [ ] 6.4 Same-key collision refused with `allow_layer_overlap=true` explicitly set; deactivate-one-of-two leaves the co-tenant's keys applied and enabled
  - [ ] 6.5 `test_circuit_claim_race.py`: two concurrent activations for one layer — exactly one wins, and the DB index is what decided it (asserted with the service pre-check disabled)
  - [ ] 6.6 `test_migration_013.py`: upgrade on a populated DB; **downgrade RUN** against a seeded two-active state leaving exactly the most recently activated circuit; round-trip; the test FAILS when the deactivation step is removed (BR-005 reachability rule)
  - [ ] 6.7 Flag-off single-active parity; startup reconciliation with 2 active rows and the flag false (EC-19.5)

- [ ] 7.0 Feature Acceptance (per instruct 008)
  - [ ] 7.1 Verify FPRD §9 criteria 1–7 + all US/EC boxes one-by-one
  - [ ] 7.2 Manual: Circuits contention section (what a claim is; why the unit is the LAYER not the feature; the two ways to resolve a refusal; what composition costs — the rung header goes away, and why that is honesty; `CIRCUIT_ALLOW_CONCURRENT` one-way-door note)
  - [ ] 7.3 Full suite green (backend ≥1597 / frontend ≥272 — the BRD floor); update CLAUDE.md Document Inventory + Current Status; `docs/mcp-contract.md` → v1.2 (additive)
  - [ ] 7.4 Record the three Open Questions' resolutions (flag default, whether `allow_layer_overlap` ships, BR-012 boundary) — the first two gate acceptance

## Coverage Audit
- FR-19.1→1.0/2.0/3.3/6.1; FR-19.2→2.2/4.3/6.2; FR-19.3→3.5/4.1/6.3; FR-19.4→2.2/3.1/6.4; FR-19.5→3.1/3.4/6.4; FR-19.6→1.2/1.3/3.7/6.6/6.7 ✓
- US-19.1→3.2/6.1; US-19.2→2.2/4.3/6.2; US-19.3→3.5/4.1/6.3; US-19.4→2.2/6.4; US-19.5→3.4/6.4; US-19.6→1.3/3.7/6.6/6.7 — implementing + testing sub-tasks each ✓
- EC-19.1→3.3/3.8; EC-19.2→3.6; EC-19.3→2.2/2.6; EC-19.4→2.5/2.6; EC-19.5→2.5/6.7; EC-19.6→3.5 (warning copy); EC-19.7→2.3/6.5 ✓
- BRD: BR-011→CLAIM-C/R/O/K/D families; RSK-007→3.7 (flag makes the split reversible); RSK-008→1.2/1.3/6.6/7.3; BR-001→3.1 (per-circuit provenance) ✓
- TDD/TID sections all mapped (registry→2.x, provenance→3.1/3.2, activation→3.3, rung→4.1, migration→1.2/1.3, call-site sweep §8→3.x, UI→5.x) ✓
- Open questions: 3 carried in FPRD §13; the flag default and `allow_layer_overlap`'s existence gate acceptance (7.4) — no spike tasks, both are product decisions not investigations ✓
- Final task is Feature Acceptance ✓

## Review rounds (goal: 3 rounds, ≥10 findings each)
- [ ] Round 1 (multi-angle /code-review, 2 finder agents): ≥10 findings — fix criticals, document deferrals.
      **Watch for, specific to contention:**
      - **Claim leakage on deactivate** — releasing circuit A clears a `(layer, feature_idx)` key belonging
        to B. Highest-consequence defect available here: a circuit the operator did not touch silently
        stops while its row still reports active. Check every `clear_steering()` / `clear_circuit_steering()`
        call site, including the activation ROLLBACK path, not only `deactivate`.
      - **A composed layer emitting a rung header** — verify BOTH suppression branches (`len != 1`, and a
        single circuit sitting on a composed layer). Check the streaming path separately from the
        non-streaming one; they set the header at two different call sites (`chat.py:141` / `:154`).
      - **The migration's downgrade** — is it RUN against a seeded multi-active state, or merely written?
        Does the index creation come AFTER the deactivation? Does the test fail if the deactivation step
        is deleted?
      - **Same-key collision slipping through the override** — is the collision check inside or after the
        `allow_layer_overlap` branch? It must be first and unconditional. Assert with the override set.
      - Also: `get_active()`'s `scalar_one_or_none()` reached from any multi-circuit path;
        `sqlite_where` missing from the new partial index (would make every contention test pass for the
        wrong reason); claim taken after the serve rather than before.
- [ ] Round 2 (post-fix verification + fresh angles): ≥10 findings — verify R1 fixes hold; hunt regressions.
      **Watch for:** the contextvar memo "simplified" onto `self` (the R2 defect `inference_service.py:790`
      records); `_rebuild_layer` made incremental "for efficiency", reintroducing key leakage; the
      incumbent's claim row not flipped to `composed`; flag-off parity drifting because only the new path
      is exercised; `allow_layer_overlap` added to the Query param but not the body schema (or vice versa).
- [ ] Round 3 (/review, 4 perspectives): ≥10 findings — fix, pin mutation survivors.
      **Apply the mutation practice** (F15 R3 found four load-bearing lines no test caught in one pass,
      including one two prior rounds had verified clean by reading). Mutate specifically: the collision
      guard, both rung-suppression branches, the release's circuit_id filter, and the downgrade's
      deactivation step — each must break a test.
- [ ] Record: `.claude/context/sessions/review_feature019_R{1,2,3}_2026-07-*.md`.
- Directive: fix latent/pre-existing defects surfaced during review, not only regressions.

## Acceptance evidence
*[To be completed at acceptance — FPRD §9 criteria 1–7 verified one-by-one, with the downgrade run and
the composed-response header omission observed on a real response, not only asserted in a unit test.]*
