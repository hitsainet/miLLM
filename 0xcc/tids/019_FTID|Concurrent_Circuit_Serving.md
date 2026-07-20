# Technical Implementation Document: Concurrent Circuit Serving

## miLLM Feature 19

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**References:** `019_FPRD|Concurrent_Circuit_Serving.md` · `019_FTDD|Concurrent_Circuit_Serving.md` · `docs/circuit-contention-model.md` (design of record) · `000_PADR|miLLM.md` (v1.3)

---

## 1. File Structure

```
millm/
├── db/models/circuit.py                     (MOD — DROP the uq_circuits_active Index from __table_args__)
├── db/models/circuit_layer_claim.py         (NEW — claim row; JSONVariant steering_keys)
├── db/repositories/circuit_claim_repository.py (NEW — live_claims/claim/release/reconcile)
├── db/repositories/circuit_repository.py    (MOD — add list_active(); get_active() no longer safe)
├── db/migrations/versions/013_add_circuit_layer_claims.py (NEW — drop index, add table, TESTED downgrade)
├── services/circuit_claim_registry.py       (NEW — LayerClaim, ContentionVerdict, assess/claim/release)
├── services/circuit_service.py              (MOD — claim phase in activate; owner-scoped deactivate+rollback)
├── services/sae_service.py                  (MOD — AttachedSAEState owner map, apply_owner/release_owner/_rebuild_layer)
├── services/profile_service.py              (MOD — _release_active_circuit iterates list_active())
├── services/inference_service.py            (MOD — _steering_circuit -> _steering_circuits; rung suppression)
├── api/schemas/circuit.py                   (MOD — allow_layer_overlap; claimed/composed_layers; active LIST)
├── api/routes/management/circuits.py        (MOD — activate param, GET /claims, active list + ?single=true)
├── core/errors.py                           (MOD — CircuitLayerContentionError + ERROR_CLASSES entry)
├── core/config.py                           (MOD — CIRCUIT_ALLOW_CONCURRENT)
admin-ui/src/components/circuits/contention/{ClaimsStrip,ContentionDialog}.tsx (NEW)
admin-ui/src/components/circuits/CircuitCard.tsx (MOD — claim chips; composed badge replaces rung badge)
tests/unit/services/test_circuit_claim_registry.py, test_sae_owner_provenance.py,
                    test_circuit_contention_gate.py, test_rung_suppression.py (NEW)
tests/integration/test_concurrent_circuit_serving.py, test_circuit_claim_race.py,
                  test_migration_013.py (NEW)
```

## 2. Load-Bearing Implementation Points (verified against live code)

- **The invariant is DB-enforced, not a service convention.** `millm/db/models/circuit.py:96-102`
  declares it inside `__table_args__`:
  ```python
  Index("uq_circuits_active", "is_active", unique=True,
        postgresql_where=(is_active == True),
        sqlite_where=(is_active == True))
  ```
  Both dialect predicates are present — the suite runs on SQLite and production on PostgreSQL, so the
  REPLACEMENT index must also supply both, or it is silently non-unique in every test.

- **`get_active()` RAISES on two active rows.** `circuit_repository.py:78-83` ends in
  `result.scalar_one_or_none()`, which throws `MultipleResultsFound` for two rows — not a wrong answer,
  an exception. It is called from `profile_service.py:343` inside `_release_active_circuit`, whose own
  docstring says *"Best-effort by design — a bookkeeping failure must not block the user's activation"*
  (`profile_service.py:334-336`). Under concurrency that call raises and blocks precisely what the
  comment promises it will not. Add `list_active()`; do not leave `get_active()` on any multi-circuit
  path. `deactivate_all()` (`:130-142`) already iterates properly and is reusable for reconciliation.

- **The apply loop CLEARS before it writes.** `sae_service.py:652-659`:
  ```python
  for key, steering in per_entry.items():
      sae = entry_by_key[key].sae
      sae.clear_steering()                    # <-- wipes a co-tenant's keys
      ...
      sae.set_steering_batch(steering)
  ```
  The comment above it (`:643-648`) explains the clear makes each serve "authoritative" so "no stale
  features from a previous circuit/cluster/manual set leak in" — correct under single-active, a
  co-tenant tear-out under Feature 19. This is the single most important line to route through
  `apply_owner`.

- **`set_steering_batch` MERGES; `clear_steering()` with no arg wipes everything.**
  `sae_wrapper.py:555-570` loops `self._steering_values[idx] = val` — an update, not a replace — and
  `clear_steering` (`:573-583`) calls `self._steering_values.clear()` when `feature_idx is None`. Both
  facts are why per-circuit provenance is mandatory: the merge is what makes same-key collision
  invisible (CLAIM-K1), and the unqualified clear is what makes naive release destructive (CLAIM-D2).
  `clear_steering(feature_idx)` DOES support single-key removal, but the FTDD §4 rebuild is preferred
  because incremental deletion cannot recover from a partially-failed release.

- **`clear_circuit_steering()` defaults to ALL attached layers.** `sae_service.py:686-702`: with
  `layers=None` it targets `[e.layer for e in self._sae_state.entries()]` and calls
  `entry.sae.clear_steering()` + `enable_steering(False)` on each. `circuit_service.deactivate`
  (`:688`) calls it unqualified, and so does the activation rollback path. Both must become
  owner-scoped.

- **Serve derives layers already; reuse it, do not recompute.** `circuit_service.py` computes
  `bound_layers` from `assess_compatibility` verdicts and then
  `served_layers = bound_layers if all_bound else bound_layers[:1]` — the slice-fallback case serves
  exactly ONE layer. The claim set is `served_layers`, not `definition.layers()` (CLAIM-C5, EC-19.1).
  Claiming declared-but-unserved layers would block a disjoint circuit for nothing.

- **Gate ordering is documented and must be extended, not reordered.** `activate`'s docstring
  (`circuit_service.py:256-266`) states "Gate order matters: 1. Evidence gate ... 2. SAE-set gate".
  The claim phase is gate 3, after the derivation and BEFORE the serve. Note the shipped co-tenant
  release runs *after* the serve succeeds ("only AFTER the serve succeeds, so a failed activation never
  leaves the user with nothing steering", `:376-382`) — the CLAIM phase is different and must run
  BEFORE, because a claim taken after a serve can lose its race with nothing to roll back to.

- **The error house style is 200 + envelope for handler-level refusals.**
  `errors.py:289-307` — both `UnvalidatedCircuitError` (`status_code = 200  # house style:
  handler-level refusal in the envelope`) and `NoActiveCircuitError` do exactly this.
  `CircuitLayerContentionError` follows them and is registered in `ERROR_CLASSES` (`errors.py:320-353`,
  which already carries `"UNVALIDATED_CIRCUIT"`, `"NO_ACTIVE_CIRCUIT"`, `"SAE_SET_INCOMPLETE"`).
  Contrast `SAESetIncompleteError` (`:209-227`), which is 422 because it reports a missing
  precondition — contention is not missing anything, so 200 is right.

- **The rung header is emitted only when the echo is truthy.** `api/routes/openai/chat.py:141`
  (streaming) and `:154` (non-streaming) both set `X-miLLM-Circuit-Rung` under a truthiness check on
  `echo_circuit_rung`. Suppression therefore needs NO route change — returning `None` from
  `active_circuit_rung()` (`inference_service.py:833-850`) is the whole mechanism.

- **The `_steering_circuit` memo MUST stay a contextvar.** `inference_service.py:780-804` documents an
  R2 defect at length: the memo was cached on `self`, but `get_inference_service` is `@lru_cache`'d, so
  the value was written once per PROCESS and "advertised a deactivated circuit's rung header forever".
  `_STEERING_CIRCUIT_MEMO` is a `ContextVar` (`:53-57`) for that reason. The plural version inherits
  the constraint exactly — do not "simplify" it onto the service.

- **Activation accepts flags on BOTH query and body.** `circuits.py:208` takes
  `acknowledge_unvalidated` as a `Query(...)`, and `:275` reads `body.acknowledge_unvalidated` from
  the schema (`api/schemas/circuit.py:281`). `allow_layer_overlap` must be added in BOTH places or the
  UI and MCP paths will disagree about whether an override was requested.

- **Migration numbering: next free is `013`.** `ls millm/db/migrations/versions` ends at
  `012_add_circuit_edge_sensing.py` (Feature 15). `down_revision = "012"`.

## 3. Key Implementations

```python
# services/sae_service.py — the co-tenant-safe apply/release core (FTDD §4)
def _rebuild_layer(self, layer: int) -> None:
    """Rebuild a layer's steering from EVERY owner. Full recompute, never incremental.

    set_steering_batch MERGES (sae_wrapper.py:555), so removing one owner's keys
    incrementally would leave them resident. Clearing and re-merging every
    remaining owner is the only shape in which release cannot leak.
    """
    entry = self._sae_state.by_layer(layer)
    if entry is None:
        return
    owners = self._circuit_keys.get(layer, {})
    merged: dict[int, float] = {}
    for circuit_id, steering in owners.items():
        clash = merged.keys() & steering.keys()
        if clash:
            # CLAIM-K1 refuses this upstream; if we are here the gate was
            # bypassed or removed. Fail loudly rather than serve a strength
            # belonging to neither author.
            raise ValidationError(
                f"Same-key collision on layer {layer}: features {sorted(clash)} "
                f"claimed by more than one circuit",
                details={"layer": layer, "feature_idx": sorted(clash)})
        merged.update(steering)
    entry.sae.clear_steering()
    if merged:
        entry.sae.set_steering_batch(merged)
        entry.sae.enable_steering(True)
    else:
        entry.sae.enable_steering(False)
```

```python
# services/circuit_claim_registry.py — assess (self-excluding, collision-first)
async def assess(self, circuit_id: str,
                 requested: dict[int, dict[int, float]]) -> ContentionVerdict:
    live = [c for c in await self.repo.live_claims() if c.circuit_id != circuit_id]
    #                                                   ^ EC-19.3: re-activating a live
    #                                                     circuit is not a self-contention
    by_layer: dict[int, list[LayerClaim]] = {}
    for c in live:
        by_layer.setdefault(c.layer, []).append(c)

    contended, incumbents, collisions = [], {}, []
    for layer, steering in requested.items():
        holders = by_layer.get(layer, [])
        if not holders:
            continue
        contended.append(layer)
        incumbents[layer] = holders[0]
        for holder in holders:                       # collision is per HOLDER, not per layer
            for idx in holder.steering_keys.keys() & steering.keys():
                collisions.append({
                    "layer": layer, "feature_idx": idx,
                    "incumbent_id": holder.circuit_id,
                    "incumbent_strength": holder.steering_keys[idx],
                    "requested_strength": steering[idx]})
    return ContentionVerdict(sorted(contended), incumbents, collisions)
```

```python
# services/inference_service.py — rung suppression, BOTH branches (CLAIM-O3)
async def active_circuit_rung(self) -> Optional[tuple[int, str]]:
    circuits = await self._steering_circuits()
    if len(circuits) != 1:
        return None                    # 0 => nothing steering; >1 => composed
    # A SINGLE circuit still omits when its layer is shared: the incumbent's
    # evidence stopped describing the response the moment someone composed onto
    # its layer. Checking only len>1 lets the incumbent keep advertising a rung
    # for output another circuit contributes to.
    if await self._claims.any_composed_layer_for(circuits[0].id):
        return None
    ...                                # shipped rung coercion unchanged (:845-850)
```

```python
# db/migrations/versions/013 — downgrade order is load-bearing
def downgrade():
    # 1. DEACTIVATE FIRST. Creating uq_circuits_active while two rows are active
    #    fails, and a failed downgrade mid-rollback is the worst time to find out.
    op.execute("""UPDATE circuits SET is_active = false, serving_mode = NULL
                  WHERE is_active = true AND id NOT IN (
                      SELECT id FROM circuits WHERE is_active = true
                      ORDER BY updated_at DESC LIMIT 1)""")
    op.drop_table("circuit_layer_claims")
    op.create_index("uq_circuits_active", "circuits", ["is_active"], unique=True,
                    postgresql_where=sa.text("is_active = true"),
                    sqlite_where=sa.text("is_active = 1"))
```

## 4. Implementation Pitfalls

1. **`get_active()` raises, it does not return the first row.** `scalar_one_or_none()`
   (`circuit_repository.py:83`) throws `MultipleResultsFound`. Every caller on a multi-circuit path
   must move to `list_active()`. The dangerous one is `profile_service.py:343`, inside a best-effort
   block whose contract is that it never blocks activation.
2. **Never `clear_steering()` a shared layer.** Both the apply loop (`sae_service.py:654`) and
   `clear_circuit_steering()` (`:699`) do a whole-dict wipe. Under co-tenancy this silently stops a
   circuit the operator did not touch while its row still reports active. Route everything through
   `apply_owner`/`release_owner`.
3. **Collision check BEFORE the override check, unconditionally.** If the same-key test sits inside
   `if not allow_layer_overlap:` — or after it — the override silently covers a case that has no
   override (CLAIM-K1). Assert directly: a test that sets `allow_layer_overlap=true` on a colliding
   pair and requires the refusal.
4. **Suppress the rung on BOTH branches.** `len(circuits) > 1` is not sufficient; a single circuit
   sharing a composed layer must also omit (§3 above). Two separate tests.
5. **Claim before serve, release after clear.** Claiming after a successful serve can lose the unique
   -index race with steering already applied and nothing to roll back to. Conversely, release must
   clear steering first and mark `released_at` second — a claim released while its keys are still
   applied lets a new claimant take a layer that is still being steered by the old one.
6. **Both dialect predicates on the new index.** `postgresql_where` AND `sqlite_where`, mirroring
   `circuit.py:100-101`. A Postgres-only predicate makes the index non-unique under the entire test
   suite, so every contention test would pass for the wrong reason.
7. **The claim set is the SERVED layers.** `served_layers`, not `definition.layers()` — a
   `slice_fallback` serve touches one layer (`circuit_service.py`, `bound_layers[:1]`). Over-claiming
   blocks disjoint circuits for no benefit (EC-19.1).
8. **Keep the memo a contextvar.** `_STEERING_CIRCUIT_MEMO` (`inference_service.py:53-57`) is a
   `ContextVar` because `get_inference_service` is `@lru_cache`'d; a `self` attribute is written once
   per process and never invalidated. The docstring at `:790-797` records exactly what that cost last
   time. The plural version has the identical constraint.
9. **`allow_layer_overlap` in both query and body.** `circuits.py:208` (Query) and
   `schemas/circuit.py:281` (body) — the UI uses one and MCP the other; adding it to only one produces
   an override that works from one surface and is silently ignored from the other.
10. **Startup reconciliation runs unconditionally.** Not only when the flag is false. It is the
    backstop for a partially-applied migration (no invariant at all) and for orphan claims (EC-19.4).
11. **Composed marks the INCUMBENT too.** When an override composes layer 13, the incumbent's claim row
    must also flip to `composed = TRUE`, or the incumbent keeps its exclusive claim (blocking a third
    circuit that should now be refused for a different reason) and keeps advertising its rung.
12. **Flag-off parity is a test, not an assumption.** With `CIRCUIT_ALLOW_CONCURRENT=false` the old
    single-active path must behave byte-identically. New tests will exercise only the new path unless
    parity is asserted explicitly.

## 5. Config Additions (millm/core/config.py)

```python
# Feature 19 — concurrent circuit serving. Default FALSE for one release: the
# first concurrent activation is a one-way door in deployed data (RSK-008), so
# enabling it must be a deliberate operator act rather than something that
# arrives with an upgrade.
CIRCUIT_ALLOW_CONCURRENT: bool = False
```

Sits beside the shipped circuit keys (`CIRCUIT_INTENSITY_MAX` at `config.py:93`,
`CIRCUIT_SENSING_FORCE_SERIAL` at `:118`).

## 6. Contradictions Between Live Code and the Design of Record

Recorded here rather than silently reconciled. None invalidates the design; all three change what the
implementation must do.

1. **The design says release "clears only the `(layer, feature_idx)` keys it wrote" (§3.5). The live
   apply path clears the ENTIRE layer before every write** (`sae_service.py:654`), and
   `clear_circuit_steering()` (`:699`) clears every attached layer. So per-circuit provenance is not an
   addition to a neutral substrate — it must REPLACE an existing clear-first design whose comment
   (`:643-648`) explicitly justifies the wipe. The FTDD's rebuild-from-owners shape (§4) is the
   reconciliation.

2. **The design's §4 table says `_steering_circuit()` "returns one circuit" and should return "the
   set". The live implementation is more constrained than that**: it is memoised in a `ContextVar`
   after an R2 defect where a `self`-cached memo advertised a deactivated circuit's rung forever
   (`inference_service.py:780-804`). The plural version inherits the memoisation discipline, which the
   design doc does not mention.

3. **The design does not note that `get_active()` RAISES.** `scalar_one_or_none()`
   (`circuit_repository.py:83`) throws `MultipleResultsFound` on two rows, and it is reached from
   `profile_service._release_active_circuit`, documented as best-effort and non-blocking
   (`profile_service.py:334-336`). Dropping `uq_circuits_active` therefore converts a documented
   never-block path into a 500 unless `list_active()` lands in the same change. This is the one item
   that must not be deferred past the migration.
