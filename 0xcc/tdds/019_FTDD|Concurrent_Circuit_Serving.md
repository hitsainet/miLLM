# Technical Design Document: Concurrent Circuit Serving

## miLLM Feature 19

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**References:** `019_FPRD|Concurrent_Circuit_Serving.md` · `BRD-MILLM-CIRCUITS-002.md` (BR-011, RSK-007, RSK-008) · `000_PADR|miLLM.md` (v1.3, "Layer-exclusive claims vs additive composition") · `docs/circuit-contention-model.md` (design of record)

---

## 1. Executive Summary

Concurrent serving replaces ONE database-enforced invariant with a DIFFERENT database-enforced
invariant. Today `uq_circuits_active` (a partial unique index on `circuits.is_active`,
`millm/db/models/circuit.py:97-102`) guarantees at most one active circuit. Feature 19 drops it and
introduces `circuit_layer_claims` with a partial unique index on `layer WHERE released_at IS NULL AND
composed = FALSE` — at most one exclusive claimant per LAYER. The invariant moves from "one circuit"
to "one circuit per layer", which is exactly the semantic BR-011 asks for, and it stays in the database
rather than in service memory so it survives restart and concurrent writers.

Three service-layer changes carry it. `CircuitService.activate` gains a claim phase between the
evidence gate and the serve: derive the claim set from the single serving derivation (BR-002),
test it against live claims, refuse or claim. `SAEService` gains per-circuit key provenance so a
serve writes `(layer, feature_idx)` keys tagged with their owning circuit and a release clears
precisely those — replacing today's `sae.clear_steering()` full-dict wipe, which is a co-tenant
tear-out the moment two circuits share a layer. `InferenceService._steering_circuit` becomes
`_steering_circuits` (plural) and the rung echo omits when the set composes.

The riskiest part is NOT the new code. It is that four shipped call sites assume "the active circuit"
is singular — `CircuitRepository.get_active()` uses `scalar_one_or_none()`, which RAISES on two rows;
`sae_service._set_circuit_steering_locked` clears each target layer before writing;
`circuit_service.deactivate` calls the unqualified `clear_circuit_steering()`; and
`profile_service._release_active_circuit` deactivates "the" active circuit. Each is correct under the
old invariant and wrong under the new one. §8 enumerates them.

### Key Technical Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Invariant location | DB partial unique index on `circuit_layer_claims.layer`, not service memory | Survives restart + concurrent writers (EC-19.7); replaces a DB invariant with a DB invariant rather than downgrading to a convention |
| Unit of contention | The LAYER, not the feature | Both circuits write one layer's steering dict and both add into the same residual sum (`modified = original + Σ strength_i·W_dec[i]`) |
| Default behaviour | REFUSE on overlap, naming the incumbent | GPU close-out: 2 layers @ strength 5 destroys generation, 2 orders of magnitude below the per-member clamp. Silent composition reliably produces garbage |
| Override | `allow_layer_overlap`, mirroring `acknowledge_unvalidated` | A logged, explicit human act; researchers studying compounding are real users |
| Rung under composition | OMIT `X-miLLM-Circuit-Rung` entirely | No single circuit's evidence describes a composed response; same rule already used for slice-fallback |
| Same-key collision | Refuse UNCONDITIONALLY, no override | `set_steering_batch` MERGES (`sae_wrapper.py:555-570`); the survivor's strength belongs to neither author. No honest composition exists |
| Release precision | Per-circuit `steering_keys` provenance; clear keys, never the dict | `clear_steering()` with no arg wipes the whole layer (`sae_wrapper.py:573-583`) — a co-tenant tear-out |
| Claim-set source | The single serving derivation (BR-002), never a second computation | Activation and every other surface agree by construction, per design-of-record §3.1 |
| Rollout | `CIRCUIT_ALLOW_CONCURRENT`, default false, one release | The first concurrent activation is a one-way door in deployed data (RSK-008) |

## 2. System Architecture

```
 POST /api/circuits/{id}/activate?allow_layer_overlap=
            │
            ▼
 ┌─────────────────────────── CircuitService.activate ───────────────────────────┐
 │ 1. evidence gate        (SHIPPED — is_validated / acknowledge_unvalidated)     │
 │ 2. serving derivation   (F18 BR-002) ──► members, served_layers               │
 │ 3. CLAIM PHASE  (NEW)                                                          │
 │      claim_set = {m.layer for m in serving members}        [CLAIM-C1/C5]      │
 │      ┌── same-key collision? ──► REFUSE, no override        [CLAIM-K1]        │
 │      ├── layer overlap && !allow_layer_overlap ──► REFUSE   [CLAIM-R1]        │
 │      └── else ──► ClaimRegistry.claim(circuit, layers, composed)              │
 │                     └─ INSERT circuit_layer_claims (unique idx enforces)      │
 │ 4. serve                (SHIPPED — _serve_full / _serve_slices)               │
 │      now writes keys TAGGED with circuit_id  ──────────────┐                  │
 │ 5. rollback on failure: release claims + clear OWN keys    │                  │
 └────────────────────────────────────────────────────────────┼──────────────────┘
                                                              ▼
                                    ┌──────────────────────────────────────┐
                                    │ SAEService (per-circuit provenance)  │
                                    │  layer -> {circuit_id -> {idx: str}} │
                                    │  apply = merge of all owners         │
                                    │  release(circuit) = clear ITS keys   │
                                    └──────────────────────────────────────┘
                                                              │
 GET /v1/chat/completions ──► InferenceService._steering_circuits() ──► [c1, c2]
                                    │
                                    └─ any composed layer? ──► OMIT X-miLLM-Circuit-Rung
```

## 3. Claim Registry (`millm/services/circuit_claim_registry.py`, NEW)

The claim registry is the single owner of "who holds which layer". It is deliberately thin: the DB
index is the enforcement, and the registry is the typed interface to it.

```python
@dataclass(frozen=True)
class LayerClaim:
    circuit_id: str
    layer: int
    composed: bool
    steering_keys: dict[int, float]      # feature_idx -> applied strength (provenance)

@dataclass(frozen=True)
class ContentionVerdict:
    """The outcome of testing a claim set against live claims."""
    contended_layers: list[int]                 # layers already exclusively claimed
    incumbents: dict[int, LayerClaim]           # layer -> current holder
    collision_keys: list[dict[str, Any]]        # [{layer, feature_idx, incumbent_strength, requested_strength}]

    @property
    def has_collision(self) -> bool: ...        # same-key: NEVER overridable  [CLAIM-K1]
    @property
    def has_contention(self) -> bool: ...       # layer overlap: overridable   [CLAIM-R1]

class CircuitClaimRegistry:
    async def live_claims(self) -> list[LayerClaim]
    async def assess(self, circuit_id: str, requested: dict[int, dict[int, float]]) -> ContentionVerdict
        # requested: layer -> {feature_idx -> strength}; EXCLUDES circuit_id's own
        # claims so re-activating a live circuit is not a contention with itself (EC-19.3)
    async def claim(self, circuit_id: str, claims: list[LayerClaim]) -> None
        # INSERT; IntegrityError on the partial unique index ⇒ lost the race ⇒ refuse (EC-19.7)
    async def release(self, circuit_id: str) -> list[int]
        # UPDATE released_at = now() WHERE circuit_id = ? AND released_at IS NULL
        # returns released layers; NEVER touches another circuit's rows  [CLAIM-D1]
    async def reconcile(self, *, allow_concurrent: bool) -> list[str]
        # startup: drop claims whose circuit row vanished (EC-19.4); with the flag
        # false, demote all but the most recently activated circuit (EC-19.5)
```

**Why `assess` takes strengths, not just layers.** The same-key check (CLAIM-K1) needs the requested
`feature_idx` set per layer, and the refusal payload names both strengths so the operator can see
exactly what would have been silently merged. Passing only layers would make the collision check
impossible without a second derivation — precisely what BR-002 forbids.

**Composed claims are exempt from the unique index.** The index is
`WHERE released_at IS NULL AND composed = FALSE`. An override inserts `composed = TRUE`, which the
index ignores, so a second claimant is permitted at exactly the point the operator acknowledged it.
The incumbent's existing row is ALSO flipped to `composed = TRUE` in the same transaction — the layer
is composed for everyone on it, and its rung must be suppressed for the incumbent too, not only the
newcomer.

## 4. Per-Circuit Key Provenance (`millm/services/sae_service.py`)

This is the load-bearing change and the one that most contradicts shipped behaviour.

**Today** (`_set_circuit_steering_locked`, `sae_service.py:640-660`): the apply loop calls
`sae.clear_steering()` on each target layer and THEN `sae.set_steering_batch(steering)`. The comment
says this makes each serve "authoritative" so "no stale features from a previous
circuit/cluster/manual set leak in" — which is correct under a single-active invariant and is a
co-tenant tear-out the instant two circuits share a layer.

**Under Feature 19**, `AttachedSAEState` gains an owner-keyed steering map:

```python
# millm/services/sae_service.py — AttachedSAEState (:178)
_circuit_keys: dict[int, dict[str, dict[int, float]]]
#              layer  -> circuit_id -> {feature_idx: strength}

def apply_owner(self, layer: int, circuit_id: str, steering: dict[int, float]) -> None:
    """Replace THIS owner's keys on this layer, then rebuild the layer from all owners."""
    self._circuit_keys.setdefault(layer, {})[circuit_id] = dict(steering)
    self._rebuild_layer(layer)

def release_owner(self, layer: int, circuit_id: str) -> list[int]:
    """Drop THIS owner's keys; rebuild from whoever remains.  [CLAIM-D2]"""
    owners = self._circuit_keys.get(layer, {})
    released = sorted(owners.pop(circuit_id, {}).keys())
    self._rebuild_layer(layer)
    return released

def _rebuild_layer(self, layer: int) -> None:
    """Full recompute: clear, then merge every remaining owner's keys.

    A full clear+rebuild (rather than an incremental delete) is deliberate:
    set_steering_batch MERGES into the existing dict, so an incremental path
    would leave a departed owner's keys resident. Rebuilding from the owner
    map is the only shape where release cannot leak.
    """
    entry = self.by_layer(layer)
    if entry is None:
        return
    merged: dict[int, float] = {}
    for owner_steering in self._circuit_keys.get(layer, {}).values():
        merged.update(owner_steering)          # same-key collisions REFUSED upstream
    entry.sae.clear_steering()                 # whole-dict clear is safe HERE...
    if merged:
        entry.sae.set_steering_batch(merged)   # ...because we immediately restore every owner
        entry.sae.enable_steering(True)
    else:
        entry.sae.enable_steering(False)
```

`merged.update(...)` is a last-writer-wins merge, and it is safe ONLY because CLAIM-K1 refuses
same-key collisions before any of this runs. That coupling is not obvious from reading either piece
alone, so it is asserted directly: a test constructs colliding owner maps and requires `_rebuild_layer`
to raise rather than silently pick a winner. If the collision gate is ever removed, the rebuild fails
loudly instead of serving a strength belonging to neither author.

## 5. Activation, Refusal and Rollback (`millm/services/circuit_service.py`)

The claim phase slots between the shipped evidence gate (`activate`, `:253-280`) and the serve
(`_serve_full` `:415` / `_serve_slices` `:435`). Ordering matters and mirrors the existing gate
ordering docstring:

1. **Evidence gate** (shipped) — rung < 2 needs `acknowledge_unvalidated`.
2. **Serving derivation** (F18) — yields `members` and `served_layers`. Note the shipped code already
   distinguishes declared from served layers: `served_layers = bound_layers if all_bound else
   bound_layers[:1]` (`circuit_service.py:~384`). The claim set is the SERVED set (CLAIM-C5, EC-19.1).
3. **Claim phase** (NEW) — `assess` → refuse or `claim`.
4. **Serve** (shipped) — through `apply_owner`, tagged with `circuit.id`.
5. **Rollback** — the shipped path already has one (`except: ... clear_circuit_steering(); raise`). It
   must become owner-scoped: release this circuit's claims and its own keys, never the global clear.

```python
# circuit_service.py::activate — the claim phase
if settings.CIRCUIT_ALLOW_CONCURRENT:
    requested = {layer: {m.feature_idx: eff for m in members if m.layer == layer}
                 for layer in served_layers}
    verdict = await self._claims.assess(circuit.id, requested)

    if verdict.has_collision:                              # CLAIM-K1 — no override
        raise CircuitLayerContentionError(
            contended_layers=sorted({k["layer"] for k in verdict.collision_keys}),
            incumbents=verdict.incumbents, requested=circuit,
            collision_keys=verdict.collision_keys)         # message says: cannot be overridden

    if verdict.has_contention and not allow_layer_overlap:  # CLAIM-R1
        raise CircuitLayerContentionError(
            contended_layers=verdict.contended_layers,
            incumbents=verdict.incumbents, requested=circuit)

    composed = set(verdict.contended_layers) if allow_layer_overlap else set()
    await self._claims.claim(circuit.id, [
        LayerClaim(circuit.id, l, l in composed, requested.get(l, {}))
        for l in served_layers])
    if composed:
        await self._claims.mark_composed(composed)          # incumbents too — §3
        logger.warning("circuit_layer_overlap_accepted", circuit_id=circuit.id,
                       incumbent_ids=[c.circuit_id for c in verdict.incumbents.values()],
                       composed_layers=sorted(composed))     # CLAIM-O4
else:
    # Flag off: preserve the pre-existing single-active path EXACTLY (CLAIM-M4).
    await self._deactivate_other_active_circuits()
```

**Atomicity (CLAIM-R3).** The refusal raises BEFORE `claim` and before any serve, so nothing is
applied and the incumbent is untouched. The reverse order — claim, serve, then discover a problem —
would leave a live claim row on a layer nothing is steering. The `claim` INSERT can still lose the
unique-index race (EC-19.7); `IntegrityError` is caught and converted to the same
`CircuitLayerContentionError`, so a race and a plain overlap are indistinguishable to the caller, which
is correct — in both cases someone else holds the layer.

**Error class** (`millm/core/errors.py`, following the shipped pattern at `:289-307`):

```python
class CircuitLayerContentionError(MiLLMError):
    """Activation whose claim set overlaps an incumbent's.

    House style 200 + success:false — nothing is missing; the operation does not
    apply. Names the incumbent so the operator's next action is obvious:
    deactivate it, edit one circuit's layers, or re-send with
    allow_layer_overlap=true (EXCEPT for same-key collisions, which have no
    override — the merge would serve a strength belonging to neither author).
    """
    code = "CIRCUIT_LAYER_CONTENTION"
    status_code = 200
```

Registered in `ERROR_CLASSES` (`errors.py:320-353`) beside `UNVALIDATED_CIRCUIT` and
`NO_ACTIVE_CIRCUIT`, both of which already use the 200 + envelope house style.

## 6. Rung Suppression Under Composition (`millm/services/inference_service.py`)

`_steering_circuit()` (`:780-804`) returns ONE circuit and is documented as "the single predicate
behind all three surfaces — the apply, the λ echo, and the rung echo. Any surface that answers 'what is
steering' must ask THIS, never re-derive it." That property is exactly right and must be preserved
while the return type becomes plural.

```python
async def _steering_circuits(self) -> list[Any]:
    """Every circuit genuinely steering right now (was: the one circuit).

    Same memoisation discipline as the singular version: a CONTEXTVAR, never
    `self` — get_inference_service is @lru_cache'd, so a self-attribute memo is
    written once per PROCESS and would advertise a deactivated circuit's rung
    forever (the R2 defect the docstring records).
    """

async def active_circuit_rung(self) -> Optional[tuple[int, str]]:
    circuits = await self._steering_circuits()
    if len(circuits) != 1:
        return None            # 0 = nothing steering; >1 = composed  [CLAIM-O3]
    if await self._claims.any_composed_layer_for(circuits[0].id):
        return None            # single circuit, but sharing a layer   [CLAIM-O3]
    ...                        # shipped rung coercion unchanged
```

Returning `None` is what already causes the header to be omitted: `chat.py:141` and `:154` set
`X-miLLM-Circuit-Rung` only when `echo_circuit_rung` is truthy, so suppression needs no route change.
Both the streaming and non-streaming paths are covered by the same predicate.

Note the second condition. A single circuit on a composed layer must ALSO omit — the incumbent's
evidence stopped describing the response the moment someone composed onto its layer. Testing only
`len(circuits) > 1` would let the incumbent keep advertising a rung for a response another circuit is
now contributing to, which is the overclaim this rule exists to prevent.

## 7. Migration (`millm/db/migrations/versions/013_add_circuit_layer_claims.py`)

`down_revision = "012"` — verified current disk tail is `012_add_circuit_edge_sensing.py`.

```python
def upgrade():
    op.drop_index("uq_circuits_active", table_name="circuits")
    op.create_table("circuit_layer_claims", ...)            # FPRD §4
    op.create_index("uq_circuit_layer_claim_live", "circuit_layer_claims", ["layer"],
                    unique=True,
                    postgresql_where=sa.text("released_at IS NULL AND composed = false"),
                    sqlite_where=sa.text("released_at IS NULL AND composed = 0"))
    # Backfill: the (at most one, by the old invariant) active circuit keeps its layers.
    # Derived from circuit_meta, so an unparseable row backfills NO claims and is
    # reconciled at startup rather than blocking the migration.

def downgrade():
    """Deterministic and TOTAL — resolves pre-existing multi-active rows (RSK-008)."""
    # 1. Keep only the most recently activated circuit.
    op.execute("""UPDATE circuits SET is_active = false, serving_mode = NULL
                  WHERE is_active = true AND id NOT IN (
                      SELECT id FROM circuits WHERE is_active = true
                      ORDER BY updated_at DESC LIMIT 1)""")
    # 2. Drop claims, restore the old invariant.
    op.drop_table("circuit_layer_claims")
    op.create_index("uq_circuits_active", "circuits", ["is_active"], unique=True,
                    postgresql_where=sa.text("is_active = true"),
                    sqlite_where=sa.text("is_active = 1"))
```

Both `postgresql_where` and `sqlite_where` are supplied, matching the shipped model's dual-dialect
declaration (`circuit.py:100-101`) — the test suite runs on SQLite and production on PostgreSQL, so a
Postgres-only predicate would make the index non-unique in every test.

The downgrade's step order matters: deactivate FIRST, then create the index. Creating
`uq_circuits_active` while two rows still have `is_active = true` fails, and a failed downgrade in the
middle of a rollback is the worst possible time to discover it. CLAIM-M3 requires this be RUN against a
seeded two-active state, not merely written — the BRD's reachability rule (BR-005) applies to
migrations too: the test must FAIL if the deactivation step is removed.

## 8. Shipped Call Sites That Assume Singularity

Each is correct under `uq_circuits_active` and wrong without it. This list is the review checklist.

| Site | Assumption | Required change |
|---|---|---|
| `circuit_repository.py:78-83` `get_active()` | `scalar_one_or_none()` — **RAISES `MultipleResultsFound` on two active rows** | Add `list_active()`; keep `get_active()` for the flag-off path or make it order+limit-1 |
| `sae_service.py:~653` apply loop | `sae.clear_steering()` before each write — wipes co-tenants | Route through `apply_owner` / `_rebuild_layer` (§4) |
| `circuit_service.py:~688` `deactivate` | unqualified `clear_circuit_steering()` — clears ALL layers | Owner-scoped release of this circuit's keys only (CLAIM-D2) |
| `circuit_service.py:376` `_release_co_tenants` | releases clusters for served layers | Unchanged (clusters are not claimants, §12) but must not release CIRCUIT co-tenants |
| `profile_service.py:330-350` `_release_active_circuit` | `repo.get_active()` then deactivate "the" circuit | Iterate `list_active()`; a profile taking layers must release every circuit holding one |
| `inference_service.py:780` `_steering_circuit` | returns one circuit | Plural + composed-layer suppression (§6) |
| `circuits.py:121` route | `await get_inference_service()._steering_circuit()` | Consume the plural predicate |
| `circuit_service.py:~350` `activate` rollback | global `clear_circuit_steering()` | Owner-scoped rollback |

`get_active()` raising on two rows is the sharpest of these: it is not a silent wrong answer but a
500 from a `scalar_one_or_none()` call in a path (`profile_service._release_active_circuit`) that is
explicitly documented as "best-effort by design — a bookkeeping failure must not block the user's
activation". A raise there would block exactly what the comment promises it will not.

## 9. Admin UI Design

`components/circuits/contention/`:
- `ClaimsStrip` — layer → claimant circuit, composed layers badged distinctly. The operator's answer to
  "who holds layer 13", fed by `GET /api/circuits/claims`.
- `ContentionDialog` — raised on a `CIRCUIT_LAYER_CONTENTION` refusal. Names the incumbent, lists the
  contended layers, offers "Deactivate '{incumbent}'" and "Compose anyway". The latter carries the same
  explicit-acknowledgement weight as the shipped `acknowledge_unvalidated` affordance and states the
  consequence in plain words: the rung badge will disappear because no single circuit's evidence
  describes a composed response. On a same-key collision the dialog shows the colliding
  `(layer, feature_idx)` pairs with both strengths and offers NO compose action.
- `CircuitCard` (MOD) — claimed-layer chips; a composed circuit shows the composed badge INSTEAD of its
  rung badge, never both.

## 10. Testing Strategy

### Unit
- `tests/unit/services/test_circuit_claim_registry.py`: assess disjoint / overlapping / same-key;
  self-exclusion on re-activation (EC-19.3); `release` touches only the caller's rows (CLAIM-D1);
  `reconcile` drops orphan claims (EC-19.4) and demotes with the flag off (EC-19.5); composed claims
  bypass the exclusive index while exclusive ones do not.
- `tests/unit/services/test_sae_owner_provenance.py`: `apply_owner` merge; `release_owner` leaves the
  co-tenant's keys applied and enabled (CLAIM-D2); `_rebuild_layer` RAISES on a colliding owner map
  (the §4 coupling assertion); release of the last owner disables steering.
- `tests/unit/services/test_circuit_contention_gate.py`: refusal payload shape (incumbent named by
  name, CLAIM-R2); collision refused with `allow_layer_overlap=true` set (CLAIM-K1); atomicity — no
  claim row and no steering after a refusal (CLAIM-R3).
- `tests/unit/services/test_rung_suppression.py`: `active_circuit_rung()` returns None for >1 circuit
  AND for a single circuit on a composed layer (CLAIM-O3, both branches).

### Integration
- `tests/integration/test_concurrent_circuit_serving.py`: two disjoint circuits serve, both steer,
  neither clears the other; deactivate one, the other survives with keys applied; contention refusal
  end-to-end through the route with the envelope asserted; override path with `composed_layers` in the
  response; **`X-miLLM-Circuit-Rung` absent from a real composed response** on both the streaming and
  non-streaming paths; flag-off single-active parity; startup reconciliation.
- `tests/integration/test_circuit_claim_race.py`: two concurrent activations for the same layer —
  exactly one succeeds, the loser gets `CIRCUIT_LAYER_CONTENTION`, and the DB index is what decided it
  (asserted by removing the service-level pre-check in the test's arrangement).
- `tests/integration/test_migration_013.py`: upgrade on a populated DB; **downgrade RUN** against a
  seeded two-active state leaving exactly the most recently activated circuit (CLAIM-M3); round-trip
  upgrade→downgrade→upgrade; the downgrade test FAILS when the deactivation step is removed (BR-005).

### E2E (post-deploy)
Circuits page: two disjoint circuits serving with claim chips; a contended activation raising the
dialog with the incumbent named; a composed serve showing the composed badge and NO rung badge.

## 11. Risks

- **The invariant window.** Between `DROP INDEX uq_circuits_active` and the first claim row, nothing
  enforces anything. The migration creates the claim table and its index in the same transaction, and
  the backfill runs inside it, so there is no window in practice — but a partially-applied migration
  (interrupted, or a dialect that auto-commits DDL) leaves the database with NO invariant at all. The
  startup reconciliation (CLAIM-M5) is the backstop and runs unconditionally, not only when the flag
  is false.
- **Claim leakage on deactivate.** The highest-consequence defect available in this feature: a release
  that clears a co-tenant's keys silently stops a circuit the operator did not touch, and the row still
  reports active. Structurally prevented by rebuilding each layer from the owner map rather than
  deleting incrementally (§4), and pinned by the co-tenant-survival test.
- **A composed layer emitting a rung header.** The overclaim this feature exists to prevent. Two
  branches must both suppress (`len != 1`, AND single-circuit-on-composed-layer); testing only the
  first leaves the incumbent advertising a rung for a response another circuit contributes to.
- **`get_active()` raising.** `scalar_one_or_none()` on two active rows raises `MultipleResultsFound`
  inside a path documented as best-effort and non-blocking (§8). Found by reading; must be pinned by a
  test that seeds two active rows and calls the profile activation path.
- **Same-key collision slipping through the override.** `allow_layer_overlap` is checked for layer
  contention; if the collision check is placed after it or shares its branch, the override silently
  covers both. The gate order in §5 puts the collision check FIRST and unconditionally, and a test
  asserts the refusal with the override explicitly set.
- **`merged.update()` last-writer-wins.** Safe only because CLAIM-K1 refuses collisions upstream. The
  coupling is invisible locally, so `_rebuild_layer` raises on a colliding owner map rather than
  trusting a guarantee made in another file.
- **Flag-off drift.** With `CIRCUIT_ALLOW_CONCURRENT=false` the old path must behave byte-identically.
  The risk is that the new code path is the only one exercised by new tests while the old one rots.
  Parity is asserted explicitly rather than assumed.
