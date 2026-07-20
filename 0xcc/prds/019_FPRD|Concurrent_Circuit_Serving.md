# Feature PRD: Concurrent Circuit Serving

## miLLM Feature 19

**Document Version:** 1.0
**Created:** July 20, 2026
**Status:** Draft
**Feature Priority:** Secondary (Increment: Circuit Consolidation)
**References:** `BRD-MILLM-CIRCUITS-002.md` (BR-011, RSK-007, RSK-008) · `000_PPRD|miLLM.md` (v1.3, FR-19.x) · `000_PADR|miLLM.md` (v1.3) · `docs/circuit-contention-model.md` (design of record)

---

## 1. Feature Overview

### Feature Name
Concurrent Circuit Serving — more than one circuit serving at once, under LAYER-EXCLUSIVE CLAIMS.

### Brief Description
miLLM's single-active-circuit invariant is lifted. A circuit claims the layers its serving members
reach; a layer is claimed by at most one active circuit, so circuits with disjoint claim sets serve
concurrently and freely. An activation whose claim set overlaps an incumbent's is REFUSED with
`CIRCUIT_LAYER_CONTENTION` (200 + `success: false`) naming the incumbent circuit and the contended
layers, so the operator learns about the conflict while they can still choose. An explicit
`allow_layer_overlap` acknowledgement permits additive composition on the contended layers — and
while any layer is composed, `X-miLLM-Circuit-Rung` is OMITTED, because no single circuit's evidence
describes a composed response. Two circuits naming the same `(layer, feature_idx)` are refused
UNCONDITIONALLY, with no override, because the merge would serve a strength belonging to neither
author. Deactivation releases exactly that circuit's own claims and steering keys and never a
co-tenant's, which requires per-circuit key provenance.

### Problem Statement
The single-active invariant is not a service-layer convention that can be relaxed by deleting a
guard — it is enforced by the `uq_circuits_active` **partial unique index** on `circuits.is_active`
(`millm/db/models/circuit.py:97-102`). Lifting it is a migration plus a semantic decision about what
happens when two active circuits both steer the same layer, and that decision cannot be deferred:
BR-001's request-scoped context must be built for N circuits from the outset, and designing it around
one circuit and generalising later would repeat precisely the mistake this increment exists to
correct (BRD assumption, RSK-007).

The decision is forced by measurement, not by taste. Steering application is
`modified = original + Σ(strength_i × W_dec[i])`, the ±200 `clamp_steering` gate bounds each member
INDIVIDUALLY, and nothing bounds the sum. The GPU close-out (2026-07-20,
`0xcc/tasks/014_FTASKS|Circuit_Dial.md` acceptance evidence) measured the consequence on
LFM2.5-1.2B-Instruct with prompt, seed and temperature held fixed:

| Configuration | Result |
|---|---|
| 1 layer, 1 member @ strength 5 | Coherent, indistinguishable from baseline |
| **2 layers**, 1 member each @ strength 5 | **Degenerate** — repeated `" lé"` tokens |
| 5 layers @ authored 20–40, λ=0.02 | Degenerate |

Cross-layer compounding destroys generation at TWO layers, two orders of magnitude below the
per-member clamp. Silent composition is therefore not a theoretical hazard but a reliable way to
produce garbage output the operator cannot distinguish from a bad circuit. This is why the model
refuses by default rather than composing by default.

### Feature Goals
1. Serve ≥2 circuits simultaneously when their claim sets are disjoint (BR-011).
2. Refuse overlapping activation with a named incumbent and named contended layers (BR-011).
3. Provide an explicit, logged override that composes — with the rung header omitted (BR-011).
4. Refuse same-`(layer, feature_idx)` collision unconditionally, with no override (BR-011).
5. Release precisely: a deactivation never tears out a co-tenant's steering (BR-011).
6. Ship reversibly: tested downgrade, claim table, `CIRCUIT_ALLOW_CONCURRENT` flag (RSK-008).

### User Value Proposition
"I have a hedging-suppression circuit on layers 10–11 and a formality circuit on 14–15. Both serve at
once. When I tried to activate a third that also wanted layer 11, miLLM told me exactly which circuit
already had it instead of quietly wrecking my output."

### Connection to Project Objectives
Delivers BR-011, the one genuinely NEW capability in the Circuit Consolidation increment and its
largest single design change (RSK-007). It also discharges the BRD success metric "≥2 circuits
serving simultaneously with a defined, tested outcome for a contended layer" (baseline: 1; activating
a second silently disarms the first).

### BRD Traceability

| BR | Requirement (abbrev.) | Coverage IDs |
|----|-----------------------|--------------|
| BR-011 | Serve >1 circuit; layer-exclusive claims; refuse with named incumbent; `allow_layer_overlap` override omits the rung header; same-key collision refused unconditionally | CLAIM-C1..C5, CLAIM-R1..R3, CLAIM-O1..O5, CLAIM-K1..K2, CLAIM-D1..D4 |
| RSK-007 | Design-first; splittable into a follow-on without invalidating the increment | CLAIM-C1, CLAIM-M4 (the flag makes the split reversible at runtime) |
| RSK-008 | Lifting a DB-enforced invariant is irreversible in deployed data; tested downgrade + deterministic resolution for pre-existing rows + explicit acceptance gate | CLAIM-M1..M5 |
| BR-001 | The request-scoped context is built for N circuits (one ring per `(request, circuit)`) | CLAIM-D3 (per-circuit provenance is the context's to own) |

---

## 2. User Stories & Scenarios

#### US-19.1: Serve two disjoint circuits at once
**As a** researcher running two independent interventions
**I want to** activate a second circuit whose layers do not overlap the first
**So that** both serve concurrently instead of the second silently disarming the first.

**Acceptance Criteria:**
- [ ] Activating circuit B (layers 14–15) while circuit A (layers 10–11) serves leaves BOTH active
- [ ] `GET /api/circuits/active` returns a LIST; each entry carries its own claimed layers
- [ ] Each circuit's steering is applied through its own layers' SAEs; neither clears the other's
- [ ] With `CIRCUIT_ALLOW_CONCURRENT=false` the second activation behaves exactly as today

#### US-19.2: Learn about a contended layer before it hurts
**As an** operator
**I want** an overlapping activation refused, naming who holds the layer
**So that** my next action is obvious rather than requiring me to guess what changed.

**Acceptance Criteria:**
- [ ] Refusal is `CIRCUIT_LAYER_CONTENTION`, 200 + `success:false` (nothing is missing; the operation does not apply)
- [ ] `details` carries `contended_layers`, `incumbent{id,name}`, `requested{id,name}`
- [ ] The refusal is atomic: NOTHING of the requested circuit is applied, and the incumbent is untouched
- [ ] The message names the incumbent circuit by NAME, not only by id

#### US-19.3: Deliberately compound, with the evidence claim withdrawn
**As a** researcher deliberately studying cross-layer compounding
**I want to** override the refusal with an explicit acknowledgement
**So that** I can run the configuration — while the runtime stops claiming a rung it cannot justify.

**Acceptance Criteria:**
- [ ] `allow_layer_overlap=true` permits activation onto a contended layer, mirroring `acknowledge_unvalidated`
- [ ] The response and `GET /api/circuits/active` carry `composed_layers` and a warning naming the summed effect
- [ ] `X-miLLM-Circuit-Rung` is OMITTED on every request while any layer is composed
- [ ] The override is logged as an explicit operator act, with both circuit ids

#### US-19.4: Never serve a strength nobody authored
**As a** user of the evidence surface
**I want** same-`(layer, feature_idx)` collisions refused even under override
**So that** a served strength always belongs to an author who can be named.

**Acceptance Criteria:**
- [ ] Two circuits naming the same `(layer, feature_idx)` are refused with NO override path
- [ ] The refusal is distinguishable from a plain layer contention (distinct `details.collision_keys`)
- [ ] `allow_layer_overlap=true` does NOT bypass it — asserted directly in tests

#### US-19.5: Deactivate without collateral
**As an** operator
**I want** deactivation to release only that circuit's claims and keys
**So that** stopping one circuit does not silently stop the other.

**Acceptance Criteria:**
- [ ] Deactivating A on a composed layer leaves B's `(layer, feature_idx)` strengths applied and enabled
- [ ] Claims are released for A only; the layer becomes claimable again only when its last claimant releases
- [ ] Releasing the last composing circuit restores the layer to single-explanation state and the rung header returns

#### US-19.6: Reversible rollout
**As an** operator upgrading a deployed instance
**I want** the capability off by default with a tested downgrade
**So that** the first concurrent activation is a deliberate act, not something that arrives with an upgrade.

**Acceptance Criteria:**
- [ ] `CIRCUIT_ALLOW_CONCURRENT` defaults to **false**; with it false, a second activation follows the pre-existing single-active path
- [ ] The migration's downgrade deactivates all but the most recently activated circuit and restores `uq_circuits_active`
- [ ] The downgrade is exercised by an automated test against a seeded multi-active state, not merely written

#### Edge Cases
**EC-19.1: Slice-fallback claim set** — **Trigger:** the activating circuit degrades to `slice_fallback`
and serves ONE layer, not every bindable one. **Behavior:** the claim set is the layers ACTUALLY
served, never the layers the definition declares. Claiming declared-but-unserved layers would block a
disjoint circuit for no reason.
**EC-19.2: Cluster/profile holds a contended layer** — **Trigger:** an active cluster or Feature 10
profile steers a layer the circuit claims. **Behavior:** unchanged from today — `_release_co_tenants`
deactivates it with a user-visible warning. Clusters/profiles are NOT claimants in v1; only circuits
claim (see §12).
**EC-19.3: Same circuit re-activated** — **Trigger:** activating a circuit that is already active.
**Behavior:** NOT a contention against itself; it re-serves, replacing its own claims idempotently.
**EC-19.4: Claim held by a circuit whose row vanished** — **Trigger:** a claim row survives its
circuit (manual DB edit, failed cascade). **Behavior:** the claim is treated as stale and reclaimable;
reconciliation logs it loudly rather than deadlocking the layer forever.
**EC-19.5: Flag flipped off while two circuits serve** — **Trigger:** `CIRCUIT_ALLOW_CONCURRENT`
set false on restart with two active rows in the DB. **Behavior:** the runtime reconciles to the most
recently activated circuit (the downgrade rule) and logs the demotion; it never serves a state the
flag forbids.
**EC-19.6: Composition pushes a layer past the clamp** — **Trigger:** two circuits' strengths sum
beyond ±200 on a composed layer. **Behavior:** the existing per-member clamp still applies per member;
the SUM is not clamped (unchanged, and precisely the hazard the refusal-by-default exists to prevent).
The composed-layer warning states this explicitly.
**EC-19.7: Concurrent activation race** — **Trigger:** two activations for disjoint-looking claim sets
arrive simultaneously. **Behavior:** the claim uniqueness is enforced by a DB unique index, not only
by the service lock, so one wins and the other is refused — the invariant survives concurrent writers
and a restart.

---

## 3. Functional Requirements

### Claims (FR-19.1)

| ID | Requirement | Priority |
|----|-------------|----------|
| CLAIM-C1 | A circuit's claim set shall be the layers reached by its SERVING members, computed by the single serving derivation (BR-002) so activation and every other surface agree by construction | Must |
| CLAIM-C2 | A layer shall be claimed by at most one active circuit; activation succeeds iff the claim set is disjoint from every currently-claimed layer | Must |
| CLAIM-C3 | Claims shall persist in `circuit_layer_claims` with a unique index on `layer` where `released_at IS NULL`, so the invariant survives a restart and concurrent writers (EC-19.7) | Must |
| CLAIM-C4 | Circuits with disjoint claim sets shall serve concurrently; `GET /api/circuits/active` shall return a LIST, each entry carrying its own claimed layers and serving mode | Must |
| CLAIM-C5 | The claim set shall reflect layers ACTUALLY served, including the single-layer set of a `slice_fallback` serve (EC-19.1) | Must |

### Refusal (FR-19.2)

| ID | Requirement | Priority |
|----|-------------|----------|
| CLAIM-R1 | Overlap shall be refused with `CIRCUIT_LAYER_CONTENTION`, house style **200 + `success:false`** — nothing is missing; the operation does not apply | Must |
| CLAIM-R2 | The refusal shall carry `details.contended_layers`, `details.incumbent{id,name}`, `details.requested{id,name}`; the message shall name the incumbent by name | Must |
| CLAIM-R3 | Refusal shall be atomic: no steering, no claim row, no `is_active` flip for the requested circuit, and the incumbent's live steering untouched | Must |

### Override (FR-19.3)

| ID | Requirement | Priority |
|----|-------------|----------|
| CLAIM-O1 | `allow_layer_overlap=true` shall permit activation onto contended layers, accepting additive composition — mirroring the `acknowledge_unvalidated` gate | Must |
| CLAIM-O2 | The response and `GET /api/circuits/active` shall carry `composed_layers: [...]` and a warning stating those layers carry the summed effect of more than one circuit | Must |
| CLAIM-O3 | `X-miLLM-Circuit-Rung` shall be OMITTED for any request while ANY layer is composed — the same rule that already omits it for slice-fallback | Must |
| CLAIM-O4 | The override shall be logged as an explicit operator act naming both circuits and the contended layers; the response shall echo `allowed_layer_overlap: true` | Must |
| CLAIM-O5 | Edge sensing on a composed layer shall record `composed: true` on affected observations, because the measured downstream activation was influenced by a circuit outside the edge's own definition | Should |

### Same-key collision (FR-19.4)

| ID | Requirement | Priority |
|----|-------------|----------|
| CLAIM-K1 | Two active circuits naming the same `(layer, feature_idx)` shall be refused UNCONDITIONALLY — no override, including under `allow_layer_overlap=true` — since `set_steering_batch` merges and the served strength would belong to neither author | Must |
| CLAIM-K2 | The collision refusal shall be distinguishable from a plain layer contention via `details.collision_keys: [{layer, feature_idx, incumbent_strength, requested_strength}]` | Must |

### Release (FR-19.5)

| ID | Requirement | Priority |
|----|-------------|----------|
| CLAIM-D1 | Deactivation shall release exactly that circuit's own claims — never a co-tenant's | Must |
| CLAIM-D2 | Deactivation shall clear only the `(layer, feature_idx)` keys that circuit wrote, never the whole layer dict, which would tear out a co-tenant's steering | Must |
| CLAIM-D3 | The registry shall hold per-circuit key provenance so release is precise; the request-scoped context (BR-001) is its natural owner | Must |
| CLAIM-D4 | When the last composing circuit on a layer releases, the layer shall return to single-explanation state and the rung header shall resume | Must |

### Migration & rollout (FR-19.6)

| ID | Requirement | Priority |
|----|-------------|----------|
| CLAIM-M1 | Migration `013` shall drop `uq_circuits_active` and create `circuit_layer_claims` with the partial unique index on `layer` | Must |
| CLAIM-M2 | The migration shall ship a TESTED downgrade that deactivates all but the most recently activated circuit and restores `uq_circuits_active` | Must |
| CLAIM-M3 | The downgrade shall be exercised by an automated test against a seeded multi-active state — written and RUN, not merely present | Must |
| CLAIM-M4 | `CIRCUIT_ALLOW_CONCURRENT` (default **false**) shall gate the capability for at least one release; with it false the pre-existing single-active path is followed exactly | Must |
| CLAIM-M5 | On startup with the flag false and >1 active row, the runtime shall reconcile to the most recently activated circuit and log the demotion (EC-19.5) | Must |

---

## 4. Data Requirements

Migration `013_add_circuit_layer_claims.py` (`down_revision` = `012_add_circuit_edge_sensing.py`, the
current disk tail — verified `ls millm/db/migrations/versions`):

```sql
-- 1. Drop the DB-enforced single-active invariant
DROP INDEX uq_circuits_active;                         -- millm/db/models/circuit.py:97-102

-- 2. Replace it with layer-exclusive claim uniqueness
CREATE TABLE circuit_layer_claims (
  id            SERIAL PRIMARY KEY,
  circuit_id    VARCHAR(50) NOT NULL,           -- FK -> circuits.id, CASCADE
  layer         INTEGER NOT NULL,
  claimed_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  released_at   TIMESTAMPTZ NULL,               -- NULL = live claim
  composed      BOOLEAN NOT NULL DEFAULT FALSE, -- claimed under allow_layer_overlap
  steering_keys JSONB NOT NULL DEFAULT '[]'::jsonb  -- per-circuit key provenance (CLAIM-D3)
);
-- The invariant: one LIVE exclusive claim per layer. Composed claims are exempt,
-- since the override's whole purpose is to permit a second claimant.
CREATE UNIQUE INDEX uq_circuit_layer_claim_live
  ON circuit_layer_claims (layer)
  WHERE released_at IS NULL AND composed = FALSE;
CREATE INDEX idx_clc_circuit ON circuit_layer_claims (circuit_id, released_at);
```

`steering_keys` is the per-circuit key provenance CLAIM-D2/D3 require: the exact
`(feature_idx, strength)` pairs this circuit wrote to this layer, so release can clear precisely those
and leave a co-tenant's intact. The model mirrors the house pattern
(`JSONVariant = JSON().with_variant(JSONB(), "postgresql")`) used by `db/models/sensing_event.py`.

**Downgrade (CLAIM-M2):** deactivate every circuit except the one with the greatest `updated_at` among
active rows, delete all claim rows, and recreate `uq_circuits_active`. Deterministic and total — it
resolves pre-existing multi-active rows rather than failing on them (RSK-008).

## 5. API Specifications

Additive to the shipped `/api/circuits/*` surface (`millm/api/routes/management/circuits.py`); the
contract moves to v1.2 with no breaking change.

- `POST /api/circuits/{circuit_id}/activate` — gains `allow_layer_overlap: bool = false` beside the
  shipped `acknowledge_unvalidated` (query param AND body field, matching the existing dual shape at
  `circuits.py:208` / `:275`). Returns `claimed_layers`, `composed_layers`, `allowed_layer_overlap`,
  and any composed-layer warnings.
- `GET /api/circuits/active` — **shape change**: returns a LIST of active circuits, each with
  `claimed_layers`, `composed_layers`, `serving_mode`. A `?single=true` compatibility mode preserves the
  v1.1 single-object shape for the shipped Open WebUI filter until it is migrated.
- `POST /api/circuits/{circuit_id}/deactivate` — releases only this circuit's claims/keys; response
  carries `released_layers` and `still_claimed_layers`.
- `GET /api/circuits/claims` — the live claim table: layer → claimant circuit, `composed` flag. The
  operator's answer to "who holds layer 13".

Refusal envelope (200 + `success:false`), per the design of record §3.2:

```json
{ "code": "CIRCUIT_LAYER_CONTENTION",
  "message": "Layers [13] are already served by circuit 'fear→threat' (circ_abc).",
  "details": { "contended_layers": [13],
               "incumbent": {"id": "circ_abc", "name": "fear→threat"},
               "requested": {"id": "circ_xyz", "name": "hedging"} } }
```

## 6. UI Requirements
The Circuits page gains a **contention state**: each circuit card shows its claimed layers; a
contended activation surfaces a refusal dialog naming the incumbent with two obvious actions
("Deactivate 'fear→threat'" / "Compose anyway"), the latter gated behind the same explicit-acknowledgement
affordance as `acknowledge_unvalidated`. A composed layer renders a distinct badge and the rung badge is
replaced by an explicit "no single circuit's evidence describes this response" note — the UI must never
show a rung for a composed serve. A claims strip (layer → claimant) makes "who holds what" readable at a
glance. Components under `components/circuits/contention/`.

## 7. Non-Functional Requirements
- Claim check is one indexed query per activation; no hot-path cost. Serving requests are unaffected.
- The claim uniqueness must hold under concurrent writers and across restart — hence DB-backed, not
  service-memory (EC-19.7).
- With `CIRCUIT_ALLOW_CONCURRENT=false`, behaviour is byte-identical to today's single-active path.
- Release must be precise under partial failure: a failed clear must not leave a claim row live, and a
  failed claim release must not leave steering applied.

## 8. Dependencies
- **Feature 13** — circuit rows, `activate`/`deactivate`, the evidence gate.
- **Feature 18** — the single serving derivation (BR-002) supplies the claim set; §3.1 of the design
  of record requires the claim set come from that ONE derivation, not a second computation.
- **Feature 17** — the request-scoped context (BR-001) owns per-circuit key provenance and the
  per-`(request, circuit)` ring.
- **Feature 12** — multi-SAE attach; claims are meaningless without per-layer SAE resolution.

## 9. Success Criteria
1. Two circuits with disjoint claim sets serve simultaneously; both steer; neither clears the other.
2. Overlapping activation is refused with `CIRCUIT_LAYER_CONTENTION` naming the incumbent and the
   contended layers, atomically — nothing applied, incumbent untouched.
3. `allow_layer_overlap=true` composes, reports `composed_layers`, and OMITS `X-miLLM-Circuit-Rung`
   for every request while any layer is composed.
4. Same-`(layer, feature_idx)` collision is refused with `allow_layer_overlap=true` set — asserted directly.
5. Deactivating one of two co-tenants leaves the other's steering applied and enabled (no key leakage).
6. The migration's downgrade, RUN against a seeded two-active state, yields exactly one active circuit
   (the most recently activated) and a restored `uq_circuits_active`.
7. With `CIRCUIT_ALLOW_CONCURRENT=false`, the full pre-existing single-active behaviour is preserved.

## 10. Testing Requirements
- **Unit:** claim-set derivation from the single derivation (incl. slice-fallback, EC-19.1); disjoint /
  overlapping / same-key predicates; release computes exactly this circuit's keys; composed-layer rung
  suppression predicate; stale-claim reconciliation (EC-19.4).
- **Integration:** two-circuit disjoint serve; contention refusal atomicity; override path incl. header
  omission observed on a real response; same-key refusal under override; deactivate-one-of-two key
  survival; flag-off single-active parity; startup reconciliation (EC-19.5); concurrent-activation race
  resolved by the DB index (EC-19.7).
- **Migration:** upgrade on a populated DB; **downgrade RUN** against a seeded multi-active state
  (CLAIM-M3); round-trip upgrade→downgrade→upgrade.
- **E2E (post-deploy):** Circuits page — two disjoint circuits serving, a refusal dialog naming the
  incumbent, a composed serve showing no rung badge.

## 11. Rollout & Migration
Migration 013 runs automatically. The capability is DORMANT until `CIRCUIT_ALLOW_CONCURRENT=true`,
which is the explicit acceptance RSK-008 requires: the first concurrent activation is a one-way door in
deployed data, so it must be a deliberate operator act rather than something that arrives with an
upgrade. Recommended sequence: ship with the flag false, exercise the downgrade on a staging copy, then
enable.

## 12. Out of Scope
Clusters and profiles as claimants (v1: only circuits claim; the existing `_release_co_tenants` /
`_release_active_circuit` behaviour is preserved unchanged for them — see §15 decision 4). Per-layer
budget splitting between contending circuits (rejected option D in the design of record). Priority or
preemption (rejected option C). Bounding the SUM of composed strengths — that is BR-012 (circuit SHAPE
warning), a separate feature. Automatic resolution of a contention (the operator chooses).

## 13. Open Questions

Carried from `docs/circuit-contention-model.md` §6 — these are for the product owner, and the first two
gate acceptance:

1. **Default for `CIRCUIT_ALLOW_CONCURRENT`** — the proposal, and what these documents implement, is
   `false` for one release (CLAIM-M4). Ship enabled instead? Shipping enabled removes the deliberate-act
   property RSK-008 asks for; shipping disabled means the increment's headline capability is dark on
   arrival.
2. **Should `allow_layer_overlap` exist at all?** It is the only path to the configuration the close-out
   proved dangerous. FOR: refusing outright makes the runtime paternalistic about a legitimate research
   action, and researchers doing deliberate compounding studies are a real user. AGAINST: a flag that
   reliably produces garbage output is a footgun with a label on it. **Design-of-record recommendation:
   keep it, with the rung header omitted** — the honesty guarantee holds either way. These documents
   implement the recommendation; removing it later is a strict simplification (delete CLAIM-O1..O5).
3. **Should the runtime warn on circuit SHAPE regardless of contention?** The close-out found a SINGLE
   circuit spanning 2+ layers at moderate strength already degenerates. That is the same underlying
   hazard but it is NOT contention — layer-exclusive claims do not address it at all. **This is BR-012
   and is a SEPARATE FEATURE**, noted here only so the boundary is explicit: Feature 19 makes sure two
   circuits do not silently sum; it makes no claim about one circuit being too large.

## 14. Documentation Requirements
Manual: a Circuits page contention section — what a claim is, why the unit of contention is the LAYER
and not the feature (both circuits write the same layer dict and both contribute to the same residual
sum), what the refusal means and the two ways to resolve it, what composition costs (the rung header
goes away, and why that is honesty rather than a bug), and the `CIRCUIT_ALLOW_CONCURRENT` rollout note
with its one-way-door warning.

## 15. Decisions from Clarifying Questions
1. **Layer-exclusive claims, refuse by default** (design of record §2 option B). Additive composition
   (A), priority/preemption (C) and budget-splitting (D) were considered and rejected — A because the
   close-out data directly contradicts it, C because it yields a half-serving circuit whose rung is a
   MIN over edges that are no longer all present, D because it silently serves an intervention nobody
   authored or validated.
2. **The unit of contention is the LAYER, not the feature.** Two circuits steering DIFFERENT features on
   the same layer still contend: both write into that layer's single steering dict and both contribute
   to the same residual-stream sum.
3. **Same-key collision has no override** (§3.4). There is no honest composition of that case.
4. **Clusters/profiles are not claimants in v1.** The shipped `_release_co_tenants` (deactivate the
   cluster, warn) and `_release_active_circuit` (the symmetric profile-side half) are preserved as-is.
   Making them claimants is a larger change to Feature 10's model and is not required by BR-011.
5. **Claims are DB-backed, not service-memory** (§5.1): enforced under `_ATTACHMENT_LOCK` in the service
   AND backed by a partial unique index, so the invariant survives restart and concurrent writers.
