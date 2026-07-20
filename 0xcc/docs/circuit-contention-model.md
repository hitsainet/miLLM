# Circuit Contention Model — design of record for BR-011

**Status:** design proposal, 2026-07-20 · **Gates:** BR-001 (the request-scoped context)
**Author:** Sean · **Source:** BRD-MILLM-CIRCUITS-002 BR-011, RSK-007, RSK-008

BR-011 lifts miLLM's single-active-circuit invariant so more than one circuit can serve at once. That
invariant is not a service-layer convention — it is enforced by the `uq_circuits_active` **partial
unique index** on `circuits.is_active`, so lifting it is a migration plus a semantic decision about
what happens when two active circuits both steer the same layer.

This document settles that decision, because BR-001's request-scoped context must be built for N
circuits from the outset. Designing it for one and generalising later would repeat precisely the
mistake this increment exists to correct.

---

## 1. What contention actually is, mechanically

Steering state on an SAE is a `dict[feature_idx → strength]`, and application is:

```
modified = original + Σ(strength_i × W_dec[i])          # sae_wrapper.py:444
```

Three consequences follow, and they constrain every option below:

1. **The unit of contention is the layer, not the feature.** Two circuits steering *different*
   features on the same layer still contend, because both write into that layer's single steering
   dict and both contribute to the same residual-stream sum.
2. **Composition is additive and unbounded in aggregate.** The ±200 `clamp_steering` bounds each
   member *individually*. Nothing bounds the sum. Two circuits each individually reasonable can
   compose into an intervention neither author intended.
3. **Last-writer-wins on a key is silent.** `set_steering_batch` merges into the dict, so if two
   circuits name the *same* `(layer, feature_idx)` with different strengths, one simply overwrites the
   other with no record that it happened.

### The empirical constraint that decides this design

The GPU close-out (2026-07-20, `0xcc/tasks/014_FTASKS` acceptance evidence) measured, holding prompt,
seed and temperature fixed on LFM2.5-1.2B-Instruct:

| Configuration | Result |
|---|---|
| 1 layer, 1 member @ strength 5 | Coherent, indistinguishable from baseline |
| **2 layers**, 1 member each @ strength 5 | **Degenerate** — repeated `" lé"` tokens |
| 5 layers @ authored 20–40, λ=0.02 | Degenerate |

**Cross-layer compounding destroys generation at two layers, two orders of magnitude below the
per-member clamp.** Any contention model that lets two circuits silently sum on a layer is therefore
not a theoretical hazard — it is a reliable way to produce garbage output, and the operator would have
no signal distinguishing it from a bad circuit.

This is why the model below refuses by default rather than composing by default.

---

## 2. Options considered

### Option A — Additive composition (rejected)

Both circuits write; strengths sum on shared keys.

*Rejected.* It is the one option the close-out data directly contradicts. Two independently-sane
circuits reliably compose into an unusable model, and neither author is at fault. It also makes
"what is steering right now" unanswerable — no single circuit explains the output.

### Option B — Layer-exclusive claim, refuse on conflict (**recommended**)

A circuit claims the layers its members touch. Activation **refuses** if any claimed layer is already
claimed, naming the incumbent circuit and the contended layers. Non-overlapping circuits serve
concurrently and freely.

*Recommended* because it delivers BR-011's actual value — several circuits serving at once — while
making the dangerous case impossible rather than merely warned about. It preserves the property that
every steered layer has exactly one explanation, which is what makes `X-miLLM-Circuit-Rung` and the
edge-sensing attribution honest. Overlap is refused, not silently resolved, so an operator learns
about the conflict at the moment they can still choose.

*Cost:* two circuits overlapping on one layer cannot both serve, even if the operator would accept the
risk. Mitigated by the explicit override below.

### Option C — Priority / preemption (rejected)

Circuits carry a priority; the higher one wins a contended layer, the lower is partially suspended.

*Rejected.* It produces a circuit that is half-serving — some layers live, others preempted — which is
exactly the partial-graph state F13 already refuses via slice-fallback disclosure, and which makes the
rung meaningless (the rung is a MIN over edges that are no longer all present).

### Option D — Per-layer budget split (rejected)

Contending circuits share a layer's budget, each scaled down.

*Rejected.* It silently changes an authored intervention into one nobody authored or validated, and
the evidence rung would then describe a circuit that is not what is running. It also cannot be
verified: neither miStudio's effect sizes nor the runtime's clamp analysis apply to a scaled-down
composite.

---

## 3. The model

### 3.1 Claims

- A circuit's **claim set** is the set of layers reached by its serving members, computed by the
  single serving derivation (BR-002) so activation and every other surface agree by construction.
- A layer is claimed by **at most one** active circuit.
- Activation succeeds iff the claim set is disjoint from every currently-claimed layer.

### 3.2 Refusal

On overlap, activation refuses with a new code `CIRCUIT_LAYER_CONTENTION`, house style **200 +
`success: false`** (nothing is missing; the operation does not apply), carrying:

```json
{ "code": "CIRCUIT_LAYER_CONTENTION",
  "message": "Layers [13] are already served by circuit 'fear→threat' (circ_abc).",
  "details": { "contended_layers": [13],
               "incumbent": {"id": "circ_abc", "name": "fear→threat"},
               "requested": {"id": "circ_xyz", "name": "hedging"} } }
```

Refusal names the incumbent so the operator's next action is obvious: deactivate it, or edit one
circuit's layers.

### 3.3 Explicit override — `allow_layer_overlap`

An operator may pass `allow_layer_overlap=true` to activate anyway, accepting additive composition on
the contended layers. This mirrors the `acknowledge_unvalidated` gate: the system refuses by default,
and a human can override with an explicit, logged act.

When overridden:

- The response and `GET /api/circuits/active` carry `composed_layers: [13]` and a warning stating that
  those layers carry the summed effect of more than one circuit.
- `X-miLLM-Circuit-Rung` is **omitted** for the request. The rung describes a *specific* circuit's
  evidence; when two circuits compose, no single rung describes what the user received, and emitting
  either one would overclaim. This is the same rule that already omits the header for slice-fallback.
- Edge sensing on a composed layer records `composed: true` on affected observations, because the
  downstream activation it measured was influenced by a circuit outside the edge's own definition.

### 3.4 Same-key collision

Even under override, two circuits naming the same `(layer, feature_idx)` are **always refused** —
`set_steering_batch` merges, so one strength would silently overwrite the other and the served value
would belong to neither author. There is no honest composition of that case, so it has no override.

### 3.5 Release

Deactivating a circuit releases exactly its own claims and clears only the `(layer, feature_idx)` keys
it wrote — never the whole layer dict, which would tear out a co-tenant's steering. This requires
per-circuit key provenance in the registry, which the request-scoped context (BR-001) is the natural
owner of.

---

## 4. Consequences for BR-001

The context must therefore be built for N circuits, not one:

| Concern | Single-circuit shape (today) | N-circuit shape (required) |
|---|---|---|
| Position counter | one per request | unchanged — position is per request, not per circuit |
| Edge ring | one per request | **one per (request, circuit)**; two circuits' edges must not match against each other |
| Event budget | one per request | per request, **attributed per circuit** so one busy circuit cannot starve another's observations |
| Steering keys | implicit | **explicit per-circuit provenance**, so release is precise |
| `_steering_circuit()` | returns one circuit | returns the **set**; the rung echo omits when more than one is composed |

The ring is the sharpest of these: it is keyed by `edge_key`, and two circuits could legitimately
contain the same edge. Sharing one ring would let circuit A's upstream fire match circuit B's
downstream fire and be recorded as an observation of an edge that never fired in either. **One ring
per (request, circuit).**

---

## 5. Migration and reversibility (RSK-008)

1. Drop `uq_circuits_active`; replace with a **layer-claim uniqueness constraint** derived from the
   active set — enforced in the service under `_ATTACHMENT_LOCK`, and backed by a `circuit_layer_claims`
   table with a unique index on `layer` where released_at IS NULL, so the invariant survives a restart
   and concurrent writers.
2. Ship the migration **with a tested downgrade** that deactivates all but the most recently activated
   circuit, so the pre-existing single-active shape is recoverable.
3. Treat the first concurrent activation as a **one-way door in deployed data** and gate the feature
   behind a config flag (`CIRCUIT_ALLOW_CONCURRENT`, default **false**) for at least one release, so
   the capability can be enabled deliberately rather than arriving with an upgrade.

---

## 6. Open questions for the product owner

1. **Default for `CIRCUIT_ALLOW_CONCURRENT`** — proposal is `false` for one release. Ship enabled
   instead?
2. **Should `allow_layer_overlap` exist at all?** It is the only path to the configuration the
   close-out proved dangerous. The argument for keeping it is that refusing outright makes the runtime
   paternalistic about a legitimate research action; the argument against is that a flag which
   reliably produces garbage output is a footgun with a label on it. **Recommendation: keep it, with
   the rung header omitted** — the honesty guarantee is preserved either way, and researchers doing
   deliberate compounding studies are a real user.
3. **Should the runtime warn on circuit SHAPE regardless of contention?** The close-out found that a
   *single* circuit spanning 2+ layers at moderate strength already degenerates. That is not
   contention, but it is the same underlying hazard, and the runtime currently has no opinion about it
   independent of miStudio's effect sizes. Proposed as a separate BR rather than folded in here.
