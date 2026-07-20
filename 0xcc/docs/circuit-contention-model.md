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
  "message": "Layers [13] are already served by circuit 'fear\u2192threat' (circ_abc). Overriding composes both circuits additively on those layers. In close-out testing, TWO steered layers at individually-harmless strength (5) destroyed generation entirely \u2014 two orders of magnitude below the per-member clamp. Pass allow_layer_overlap=true only if you intend a compounding study; the circuit-rung header is omitted while any layer is composed, because no single circuit's evidence describes the response.",
  "details": { "contended_layers": [13],
               "incumbent": {"id": "circ_abc", "name": "fear\u2192threat"},
               "requested": {"id": "circ_xyz", "name": "hedging"},
               "override_param": "allow_layer_overlap",
               "measured_hazard": {
                 "source": "GPU close-out 2026-07-20, LFM2.5-1.2B-Instruct",
                 "one_layer_at_strength_5": "coherent, indistinguishable from baseline",
                 "two_layers_at_strength_5": "degenerate output",
                 "note": "one model, one fixture \u2014 indicative, not exhaustive"
               },
               "rung_header_suppressed_if_overridden": true } }
```

Refusal names the incumbent so the operator's next action is obvious: deactivate it, or edit one
circuit's layers. It also carries the MEASUREMENT behind the refusal, per the settled retention
condition in §6.2 — an operator overriding this has been told what happened last time, and the
hazard block is explicit that it is one model and one fixture rather than an exhaustive claim.

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

## 6. Settled decisions (product owner, 2026-07-20)

The three items below were open when this document was written. Two are now decided; the third is
promoted to its own requirement.

### 6.1 `CIRCUIT_ALLOW_CONCURRENT` defaults to `false` for ONE release — with a dated flip

**Decided: ship `false`, commit the flip date now, and drop the flag entirely if no other deployment
exists at flip time.**

The flag exists for exactly one reason: the first concurrent activation is a **one-way door in
deployed data**. Once two circuits have been active simultaneously, `uq_circuits_active` cannot be
restored without choosing which row to destroy. That asymmetry — trivial to enable, destructive to
reverse — is what justifies a gate. It is *not* a statement of doubt about the implementation.

Two conditions attach, because a default-off flag has its own failure mode:

- **The flip is dated, not deferred.** An unflipped flag makes a shipped capability unreachable, which
  is the precise defect class this increment exists to eliminate. Shipping `millm_circuits` MCP tools
  that fail against a disabled flag would be self-defeating in the most literal sense.
- **`false` means refuse LOUDLY**, naming configuration as the reason. It must not fall back to the
  silent single-active disarm that Feature 19 replaces — that is the bug, not the safe default.

Tracked as **BR-011a**.

### 6.2 `allow_layer_overlap` is RETAINED — on condition it is loud and informed

**Decided: keep it.** This was the closer call, and the argument against is recorded honestly: it is a
flag whose only function is to permit a configuration the close-out measured as destructive. "A
footgun with a label on it" is fair.

It is retained for three reasons, in order of weight:

1. **The measurement is thinner than it sounds.** One model, arbitrarily chosen feature indices,
   invented `max_activation` values. It proves *that fixture* compounds destructively — not that all
   overlapping circuits do. A circuit built from mined features at calibrated strengths might compose
   fine; there is no data either way. Hard-refusing a legitimate research action on one
   unrepresentative data point is overreach.
2. **Deliberate compounding studies are a real use case.** This is an interpretability tool. "What
   happens when two circuits contend for a layer?" is a legitimate question, and a runtime that
   forbids asking it is the wrong tool.
3. **The honesty guarantee holds either way.** The rung header is omitted when composed, so the system
   never claims evidence it does not have. The danger was never "the user got garbage" — it was "the
   user got garbage while being told a validated circuit produced it."

**The retention condition is binding: the override must be LOUD AND INFORMED.**

- The refusal that precedes it **carries the measurement**, not merely the fact of contention. An
  override chosen knowing that two steered layers at individually-harmless strength destroyed
  generation in testing is a research decision; one chosen blind is a footgun.
- Every use is **echoed in the response, logged, and surfaced in the UI**, mirroring
  `acknowledged_unvalidated`, which is already echoed back at `circuit_service.py:349-351`.

### 6.3 Warning on circuit SHAPE is a separate requirement

Not a contention concern — a **single** circuit spanning 2+ layers already degenerates, so
layer-exclusive claims do not address it. Promoted to **BR-012** and its own feature-level work rather
than folded here.

---

## 7. Superseded: original open questions

### The questions as originally posed

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
