---
sidebar_position: 7
title: Circuits
---

# Circuits

A **circuit** is a multi-layer intervention: several SAEs, on different layers, steered together
as one thing. Where a [cluster](./clusters.md) is a set of features on a single layer, a circuit
is a directed graph — features on layer 10 feeding features on layer 13 — authored and validated
in miStudio, then imported here as a `mistudio.circuit-definition/v1` document.

Circuits carry something clusters do not: an **evidence rung** stating how well-supported the
claim is that these features actually relate to each other.

## The evidence ladder

Every circuit and every edge inside it carries a rung, mirrored verbatim from miStudio:

| Rung | Language | What it means |
|------|----------|---------------|
| 0 | `associated` | Mined co-occurrence only. The features appear together. Nothing more is claimed. |
| 1 | `suggested (attribution-supported)` | Attribution methods point at a relationship. Still not a causal claim. |
| 2 | `causally validated (edge)` | Each edge was causally validated by intervention in miStudio. |
| 3 | `faithfulness-tested (circuit)` | The circuit as a whole passed a faithfulness test. |

A circuit's rung is the **minimum over its edges** — one weakly-supported edge caps the whole
circuit, because a chain is only as strong as its least-evidenced link.

:::caution The word "causal"
Below rung 2, nothing in miLLM will describe a circuit as causal — not the UI, not the API, not
the logs. The phrase you see is rendered from the ladder and never composed per-surface, and a
build-time copy audit fails the build if hand-written causal language appears anywhere on a
runtime surface. **Activating a circuit below rung 2 requires an explicit acknowledgement.**

This matters because the failure mode is silent: a mined correlation described as a validated
mechanism reads exactly like the real thing.
:::

## Serving a circuit

Activation attaches every SAE the circuit references and steers each member through **its own
layer's** SAE, with per-layer budgets under one global λ.

If not every referenced SAE can be attached, activation **degrades to slice-fallback**: the
circuit is served as a per-layer cluster slice instead of the whole graph. The Circuits page says
so explicitly — a slice is never presented as the whole circuit.

Dial a serving circuit per request with the `steering_intensity` extension field; see
[the OpenAI-compatible API reference](../api/openai-compatible.md) and
[the Open WebUI tutorial](../tutorials/open-webui.md).

## Edge Sensing

With a circuit active, **edge sensing** watches live traffic for moments where an edge's
**upstream member fires and its downstream partner then fires shortly after**. Toggle it per
circuit on the circuit card; it arms when the circuit is active and its SAEs are attached, and
the **Edge Sensing** panel shows observations live.

**What counts as an observation.** At a token position a member *fires* when its activation
exceeds `θᵢ = max(θ_floor, ε · max_activationᵢ)` (ε = 0.1 by default) — the same rule cluster
sensing uses. An **edge observation** is an upstream fire followed by a downstream fire, in the
authored direction, within the **lag window** (8 tokens by default). Three things deliberately do
*not* count:

- **A lone upstream fire.** Upstream alone says nothing about the edge.
- **A reversed pair** (downstream, then upstream). That is not the authored direction.
- **A same-position co-fire.** Simultaneous firing is co-activation, not a sequence — reporting
  it as up→down would assert an ordering that was never observed.

When several upstream fires sit inside the window, the **nearest** one is reported: the closest
antecedent is the most defensible attribution.

:::danger An observation is not validation
Watching an edge fire is **co-activation evidence in the authored direction** — it is not
evidence that the edge is causal, and it never raises the edge's rung. Every recorded observation
stores the rung language *as of the moment it was observed*, so a later re-validation in miStudio
cannot retroactively upgrade months-old observations.
:::

### Edges that cannot be watched

Some edges are **unsensable**, and the panel lists them with a reason. This is deliberate and
load-bearing: without it, "no events" is indistinguishable from "the edge never fired", and
absence of observation would read as evidence of absence.

| Reason | Meaning |
|--------|---------|
| `layer_not_attached` | No unambiguously attached SAE on one of the edge's layers. Common under slice-fallback, which serves a single layer. |
| `no_activation_threshold` | An endpoint has no usable `max_activation` and no positive floor is configured, so it would either never fire or fire on everything. |
| `endpoint_not_a_feature` | An endpoint is a cluster supernode rather than a single feature, so there is no single activation to threshold. |

### Limits and exclusions

- **Serial routing.** An armed circuit forces serial generation: continuous-batching rows cannot
  be attributed to a request, so positions would be meaningless. Set
  `CIRCUIT_SENSING_FORCE_SERIAL=false` to prefer throughput — batched requests then simply go
  unsensed rather than mis-attributed.
- **Speculative decoding is excluded.** Verification passes advance the position counter by a
  whole candidate block and rejected tokens re-run, so absolute positions — which the matcher
  depends on — diverge.
- **Per-request cap.** At most 20 observations per request; beyond that the request is marked
  `truncated` rather than growing without bound.
- **Retention.** Observations are capped per circuit (1000) and pruned past an age window
  (7 days), on write and periodically on read. Deleting a circuit deletes its observations.

### Privacy

Observations store a small decoded context window (±16 tokens by default) so you can see what the
model was reading. That is **prompt content**. Two consequences worth knowing:

- Set `CIRCUIT_SENSING_CONTEXT_TOKENS=0` to store observations with no decoded text at all.
- The live WebSocket broadcast **never** carries context text — only the structural fields.

## Configuration

| Setting | Default | Purpose |
|---------|---------|---------|
| `CIRCUIT_SENSING_MAX_TOKEN_LAG` | `8` | Tokens between upstream and downstream for a pair to count. Capped at 64 — a wider window stops being an attribution and becomes a coincidence detector. |
| `CIRCUIT_SENSING_EPSILON` | `0.1` | Threshold fraction of each member's max activation. |
| `CIRCUIT_SENSING_THETA_FLOOR` | `0.0` | Absolute activation floor. |
| `CIRCUIT_SENSING_CONTEXT_TOKENS` | `16` | Context window per observation; `0` stores none. |
| `CIRCUIT_SENSING_MAX_EVENTS_PER_REQUEST` | `20` | Per-request cap. |
| `CIRCUIT_SENSING_MAX_EVENTS_PER_CIRCUIT` | `1000` | Retention cap. |
| `CIRCUIT_SENSING_MAX_AGE_DAYS` | `7` | Retention window. |
| `CIRCUIT_SENSING_FORCE_SERIAL` | `true` | Force serial routing while armed. |
| `CIRCUIT_SENSING_MAX_OVERHEAD_MS` | `5.0` | Per-request overhead above which a warning is logged. |

A circuit document may carry its own `sensing` block (`epsilon`, `theta_floor`, `max_token_lag`,
`context_tokens`) overriding the server defaults. Out-of-range values degrade to the default
rather than clamping — a negative epsilon would otherwise resurrect a fire-on-anything threshold.

## API

| Route | Purpose |
|-------|---------|
| `GET /api/circuit-sensing/status` | Armed state, layers, sensable and unsensable edges, overhead. |
| `GET /api/circuit-sensing/events` | Observations; filter by `circuit_id`, `edge_key`, `since`. |
| `GET /api/circuit-sensing/events/{id}` | One observation, with its context window. |
| `DELETE /api/circuit-sensing/events` | Clear observations, optionally for one circuit. |
| `POST /api/circuit-sensing/{id}/enable` · `/disable` | Persist the intent and arm/disarm live. |

Enabling records **intent** even when the circuit cannot arm right now (it is not active, or its
SAEs are not attached) — the response says which, rather than silently doing nothing. The live
WebSocket event is `circuit:sensing:event`.
