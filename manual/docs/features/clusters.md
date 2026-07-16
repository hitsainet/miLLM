---
sidebar_position: 6
title: Clusters
---

# Clusters — Imported Steering Definitions

The **Clusters** page runs portable cluster definitions — groups of SAE features tuned together in
miStudio (or shared by the community on Hugging Face) — as one-click steering profiles. A cluster
carries its author's narrative, every member's tuned strength, a validated strength budget, and full
provenance; activating it applies **all members together** with zero manual tuning.

## Importing

Click **Import** on the Clusters page. Three ways in:

- **Paste JSON** — paste a `mistudio.cluster-definition/v1` document (or a
  `mistudio.cluster-bundle/v1` with up to 50 definitions).
- **Upload file** — a `.cluster.json` exported from miStudio.
- **Hugging Face** — browse public cluster packs (repos tagged
  `mistudio-cluster-definition`), optionally filtered to your loaded model, and import
  anonymously — no Hugging Face account needed.

Imports are validated strictly against the frozen v1 contract (size ≤ 1 MB, ≤ 20 members, ≤ 50
definitions per bundle, no filesystem paths, nothing is ever executed). Bundle items import
independently — one bad definition never blocks the rest.

### Compatibility

Each import is assessed against the attached SAE:

| Outcome | Meaning |
|---|---|
| imported | Bound to the attached SAE (warnings shown for model/layer differences) |
| imported unbound | No SAE attached, or the definition's feature space differs — the cluster imports for inspection and binds later |
| blocked at activation | The hard gate: a cluster whose declared `n_features` doesn't match the attached SAE, or whose member indices are out of range, refuses to activate (nothing is partially applied) |

An **unbound** cluster binds automatically the first time it activates successfully against a
compatible SAE.

## Activating & the intensity dial (λ)

**Activate** applies every member at `sign x strength x λ`, clamped to the ±200 steering range.
Member strengths are stored exactly as authored (λ=1 basis) — the **intensity dial** scales them at
apply time, within the definition's declared safe range (`budget.intensity_range`; dialing to 0 is
always allowed). Changing λ while the cluster is active re-applies immediately; a failed re-apply
rolls the dial back so the display never lies about what's running.

Clusters are profiles under the hood: activating one deactivates any active manual profile (one
active steering configuration at a time), and the per-request `profile` parameter on
`/v1/chat/completions` accepts cluster names too — scaled by the cluster's current λ.

:::caution Editing
An imported cluster's steering values can't be edited directly (that would silently diverge from
the stored definition and double-scale under λ). Adjust the dial, or re-import an updated
definition. Name and description edits are fine.
:::

## Exporting

**Export** re-emits the exact original document — byte-for-byte lossless, including any additive
fields from newer producers. The file it downloads is the portable artifact; share it, publish it
to Hugging Face, or import it into another miLLM/miStudio instance.

## Co-Activation Sensing

With a cluster active, **sensing** watches every forward pass for moments where the cluster's
members fire *together* — evidence that the concept the cluster encodes is live in the model's
processing. Toggle it per cluster (the pulse icon on the cluster card); it arms automatically
whenever that cluster is active with an SAE attached, and the **Co-Activation Sensing** panel
shows events live.

**What counts as an event:** at a token position, a member *fires* when its activation exceeds
`θᵢ = max(θ_floor, ε · max_activationᵢ)` (ε = 0.1 by default; definitions without
`max_activation` data fall back to the floor and the panel shows a *floor-only thresholds*
warning). An **event** is a position where at least `min_k` members fire at once (default:
~30% of the cluster, minimum 2). Consecutive positions merge into one span.

**Attribution convention:** an event attaches to the token being *read* at that position — the
token the model emits next is unknowable at sensing time. Under speculative decoding, spans may
include positions of later-rejected draft tokens (accepted v1 inaccuracy). The
`ambient_fired_count` field (how many features across the *whole* SAE fired — the
"alone vs. within a crowd" signal) is best-effort: it is filled only when full-width monitoring
happens to be running, and is never estimated.

**Context & privacy:** each event stores a ±K token context window (default 16, max 64;
set `context_tokens: 0` in the definition's `sensing` block to keep events without text).
Context is user content in the database — retention is bounded by construction
(newest 1000 events per cluster, 7-day age cap, pruned automatically), events can be cleared
from the panel, and WebSocket payloads never carry the context text.

**Performance:** sensing is a member-only encode (≤ 20 encoder columns) per pass — microseconds.
The panel shows per-request overhead and warns above 5 ms. Armed sensing forces requests onto
the serial path so events attribute to the right request; with sensing off the cost is one
boolean per pass.

## API

Everything is available programmatically under `/api/clusters` — list, import, hub search
(`/api/clusters/hub/search`), activate/deactivate, intensity (`PUT /api/clusters/{id}/intensity`,
`PUT /api/clusters/active/intensity`), and export. See the API reference for shapes.
