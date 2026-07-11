---
sidebar_position: 3
title: How Monitoring Works
---

# How Monitoring Works

Monitoring is the observational half of miLLM: instead of writing into the residual stream (steering), it reads from it — running the SAE encoder on live activations to see which features fire during inference.

## The capture path

With monitoring enabled, the same hook that applies steering also does:

```
feature_acts = ReLU(hidden_states @ W_enc + b_enc)   # (seq_len, d_sae)
```

on every forward pass through the attached layer. After a generation request completes, the captured activations are handed to the monitoring service, which:

1. Records an **activation entry** in a ring-buffer history (default 100 entries), tagged with the request ID
2. Updates **running statistics** per feature (count, mean, std, min, max, active ratio)
3. Emits a throttled **`monitoring:activation`** WebSocket event for live UIs

:::info What exactly is captured
The hook's capture is overwritten on every forward pass, and it is read once, after generation finishes. What survives is the **final forward pass** — i.e. the activations of the **last generated token**, not the whole sequence. Per-request monitoring data should therefore be read as "which features were active at the end of this generation," not a token-by-token trace. Prompt-token and intermediate-step activations are not currently recorded.
:::

Monitoring observes the stream **before** steering is added at that layer, so monitored activations reflect what the model computed naturally at the hook layer — but note that on layers *downstream* of active steering, the stream (and hence what an SAE there would see) is already steered.

## Monitoring modes

### All features (default)

Every feature is captured; only non-zero activations are stored (sparsity makes this cheap — typically 30–100 non-zero features per token). The **top-k** highest activations (default 10, configurable 1–1000) are highlighted in each record and in WebSocket events.

### Watched subset

Configure a specific feature list to monitor only those indices:

```bash
curl -X POST http://localhost:8000/api/monitoring/configure \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "features": [12082, 4517, 9001], "top_k": 10}'
```

This reduces capture memory (only the selected columns are kept) and makes history entries report exactly your watchlist. Feature indices are validated against the attached SAE's `d_sae` — an out-of-range index returns `400 INVALID_FEATURE_INDEX` instead of breaking inference.

## Statistics semantics

Statistics accumulate across requests until reset (`DELETE /api/monitoring/statistics`):

| Metric | Meaning |
|--------|---------|
| `count` | Number of recorded observations of the feature |
| `mean`, `std` | Distribution of its activation values |
| `min`, `max` | Range observed |
| `active_ratio` | Fraction of observations where the activation was > 0 |

`POST /api/monitoring/statistics/top` ranks features by any of these metrics — useful for "what fires most in my workload?" exploration.

## Interaction with steering & backends

- **Monitoring + steering together** is the standard experimental setup: steer feature A, watch whether features B and C shift.
- **Embeddings requests** never pollute monitoring — the hook is suppressed during `/v1/embeddings`.
- **Continuous batching**: in a CBM batch, capture is per batch-slot and slots don't map to request IDs, so attribution is approximate (entries are tagged `request:batch_N`). Set `CBM_FORCE_SERIAL_MONITORING=true` to route monitored requests through the serial path for exact per-request attribution. See [Architecture](/concepts/architecture).
- **Overhead**: one extra matrix multiply per forward pass — `(1 × d_in) @ (d_in × d_sae)`. Negligible for 16k SAEs; measurable but small for 131k.

## Where to see it

The **Probe** page in the Admin UI renders live activations, the history, and per-feature statistics with direct Neuronpedia links per feature — see [Probe Monitoring](/features/probe-monitoring). The raw data is available at [`/api/monitoring/*`](/api/monitoring).
