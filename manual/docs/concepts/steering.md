---
sidebar_position: 2
title: How Steering Works
---

# How Steering Works

Steering is a direct intervention on the model's residual stream. This page explains exactly what miLLM does when steering is enabled, how to calibrate strengths, and how to verify the intervention is really happening.

## The formula

For every steered feature *i* with strength *sᵢ*, miLLM computes a single **steering delta** in residual-stream space:

```
delta = Σᵢ  sᵢ × W_dec[i]
```

where `W_dec[i]` is feature *i*'s decoder direction (a `d_in`-dimensional vector). During every forward pass, a hook on the attached layer adds this delta to the layer's output:

```
hidden_states ← hidden_states + delta
```

Key properties:

- **All token positions** receive the delta — prompt tokens and generated tokens alike. This matches miStudio and Neuronpedia semantics, so strengths transfer between tools.
- **No reconstruction.** miLLM does *not* run the full SAE encode→decode during steering; it adds the raw decoder directions. Model behavior outside the steered directions is untouched.
- **Accumulated once.** Multiple features fold into one delta vector, so steering 1 feature and 100 features cost the same at inference time (the delta is rebuilt only when values change).
- The delta rides through all **downstream layers** — steering layer 12 influences layers 13+, which is why mid-network layers are the usual choice.

## Strength calibration

Strengths are raw coefficients on unit-scale decoder directions, Neuronpedia-compatible. The API accepts **−200 to +200**.

| Strength | Typical effect |
|----------|---------------|
| 0 | No intervention |
| 1 – 10 | Subtle: mild topical drift, detectable in A/B comparison |
| 10 – 50 | Moderate: concept clearly surfaces in output |
| 50 – 100 | Strong: output is dominated by the concept |
| 100 – 200 | Extreme: often degrades into repetition or incoherence |
| Negative | Suppression: pushes the concept *out* of the stream |

:::warning Start low
Begin at 10–30 and increase. The effective scale varies by SAE, layer, and feature — some features saturate at 40, others need 120. Past the coherence cliff the model produces loops and word salad, which proves nothing except that you broke the forward pass.
:::

**Negative steering** subtracts the direction. It suppresses a concept the prompt would otherwise evoke, but the effect is usually weaker and noisier than amplification — a feature that never activates can't be pushed much further down.

## Scope of the intervention

When steering is enabled it applies to **every generation request** — chat completions, text completions, streaming, from every client. Two deliberate exceptions:

- **`/v1/embeddings` is never steered.** Embedding vectors are computed with the steering hook suppressed so they reflect the unmodified model.
- **Per-request profiles** (`profile` parameter on chat completions) temporarily replace the global steering for the duration of that one request, then restore it. See [Profiles](/features/profiles).

## Verifying steering is active

Interventions you can't verify are experiments you can't trust. miLLM exposes three checks:

1. **`steering_apply_count`** — returned by `GET /api/saes/attachment`. This counter increments every forward pass in which the delta was actually applied. If you generate tokens while steering is enabled and the counter doesn't move, the hook isn't firing — stop and check the [troubleshooting guide](/troubleshooting).
2. **`layer_module_path`** — returned by the attach call (e.g. `"model.layers.12"`). Confirms the hook landed on the module you intended, which matters on exotic architectures.
3. **A/B comparison** — same prompt, temperature 0, steering on vs. off. Deterministic decoding makes the diff attributable to the intervention alone.

## Lifecycle & interactions

| Event | Effect on steering |
|-------|--------------------|
| Set/update a feature (`POST /api/saes/steering`) | Delta rebuilt; steering auto-enabled |
| Disable (`POST /api/saes/steering/disable`) | Values kept, delta not applied |
| Clear all (`DELETE /api/saes/steering`) | Values removed **and steering disabled** |
| Remove last single feature | Steering auto-disabled (no empty-delta "enabled" state) |
| Detach SAE | Steering values cleared — re-attaching starts clean |
| Activate a profile | Replaces current values with the profile's (an empty-steering profile clears steering) |

Steering changes take effect on the **next forward pass**. A request already generating mid-change will see the new delta for its remaining tokens — for clean experiments, change steering between requests, not during them.

Steering composes with the performance stack: it works under `torch.compile`, with speculative decoding (the draft model proposes unsteered, but every accepted token is verified by the steered main model, so output correctness is preserved), and with continuous batching (steering is global, so all batched requests share it — which is also why per-request profiles route to the serial path). Details in [Architecture](/concepts/architecture).
