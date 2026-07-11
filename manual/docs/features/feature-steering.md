---
sidebar_position: 3
title: Feature Steering
---

# Feature Steering

Steering manipulates model behavior by adding scaled feature directions to the residual stream during inference. This is the primary tool for proving that an SAE feature **causally influences** model output.

For the underlying math and calibration guidance, see [Concepts: How Steering Works](/concepts/steering). For a complete worked example, see the [Steering Gemma tutorial](/tutorials/steering-gemma).

## Prerequisites

Both a **model** and an **SAE** must be loaded and attached before steering is available. The Steering page shows a waiting state until then.

## Adding Features

### Single Feature

Enter a feature index (`0` to `d_sae−1`) and click **Add**. Indices are validated against the attached SAE — out-of-range indices are rejected with a clear error rather than accepted silently.

### Batch Add

Click **Batch Add** and enter multiple features:

- One per line or comma-separated
- Optionally specify strength: `1234:2.5`
- Features without a strength use the default (1.0)
- Up to 1000 features per batch

### Finding feature indices

Feature indices come from exploration: browse [Neuronpedia](https://neuronpedia.org) for your model/SAE combination, or use [Probe Monitoring](/features/probe-monitoring) to see which features fire on your own prompts.

## Strength Values

Strengths are **raw coefficients** added to the residual stream, compatible with Neuronpedia's scale. The API accepts **−200 to +200**:

| Range | Effect |
|-------|--------|
| **0** | No intervention |
| **1 – 10** | Subtle influence, visible in A/B comparison |
| **10 – 50** | Moderate, concept clearly surfaces |
| **50 – 100** | Strong effect |
| **100 – 200** | Extreme — frequently repetitive or incoherent |
| **Negative** | Suppression (inhibits the concept) |

:::warning Strength Calibration
Start with low values (10–30) and increase gradually. The effective range depends on the specific SAE, layer, and feature. Values past the coherence cliff prove nothing except that you broke the forward pass.
:::

## How Steering Works (short version)

For each steered feature, miLLM:

1. Takes the **decoder direction** (a column of the SAE's decoder matrix)
2. Computes `delta = Σ strength × decoder_direction` across all steered features
3. Adds the delta to **all token positions** at the hooked layer's output, every forward pass

Multiple features accumulate into a single delta vector, so steering many features has no extra inference cost.

## Enable / Disable / Clear

| Action | Values | Applied? |
|--------|--------|----------|
| **Toggle off** | Preserved | No — flip back on without reconfiguring |
| **Remove one feature** | That feature removed | Remaining features still applied; removing the *last* feature disables steering |
| **Clear all** | Removed | Steering disabled |
| **Detach SAE** | Removed | — (re-attach starts clean) |

Steering state is global: when enabled, it applies to every completion request from every client (embeddings are the [deliberate exception](/concepts/steering#scope-of-the-intervention)).

## Verifying the Intervention

`GET /api/saes/attachment` returns `steering_apply_count` — the number of forward passes where the delta was actually applied. Generate some tokens with steering enabled and confirm the counter advanced; if it didn't, the hook isn't firing ([troubleshooting](/troubleshooting#steering-has-no-effect)). For rigorous experiments, A/B the same prompt at `temperature: 0` with steering on and off.

## Save as Profile

Click **Save as Profile** to store the current feature configuration for later use — including per-request application via the API. See [Profiles](/features/profiles).
