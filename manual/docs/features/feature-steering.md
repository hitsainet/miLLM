---
sidebar_position: 3
title: Feature Steering
---

# Feature Steering

Steering manipulates model behavior by adding scaled feature directions to the residual stream during inference. This is the primary tool for proving that an SAE feature causally influences model output.

## Prerequisites

Both a **model** and an **SAE** must be loaded and attached before steering is available.

## Adding Features

### Single Feature
Enter a feature index (0 to d_sae-1) and click **Add**.

### Batch Add
Click **Batch Add** and enter multiple features:
- One per line or comma-separated
- Optionally specify strength: `1234:2.5`
- Features without strength use the default (1.0)

## Strength Values

Strengths are **raw coefficients** added to the residual stream, compatible with Neuronpedia's scale:

| Range | Effect |
|-------|--------|
| **0** | No intervention |
| **0.1 – 5** | Subtle influence |
| **5 – 50** | Moderate effect |
| **50 – 100** | Strong effect |
| **100 – 300** | Very strong / extreme |
| **Negative** | Suppression (inhibits the feature) |

The input field accepts values from **-300 to +300** with **0.1 step** precision via arrow keys.

:::warning Strength Calibration
Start with low values (5–20) and increase gradually. Values above ±100 frequently cause repetitive or incoherent output. The effective range depends on the specific SAE and layer.
:::

## How Steering Works

For each steered feature, miLLM:
1. Gets the **decoder direction** (column from SAE's decoder weight matrix)
2. Computes `steering_delta = strength × decoder_direction`
3. Adds the delta to **all token positions** in the residual stream
4. The modification happens in-place during the forward pass

Multiple features are accumulated into a single delta vector before application.

## Enable/Disable

The steering toggle enables or disables all configured features at once. Feature configurations are preserved when disabled — you can toggle steering on and off without reconfiguring.

## Save as Profile

Click **Save as Profile** to store the current feature configuration for later use. See [Profiles](/features/profiles) for details.
