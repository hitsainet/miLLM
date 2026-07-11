---
sidebar_position: 4
title: Probe Monitoring
---

# Probe Monitoring

The Probe page provides real-time visibility into which SAE features activate during inference — the observational counterpart to steering, with no effect on model output.

For capture semantics and internals, see [Concepts: How Monitoring Works](/concepts/monitoring).

## How It Works

When monitoring is enabled, miLLM:

1. Captures the residual-stream activations at the SAE's hooked layer
2. Encodes them through the SAE to get feature activations
3. Records the **top-K** most active features per completed request
4. Emits results via WebSocket (`monitoring:activation`) for real-time display

:::info What a record represents
Each activation record reflects the **final forward pass** of a generation — the last generated token's activations — not a token-by-token trace of the whole sequence. This tells you which features were active as the model finished responding.
:::

## Controls

| Control | Description |
|---------|-------------|
| **Enable/Disable** | Toggle monitoring on or off (configuration is preserved when off) |
| **Pause/Resume** | Freeze the display while monitoring continues in the background |
| **Top-K** | Number of top features to track: 5, 10, 20, 50, or 100 |
| **Watched features** | Optional subset of feature indices to monitor exclusively — cheaper, and focused on your hypothesis |

Watched-feature indices are validated against the attached SAE's `d_sae`; invalid indices are rejected up front.

## Live Activations Chart

A bar chart of the most recently activated features — feature index on the X-axis, activation magnitude on the Y-axis — updating with each inference request. Each bar links to the feature's **Neuronpedia page** so you can immediately see what a firing feature means.

## Latest Activations & History

The Latest Activations panel and history table show recent records:

- **Timestamp** and **Request ID** for correlating with API calls
- **Top features** with activation values
- Per-feature links to Neuronpedia

History is a ring buffer (default 100 entries, configurable). **Clear History** resets it; statistics can be reset independently.

## Statistics Panel

Running statistics per feature, accumulated across requests until reset:

| Metric | Meaning |
|--------|---------|
| **Count** | Observations recorded for the feature |
| **Mean / Std** | Distribution of activation values |
| **Min / Max** | Observed range |
| **Active ratio** | Fraction of observations with activation > 0 |

The **top features** view ranks by any of these metrics — a quick answer to "what does my workload actually activate?"

## Suggested Workflows

:::tip Discover → Steer
Enable monitoring, send prompts about your topic of interest via the OpenAI API or Open WebUI, and note which features consistently top the chart. Those indices are your steering candidates — verify their meaning on Neuronpedia, then dial them up on the [Steering page](/features/feature-steering).
:::

:::tip Steer → Observe
The reverse loop: steer feature A and watch whether related features B and C shift in the statistics. Monitoring reads the stream *before* the steering delta is added at the hooked layer, so at that layer you observe the model's natural response to the steered context flowing in from below.
:::

## API

Everything here is available at [`/api/monitoring`](/api/monitoring): configure, toggle, history (filterable by request ID), statistics, and top-features ranking.
