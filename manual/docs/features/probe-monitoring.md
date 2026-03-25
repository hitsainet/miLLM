---
sidebar_position: 4
title: Probe Monitoring
---

# Probe Monitoring

The Probe page provides real-time visibility into which SAE features activate during inference — without modifying the model's output.

## How It Works

When monitoring is enabled, miLLM:
1. Captures the residual stream activations at the SAE's hooked layer
2. Encodes them through the SAE to get feature activations
3. Records the **top-K** most active features per forward pass
4. Emits results via WebSocket for real-time display

## Controls

| Control | Description |
|---------|-------------|
| **Enable/Disable** | Toggle monitoring on or off |
| **Pause/Resume** | Temporarily freeze the display while keeping monitoring active |
| **Top-K** | Number of top features to track: 5, 10, 20, 50, or 100 |

## Live Activations Chart

A bar chart showing the most recently activated features:
- **X-axis:** Feature index
- **Y-axis:** Activation magnitude
- Updates in real-time with each inference request

## Statistics Panel

Aggregated statistics for each monitored feature:
- **Count:** Number of times the feature appeared in top-K
- **Mean, Min, Max, Std:** Activation value statistics

## Activation History

A table of recent activation records showing:
- **Timestamp** of the inference request
- **Request ID** for correlation
- **Top features** that activated
- **Token position** in the sequence

Use **Clear History** to reset the buffer.

:::tip Workflow
Enable monitoring, then send inference requests via the OpenAI API or Open WebUI. Watch which features light up for different prompts — this helps identify which features to steer.
:::
