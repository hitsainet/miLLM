---
sidebar_position: 1
title: Model Management
---

# Model Management

Everything starts with a loaded model. miLLM downloads models from HuggingFace (or imports from a local path), optionally quantizes them at download time, and loads one model at a time onto the GPU.

![miLLM Models Page](/img/miLLM_Models_01.jpg)

## Downloading a Model

1. Navigate to **Models** in the sidebar
2. Enter a HuggingFace repository ID (e.g., `google/gemma-2-2b`) — or choose **Local Path** to import weights already on disk
3. Click **Preview** to see the model's size, architecture, and estimated memory per quantization before committing
4. Select **Quantization**:

| Mode | Bits | VRAM Savings | Quality | Best For |
|------|------|-------------|---------|----------|
| **FP16** | 16 | Baseline | Maximum | Precision research; enables `torch.compile` |
| **Q8** | 8 | ~50% | Minimal loss | Good balance |
| **Q4** | 4 | ~75% | Moderate loss | Consumer GPUs |
| **Q2** | 2 | ~87% | Significant loss | Maximum compression |

5. Optionally enter a **HuggingFace Token** for gated models (Gemma and Llama are gated — accept the license on the model page first)
6. Check **Trust Remote Code** only if the model requires custom code (explicit opt-in, per download)
7. Click **Download & Load Model**

Quantization happens **at download time** — miLLM saves the quantized weights to disk, so subsequent loads skip re-quantization. Download progress streams over WebSocket to the UI; downloads can be cancelled but not paused.

:::tip Choosing quantization for steering work
Prefer **FP16** for a model that fits: quantized (bitsandbytes) models cannot use `torch.compile`, so FP16 decodes faster on capable GPUs despite the extra memory. See [Hardware Requirements](/getting-started/hardware) for sizing tables.
:::

## Loading & Unloading

One model is resident on the GPU at a time. Clicking **Load** on another ready model unloads the current one first. Before loading, miLLM estimates the memory requirement and warns if it exceeds free VRAM.

Unloading is **graceful**: in-flight inference requests get up to `GRACEFUL_UNLOAD_TIMEOUT` (default 30 s) to complete before the model is released.

To load a model automatically at server startup, set `AUTO_LOAD_MODEL` — see [Configuration](/reference/configuration).

:::warning Hybrid Models (Mamba/SSM)
Models with Mamba/SSM layers (e.g., `granite-4.0-h-*`) require the `mamba-ssm` package for efficient inference. Without it, the naive fallback creates massive intermediate tensors that cause OOM errors. miLLM automatically selects the hybrid KV-cache these architectures need.
:::

## Model Locking

When an SAE is attached, the model is automatically **locked** — the unload and delete actions are refused (`409 MODEL_LOCKED`) so a steering experiment can't lose its substrate mid-run. Detaching the SAE unlocks the model automatically; you can also lock/unlock manually from the model details or via [`POST /api/models/{id}/lock`](/api/models).

## Deleting

Delete removes the model from disk and the registry (hard delete). A loaded or locked model must be unloaded/unlocked first.

:::info Dynamic Architecture Support
miLLM uses dynamic layer discovery to support any transformer architecture — Llama, Gemma, GPT-2, LFM, Granite, Mistral, Phi, and more. No configuration needed. When you attach an SAE, the attach response reports the exact module hooked (`layer_module_path`) so you can verify layer resolution on unusual architectures.
:::

## API

All of the above is scriptable — see the [Models API reference](/api/models). The model list, download, load/unload, lock/unlock, preview, and delete operations map 1:1 to endpoints.
