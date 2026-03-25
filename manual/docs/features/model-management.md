---
sidebar_position: 1
title: Model Management
---

# Model Management

## Loading a Model

1. Navigate to **Models** in the sidebar
2. Enter a HuggingFace repository ID (e.g., `google/gemma-2-2b-it`)
3. Select **Quantization**:

| Mode | Bits | VRAM Savings | Quality | Best For |
|------|------|-------------|---------|----------|
| **FP16** | 16 | Baseline | Maximum | Precision research |
| **Q8** | 8 | ~50% | Minimal loss | Good balance |
| **Q4** | 4 | ~75% | Moderate loss | Consumer GPUs (recommended) |
| **Q2** | 2 | ~87% | Significant loss | Maximum compression |

4. Select **Device** (`auto` recommended — places model on GPU with CPU offload if needed)
5. Optionally enter a **HuggingFace Token** for gated models (e.g., Llama)
6. Check **Trust Remote Code** if required by the model
7. Click **Download &amp; Load Model**

:::warning Hybrid Models (Mamba/SSM)
Models with Mamba/SSM layers (e.g., `granite-4.0-h-*`) require the `mamba-ssm` package for efficient inference. Without it, the naive fallback creates massive intermediate tensors that cause OOM errors. Check that `mamba-ssm` is installed in your deployment.
:::

## Model Locking

When an SAE is attached, the model is automatically **locked** — preventing accidental unloading during steering experiments. Unlock manually from the model details if needed.

## Downloaded Models

Previously downloaded models appear in a list below the load form. Click **Load** to switch to any ready model. The previous model is unloaded first to free GPU memory.

:::info Dynamic Architecture Support
miLLM uses dynamic layer discovery to support any transformer architecture — Llama, Gemma, GPT-2, LFM, Granite, Mistral, Phi, and more. No configuration needed.
:::
