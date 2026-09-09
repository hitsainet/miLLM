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

## GGUF models and quantization

miLLM serves **GGUF** files the way Ollama does, alongside HuggingFace checkpoints. A GGUF repository usually publishes several quantizations of the same weights, and **Preview** lists each one with its file size so you can pick before downloading.

The distinction that matters against the table above: bitsandbytes quantization happens *at download time* from a full-precision checkpoint, while a GGUF file was quantized ahead of time by whoever published it. What you choose is a file, not a mode — `IQ4_XS`, `Q4_K_M`, `Q5_K_M` and so on.

### Several quantizations of one repository can coexist

They are different models with different sizes and different quality, so miLLM keeps them apart by naming a GGUF model **`repo:QUANT`** — the convention Ollama already established:

```
gemma-4-31b-it-3MPER0RR-abliterated-GGUF:IQ4_XS
gemma-4-31b-it-3MPER0RR-abliterated-GGUF:Q4_K_M
```

Both appear separately in `/v1/models` instead of one shadowing the other. A **custom name** you supply is left alone — that is your choice, not a derived one.

A **bare** repository name still resolves while exactly one quantization of it exists, so existing scripts and saved client selections keep working. Once a second is downloaded the bare name becomes ambiguous, and miLLM returns a **400** naming the tags that exist rather than picking one:

```json
{ "error": { "code": "AMBIGUOUS_MODEL_NAME",
             "message": "'…-GGUF' matches 2 quantizations: …:IQ4_XS, …:Q4_K_M. Name one of them exactly." } }
```

Choosing silently would make the served model depend on which was downloaded first, with nothing on the wire to say which answered.

### Context window

A GGUF model's context is **derived, not configured**: miLLM reads what the file declares, predicts what free VRAM can hold, and loads at the largest window that fits under the [`GGUF_CONTEXT_LENGTH`](/reference/configuration#gguf-models) ceiling. See [Hardware Requirements](/getting-started/hardware#gguf-context-windows-are-derived-not-guessed) for the arithmetic and the KV-cache lever that triples it.

### Embeddings

GGUF models are loaded with embedding output enabled so `/v1/embeddings` works without a second load, at a measured ~6.7% generation cost (113.5 → 105.9 tok/s on `zora-v1.13 Q5_K_M`). Set [`GGUF_ENABLE_EMBEDDINGS=false`](/reference/configuration#gguf-models) to buy that back on a deployment that never embeds.

Some architectures **refuse** the pooling mode embeddings require, and fail context creation at every length rather than reporting anything useful. When that happens miLLM retries the load without embeddings and records `supports_embeddings: false` on the model, so a caller is told up front instead of discovering it from a confusing failure at `/v1/embeddings`. Serving the model matters more than embedding it.

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
