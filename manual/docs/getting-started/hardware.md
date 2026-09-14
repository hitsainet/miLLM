---
sidebar_position: 4
title: Hardware Requirements
---

# Hardware Requirements

miLLM runs the model, the SAE, and (optionally) a speculative-decoding draft model on a single GPU. This page helps you size that GPU.

## Minimum & Recommended

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA, 8 GB VRAM, CUDA 12.x | 16–24 GB VRAM (RTX 4090 / A5000 / L4 class) |
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32 GB (model weights pass through host RAM during load) |
| Disk | 30 GB free | 100 GB+ SSD (each model is 5–20 GB; SAEs 300 MB–2 GB each) |

CPU-only operation works for API smoke tests but is impractically slow for generation.

## VRAM Budget

Total VRAM ≈ **model weights + KV cache + SAE + overhead (~1 GB)**.

### Model weights by quantization

| Model | FP16 | Q8 (int8) | Q4 (int4) |
|-------|------|-----------|-----------|
| Gemma 2 2B | ~5.5 GB | ~3 GB | ~2 GB |
| Llama 3.1 8B / Gemma 2 9B | ~16–18 GB | ~9 GB | ~5.5 GB |
| Gemma 2 27B | ~54 GB | ~28 GB | ~15 GB |

Quantization is chosen **at download time** — miLLM saves the quantized weights to disk, so a Q4 download loads directly without a full-precision intermediate.

The table above is **bitsandbytes** quantization of a HuggingFace checkpoint. **GGUF** files are quantized ahead of time by whoever published them, so the figure that matters is the file size on the Hub — an `IQ4_XS` build of a 31B model is ~16 GB, a `Q5_K_M` ~22 GB. miLLM shows the size of each quantization before you download it, and several quantizations of one repository can coexist; see [Model Management](/features/model-management#gguf-models-and-quantization).

:::warning Quantization vs torch.compile
bitsandbytes-quantized models (Q4/Q8) are incompatible with `torch.compile`; miLLM detects this and disables compilation automatically. FP16 models get compiled decoding (faster tokens/sec) by default on CUDA. See [Configuration](/reference/configuration).
:::

### SAE memory

An SAE's footprint is roughly `2 × d_in × d_sae × 2 bytes` (encoder + decoder in bf16):

| SAE width | For Gemma 2 2B (d_in = 2304) | For 9B (d_in = 3584) |
|-----------|------------------------------|----------------------|
| 16k features | ~300 MB | ~470 MB |
| 65k features | ~1.2 GB | ~1.9 GB |
| 131k features | ~2.4 GB | ~3.8 GB |

The Admin UI shows the measured footprint after attach (`memory_usage_mb` in the attachment status).

### KV cache

Grows with context length and concurrent requests. For a 2B model at 4k context, budget ~1 GB; larger models and longer contexts scale roughly linearly.

For a **GGUF** model the cache is allocated in full when the context is created, and its size per token is

```
2 × n_layer × n_head_kv × head_dim × bytes_per_element
```

which is worth doing once for a large model. `gemma-4-31b` at F16 spends **630 KB per token** (60 layers × 16 KV heads × 168 head_dim × 2 tensors × 2 bytes). At 4096 tokens that is 2.6 GB — more than a third of the free VRAM on a 24 GB card — to hold roughly three thousand words.

**Quantizing the cache is the single biggest lever on how much context fits.** Measured on gemma-4-31b IQ4_XS, RTX 3090:

| KV cache | Flash attention | Largest context that loads |
|---|---|---|
| `f16` | on | 4,096 |
| `q8_0` | on | **12,288** |
| `q4_0` | on | 16,384 |

miLLM defaults to `q8_0`, which is near-lossless and triples the window. `q4_0` buys another third at a real accuracy cost — the wrong trade when the model is acting as a judge, where discrimination is the job.

:::warning Quantized KV cache requires flash attention
This is a dependency, not an optimisation: `q8_0` **without** flash attention fails at 8192 where `q8_0` with it reaches 12288. The loader refuses to pair a quantized cache with `GGUF_FLASH_ATTENTION=false` rather than loading something that will fail later.
:::

### GGUF context windows are derived, not guessed

miLLM does not pick a context length for a GGUF model and hope. It reads what the file itself declares (`n_ctx_train`, via a ~1.2-second CPU probe that costs no VRAM), predicts what free VRAM can hold using the arithmetic above, and starts there — bounded above by [`GGUF_CONTEXT_LENGTH`](/reference/configuration#gguf-models).

That ceiling is a **limit, not a target**, and it is needed in both directions. llama.cpp's own default is 512 tokens, which truncates almost any real conversation. Unbounded is the opposite trap: `ByteOtter/Qwen3.8-27B-TAK-Reasoning-GGUF` declares 262,144 tokens, and a quarter-million-token window on a 31B model is tens of gigabytes of cache — on a 24 GB card the load simply fails after ~19.9 GB of weights, and even when it succeeds it reserves a GPU that extraction, training and steering work shares.

## Worked Examples

| Setup | VRAM needed | Fits on |
|-------|-------------|---------|
| Gemma 2 2B FP16 + 16k SAE | ~7.5 GB | 8 GB card (tight), 12 GB comfortably |
| Gemma 2 2B FP16 + 65k SAE + monitoring | ~9 GB | 12 GB card |
| Gemma 2 9B Q8 + 16k SAE | ~11 GB | 16 GB card |
| Gemma 2 9B FP16 + 131k SAE | ~23 GB | 24 GB card |

miLLM estimates memory before loading and warns (but does not block) when the estimate exceeds free VRAM. On an out-of-memory event during inference with an SAE attached, miLLM degrades gracefully: the SAE is disabled and the base model continues serving.

## Multi-GPU

A model goes on **one card whenever one card holds it**: with **Auto**, the card with the most free memory. The free memory is read live, and nothing is held back for other applications on the node.

A model that no single card can hold is **split across GPUs**. The cards with the most free memory are taken first, and only as many as the model needs. Each card of a split keeps 1 GB back for its CUDA context and for the activations of the layers it runs. Splitting costs a copy between cards at every layer boundary, and a split model is not compiled, so a model that fits one card is never split unless you ask.

**A transformers model never runs from CPU memory or disk.** This covers FP16/BF16 and bitsandbytes Q8/Q4. If the cards together cannot hold it, the load is refused before anything is unloaded, and the refusal gives the figures for each card. miLLM also checks where the weights actually landed, and refuses the load if any part ended up on the CPU or disk. Only GGUF models may run partly on the CPU.

Choose **All GPUs (split)** in the GPU selector, or send `"gpu": "all"`, to split a model across every card even when one card could hold it. This is useful for checking that a split model generates what the single-card model does. Like a named card, "all" is honoured or refused, never swapped.

A GGUF model that needs more than one card uses a llama.cpp layer split over the cards the plan chose, and the other cards get none of it. By default, layers are divided in proportion to each card's free memory. [`GGUF_TENSOR_SPLIT`](/reference/configuration#gguf-models) sets the proportions yourself.

SAEs attach on the device that hosts their layer, and steering and monitoring work unchanged on a split model.
