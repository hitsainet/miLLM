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

## Worked Examples

| Setup | VRAM needed | Fits on |
|-------|-------------|---------|
| Gemma 2 2B FP16 + 16k SAE | ~7.5 GB | 8 GB card (tight), 12 GB comfortably |
| Gemma 2 2B FP16 + 65k SAE + monitoring | ~9 GB | 12 GB card |
| Gemma 2 9B Q8 + 16k SAE | ~11 GB | 16 GB card |
| Gemma 2 9B FP16 + 131k SAE | ~23 GB | 24 GB card |

miLLM estimates memory before loading and warns (but does not block) when the estimate exceeds free VRAM. On an out-of-memory event during inference with an SAE attached, miLLM degrades gracefully: the SAE is disabled and the base model continues serving.

## Multi-GPU

Models load with `device_map="auto"`, so a model larger than one GPU spreads across available GPUs automatically. The SAE attaches to a single layer and lives on the device that hosts that layer. Steering and monitoring work unchanged.
