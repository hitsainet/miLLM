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

Quantization is applied **when the model is loaded**, not when it is downloaded. The download stores the repository's checkpoint as published, so a `Q4` download takes as much disk as `FP16`. While loading, bitsandbytes quantizes each weight on the card it is placed on, so the GPU never holds a full-precision copy of the model.

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

miLLM checks each card before loading and refuses a load that does not fit (see below), and refuses an SAE whose card cannot also keep the model's KV cache. If a request still runs out of memory during generation, that request fails with a typed error naming the card, its memory is released, and later requests are served; nothing is detached or disabled.

## Multi-GPU

A model goes on **one card whenever one card holds it**: with **Auto**, the card with the most free memory. The free memory is read live, and nothing is held back for other applications on the node.

A model that no single card can hold is **split across GPUs**. The cards with the most free memory are taken first, and only as many as the model and its KV cache need. Each card's share of the weights is its free memory less its CUDA context and the KV cache of the layers it holds, so the layers are not packed into the room that card needs for its context. Splitting costs a copy between cards at every layer boundary, and a split model is not compiled, so a model that fits one card is never split unless you ask.

### How miLLM decides a transformers model fits

A transformers model is judged **card by card**. From the checkpoint's configuration, without reading any weights, miLLM works out how much memory the weights take once loaded and, for a split, which layers transformers will put on each card. Each card the model uses must then have room for:

- the weights on that card;
- the **KV cache** of that card's layers at a minimum context, [`TRANSFORMERS_MIN_CONTEXT`](/reference/configuration#transformers-models) (4,096 tokens by default), or at the model's own maximum context if that is shorter, counted at bfloat16;
- a request's **working memory** on that card. This is the most memory a request of that length holds at once while the card's part of the model runs: the activations of its prefill, beside the KV cache. miLLM measures it by running the model's own forward pass on the meta device, which allocates nothing, and following every tensor it creates. It adds room for what PyTorch's memory allocator keeps reserved and cannot hand back, 40% of that peak and of the card's KV cache. For OLMo-2-13B at 4,096 tokens this is about 1,100 to 1,600 MiB a card, and for Qwen2.5-7B on one card about 870 MiB;
- for a bitsandbytes Q8 or Q4 load, what quantizing the weights as they load leaves stranded on the card: 7% of the bfloat16 size of the weights quantized there;
- a **CUDA context**, [`TRANSFORMERS_CUDA_CONTEXT_MB`](/reference/configuration#transformers-models) (500 MB by default; 250 to 400 MiB measured on the node).

A model whose forward pass cannot run on the meta device, such as one with FP8 kernels, a mixture of experts or custom remote code, gets a working-memory estimate instead. The estimate sits above every model that was measured, and miLLM logs an error naming the architecture when it uses one (`transformers_fit_working_memory_estimated`).

A sliding-window layer is counted up to its window. In a hybrid model only the attention layers are counted; the fixed-size state of a Mamba or convolution layer is not. The minimum context is a floor for loading, not a limit on requests: a card with more room serves longer contexts. Requests are limited only by the model's own maximum context, so a request whose KV cache needs more than its cards have left runs out of GPU memory during generation. That request is refused with `503 insufficient_memory` (type `invalid_request_error`) on `/v1`, or `507 INSUFFICIENT_MEMORY` on the management API, naming the card torch ran out on, the prompt's tokens and `max_tokens`; a streamed response ends with the same error event and `[DONE]`. The failed request's memory is released and later requests are served. Raise `TRANSFORMERS_MIN_CONTEXT` to load only where the contexts you serve fit.

The same test decides everything: whether Auto puts the model on one card or splits it, whether a named card is accepted, and whether a split is accepted. When a split leaves one card short while another has room, miLLM moves layers off the short card and works out the layout again, until every card fits or no arrangement can. A refusal lists every card the model would use, with its free memory, the weights, the KV cache and the context allowance, and how many MiB the short card is missing. For example, with an RTX 3080 Ti at 11,500 MB free and an RTX 3090 at 23,500 MB free:

| Model (BF16) | At 4,096 tokens | At 8,192 tokens |
|---|---|---|
| Qwen2.5-14B | Split; 1,399 and 3,660 MiB to spare | Split; 1,159 and 3,132 MiB to spare |
| OLMo-2-13B | Split; 429 and 4,208 MiB to spare | The same: the model serves at most 4,096 tokens |
| Vicuna-13B | Split, one layer moved off the RTX 3080 Ti; 412 and 5,562 MiB to spare | The same: the model serves at most 4,096 tokens |

Qwen2.5-14B at 32,768 tokens is refused on those cards: its weights, its cache and two CUDA contexts need 35,317 MiB of the 35,000 free, however the layers are divided.

:::note The CUDA context and the memory a request leaves behind
The 500 MB CUDA context allowance was measured on the node on 2026-09-14. A context holds 250 to 256 MiB idle, about 330 MiB after a first generation, and grows to about 400 MiB over a session, on both the RTX 3080 Ti and the RTX 3090. A request's prefill activations and the allocator's reserve are counted separately, as its working memory.

When a request finishes, PyTorch keeps the memory it freed, and nvidia-smi shows that memory as used. Once a transformers model's request queue has been idle for [`TRANSFORMERS_IDLE_CACHE_RELEASE_S`](/reference/configuration#transformers-models) seconds (5 by default), miLLM returns it to the cards, so miStudio and other tenants of the node can place work there. This never happens while a request is running, and never while continuous batching is on, because its requests do not pass through the request queue.
:::

A model whose KV cache miLLM cannot work out from its configuration, such as one with DeepSeek's multi-head latent attention, or an encoder-decoder, is judged the older way instead: its weight estimate plus 20%, against each card's free memory. miLLM logs an error naming the architecture when it does this.

**A transformers model never runs from CPU memory or disk.** This covers FP16/BF16 and bitsandbytes Q8/Q4. If the cards together cannot hold it, the load is refused before anything is unloaded, and the refusal gives the figures for each card. A split is also checked layer by layer before anything is unloaded: miLLM works out where each layer of the model would go from the checkpoint's configuration, without reading any weights, and refuses a split that would put any layer on the CPU or disk. The refusal lists how much would land on each device. After loading, miLLM checks again where the weights actually landed.

A GGUF model runs on the CPU only when no card has room for it, and then it runs there entirely. A GGUF model too big for all the cards together is still loaded with every layer on the GPUs, at the largest context that fits. If its weights alone do not fit, the load fails. miLLM does not yet offload part of a GGUF model to the CPU.

Choose **All GPUs (split)** in the GPU selector, or send `"gpu": "all"`, to split a model across every card even when one card could hold it. This is useful for checking that a split model generates what the single-card model does. Like a named card, "all" is honoured or refused, never swapped. A split places whole layers, so a small model can come out on one card: a card's share may be too small for any layer, or the first card may hold all of it. miLLM works out the layout before anything is unloaded, and refuses such a request (`409 SPLIT_NOT_HONOURED`) with the amount each card would get, rather than loading on fewer cards than you asked for. It is refused the same way when a card is too full to take any share, for example while another job holds most of it. Occasionally the layout cannot be worked out in advance, such as for a model class miLLM cannot build without reading its weights. The load then checks where the model actually landed and refuses at that point, and by then the previous model has already been unloaded.

A GGUF model that needs more than one card uses a llama.cpp layer split over the cards the plan chose, and the other cards get none of it. By default, layers are divided in proportion to each card's free memory. [`GGUF_TENSOR_SPLIT`](/reference/configuration#gguf-models) sets the proportions yourself.

SAEs attach on the device that hosts their layer, and steering and monitoring work unchanged on a split model. An SAE is refused (`507 INSUFFICIENT_MEMORY`) when its card cannot hold it and still keep room for the KV cache the model was loaded with; the refusal names the card, the SAE's size, the cache it keeps and what is free there. Continuous batching (`ENABLE_CONTINUOUS_BATCHING`) is not started for a split model: its paged KV cache lives on one card, so requests are served one at a time instead.
