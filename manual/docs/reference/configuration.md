---
sidebar_position: 1
title: Configuration
---

# Configuration Reference

miLLM is configured entirely through environment variables (with `.env` file support, case-sensitive names). In Docker Compose these go in the `api` service's `environment:` block; in Kubernetes, in the deployment's `env:`.

## Core

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/millm` | PostgreSQL DSN (async driver required) |
| `MODEL_CACHE_DIR` | `/app/model_cache` | Where downloaded model weights live |
| `SAE_CACHE_DIR` | `/app/sae_cache` | Where downloaded SAEs live |
| `HF_TOKEN` | — | Server-wide HuggingFace token for gated repos; per-request tokens in API calls take precedence and are never persisted |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Bind address and port |
| `DEBUG` | `false` | Enables debug behavior; never use in production |
| `REDIS_URL` | — | Optional Redis for cache/real-time state |
| `AUTO_LOAD_MODEL` | — | Model ID or name to load automatically at startup — recommended for unattended deployments |

## HTTP & logging

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins. Set explicitly (e.g. `http://localhost:3000,https://webui.example.com`) when browsers call the API cross-origin |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `LOG_FORMAT` | `console` | `console` for humans, `json` for log pipelines |

## Concurrency & timeouts

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONCURRENT_REQUESTS` | `1` | **Must stay `1`.** This is a correctness constraint, not a throughput knob: values above `1` race on steering apply/restore and on the shared sensing buffer. Leave it at `1` |
| `MAX_PENDING_REQUESTS` | `10` | Queue depth beyond the concurrent slots; overflow returns `503` (`QUEUE_FULL`, backpressure) |
| `MAX_DOWNLOAD_WORKERS` | `2` | Parallel model/SAE downloads |
| `MAX_LOAD_WORKERS` | `1` | Parallel model loads (keep at 1) |
| `GRACEFUL_UNLOAD_TIMEOUT` | `30.0` | Seconds to wait for in-flight requests before unloading a model |
| `DOWNLOAD_TIMEOUT` | `3600.0` | Max seconds for a single download |

## Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `TORCH_COMPILE` | *(auto)* | `true`/`false`/unset. Unset = auto: compile on CUDA for non-bitsandbytes models. Compilation speeds up decoding significantly; first request after model load (and after SAE attach/detach) pays a one-time recompile (~20 s). The SAE hook is fully honored under compilation — see [Architecture](/concepts/architecture#torchcompile-and-the-hook) |
| `TORCH_COMPILE_MODE` | `reduce-overhead` | `default` / `reduce-overhead` / `max-autotune` |
| `KV_CACHE_MODE` | `dynamic` | `static` enables compiled static KV cache (needs a C compiler for triton in the image) |
| `SPECULATIVE_MODEL` | — | HF model ID of a small draft model for speculative decoding. Works with steering: draft proposes, the steered main model verifies — output correctness preserved, acceptance rate lower |
| `SPECULATIVE_NUM_TOKENS` | `5` | Tokens the draft proposes per step |

## Continuous batching (opt-in)

High-throughput batched inference via HuggingFace `ContinuousBatchingManager`. Trade-offs in [Architecture](/concepts/architecture#continuous-batching-opt-in).

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_CONTINUOUS_BATCHING` | `false` | Start CBM at model load |
| `CBM_MAX_QUEUE_SIZE` | `256` | Manager queue size |
| `CBM_DEFAULT_TEMPERATURE` | `0.7` | Fixed sampling temperature for batched requests; requests with a different value fall back to serial |
| `CBM_DEFAULT_TOP_P` | `0.95` | Fixed top-p, same fallback rule |
| `CBM_DEFAULT_MAX_TOKENS` | `512` | Default generation length |
| `CBM_FORCE_SERIAL_MONITORING` | `false` | Route monitored requests to the serial path for exact per-request activation attribution (trades throughput for fidelity) |

## GGUF models

Settings that apply only when the loaded model is a GGUF file. See [Hardware Requirements](/getting-started/hardware#kv-cache) for the measurements behind the defaults.

| Variable | Default | Notes |
|---|---|---|
| `GGUF_CONTEXT_LENGTH` | `32768` | **A ceiling, not a target.** miLLM reads what the file declares and loads the largest window that fits under this. `0` means "whatever the file declares", bounded only by what fits. |
| `GGUF_KV_CACHE_TYPE` | `q8_0` | KV-cache precision — the biggest lever on how much context fits. `q8_0` is near-lossless and roughly triples the window over `f16`. `q4_0` buys another third at a real accuracy cost. `f16` restores llama.cpp's default behaviour. |
| `GGUF_FLASH_ATTENTION` | `true` | **Required** by a quantized KV cache, not merely an optimisation. The loader refuses to pair the two with this off. |
| `GGUF_ENABLE_EMBEDDINGS` | `true` | Loads with embedding output so `/v1/embeddings` works without a second load. Costs ~6.7% generation throughput, measured. Must be decided at load time — it cannot be switched on later. |

:::tip Why the ceiling is needed in both directions
llama.cpp's own default is 512 tokens, which truncates almost any real conversation. Unbounded is the opposite trap: a model declaring 262,144 tokens needs tens of gigabytes of KV cache, which either fails the load outright or reserves a GPU that other work shares.
:::

## Example configurations

```bash title=".env — single-GPU research box"
AUTO_LOAD_MODEL=gemma-2-2b
MAX_CONCURRENT_REQUESTS=1
CORS_ORIGINS=http://localhost:3000
LOG_FORMAT=console
```

```bash title=".env — shared inference server"
ENABLE_CONTINUOUS_BATCHING=true
CBM_FORCE_SERIAL_MONITORING=true
MAX_PENDING_REQUESTS=32
LOG_FORMAT=json
LOG_LEVEL=INFO
```
