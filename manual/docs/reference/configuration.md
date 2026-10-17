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
| `EMBEDDINGS_MAX_INPUTS` | `256` | Most strings one `/v1/embeddings` request may carry (1–2048); more is refused with `400` before any model load. A capped request holds the only request slot, so this bounds how long a chat waits behind it. **Provisional:** to be set from latency measured on the RTX 3090 — the largest power of two with cap × p95 seconds per input ≤ 30 s |
| `MAX_DOWNLOAD_WORKERS` | `2` | Parallel model/SAE downloads |
| `MAX_LOAD_WORKERS` | `1` | Parallel model loads (keep at 1) |
| `GRACEFUL_UNLOAD_TIMEOUT` | `30.0` | Seconds an unload waits for the requests already running on the model before it moves any weight. Requests that arrive once the unload has begun are refused with `503 model_busy` rather than waited for. If the running requests outlast this, the unload goes ahead and they fail. (Until 2026-09-14 this setting was read by nothing and the wait was a fixed 5 s.) |
| `DOWNLOAD_TIMEOUT` | `3600.0` | Max seconds for a single download |

## Model lease and backpressure

| Variable | Default | Description |
|----------|---------|-------------|
| `LEASE_DEFAULT_TTL_SECONDS` | `7200` | TTL of a lease taken without `ttl_seconds`. Must not exceed the maximum (checked at startup) |
| `LEASE_MAX_TTL_SECONDS` | `7200` | Largest `ttl_seconds` accepted; larger values are refused with `400`, never clamped |
| `LEASE_HOLDER_MAX_CHARS` | `128` | Longest `holder` label, after stripping |
| `LEASE_REASON_MAX_CHARS` | `512` | Longest `reason`, after stripping |
| `LEASE_ENDED_MEMORY` | `64` | Ended leases remembered, so renew/release of a seen ID answers `409 LEASE_EXPIRED` with its reason instead of `404`. Lost at restart |
| `QUEUE_DURATION_WINDOW` | `50` | Recent slot-holding durations kept for `estimated_wait_seconds` |
| `RETRY_AFTER_QUEUE_DEFAULT_S` | `5` | `QUEUE_FULL` with no estimate |
| `RETRY_AFTER_MAX_S` | `60` | Ceiling on the `QUEUE_FULL` estimate |
| `RETRY_AFTER_LOAD_S` | `15` | `MODEL_BUSY` / `MODEL_LOADING` while a load runs |
| `RETRY_AFTER_UNLOAD_S` | `5` | `MODEL_BUSY` while an unload runs |
| `RETRY_AFTER_NOT_LOADED_S` | `30` | `MODEL_NOT_LOADED` |
| `RETRY_AFTER_MEMORY_S` | `30` | `INSUFFICIENT_MEMORY` on `/v1` |
| `RETRY_AFTER_READINESS_S` | `5` | The readiness probe's `503` |
| `RETRY_AFTER_FALLBACK_S` | `10` | Any other `503` that reached the response without `Retry-After`; logged as `retry_after_defaulted`. Deliberately distinct from every other value |
| `RETRY_AFTER_WEIGHTS_TORN_S` | `30` | `WEIGHTS_TORN`: the resident weights were left mid-write. Serving returns after a late `commit-end` or a weights load, not in seconds |

Leases live in process memory: a restart ends every lease. miLLM runs one worker process; several
workers would each hold their own lease registry.

## Attachable weights

Settings for weight attachment and the step barrier (Feature 31), checkpoints (Feature 32) and
adapter serving (Feature 33). With `WEIGHTS_TOKENS` empty the features are inert: every gated
`/api/weights` call is refused and serving is unaffected.

| Variable | Default | Description |
|----------|---------|-------------|
| `WEIGHTS_TOKENS` | empty | Bearer tokens for the gated `/api/weights` routes: a JSON array of `{"label", "role", "sha256"}`, where `role` is `trainer` or `operator`, labels are unique and `sha256` is the token's SHA-256 digest in hex (`printf %s "$TOKEN" \| sha256sum`). miLLM stores digests only; the plaintext stays in the trainer's own Secret. Empty refuses every gated call with `401 WEIGHTS_UNAUTHORIZED` (`reason: no_tokens_configured`). A malformed value does **not** stop the server: gated calls are refused with `reason: tokens_malformed`, the log names the faulty entry by index, and the value is never echoed. Serving an adapter with no trainer still needs an `operator` token. In Kubernetes it comes from the optional key `WEIGHTS_TOKENS` of the `millm-secrets` Secret (absent is fine: the pod starts and the gate refuses). Use long random tokens: a digest of a short token can be brute-forced from a leaked configuration. A model is loaded **attachable** (its parameters allocated so they can be shared with a trainer) only when at least one `trainer` token is configured at the moment it loads: add the token, then load or reload the model |
| `WEIGHTS_HOLD_TIMEOUT_S` | `5.0` | A quiesce the holder leaves open longer than this resumes serving on its own, only while the weights are `clean` (never out of a commit or a `torn` set). Suspended while a commit is open; restarted in full by commit-end |
| `WEIGHTS_COMMIT_WATCHDOG_MS` | `200` | A commit open longer than this marks the weights `torn`; a late commit-end from the same attachment still restores `clean` and is recorded as an overrun. Measured on the RTX 3090 (2026-10-16): ten times the 99.9th-percentile commit (15.9 ms) of 10,000 full-parameter commits, rounded up. Raise it for a trainer whose commits are slower |
| `WEIGHTS_HEARTBEAT_TIMEOUT_S` | `30.0` | An attachment with no heartbeat for this long is stale. Must be above the hint (checked at startup) |
| `WEIGHTS_HEARTBEAT_INTERVAL_HINT_S` | `5.0` | The heartbeat interval the attach response suggests |
| `WEIGHTS_QUIESCE_DRAIN_TIMEOUT_S` | `60.0` | How long a quiesce or a serving pause waits for in-flight requests to finish before it is refused with `MODEL_BUSY` (`reason: drain_timeout`) |
| `WEIGHTS_CHECKPOINT_DIR` | `/app/checkpoints` | The checkpoint root miLLM writes saves to and reads from. Kubernetes: `/data/checkpoints` on the `millm-data` volume (the init container creates it). Docker Compose: `/app/checkpoints` on the named volume `millm-checkpoints`. Adapters are loaded from under the checkpoint roots too |
| `WEIGHTS_CHECKPOINT_READ_ROOTS` | empty | Further checkpoint roots, **read only**, as comma-separated absolute paths (a relative entry fails settings validation at startup) (for example miForge's checkpoint volume mounted read-only). Loads and lineage resolve across them; saves never write there and temporary-file cleanup never runs there. A run present under two roots is refused as ambiguous. Kubernetes: `k8s/base/backend.yaml` carries a commented, disabled example that mounts miForge's checkpoint claim read-only at `/mnt/miforge-checkpoints`; fill in the claim name and uncomment the env entry, the mount and the volume together |
| `WEIGHTS_LINEAGE_MAX_DEPTH` | `10000` | A lineage walk stops after this many links and reports `depth_limit` |
| `WEIGHTS_SAVE_FREE_SPACE_FACTOR` | `1.10` | A save needs free disk of at least its estimated size times this (at least `1.0`) |
| `WEIGHTS_SAVE_OVERHEAD_MB` | `64` | MB added to a save's estimate for the configuration and tokenizer files |

## Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `TORCH_COMPILE` | *(auto)* | `true`/`false`/unset. Unset = auto: compile on CUDA for non-bitsandbytes models. Compilation speeds up decoding significantly; first request after model load (and after SAE attach/detach) pays a one-time recompile (~20 s). The SAE hook is fully honored under compilation — see [Architecture](/concepts/architecture#torchcompile-and-the-hook) |
| `TORCH_COMPILE_MODE` | `reduce-overhead` | `default` / `reduce-overhead` / `max-autotune` |
| `KV_CACHE_MODE` | `dynamic` | `static` enables compiled static KV cache (needs a C compiler for triton in the image) |
| `SPECULATIVE_MODEL` | — | HF model ID of a small draft model for speculative decoding. Works with steering: draft proposes, the steered main model verifies — output correctness preserved, acceptance rate lower |
| `SPECULATIVE_NUM_TOKENS` | `5` | Tokens the draft proposes per step |

## Circuits

| Variable | Default | Description |
|----------|---------|-------------|
| `CIRCUIT_ALLOW_CONCURRENT` | `true` | Several [circuits](/features/circuits) may serve at once on disjoint layers, and an operator may compose two on a shared layer with `allow_layer_overlap`. `false` refuses that override (naming configuration as the reason) and keeps only the most recently activated circuit across a restart. Was `false` for the first release |

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

## Transformers models

Settings that decide whether a transformers (non-GGUF) model fits the GPUs. See [Hardware Requirements](/getting-started/hardware#how-millm-decides-a-transformers-model-fits) for how they are used.

| Variable | Default | Notes |
|---|---|---|
| `TRANSFORMERS_MIN_CONTEXT` | `4096` | The context, in tokens, whose KV cache every card a model uses must have room for, or the model's own maximum context if that is shorter. A floor for loading, not a limit on requests. SAE attachment keeps the same room free. Raise it to refuse a model that would only fit with a short context. |
| `TRANSFORMERS_CUDA_CONTEXT_MB` | `500` | Memory each card keeps for its CUDA context, in MB. Measured on the node at 250–400 MiB a card. It covers the context only: a request's working memory (its prefill activations and what PyTorch's allocator keeps reserved) is sized per model and per card, and a bitsandbytes load's staging is counted too (see [Hardware Requirements](/getting-started/hardware#how-millm-decides-a-transformers-model-fits)). |
| `TRANSFORMERS_IDLE_CACHE_RELEASE_S` | `5.0` | Seconds a transformers model's cards stay idle before PyTorch's unused cached memory is returned to them. Without this, memory a finished request freed stays reserved, and nvidia-smi, which miStudio and other tenants of the node read, shows it as used until the model is unloaded. The release waits for the request queue to be idle and takes its slot, so it never runs during a request, and a request that arrives first cancels it. The next request then reserves that memory again, which costs tens of driver calls. `0` releases as soon as the queue is idle; a negative value never releases. Nothing is released while continuous batching is running, because its requests do not pass through the request queue. GGUF models are not affected. |

## GGUF models

Settings that apply only when the loaded model is a GGUF file. See [Hardware Requirements](/getting-started/hardware#kv-cache) for the measurements behind the defaults.

| Variable | Default | Notes |
|---|---|---|
| `GGUF_CONTEXT_LENGTH` | `32768` | **A ceiling, not a target.** miLLM reads what the file declares and loads the largest window that fits under this. `0` means "whatever the file declares", bounded only by what fits. |
| `GGUF_TENSOR_SPLIT` | *(empty)* | How a GGUF model that needs more than one GPU divides its layers. Empty divides them in proportion to each used card's free memory. Otherwise give one proportion per card the split uses, in index order (`1,3` puts a quarter on the lower-index card). A value with the wrong number of entries for a load is refused, not stretched. It has no effect on a model that fits one card. |
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
