---
sidebar_position: 4
title: Architecture
---

# Architecture

How miLLM is put together, and what happens to a request between the HTTP socket and the GPU.

## Component map

```
                   ┌────────────────────────────────────────────┐
 OpenAI clients ──▶│  /v1/*   OpenAI-compatible API             │
 Admin UI / curl ─▶│  /api/*  Management API        FastAPI     │
 Socket.IO ───────▶│  WebSocket events (progress, metrics)      │
                   ├────────────────────────────────────────────┤
                   │  Services: model · sae · inference ·       │
                   │  monitoring · profile                      │
                   ├────────────────────────────────────────────┤
                   │  ML core: model loader · SAE loader ·      │
                   │  forward hook (steer + monitor) ·          │
                   │  generation backends                       │
                   ├──────────────┬─────────────────────────────┤
                   │ PostgreSQL   │ GPU: model + SAE weights    │
                   │ (metadata)   │ Redis (cache/realtime)      │
                   └──────────────┴─────────────────────────────┘
```

PostgreSQL stores metadata only (model/SAE registry, profiles); weights live on disk caches and are loaded into GPU memory on demand. One model and one SAE are resident at a time.

## The forward hook

Steering and monitoring share a single mechanism: a PyTorch **forward hook** registered on one decoder layer of the loaded model. On every forward pass it:

1. (If monitoring) encodes the layer output through the SAE encoder and captures activations
2. (If steering) adds the precomputed steering delta to the layer output

The hook is installed at SAE attach and removed at detach. Attach/detach waits for in-flight requests to drain before touching the hook, and the attach response reports the exact module path hooked (`layer_module_path`).

### torch.compile and the hook

On CUDA with non-quantized models, miLLM compiles the model's forward (`TORCH_COMPILE` auto-detect, mode `reduce-overhead`). Compiled graphs would normally *ignore* hooks registered after compilation — silently disabling steering. miLLM prevents this by enabling Dynamo's nn-module hook guards before compiling and resetting the compile cache on attach/detach, at the cost of a one-time recompile (~20 s) on the first request after an attach or detach. The `steering_apply_count` diagnostic exists to make this class of failure visible.

## Generation backends

### Serial queue (default)

Requests acquire a slot in a semaphore-guarded queue (`MAX_CONCURRENT_REQUESTS`, with `MAX_PENDING_REQUESTS` waiting slots; overflow returns `429`). Each generation runs `model.generate()` in a worker thread; streaming bridges tokens to SSE via `TextIteratorStreamer`. The serial path supports everything: per-request sampling params, per-request profiles, stop sequences with prompt cancellation, speculative decoding, and exact monitoring attribution.

### Continuous batching (opt-in)

`ENABLE_CONTINUOUS_BATCHING=true` starts a HuggingFace `ContinuousBatchingManager` at model load, batching many requests into shared forward passes for throughput. Its trade-offs:

| Property | Behavior |
|----------|----------|
| Sampling params | Fixed at manager creation (`CBM_DEFAULT_TEMPERATURE`, `CBM_DEFAULT_TOP_P`); requests with different values **fall back to serial** |
| Per-request profiles | Not supported in batch — such requests **fall back to serial** |
| Monitoring attribution | Batch slots ≠ request IDs; set `CBM_FORCE_SERIAL_MONITORING=true` for exact attribution |
| Steering | Applies to the whole batch (it's global state) |

`GET /api/health/inference` reports which backend is active and its capabilities.

### Speculative decoding (opt-in)

Set `SPECULATIVE_MODEL` to a small draft model (e.g. `google/gemma-2-2b` drafting for 27B). The draft proposes `SPECULATIVE_NUM_TOKENS` tokens; the main model verifies them in one pass. With an SAE attached, the draft is unsteered — acceptance rate drops, but **correctness is preserved** because every accepted token was verified by the steered main model, and monitoring captures real main-model activations.

## Request lifecycle (serial chat completion)

1. Request validated (`ChatCompletionRequest`), routed serial vs CBM
2. Queue slot acquired
3. If `profile` set: profile steering applied (validated against the attached SAE; failures return 4xx rather than silently generating with wrong steering)
4. Chat template applied, prompt tokenized, context-length checked
5. `model.generate()` in a worker thread — the hook steers/monitors every forward pass
6. Stop sequences enforced (streaming: generation is cancelled promptly when a stop sequence matches, not run to `max_tokens`)
7. Monitoring notified; usage counted; profile steering restored
8. Response (or final SSE chunk with `usage`) returned; slot released

## Graceful degradation

- **OOM with SAE attached** → SAE disabled, base model keeps serving
- **Model unload** → waits `GRACEFUL_UNLOAD_TIMEOUT` for in-flight requests
- **SAE detach** → drains both the serial queue and CBM in-flight requests before removing the hook
- **Circuit breakers** guard HuggingFace calls; status at `GET /api/health/circuits`
