---
sidebar_position: 2
title: OpenAI-Compatible API
---

# OpenAI-Compatible API

miLLM exposes an OpenAI-compatible API at `/v1`, making it a drop-in replacement backend for the OpenAI SDK, Open WebUI, LangChain, LlamaIndex, and anything else that speaks the OpenAI protocol. When steering is active on the server, it applies transparently to every completion (never to embeddings).

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (streaming and non-streaming) |
| `/v1/completions` | POST | Text completion |
| `/v1/embeddings` | POST | Text embeddings — always **unsteered** |
| `/v1/models` | GET | List available models |
| `/v1/models/{id}` | GET | Model metadata |

## Chat completions

### Request parameters

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `model` | string | required | Must match the loaded model's name (see `/v1/models`) |
| `messages` | array | required | Roles: `system`, `user`, `assistant`, `tool`, `function` |
| `stream` | bool | `false` | SSE streaming |
| `temperature` | float | `1.0` | `0` = greedy/deterministic. Range 0–2 |
| `top_p` | float | `1.0` | Nucleus sampling, 0–1 |
| `n` | int | `1` | Number of choices (serial backend only) |
| `max_tokens` | int | server default | Validated against the model's context window |
| `stop` | string \| string[] | — | Stop sequences, enforced in both streaming and non-streaming |
| `frequency_penalty` | float | `0.0` | −2 to 2; mapped to repetition penalty internally |
| `presence_penalty` | float | `0.0` | −2 to 2; mapped to repetition penalty internally |
| `profile` | string | — | **miLLM extension**: apply a saved [steering profile](/features/profiles) for this request only |
| `steering_intensity` | float \| string | — | **miLLM extension** (chat completions only): per-request steering dial — a λ in `0`–`2`, or `"off"` / `"min"` / `"max"` |

:::note Intensity coupling
When the steering base is an imported **cluster**, its stored strengths are scaled by an intensity dial (λ) before applying. Without `steering_intensity`, the cluster's persistent dial (set on the Clusters page) applies; with it, the request's λ **overrides** the stored one for that request only. Symbolic `"min"`/`"max"` resolve to the cluster's declared `intensity_range` bounds (intersected with the `[0, 2]` dial envelope), and numeric λ is capped at the range's **maximum** (or the server's configured maximum for clusters without a declared range) — dialing *down* below the declared floor (toward off) is always honored, matching the management API's bounds of `[0, max]`. The base is the named `profile` if given, else the active profile, else the live steering values. `0`/`"off"` disables steering for the request without validating the base (a profile that would 400 at λ=0.01 still turns steering off at λ=0).
:::

When the steering base is an imported **circuit** (a multi-layer intervention spanning several SAEs), one λ scales **every layer together** — each member through its own layer's SAE. Two differences from the cluster rule above:

- **Both ends are clamped.** Numeric λ is clamped into the circuit's declared `intensity_range` intersected with the configured envelope — the floor as well as the ceiling. `0.1` against an authored `[0.5, 1.5]` resolves to `0.5`, not `0.1`. Only an exact `0`/`"off"` is honored below the floor.
- **The default floor is 0, not 0.5.** Circuits use `CIRCUIT_INTENSITY_MIN` (default `0.0`) where clusters use `CLUSTER_INTENSITY_MIN` (default `0.5`). A circuit whose document declares no `intensity_range` therefore makes `"min"` identical to `"off"`.

Members are re-derived from the strengths the circuit was **authored** with, so the dial is absolute rather than compounding on the circuit's stored intensity. Each member is clamped to miLLM's ±200 steering range at apply time, so at a high λ a strong member can reach the ceiling while weaker ones keep scaling, compressing their relative proportions.

A circuit serving in `slice_fallback` mode is steered by its backing **cluster profile**, so the cluster rule above applies to it — including the 0.5 floor.

### `X-miLLM-Circuit-Rung`

Responses carry `X-miLLM-Circuit-Rung` when a circuit is genuinely steering generation, in [RFC 8941](https://www.rfc-editor.org/rfc/rfc8941) structured form:

```
X-miLLM-Circuit-Rung: 2; language="causally validated (edge)"
```

The rung is a bare integer and the phrase a quoted-string, so ladder punctuation cannot break a naive parser. The phrase is rendered server-side from the evidence ladder and **never composed per-request** — a circuit below rung 2 is never described as causal:

| Rung | `language` | Meaning |
|------|-----------|---------|
| 0 | `associated` | Mined co-occurrence only — unvalidated |
| 1 | `suggested (attribution-supported)` | Attribution evidence, not causal |
| 2 | `causally validated (edge)` | Each edge causally validated |
| 3 | `faithfulness-tested (circuit)` | The whole circuit was faithfulness-tested |

The header is **omitted** whenever the circuit is not actually steering — no active circuit, a slice-fallback serve, an unparseable definition, or no SAE attached on any member layer. Its absence never means "rung 0"; it means "no circuit-attributable steering on this response". Clients displaying evidence language must read this header (or the `steering` field on `GET /api/circuits/active`) rather than deriving it from whether a circuit row is active.

An operator changing steering through the management API **while a request is generating** wins: the request's restore is skipped rather than reverting them, and the management response reports whether the value is actually live. A per-request dial therefore never silently undoes a concurrent operator action.

Responses to dialed requests carry an `X-miLLM-Steering-Intensity` header echoing the resolved λ. The echo is best-effort: it is omitted whenever the dial will not change steering (no SAE attached, unknown profile, steering disabled with a dial-only request, or an empty steering base), and a concurrent profile switch while the request queues can in rare cases make a symbolic echo differ from the applied λ.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-2-2b",
    "messages": [
      {"role": "system", "content": "You are concise."},
      {"role": "user", "content": "What is a sparse autoencoder?"}
    ],
    "temperature": 0.7,
    "max_tokens": 150
  }'
```

```json title="Response"
{
  "id": "chatcmpl-9f3a2b...",
  "object": "chat.completion",
  "created": 1783761600,
  "model": "gemma-2-2b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "A sparse autoencoder is..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 24, "completion_tokens": 87, "total_tokens": 111}
}
```

`finish_reason` is `"stop"` for EOS or a stop sequence, `"length"` when `max_tokens` was reached.

### With the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # miLLM doesn't require auth
)

response = client.chat.completions.create(
    model="gemma-2-2b",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
)
print(response.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="gemma-2-2b",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

Streaming uses standard SSE (`data: {json}\n\n` frames, terminated by `data: [DONE]`). The first chunk carries the `role`, the final chunk carries `finish_reason` and `usage`. If generation fails mid-stream, an SSE error event is emitted before `[DONE]` (the HTTP status is already 200 by then — check for `error` objects in-stream). When a stop sequence matches during streaming, generation is cancelled promptly rather than running to `max_tokens`.

### Per-request steering with `profile`

```python
response = client.chat.completions.create(
    model="gemma-2-2b",
    messages=[{"role": "user", "content": "What is truth?"}],
    extra_body={"profile": "honesty-amplification"},
)
```

The named profile's steering replaces the global configuration for this one request, then the previous state is restored. Unknown profile → `404`; profile invalid for the attached SAE → `400`; no SAE attached → the request runs unsteered. Requests with `profile` always use the serial backend. Details: [Profiles](/features/profiles#per-request-profiles-api).

## Text completions

`POST /v1/completions` accepts `prompt` (string or list of strings; each list entry produces a choice) plus the same sampling parameters as chat. No chat template is applied — the prompt goes to the model verbatim, which is often preferable for base-model steering experiments.

## Embeddings

`POST /v1/embeddings` with `input` (string or list) returns mean-pooled last-hidden-layer embeddings. `encoding_format` may be `"float"` (default) or `"base64"`.

:::info Embeddings are never steered
The steering hook is suppressed during embedding computation, so embeddings always reflect the unmodified model — making them a neutral measuring stick for comparing steered vs. unsteered generations.
:::

## Models

```bash
curl http://localhost:8000/v1/models
```

```json
{"object": "list", "data": [{"id": "gemma-2-2b", "object": "model", "created": 1774046346, "owned_by": "google/gemma-2-2b"}]}
```

By default `/v1/models` lists **all available models** (READY, LOADED, LOADING). When a model is **locked for steering**, only that locked model is listed — so a steering-locked server presents a single stable model id to OpenAI clients. Use the [Management API](/api/models) to see everything on disk (including states not surfaced here).

## Errors & backpressure

`/v1` errors use OpenAI's format so SDK exception handling works unchanged. Notable cases:

| Situation | Status | `code` |
|-----------|--------|--------|
| No model loaded | 503 | `model_not_loaded` |
| Unknown model name in request | 404 | `model_not_found` |
| Prompt + `max_tokens` exceeds context window | 400 | `context_length_exceeded` |
| Steering error on an in-flight stream (mismatched cluster, bad index) | SSE `error` event, then `[DONE]` | `invalid_feature_index` |
| Request queue full (backpressure) | 503 | `queue_full` |
| Unknown `profile` | 404 | `profile_not_found` |
| Invalid `steering_intensity` (outside 0–2 / unknown symbol) | 400 | `invalid_parameter` |

## Behavior under continuous batching

If the opt-in [CBM backend](/concepts/architecture#continuous-batching-opt-in) is enabled, requests matching the server's fixed sampling parameters are batched for throughput; requests with different `temperature`/`top_p`, a `profile` or `steering_intensity` parameter, or (optionally) active monitoring fall back to the serial path automatically. `GET /api/health/inference` shows which backend is active.

:::tip Integration with Other Tools
- **Open WebUI:** set the OpenAI API base URL to `http://<host>:8000/v1` — [tutorial](/tutorials/open-webui)
- **miStudio:** use the "OpenAI Compatible" method pointed at miLLM's `/v1`
- **LangChain / LlamaIndex:** use the OpenAI provider with a custom base URL
:::
