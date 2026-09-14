---
sidebar_position: 2
title: Error Codes
---

# Error Codes

Machine-readable error codes returned by the management API in the [error envelope](/api/overview#the-management-envelope) (`error.code`), with their HTTP status. The `/v1` OpenAI-compatible endpoints translate the same conditions into OpenAI-format errors.

## Model errors

| Code | HTTP | Meaning |
|------|------|---------|
| `MODEL_NOT_FOUND` | 404 | No model with that ID |
| `MODEL_ALREADY_EXISTS` | 409 | Same repo + quantization already downloaded |
| `MODEL_LOAD_FAILED` | 500 | Load crashed — see server logs |
| `MODEL_NOT_LOADED` | 400 | Operation needs a loaded model |
| `MODEL_ALREADY_LOADED` | 400 | Load called on the loaded model |
| `MODEL_BUSY` | 409 | Operation conflicts with one in progress: a load is running, or the model is being unloaded (a second unload, or a load that would unload it again). On `/v1` it is `503 model_busy`, typed `server_error`: the request is fine and succeeds once the other operation finishes, including a request for a model that is being unloaded |
| `MODEL_LOCKED` | 409 | Unload/delete refused; detach the SAE or unlock first |

## Resource errors

| Code | HTTP | Meaning |
|------|------|---------|
| `INSUFFICIENT_MEMORY` | 507 | A card the model would use cannot hold its weights, its KV cache at `TRANSFORMERS_MIN_CONTEXT` tokens, a request's working memory at that length and its CUDA context. `details.per_card` gives each card's free, weights, KV, working memory (`working_mb`), bitsandbytes staging (`staging_mb`) and context MiB; `details.working_memory` says how the working memory was sized (`method`: `traced` or `estimated`) and at how many tokens; `details.short_devices` names the short cards; `details.min_context_tokens` is the context the cache was sized at and `details.model_max_context_tokens` the model's own limit; `details.rebalance_passes` says how many times a split was re-planned before it was refused. An SAE attachment refused for memory carries `device`, `projected_mb`, `kv_reserve_mb`, `kv_context_tokens`, `working_reserve_mb` (the request working memory the load kept on that card) and `available_mb`. For a model whose KV cache miLLM cannot size, the estimate exceeds what is free. A request on `/v1` that loads the model gets `503 insufficient_memory`. A request that runs a card out of memory while generating gets this code too, typed `invalid_request_error` on `/v1` (`503`; a stream ends with that error event and `[DONE]`), with `details.device`, `device_name`, `prompt_tokens`, `max_new_tokens` and `batch_rows`: send a shorter prompt or fewer `max_tokens` |
| `INSUFFICIENT_DISK` | 507 | Not enough disk for the download |
| `UNSUPPORTED_QUANTIZATION` | 400 | The load asks for a quantization its engine cannot apply (Q2 on a transformers checkpoint that is not already quantized) |
| `GPU_NOT_FOUND` | 404 | The `gpu` named in a load is not visible to miLLM |
| `SPLIT_NOT_HONOURED` | 409 | `"gpu": "all"` was requested, and at least one card would get none of the model: it is too full to take a share, or the model's whole layers leave it empty. `details.unused_devices` names the cards. `details.mapped_mb_by_device` shows the layout when it was worked out before loading; otherwise `details.landed_on_devices` shows where the model landed |
| `INVALID_GGUF_TENSOR_SPLIT` | 500 | `GGUF_TENSOR_SPLIT` does not name one proportion per card the GGUF split uses; only the setting fixes it |

## Download errors

| Code | HTTP | Meaning |
|------|------|---------|
| `DOWNLOAD_FAILED` | 502 | Upstream (HuggingFace) failure |
| `DOWNLOAD_CANCELLED` | 499 | Cancelled by user |
| `REPO_NOT_FOUND` | 404 | Repository doesn't exist |
| `GATED_MODEL_NO_TOKEN` | 401 | Gated repo, no token — accept the license and supply `hf_token` |
| `INVALID_HF_TOKEN` | 401 | Token rejected by HuggingFace |
| `INVALID_LOCAL_PATH` | 400 | Local import path missing, malformed, or in a protected system directory |

## SAE & steering errors

| Code | HTTP | Meaning |
|------|------|---------|
| `SAE_NOT_FOUND` | 404 | No SAE with that ID |
| `SAE_NOT_ATTACHED` | 400 | Steering/monitoring requires an attached SAE |
| `SAE_ALREADY_ATTACHED` | 409 | That exact `(sae_id, layer)` is already attached (re-attach is rejected; multi-SAE `attach_set` on other layers is fine); also raised deleting an attached SAE |
| `SAE_INCOMPATIBLE` | 400 | `d_in` doesn't match the loaded model's hidden size |
| `SAE_LOAD_FAILED` | 500 | Weight loading crashed |
| `INVALID_FEATURE_INDEX` | 400 | Feature index outside `[0, d_sae)` — `details` names the offender |
| `SAE_SET_INCOMPLETE` | 422 | Serving a cross-layer circuit but a member's layer has no (unique) attached SAE — `details.offenders` names each `{feature_idx, layer, sae_id?, reason?}` (Feature 12 multi-SAE) |

:::note Steering value range: reject on set, clamp on dial
The two paths differ. **Setting** a steering value directly — `POST /api/saes/steering` and `/steering/batch` — validates against `[-200, 200]`, so an out-of-range value is **rejected** with `422 VALIDATION_ERROR` (the schema bound), not silently clamped. The ±200 **clamp** applies only on the **dial/intensity path** — profile activation and per-request/cluster/circuit λ — where a profile's stored strengths are *scaled* by λ and the scaled result is clamped to ±200 at apply time rather than failing the request. So a value you type is checked; a value the dial produces is clamped.
:::

## Profile errors

| Code | HTTP | Meaning |
|------|------|---------|
| `PROFILE_NOT_FOUND` | 404 | Unknown profile ID/name (including the per-request `profile` parameter) |
| `PROFILE_ALREADY_EXISTS` | 409 | Duplicate name |
| `PROFILE_INCOMPATIBLE` | 400 | Profile can't apply to the current configuration |
| `VALIDATION_ERROR` | 200† | Malformed import payload — steering entries that can't convert to `int→float` (returned in the envelope, `details.invalid_keys` names them) |
| `IS_CLUSTER_DOCUMENT` | 200† | A **cluster** definition/bundle was posted to `/api/profiles/import`; import it via `/api/clusters/import` instead (the flat profile format has no member/budget semantics) |

† Handler-level refusal returned inside the `{success:false, error}` envelope with HTTP 200 — see the [200-envelope note](#the-200-envelope-house-style) below.

## Circuit & sensing errors

Circuit serving spans several SAEs and carries an evidence rung; several of these are **handler-level refusals returned in the envelope with HTTP 200** (see the note below) rather than HTTP error statuses, so the client can surface the rung/contention and re-send with an acknowledgement.

| Code | HTTP | Meaning |
|------|------|---------|
| `CIRCUIT_NOT_FOUND` | 404 | No circuit with that ID |
| `UNVALIDATED_CIRCUIT` | 200† | Activating a circuit below rung 2 (`CAUSALLY_VALIDATED`) without acknowledgement — re-send with `acknowledge_unvalidated=true`. The payload carries the evidence rung so the override is deliberate |
| `CIRCUIT_LAYER_CONTENTION` | 200† | The circuit's layers are already served by another active circuit. `details` names the incumbent(s) and the measured hazard; overridable with `allow_layer_overlap=true` **unless** a same-key collision (`colliding_keys` present) makes it non-overridable |
| `NO_ACTIVE_CIRCUIT` | 200† | An operation needing an active circuit (e.g. `PUT /api/circuits/active/intensity`) was called with none serving |
| `AMBIGUOUS_ACTIVE_CIRCUIT` | 200† | Several circuits serve, so there is no single "active circuit" to dial — `details.active_circuits` lists them; deactivate all but one, or dial through the owning cluster |
| `SAE_SET_INCOMPLETE` | 422 | A circuit member's layer has no (unique) attached SAE (also listed under SAE & steering above) |
| `CIRCUIT_SENSING_EVENT_NOT_FOUND` | 404 | Circuit **edge** sensing event id that doesn't exist (pruned, cleared, or never existed) |
| `SENSING_EVENT_NOT_FOUND` | 404 | Cluster **co-activation** sensing event id that doesn't exist |

† Handler-level refusal in the envelope with HTTP 200 — see below.

### The 200-envelope house style

Most errors map to an HTTP error status. A few **circuit refusals** deliberately do not: `UNVALIDATED_CIRCUIT`, `CIRCUIT_LAYER_CONTENTION`, `NO_ACTIVE_CIRCUIT`, `AMBIGUOUS_ACTIVE_CIRCUIT` (and the profile import guards `VALIDATION_ERROR` / `IS_CLUSTER_DOCUMENT`) return **HTTP 200** with the standard `{success:false, data:null, error:{code, message, details}}` envelope. These are *decisions the handler makes about a well-formed request* — an evidence-rung gate, a layer-contention gate, "no/ambiguous active circuit" — rather than malformed input or a missing resource. The `code` is still stable and machine-readable; clients must branch on `success`/`error.code`, not on the HTTP status, for these. The rich `details` (the rung, the incumbent, the measured hazard) is what lets the caller re-send with `acknowledge_unvalidated` or `allow_layer_overlap`.

## General

| Code | HTTP | Meaning |
|------|------|---------|
| `VALIDATION_ERROR` | 422 | Request body failed schema validation (also FastAPI's native 422s). A few handlers also return this code in a 200 envelope for semantic input problems — see the [200-envelope note](#the-200-envelope-house-style) |
| `CONTEXT_LENGTH_EXCEEDED` | 400 | A `/v1` request whose prompt plus `max_tokens` (or an embeddings input) is longer than the model's context window. `details` carries `max_context_tokens`, `requested_tokens`, `prompt_tokens` and `max_tokens`. On `/v1` it is `400 context_length_exceeded`, typed `invalid_request_error`; a streamed request is refused before its stream starts. Retrying unchanged cannot succeed: shorten the prompt or ask for fewer `max_tokens` |

## Handling errors in code

```python
r = requests.post(f"{MILLM}/api/saes/steering",
                  json={"feature_idx": 999999, "value": 40})
body = r.json()
if not body["success"]:
    code = body["error"]["code"]          # "INVALID_FEATURE_INDEX"
    details = body["error"]["details"]    # {"feature_idx": 999999, "d_sae": 16384}
```

For `/v1` endpoints, OpenAI SDKs raise their native exceptions (`NotFoundError`, `RateLimitError`, …) based on the HTTP status.
