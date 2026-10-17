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
| `MODEL_BUSY` | 409 | Operation conflicts with one in progress: a load is running, or the model is being unloaded (a second unload, or a load that would unload it again), or a weights operation (a checkpoint load or a serving pause) holds the resident weights (`details.reason: weights_operation`, `details.operation`). On `/v1` it is `503 model_busy`, typed `server_error`: the request is fine and succeeds once the other operation finishes, including a request for a model that is being unloaded |
| `MODEL_LOCKED` | 409 | Another model is locked for steering: a lock request, or a `/v1` request that would auto-load over it (`model_locked`). The management load and unload do not read it |
| `MODEL_LEASED` | 409 | A load, unload or swap while another holder leases the resident model. `details`: `holder`, `reason`, `expires_at`, `leased_model_id`, `leased_model_name`, `operation`, `target_model_id`. On `/v1` it is `409 model_leased`, typed `invalid_request_error` — never `model_locked`, even when the model is also locked. The holder proceeds by sending `X-miLLM-Lease` |
| `MODEL_NOT_RESIDENT` | 409 | A lease was requested on a model that is not the resident, loaded one; `details` name the resident model. On `/v1`, `model_not_resident` answers `X-miLLM-Load-Policy: refuse` for a model that is not resident: nothing loads |
| `LEASE_NOT_FOUND` | 404 | Renew or release with an unknown lease ID, or one for another model. The message says that a restart ends every lease |
| `LEASE_EXPIRED` | 409 | Renew or release with a lease ID that has ended; `details.end_reason` is `released`, `expired`, `model_unloaded` or `restart` |
| `INVALID_LEASE_REQUEST` | 400 | `holder` (1–128 chars), `reason` (1–512), `ttl_seconds` (1–7200, integer, never clamped) or the `X-miLLM-Lease` header is missing or out of bounds; `details.param` names it and the limit |
| `MODEL_LOADING` (`/v1`) | 503 | `X-miLLM-Load-Policy: refuse` and the model named is being loaded now; retry after `Retry-After` |

## `Retry-After` on every 503

Every response miLLM sends with status `503`, on any route, carries a `Retry-After` header in whole
seconds (at least 1). The error envelope, type and code are unchanged — the header is the only
addition.

| Code | `Retry-After` |
|---|---|
| `QUEUE_FULL` | The [estimated wait](/api/management-api#queue-state-the-inference-block-of-apihealthdetailed) rounded up, at most `RETRY_AFTER_MAX_S` (60); `RETRY_AFTER_QUEUE_DEFAULT_S` (5) with no estimate |
| `MODEL_BUSY`, `MODEL_LOADING` | `RETRY_AFTER_LOAD_S` (15) for a load in progress; `RETRY_AFTER_UNLOAD_S` (5) for an unload |
| `MODEL_NOT_LOADED` | `RETRY_AFTER_NOT_LOADED_S` (30) — a retry succeeds only after a model is loaded |
| `INSUFFICIENT_MEMORY` (`/v1`) | `RETRY_AFTER_MEMORY_S` (30) |
| `HUB_UNAVAILABLE` | The hub circuit breaker's remaining recovery time (1 when it is closed) |
| `GET /api/health/ready` not ready | `RETRY_AFTER_READINESS_S` (5) |
| `WEIGHTS_TORN` | `RETRY_AFTER_WEIGHTS_TORN_S` (30) — serving returns after a late `commit-end` or a weights load |
| any other 503 | `RETRY_AFTER_FALLBACK_S` (10), added by a safety net that also logs `retry_after_defaulted` |

A refusal raised inside a stream whose `200` is already sent cannot carry a header; its in-stream
error event carries `"retry_after": <seconds>` instead.

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

## Attachable weights errors

Raised by the `/api/weights` routes (weight attachment and the step barrier, checkpoints, adapter
serving). No refusal ever carries a token, an attachment ID or a path outside the checkpoint roots.

### Attachment, step barrier and tokens

| Code | HTTP | Meaning |
|------|------|---------|
| `WEIGHTS_UNAUTHORIZED` | 401 | A gated weights call without a valid bearer token. The response carries `WWW-Authenticate: Bearer`. `details.reason`: `missing` (no `Authorization: Bearer` header), `invalid_token`, `no_tokens_configured` (`WEIGHTS_TOKENS` is empty, so the gate fails closed) or `tokens_malformed` (the setting does not parse; the server log names the faulty entry) |
| `WEIGHTS_FORBIDDEN` | 403 | The token's role may not perform this call; `details.required_roles` names the roles that may |
| `NOT_ATTACHMENT_HOLDER` | 403 | A holder-only call whose `X-miLLM-Attachment` is missing or names another attachment, or whose token label is not the one that attached |
| `ALREADY_ATTACHED` | 409 | A trainer holds the resident weights (active or stale); `details` name its role, label, state, `attached_at` and the refused operation. On `/v1` an auto-load refused for this reason is `409 already_attached`, typed `invalid_request_error` |
| `WEIGHTS_UNSAVED` | 409 | A `/v1` request named a model other than the resident while the resident weights carry commits no checkpoint records (commits since the weights were last loaded or last saved). Answering it would auto-load the other model and discard those commits, so nothing is loaded or unloaded. `details`: `operation` (`auto_load`), `resident` (its served name), `resident_model_id`, `unsaved_commits`, `requested_model`, `requested_model_id`, `recovery`. Save the commits (`POST /api/weights/save`, as the attachment holder) or replace them deliberately (`POST /api/weights/load`, or the management load `POST /api/models/{id}/load`); either clears it. On `/v1`, `409 weights_unsaved` typed `invalid_request_error`; the message carries the same names and count (decision D-45) |
| `NOT_ATTACHABLE` | 400 | The resident model cannot be attached. `details.reason`: `gguf`, `quantized`, `dtype`, `not_on_gpu`, `continuous_batching_running`, `allocator` (not loaded as an attachable load), `concurrency` (`MAX_CONCURRENT_REQUESTS` above 1) or `model_transition` (a load, unload or internal write in progress); `details.parameter` names the first offending tensor where there is one |
| `NOT_ATTACHED` | 409 | A holder-only call while nothing is attached |
| `NOT_QUIESCED` | 409 | `commit-begin` while serving, or `resume` while a quiesce is still draining (`details.reason: quiesce_in_progress`); a `resume` while serving is a no-op success |
| `ALREADY_QUIESCED` | 409 | `quiesce` while already quiesced |
| `NO_OPEN_COMMIT` | 409 | `commit-end` with no open commit, or from someone other than its owner |
| `ATTACHMENT_NOT_STALE` | 409 | `clear-stale` against an attachment that is still sending heartbeats |
| `WEIGHTS_NOT_CLEAN` | 409 | The operation needs clean weights and a commit is open (`writing`) or the weights are `torn`; `details.weights_state` says which. A holder that re-attached after a watchdog tear gets this from `commit-begin` and `resume` (it is checked before the holder check) |
| `WEIGHTS_TORN` | 503 | The resident weights were left mid-write and are not served. A late `commit-end` from the writer or a load through `POST /api/weights/load` restores serving. On `/v1`, `503 weights_torn` typed `server_error`, with `Retry-After` |
| `PAUSE_IN_PROGRESS` | 409 | A serving pause (a save snapshot, an adapter load or unload) is in progress, so a `commit-begin`, a second pause or a concurrent save is refused; `details.operation` names the pause. No `Retry-After` (a 409) |
| `BARRIER_UNAVAILABLE` | 409 | Serving cannot be paused under this deployment's settings. `details.reason`: `cbm` (continuous batching is enabled and generates without the request slot) or `concurrency` (`MAX_CONCURRENT_REQUESTS` above 1). A retry does not help; change the setting |

### Checkpoints

| Code | HTTP | Meaning |
|------|------|---------|
| `INVALID_RUN_ID` | 400 | A `run_id` outside the save rule or the load rule, or too long for a model name; `details`: `run_id`, `pattern` |
| `MANIFEST_FIELD_NOT_CALLERS` | 400 | The save request sets manifest fields that miLLM writes itself; `details.fields` names them |
| `INVALID_MANIFEST` | 400 | A manifest fails the vendored schema; `details.errors[]` gives each `json_path` and `message` |
| `CHECKPOINT_EXISTS` | 409 | A checkpoint already exists at that step; `details.path` |
| `CHECKPOINT_AMBIGUOUS` | 409 | The run exists under more than one checkpoint root (`WEIGHTS_CHECKPOINT_DIR` and `WEIGHTS_CHECKPOINT_READ_ROOTS`); `details.roots` |
| `INSUFFICIENT_STORAGE` | 507 | Free disk below the checkpoint's estimate times `WEIGHTS_SAVE_FREE_SPACE_FACTOR`, refused before serving pauses; `details`: `need_bytes`, `free_bytes`, `factor` |
| `CHECKPOINT_WRITE_FAILED` | 500 | Writing the checkpoint failed after the snapshot; nothing was published. `details`: `stage`, `error` |
| `INVALID_CHECKPOINT_PATH` | 400 | The path resolves outside every checkpoint root; `details.input` is what was sent, never the resolved path |
| `CHECKPOINT_NOT_FOUND` | 404 | No checkpoint by that name or path |
| `CHECKPOINT_HASH_MISMATCH` | 422 | The weights file's SHA-256 differs from the manifest (`expected`, `actual`), or it changed after hashing (`changed_since_hash: true`). Nothing was written |
| `CHECKPOINT_NOT_LOADABLE` | 400 | A verified checkpoint miLLM will not load as it stands. `details.reason`: `dtype` (weights not 16-bit; never cast silently) or `remote_code` (its config names custom code and the request did not set `trust_remote_code: true` with an operator token) |
| `WEIGHTS_LOAD_FAILED` | 500 | A weights load failed after its first write; the weights are `torn`. `details`: `stage`, `error`, `weights_state` |
| `LOCAL_PATH_REGISTERED` | 409 | The checkpoint directory is already registered as a plain local model; delete or re-register that model first (`details.model_id`) |
| `LOAD_THROUGH_WEIGHTS_API` | 409 | A checkpoint model named by the management load or by `/v1`; it loads only through `POST /api/weights/load`, which verifies its hash. `details`: `model_id`, `weights_origin` |
| `MODEL_NAME_TAKEN` | 409 | Another model already uses the name a checkpoint needs; `details.name` |
| `NOT_INSTANTIABLE` | 400 | Reserved for instantiate, which is deferred |

### Adapters

| Code | HTTP | Meaning |
|------|------|---------|
| `ADAPTER_NOT_SERVABLE` | 400 | The resident model cannot serve an adapter. `details.reason`: `engine` (GGUF), `quantization` or `cbm` (continuous batching enabled) |
| `ADAPTER_INVALID` | 400 | The adapter directory is outside the checkpoint roots, a file is missing or not a regular file, or the manifest names a scaling miLLM does not implement. The path is never echoed beyond the root |
| `ADAPTER_MANIFEST_INVALID` | 400 | `adapter_manifest.json` fails the vendored schema; each failing field is named |
| `ADAPTER_HASH_MISMATCH` | 400 | The adapter weights' SHA-256 differs from the manifest; `details`: `expected`, `actual`. Nothing reached the GPU |
| `ADAPTER_BASE_MISMATCH` | 400 | The manifest names another base model (name or manifest hash); both identities in `details` |
| `ADAPTER_SHAPE_MISMATCH` | 400 | A target module is missing, not linear, the wrong shape or rank, fused by transformers, or the adapter carries tensors other than LoRA's A and B. `details.reason` and the first offender with both shapes |
| `ADAPTER_SCHEMA_NOT_VENDORED` | 409 | The adapter manifest schema is not installed, so every adapter load is refused |
| `ADAPTER_ALREADY_LOADED` | 409 | Another adapter is applied (named in `details`); unload it first |
| `NO_ADAPTER_LOADED` | 409 | Adapter unload with nothing applied |
| `ADAPTER_BASE_CHANGED` | 409 | The base model was replaced between verification and the pause; nothing was applied |
| `ADAPTER_INSTALL_FAILED` | 500 | Installing the adapter failed partway and every installed hook was removed; `details.error` |
| `ADAPTER_APPLIED` | 409 | Probe parity (a comparison against the base) refused while an adapter is applied; `details.adapter` names it |

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
