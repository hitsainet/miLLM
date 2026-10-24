---
sidebar_position: 1
title: Configuration
---

# Configuration Reference

miLLM is configured entirely through environment variables (with `.env` file support, case-sensitive names). In Docker Compose these go in the `api` service's `environment:` block; in Kubernetes, in the deployment's `env:`.

Thirty of these settings can also be changed **while miLLM runs**, on the Admin UI's **Settings → Environment** section, with no restart. See [Changing settings while miLLM runs](#changing-settings-while-millm-runs).

## Core

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/millm` | PostgreSQL DSN (async driver required) |
| `MODEL_CACHE_DIR` | `/app/model_cache` | Where downloaded model weights live |
| `SAE_CACHE_DIR` | `/app/sae_cache` | Where downloaded SAEs live |
| `HF_TOKEN` | — | Server-wide Hugging Face token for gated and private repos. A token saved on the **Settings** page outranks it, and a per-request token (never persisted) outranks both. Empty or `none` counts as unset. See [Hugging Face token](#hugging-face-token) |
| `SETTINGS_ENCRYPTION_KEY` | — | Key that encrypts secrets saved in the database (today: the Settings page's Hugging Face token). At least 32 characters; generate with `openssl rand -base64 32`. **Unset, saving a token is refused** — nothing is ever stored unencrypted and no key is generated for you. Keep it: a saved token cannot be read under a different key. See [Hugging Face token](#hugging-face-token) |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Bind address and port |
| `DEBUG` | `false` | Enables debug behavior; never use in production |
| `AUTO_LOAD_MODEL` | — | Model ID or name to load automatically at startup — recommended for unattended deployments |

## HTTP & logging

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins. Set explicitly (e.g. `http://localhost:3000,https://webui.example.com`) when browsers call the API cross-origin |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL`. Changeable while miLLM runs (takes effect at once). No level adds request text or error details to responses, and chat text is not written to the log at any level |
| `LOG_FORMAT` | `console` | `console` for humans, `json` for log pipelines |

## Concurrency & timeouts

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONCURRENT_REQUESTS` | `1` | **Must stay `1`.** This is a correctness constraint, not a throughput knob: values above `1` race on steering apply/restore and on the shared sensing buffer. Leave it at `1` |
| `MAX_PENDING_REQUESTS` | `10` | Queue depth beyond the concurrent slots; overflow returns `503` (`QUEUE_FULL`, backpressure) |
| `EMBEDDINGS_MAX_INPUTS` | `128` | Most strings one `/v1/embeddings` request may carry (1–2048); more is refused with `400` before any model load. A capped request holds the only request slot, so this bounds how long a chat waits behind it. Set from measured latency (2026-10-07): the largest power of two with cap × p95 seconds per input ≤ 30 s for the largest model served (Llama-3.1-8B-Instruct: 128). Changeable while miLLM runs |
| `GRACEFUL_UNLOAD_TIMEOUT` | `30.0` | Seconds an unload waits for the requests already running on the model before it moves any weight. Requests that arrive once the unload has begun are refused with `503 model_busy` rather than waited for. If the running requests outlast this, the unload goes ahead and they fail. (Until 2026-09-14 this setting was read by nothing and the wait was a fixed 5 s.) |

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
| `WEIGHTS_COMMIT_WATCHDOG_MS` | `800` | A commit open longer than this marks the weights `torn`; a late commit-end from the same attachment still restores `clean` and is recorded as an overrun. Ten times the measured 99.9th-percentile commit, rounded up to the next 100 ms: miForge's real trainer on the RTX 3090 (2026-10-17, 1,000 training steps of a 1.2B model with 8-bit AdamW under chat load) committed in 56.4 ms median and 71.0 ms at the 99.9th percentile. Raise it for a trainer whose commits are slower |
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

## Float32 scoring

Settings for scoring requests that ask for `scoring_precision: "float32"`. See [Float32 scoring](/api/openai-compatible#float32-scoring).

| Variable | Default | Notes |
|---|---|---|
| `SCORING_FLOAT32_DEVICE` | `auto` | Where the request's float32 copy of the model is built. `auto` uses the GPU the model's inputs are on if the copy fits there, then the CPU if it fits there, and otherwise refuses with reason `memory`. `cpu`, `cuda` (the model's GPU) or `cuda:N` choose one device. If the chosen device can't hold the copy, the request is refused; miLLM never moves it somewhere else. Any other value stops the server at startup. |
| `SCORING_FLOAT32_HEADROOM_MB` | `1024` | Memory in MB that must be left free on the device after the copy (for the forward pass and the allocator). A copy that would leave less is refused with reason `memory` before anything is allocated. CPU memory is the smaller of the host's available memory and the container's memory limit. |

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

## Telemetry export

OpenTelemetry export of probe results, spans and counters (Feature 38). **Off by default**: nothing
leaves until a switch is on and an endpoint is named. Every one of these is environment only: no
route, and not the Settings page, can change them, and the listing there shows the endpoint
without credentials and the secrets only as set or not set. The names avoid the `OTEL_` prefix,
so OpenTelemetry's own `OTEL_EXPORTER_*` variables are never read. Guide:
[Telemetry Export](/features/telemetry-export).

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEMETRY_TRACES_ENABLED` | `false` | Send one span per inference request (and its queue-wait and probe-evaluation children) |
| `TELEMETRY_METRICS_ENABLED` | `false` | Send the probe coverage counters and two histograms as OTLP metrics. The Prometheus route has the counters either way; feed a store from one of the two, not both |
| `TELEMETRY_LOGS_ENABLED` | `false` | Send one `mi.probe.result` record per recorded result and one `mi.probe.lifecycle` record per lifecycle step |
| `TELEMETRY_ENDPOINT` | — | The receiver's base URL; `/v1/traces`, `/v1/metrics` and `/v1/logs` are appended to its path. A switch on with no endpoint is refused (`NoEndpoint`), never defaulted. **Put no credential in it**: the path is shown in health |
| `TELEMETRY_PROTOCOL` | `http/protobuf` | The only value accepted |
| `TELEMETRY_HEADERS` | — | `key=value,key=value` headers sent with every export: where credentials go. Secret. A malformed entry refuses the whole value (`InvalidHeaders`) |
| `TELEMETRY_ALLOW_PLAINTEXT` | `false` | Required for an `http://` endpoint (`PlaintextNotAllowed` otherwise): the recorded choice that the path is trusted and local |
| `TELEMETRY_TLS_CA_FILE` | — | A CA bundle to verify the receiver with, instead of the system bundle |
| `TELEMETRY_TLS_CLIENT_CERT_FILE`, `TELEMETRY_TLS_CLIENT_KEY_FILE` | — | Client certificate and key for mutual TLS; both or neither. The key path is secret |
| `TELEMETRY_SERVICE_NAME` | `millm` | Resource attribute `service.name` |
| `TELEMETRY_SERVICE_INSTANCE_ID` | random per process | Resource attribute `service.instance.id` |
| `TELEMETRY_DEPLOYMENT_ENVIRONMENT` | — | Resource attribute `deployment.environment.name` |
| `TELEMETRY_QUEUE_SIZE` | `8192` | Records queued per signal before new ones are dropped (and counted) |
| `TELEMETRY_BATCH_SIZE` | `512` | Records per export request |
| `TELEMETRY_EXPORT_TIMEOUT_S` | `10` | Bounds connecting and each read of the answer (not the whole exchange) |
| `TELEMETRY_RETRY_HORIZON_S` | `60` | How long a retryable failure (connection error, 408, 429, 502, 503, 504) is retried before its records count as failed |
| `TELEMETRY_DRAIN_DEADLINE_S` | `5` | At shutdown, how long queued records may still be sent; the rest count as dropped (at most 25) |
| `TELEMETRY_METRIC_INTERVAL_S` | `15` | How often metrics are collected and sent |
| `TELEMETRY_TRACES_SAMPLER_RATIO` | `1.0` | Share of traces kept (parent-based). Result records are not sampled |
| `TELEMETRY_SERIES_CAP` | `2000` | Most series per metric instrument; more fold into one overflow series, counted in `millm_probe_series_overflow_total` |
| `TELEMETRY_STATELESS_EXPORT_PERMITTED` | `false` | The operator's half of exporting `POST /api/probes/score` results; the request's `export_results: true` is the other |
| `TELEMETRY_RESULT_LOG_ORIGINS` | `live,batch` | Which recorded results become log records |
| `TELEMETRY_HEALTH_DEGRADED_WINDOW_S` | `60` | How long a drop keeps a signal `degraded` in health |
| `TELEMETRY_HEALTH_UNAVAILABLE_AFTER_S` | `300` | How long failing without a success turns a signal `unavailable` |
| `OTEL_SDK_DISABLED` | `false` | The one standard OpenTelemetry variable honoured: `true` forces every signal off |

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

## Hugging Face token

Every call miLLM makes to the Hugging Face Hub — model and SAE downloads, model and SAE previews,
cluster and probe pack browsing and import, and the speculative-decoding draft model — chooses its
credential by one rule, in this order:

1. the token given with the request (the optional field on the download and preview forms, or
   `hf_token` in an API call);
2. the token saved on the Admin UI's **Settings** page;
3. the `HF_TOKEN` environment variable;
4. none (anonymous).

A value that is empty, blank or `none` counts as absent at every step. With nothing configured the
Hub is called with no credential at all — never a token picked up from elsewhere on the host.

**Saving a token on the Settings page** stores it in the database encrypted with AES-256-GCM, under
the key in `SETTINGS_ENCRYPTION_KEY`. The server never sends the token back: the page shows a masked
form (`hf_…abcd`), which source is active, and when the token was last verified (Test calls
Hugging Face's `whoami` and records the account name).

**Create the key once** (Kubernetes; the `millm-secrets` Secret already exists):

```bash
KEY=$(openssl rand -base64 32)
kubectl -n millm patch secret millm-secrets --type merge \
  -p "{\"stringData\":{\"SETTINGS_ENCRYPTION_KEY\":\"$KEY\"}}"
kubectl -n millm delete pod -l app=millm-backend   # the pod reads the key at start
```

With Docker Compose, put `SETTINGS_ENCRYPTION_KEY=<value>` in `.env` and recreate the `api`
container. Keep a copy of the value somewhere safe: if it changes, the saved token cannot be decrypted
(the Settings page says so, Hub calls fall back to `HF_TOKEN` or anonymous, and saving the token
again fixes it). Without the key, Save is disabled on the page and `PUT /api/settings/hf-token`
answers `409 SETTINGS_KEY_MISSING` with these instructions.

| Route | Purpose |
|---|---|
| `GET /api/settings/hf-token` | Status: `configured`, `source` (`settings` / `env` / `none`), `masked`, `last_verified_at`, `user`, `stored`, `stored_error`, `encryption_available` |
| `PUT /api/settings/hf-token` | Save `{"token": "hf_…"}` (encrypted) |
| `DELETE /api/settings/hf-token` | Clear the saved token |
| `POST /api/settings/hf-token/test` | Check the active token, or `{"token": "…"}` without saving it: the account name, or `401 INVALID_HF_TOKEN` / `503 HF_HUB_UNREACHABLE` |

No route ever returns the token, and it is never written to the logs.

## Changing settings while miLLM runs

The Admin UI's **Settings** page has an **Environment** section (Feature 35). The environment stays the source of truth: a value saved there is an **override**, stored in miLLM's database, that replaces the environment's value until **Use the environment value** removes it. Precedence is override, then the environment variable, then the code default. Overrides are loaded when the server starts, before anything reads a setting, so they survive a restart; an override that no longer passes the setting's rules is ignored and logged (`runtime_override_ignored`), never applied.

Only these settings can be changed this way. Each is checked by its own rules (type, range, and rules across settings such as a lease default not above the maximum), and a refused value is answered `422 SETTING_INVALID` with the rule that refused it.

| Setting | Takes effect | Allowed while running |
|---|---|---|
| `LOG_LEVEL` | now | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `MAX_PENDING_REQUESTS` | next queue admission | 1–1000 |
| `CLUSTER_HUB_CACHE_TTL_S`, `PROBE_HUB_CACHE_TTL_S` | next Hub lookup | 0–86400 |
| `BATCH_RETENTION_INTERVAL_S` | within 30 seconds | 60–604800 |
| `EMBEDDINGS_MAX_INPUTS` | next request | 1–2048 |
| `TRANSFORMERS_IDLE_CACHE_RELEASE_S` | when the next request ends | −1–86400 |
| `GRACEFUL_UNLOAD_TIMEOUT` | next unload | 1–600 |
| `LEASE_DEFAULT_TTL_SECONDS`, `LEASE_MAX_TTL_SECONDS` | next lease grant | 1–604800; default ≤ maximum; `BATCH_LEASE_TTL_S` ≤ maximum |
| `BATCH_MAX_ROWS`, `BATCH_MAX_FILE_BYTES` | next upload | 1–1,000,000; 1 MiB–2 GiB |
| `BATCH_FILE_RETENTION_DAYS` | next upload | 1–365 |
| `BATCH_CHUNK_ROWS` | next chunk | 1–256 |
| `PROBE_MAX_ARMED` | next arm | 1–64 (default 64: the armed-probe backstop) |
| `PROBE_MAX_ARMED_BEHAVIOURS` | next arm | 1–64 (default 2: behaviours armed at once on a model with no approved limit) |
| `PROBE_MAX_PER_BEHAVIOUR`, `PROBE_MAX_IMPORTED` | next arm; next import | 1–256 (default 32); 1–1,000,000 (default 10,000) |
| `PROBE_VERDICT_HEADER_MAX_BYTES` | next request | 256–65536 (default 2,048) |
| `PROBE_LIST_PAGE_SIZE` | next request | 10–1000 (default 200) |
| `RULE_SIGNALS_ENABLED` | next request | true or false (default true). Off: every attached rule signal reports `rule_signals_disabled`; attachments are kept |
| `RULE_TIME_BUDGET_MS` | next request | 1–1000 (default 20): how long one request's rule signals may run; a signal still running reports `rule_timeout` |
| `RULE_SETS_MAX_IMPORTED`, `RULE_SET_MAX_SIGNALS`, `RULE_SET_MAX_BYTES` | next import | 1–100,000 (default 1,000); 1–1,024 (default 64); 1 KiB–16 MiB (default 1 MiB) |
| `PROBE_RECOMMEND_TARGET_SETS`, `PROBE_RECOMMEND_MAX_DECODE_SHARE`, `PROBE_RECOMMEND_MAX_PREFILL_SHARE` | next recommendation | 1–16 (default 2); 0.01–1 (default 0.10); 0.01–1 (default 0.05) |
| `PROBE_MAX_AGE_DAYS`, `PROBE_MAX_EVENTS_PER_PROBE` | next recorded event | 1–3650; 1–1,000,000 |
| `SENSING_MAX_AGE_DAYS`, `CIRCUIT_SENSING_MAX_AGE_DAYS` | next recorded event | 1–3650 |
| `SENSING_EPSILON`, `CIRCUIT_SENSING_EPSILON`, `CIRCUIT_SENSING_MAX_TOKEN_LAG` | next arm (armed clusters and circuits keep theirs) | 0–1; 0–1; 0–256 |
| `CLUSTER_INTENSITY_MIN/MAX`, `CIRCUIT_INTENSITY_MIN/MAX` | next request | 0–10, minimum below maximum |
| `MULTISAE_VRAM_ENVELOPE_MB` | next attach or status read | 1–1,048,576 |
| `SCORING_FLOAT32_DEVICE` | next float32 scoring request | `auto`, `cpu`, `cuda` (a card cannot be named by number here; the environment may still set `cuda:N`) |
| `SCORING_FLOAT32_HEADROOM_MB` | next float32 scoring request | 0–65536 |
| `PROBE_SCORE_MAX_INPUTS` | next request | 64–1024 (miStudio's probe tool sends up to 64) |

The page also shows, read-only, the settings that apply at the **next model load** (`TORCH_COMPILE`, the `GGUF_*` settings, `TRANSFORMERS_MIN_CONTEXT`, `TRANSFORMERS_CUDA_CONTEXT_MB`, `MULTISAE_ATTACH_DTYPE`, `STRUCTURED_OUTPUT_GRAMMAR_CACHE`), those that need a **restart** of `millm-backend` (continuous batching, `KV_CACHE_MODE`, `SPECULATIVE_*`, `AUTO_LOAD_MODEL`, `LOG_FORMAT`, `QUEUE_DURATION_WINDOW`, `LEASE_ENDED_MEMORY`; a restart ends running requests and batches), and every other setting. **Secrets** (`DATABASE_URL`, `HF_TOKEN`, `SETTINGS_ENCRYPTION_KEY`, `WEIGHTS_TOKENS`) are shown only as set or not set, and can never be changed here. Correctness guarantees (`MAX_CONCURRENT_REQUESTS`, the `*_FORCE_SERIAL` switches, the `PROBE_PARITY_*` floors, `BATCH_PACK_DEFAULT`, `TORCH_COMPILE_MODE`) and cross-repository values (`CLUSTER_HUB_TAG`, `PROBE_HUB_TAG`, `WEIGHTS_COMMIT_WATCHDOG_MS`, `LEASE_HOLDER_MAX_CHARS`, `LEASE_REASON_MAX_CHARS`) are environment-only.

`PROBE_REFERENCE_SET_SHAPE` (the probe set each model's recommendation is costed for; default `{"probes": 9, "layers": 3, "rules": {"mean": 3, "max": 3, "attention": 3}}`) is a JSON value and is set in the environment only.

Every save and reset is kept in an append-only history with the previous and new value, who made it and an optional reason; the page shows the last 50.

### API

| Method and path | Does |
|---|---|
| `GET /api/environment` | Every setting: value in force, the environment's value, `source` (`env` or `override`), group, when a change takes effect, last change |
| `PUT /api/environment/{key}` | Save an override: `{"value": 4, "reason": "optional"}`. `422 SETTING_NOT_RUNTIME` for a setting that cannot change while running, `404 SETTING_NOT_FOUND` for an unknown name |
| `DELETE /api/environment/{key}?reason=` | Use the environment's value again. `404 SETTING_NOT_OVERRIDDEN` when there is no override |
| `GET /api/environment/history?limit=50` | The change history, newest first (1–200) |

miLLM runs one process, so an override reaches every part of the server at once. Like every management route, these have no login (the suite's decision); there is no MCP tool that writes them.

### Settings that were removed

`REDIS_URL`, `MAX_DOWNLOAD_WORKERS`, `MAX_LOAD_WORKERS`, `DOWNLOAD_TIMEOUT`, `CIRCUIT_HUB_TAG`, `CIRCUIT_MAX_LAYERS`, `CIRCUIT_MAX_EDGES` and `CIRCUIT_MAX_MEMBERS_PER_LAYER` were declared but never read, so setting them changed nothing. They were removed on 2026-10-20. A `.env` file that still names one starts normally with a warning; remove the line.

