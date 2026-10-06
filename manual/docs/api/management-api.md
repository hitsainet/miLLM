---
sidebar_position: 8
title: Endpoint Index
---

# Management API — Endpoint Index

A single-page index of every `/api` endpoint. Each area has a dedicated page with request/response examples — follow the section links.

All endpoints return the [management envelope](/api/overview#the-management-envelope): `{success, data, error}`.

## Models (`/api/models`) — [full reference](/api/models)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/models` | GET | List all models |
| `/api/models` | POST | Download a model (HF or local path) |
| `/api/models/preview` | POST | Preview model metadata & memory estimates |
| `/api/models/{id}` | GET | Get model details |
| `/api/models/{id}` | DELETE | Delete model from disk |
| `/api/models/{id}/load` | POST | Load model to GPU |
| `/api/models/{id}/unload` | POST | Unload model from GPU |
| `/api/models/{id}/lock` | POST | Lock model (prevent unload) |
| `/api/models/{id}/unlock` | POST | Unlock model |
| `/api/models/{id}/cancel` | POST | Cancel in-progress download |
| `/api/models/{id}/lease` | POST | Lease the resident model (returns `lease_id` once) |
| `/api/models/{id}/lease` | GET | Live lease and last ended lease (never the ID) |
| `/api/models/{id}/lease/renew` | POST | Renew (header `X-miLLM-Lease`) |
| `/api/models/{id}/lease` | DELETE | Release (header `X-miLLM-Lease`) |

`POST …/load` and `…/unload` read `X-miLLM-Lease`; see [Model lease](/api/models#model-lease).

## SAEs (`/api/saes`) — [full reference](/api/saes)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/saes` | GET | List SAEs + attachment status |
| `/api/saes/download` | POST | Download SAE from HuggingFace |
| `/api/saes/preview` | POST | Preview SAE repository files |
| `/api/saes/attachment` | GET | Single-SAE attachment status (incl. `steering_apply_count`) |
| `/api/saes/attachments` | GET | Multi-SAE attachment status — every attached `(sae_id, layer)` entry, total VRAM and the `vram_envelope_mb` warning (Feature 12 cross-layer circuit serving) |
| `/api/saes/attach-set` | POST | Attach a **set** of SAEs at once for cross-layer circuit serving — loads only the referenced SAEs (fp16), installs one hook per `(sae_id, layer)`, idempotent per key |
| `/api/saes/{id}` | GET / DELETE | Get / delete an SAE |
| `/api/saes/{id}/compatibility` | GET | Dry-run compatibility check |
| `/api/saes/{id}/attach` | POST | Attach to loaded model |
| `/api/saes/{id}/detach` | POST | Detach |
| `/api/saes/{id}/cancel` | POST | Cancel in-progress download |
| `/api/saes/monitoring` | POST | Configure monitoring (alias of `/api/monitoring/configure`) |

## Steering (`/api/saes/steering`) — [full reference](/api/saes#steering)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/saes/steering` | GET | Get steering status and values |
| `/api/saes/steering` | POST | Set single feature value |
| `/api/saes/steering/batch` | POST | Set multiple feature values |
| `/api/saes/steering/enable` | POST | Enable steering |
| `/api/saes/steering/disable` | POST | Disable steering (keep values) |
| `/api/saes/steering/{idx}` | DELETE | Remove single feature |
| `/api/saes/steering` | DELETE | Clear all values & disable |

## Monitoring (`/api/monitoring`) — [full reference](/api/monitoring)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/monitoring` | GET | Get monitoring state |
| `/api/monitoring/configure` | POST | Configure (features, history size, top-k) |
| `/api/monitoring/enable` | POST | Toggle without reconfiguring |
| `/api/monitoring/history` | GET / DELETE | Activation history / clear |
| `/api/monitoring/statistics` | GET / DELETE | Feature statistics / reset |
| `/api/monitoring/statistics/top` | POST | Top features by metric |

## Profiles (`/api/profiles`) — [full reference](/api/profiles)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/profiles` | GET / POST | List / create |
| `/api/profiles/save-current` | POST | Save live steering as a profile |
| `/api/profiles/active` | GET | Get active profile |
| `/api/profiles/{id}` | GET / PATCH / DELETE | Get / update (partial) / delete |
| `/api/profiles/{id}/activate` | POST | Activate (replaces current steering) |
| `/api/profiles/{id}/deactivate` | POST | Deactivate |
| `/api/profiles/{id}/export` | GET | Export as JSON |
| `/api/profiles/import` | POST | Import from JSON |

## Clusters (`/api/clusters`) — imported cluster definitions ([concepts](/features/clusters))

Clusters are cluster-typed steering documents imported from miStudio (or a Hugging Face pack). One cluster serves at a time, behind a global intensity dial.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/clusters` | GET | List imported clusters with bound state, warnings and intensity |
| `/api/clusters/import` | POST | Import a cluster definition or bundle |
| `/api/clusters/hub/search` | GET | Search public cluster packs on Hugging Face |
| `/api/clusters/hub/{repo_id}/definitions` | GET | List a Hub repo's cluster definitions |
| `/api/clusters/hub/import` | POST | Import one definition from a Hub repo |
| `/api/clusters/{cluster_id}/activate` | POST | Activate a cluster (hard compatibility gate) |
| `/api/clusters/{cluster_id}/deactivate` | POST | Deactivate a cluster |
| `/api/clusters/{cluster_id}` | DELETE | Delete a cluster profile |
| `/api/clusters/active/intensity` | PUT | Set the **active** cluster's intensity (global λ dial) |
| `/api/clusters/{cluster_id}/intensity` | PUT | Set a specific cluster's intensity (λ) |
| `/api/clusters/{cluster_id}/export` | GET | Re-export the lossless original definition |

## Circuits (`/api/circuits`) — imported multi-layer circuits ([concepts](/concepts/interpretability))

Circuits are multi-layer interventions spanning several SAEs, each carrying an evidence rung (0–3). A circuit below rung 2 is refused unless activation passes `acknowledge_unvalidated`. Serving requires an SAE attached on every member layer (Feature 12 multi-SAE).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/circuits` | GET | List imported circuits with evidence rung, layers and serveability (`?min_rung=`, `?serveable=`, `?limit=`, `?offset=`) |
| `/api/circuits/active` | GET | The currently serving circuit(s) and their `steering` state |
| `/api/circuits/claims` | GET | Which circuit holds which layer |
| `/api/circuits/claims/release` | POST | Release a stuck layer claim |
| `/api/circuits/import` | POST | Import a circuit definition |
| `/api/circuits/{circuit_id}/activate` | POST | Activate (serve) a circuit — refused `UNVALIDATED_CIRCUIT` below rung 2 unless acknowledged; `CIRCUIT_LAYER_CONTENTION` when another circuit already holds a layer |
| `/api/circuits/{circuit_id}/deactivate` | POST | Stop serving a circuit |
| `/api/circuits/active/intensity` | PUT | Set the active circuit's global intensity (λ) — `NO_ACTIVE_CIRCUIT` / `AMBIGUOUS_ACTIVE_CIRCUIT` when zero or several circuits serve |
| `/api/circuits/{circuit_id}` | DELETE | Delete an imported circuit |
| `/api/circuits/{circuit_id}/export` | GET | Export the raw circuit definition |

## Circuit sensing (`/api/circuit-sensing`) — observed edge firings ([WebSocket event](/api/websockets))

Records when a served circuit's **edges** co-activate during generation. This is an observation of firing, not a causal claim — an edge earns a causal reading only from rung-2 edge validation upstream in miStudio, never from a sensing event. Instrument, not gate — context text is never sent over WebSocket; fetch event detail for the ±K window.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/circuit-sensing/status` | GET | Edge-sensing status: sensable/unsensable edges, per-request overhead, `enabled_circuits` |
| `/api/circuit-sensing/events` | GET | List observed edge firings newest-first (`?limit=`, `?since=`, scoping params) |
| `/api/circuit-sensing/events/{event_id}` | GET | One observed edge firing incl. the ±K context window (404 `CIRCUIT_SENSING_EVENT_NOT_FOUND` when pruned/cleared) |
| `/api/circuit-sensing/events` | DELETE | Clear observed edge firings |
| `/api/circuit-sensing/{circuit_id}/enable` | POST | Enable edge sensing for a circuit |
| `/api/circuit-sensing/{circuit_id}/disable` | POST | Disable edge sensing for a circuit |

## Sensing (`/api/sensing`) — cluster co-activation events ([concepts](/features/clusters#co-activation-sensing))

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sensing/status` | GET | Armed state, threshold mode, per-request overhead, retention limits, and `enabled_clusters` (persistent intent, distinct from `armed`) |
| `/api/sensing/events` | GET | Events newest-first (`?profile_id=`, `?limit=`, `?since=`); age-expired rows are pruned on read |
| `/api/sensing/events/{id}` | GET | Event detail incl. the ±K context window (404 when pruned/cleared) |
| `/api/sensing/events` | DELETE | Clear events (`?profile_id=` scopes to one cluster) |
| `/api/sensing/{profile_id}/enable` | POST | Enable sensing for a cluster (arms live when that cluster is active with an SAE attached) |
| `/api/sensing/{profile_id}/disable` | POST | Disable sensing (disarms live) |
| `/api/sensing/{profile_id}/config` | PUT | Runtime overrides: `{"min_k": n}` sets the quorum (validated against the sensable-member ceiling), `{"min_k": null}` restores the default (all sensable members). Stored miLLM-locally — exports stay lossless. Re-arms live |

Event detail responses include `context_parts` `{before, span, after}` — the window split at the
fired span so clients can highlight it (older events predate the field and carry `null`).

## Health & operations (`/api/health`) — [full reference](/api/overview#health--operations-endpoints)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Liveness |
| `/api/health/ready` | GET | Readiness |
| `/api/health/detailed` | GET | Component breakdown, queue state (`inference`) and the live `lease` |
| `/api/health/gpus` | GET | Per-card memory, including what miLLM holds |
| `/api/health/inference` | GET | Active backend & capabilities |
| `/api/health/metrics` | GET | App metrics |
| `/api/health/metrics/prometheus` | GET | Prometheus format |
| `/api/health/circuits` | GET | Circuit breakers |
| `/api/health/circuits/{name}/reset` | POST | Reset a breaker |
| `/api/health/version` | GET | Version |

### Queue state: the `inference` block of `/api/health/detailed`

A stable, typed contract (Feature 29). The block is **always present**; when it cannot be read it
carries `error` and the other fields are `null`. Unmeasured is `null`, never `0`.

| Field | Type | Meaning |
|---|---|---|
| `backend` | `"queue"` \| `"cbm"` | The serial queue, or continuous batching while it runs |
| `cbm_enabled`, `cbm_running` | bool | Continuous batching configured / serving now |
| `queue_pending` | int | Interactive requests **waiting plus holding** a slot (it is incremented before the slot is taken — not "waiting" alone) |
| `queue_max_concurrent`, `queue_max_pending` | int | `MAX_CONCURRENT_REQUESTS`, `MAX_PENDING_REQUESTS` |
| `in_flight` | int \| null | Slots held now. The idle cache release holds a slot too and counts, because a request arriving then waits for it. Batch chunks count once the batch API ships. `null` while continuous batching runs |
| `queue_waiting` | int \| null | Interactive requests waiting: `queue_pending − holding`. `null` while continuous batching runs |
| `batch_backlog_rows` | int \| null | Rows not yet run across in-progress batches; `null` until a batch API exists (not `0`) |
| `estimated_wait_seconds` | float \| null | An **estimate** of how long a request arriving now waits for a slot |
| `error` | string \| null | Why the block could not be read |

The estimate is

```
estimated_wait_seconds = median(last 50 slot-holding durations) × (queue_waiting + in_flight) / queue_max_concurrent
```

`null` with fewer than 3 measured durations, and while continuous batching runs (its requests
hold no queue slot). The batch backlog is not added: interactive requests go first at every
batch chunk boundary. It is an estimate, not a promise.

`/api/health/detailed` also carries **`lease`**: the live lease on the resident model
(`model_id`, `model_name`, `holder`, `reason`, `acquired_at`, `renewed_at`, `expires_at`,
`ttl_seconds`, `seconds_remaining`) or `null`. Never the lease ID. No `nvidia-smi` runs on this
endpoint.

### Per-card memory: `GET /api/health/gpus`

```json
{"read_at": "2026-10-06T12:00:00Z", "reason": null, "cards": [
  {"smi_index": 0, "uuid": "GPU-247a…", "name": "NVIDIA GeForce RTX 3090",
   "total_mb": 24576, "used_mb": 17396, "free_mb": 7180,
   "torch_index": 0, "torch_measured": true, "millm_allocated_mb": 0, "millm_reserved_mb": 16820,
   "engine_memory": null, "processes": [{"pid": 4242, "used_mb": 17100}], "processes_reason": null}]}
```

- Total, used and free come from `nvidia-smi`. `torch_index` is torch's index, matched by UUID;
  a card torch cannot see is listed with `torch_index: null`, not dropped.
- `millm_allocated_mb` / `millm_reserved_mb` are torch's allocator on that card. They are read
  **only** on cards miLLM has placed a transformers model on (`torch_measured: true`); on any
  other card they are `null`. Reading a card never creates a CUDA context on it. torch's
  reserved figure excludes the CUDA context itself, which `nvidia-smi` counts per process.
- A resident GGUF model's memory is llama.cpp's, not torch's: its cards say
  `engine_memory: "not_measured_by_torch"`, and `processes` (nvidia-smi's per-process list,
  where the container can read it) is the only measurement of it.
- Without `nvidia-smi` the answer is `200` with `cards: []` and `reason` — never zeros.
- The read runs in a worker thread and takes no request-queue slot, so it is safe to poll
  during a generation.
