# Technical Design: Model Lease, Backpressure and GPU Visibility

## miLLM Feature 29

**Version:** 1.0 · **Created:** 2026-10-06 · **Status:** Planned
**References:** `029_FPRD|Model_Lease_Backpressure_And_GPU_Visibility.md` v1.1 · BRD-04 §5.10–§5.12
(R-04.38 – R-04.45) · PPRD v1.5 Feature 29 · PADR v1.5 §10 "A model lease with holder and expiry vs
extending the `locked` flag" and "`Retry-After` and documented queue fields vs leaving backoff to the
client" · checkpoint decision C8 and the Feature-PRD decisions P-05, X-01, X-08 and the 2-hour default
TTL (`~/app/miDataworks/0xcc/docs/brd-decisions-2026-10-05.md`) · register T-66, T-84 – T-90
(`~/app/miDataworks/0xcc/docs/fprd-open-questions-2026-10-06.md`)
**Siblings:** 026 (`026_FTDD|Batch_API.md`, which consumes this lease API in process and extends the
request queue this design changes) · 025, 027, 028, 030 (FPRD level; they add no lease behaviour)
**Clients:** miStudio `034_FPRD` FR-18 (lease tools), FR-19 (refuse-load header); miDataworks 005
(lease, `Retry-After`, queue state), 007 and 009 (lease)

Code references are to miLLM at `7aa659c`, verified 2026-10-06. Clarifying rounds were waived; every
decision is in §14 with its source.

---

## 1. Executive Summary

Feature 29 makes miLLM safe to share between a long job and interactive users. A caller pins the
resident model with a lease; a busy server says when to retry; queue state and per-card memory become
readable contracts (FPRD §1).

| Area | Decision | Why |
|---|---|---|
| Lease storage | An in-process `LeaseRegistry` singleton keyed by model; no table, no migration | X-01: a restart ends every lease, so there is nothing to persist |
| Lease scope | Granted only on the resident, `LOADED`, not-unloading model; ends when that model stops being resident | T-85; at most one live lease follows from one resident model (X-08) |
| Enforcement point | One helper, `ModelService._refuse_if_leased`, called from `load_model`, `unload_model` and `load_model_and_wait` | R-04.39 names five entry points; routes cannot forget a check they never make |
| Proof of holding | A random lease ID returned once; only its SHA-256 digest is stored; it travels in the `X-miLLM-Lease` header, never in a path | FR-29.1.6; paths reach access logs, headers do not |
| Restart | `clear_leases_on_startup()` named function, called from `lifespan`, plus a self-healing read | X-01; memory `startup-reset-lists-hide-omissions` |
| Refuse-load policy | One helper `apply_load_policy` used by the three `/v1` routes before any auto-load | R-04.41; FR-29.4 |
| `Retry-After` | Set by the error builders from one policy function; a pure ASGI middleware adds a distinct fallback value and logs a warning when a `503` arrives without one | FR-29.6.2: one choke point and a detector for a site that bypassed it |
| Queue state | `RequestQueue` gains a holding counter and a window of slot-holding durations; `/api/health/detailed` gets a typed `inference` block | FR-29.7; the counter is also what Feature 26 reads |
| GPU read | `GET /api/health/gpus`, `nvidia-smi` in a worker thread, torch read only on cards miLLM placed a transformers model on | FR-29.8.4: never create a CUDA context to report one |
| Admin UI | Lease badge on the Models page and the loaded-model card, display only | R-04.42; T-86 |

## 2. System Architecture

```
                      ┌────────────────────────── miLLM API process ───────────────────────────┐
 client ─ /v1/* ────► │ route ─► apply_load_policy ─► ModelService.load_model_and_wait(lease_id)│
   X-miLLM-Lease      │                                   │                                     │
   X-miLLM-Load-Policy│                                   ▼                                     │
                      │                      _refuse_if_leased ◄──── LeaseRegistry (memory)     │
 client ─ /api/models/{id}/load|unload ─► ModelService.load_model / unload_model(lease_id)      │
 client ─ /api/models/{id}/lease ───────► ModelService.acquire|renew|release|get_lease          │
 Feature 26 runner ─ in process ────────► ModelService.acquire|renew|release|resolve_lease      │
                      │                                                                         │
 any 503 ─► error builders (retry_after_for) ─► RetryAfterMiddleware (fallback + warning) ─► out │
 /api/health/detailed ◄─ RequestQueue (holding, durations) + LeaseRegistry.current()            │
 /api/health/gpus ◄─ nvidia_smi.query_gpus + query_compute_apps + torch on touched cards         │
 lifespan startup ─► clear_leases_on_startup()                                                   │
                      └─────────────────────────────────────────────────────────────────────────┘
```

**Data flow of a guarded auto-load.** A `/v1` request names a model that is not resident. The route
parses the two headers. Under `refuse`, `apply_load_policy` answers `409 model_not_resident` (or
`503 model_loading` with `Retry-After`) and nothing loads. Under `auto`, the route calls
`load_model_and_wait(model.id, lease_id=...)`. That method calls `_refuse_if_leased` before its
`locked` check (`millm/services/model_service.py:1472-1479`), so the lease answer wins (FPRD D9).
`load_model` repeats the check immediately before it claims the load slot
(`model_service.py:832-837`), with no `await` between check and claim. The internal unload a load
performs (`model_service.py:884-892`) passes the same lease ID, so the holder's swap is not refused by
its own lease.

**Integration points with existing systems:**
- `ModelService` (`model_service.py:782`, `1194`, `1442`) — enforcement and the lease API.
- `LoadedModelState` (`millm/ml/model_loader.py:277`) — residency, read by the registry's
  self-healing check; records which cards torch touched.
- `RequestQueue` (`millm/services/request_queue.py:37`) — holding counter and duration window.
- Error builders (`millm/api/routes/openai/errors.py:26`, `150`, `280`, `305`), the registered
  handler `millm_error_handler` (`millm/api/exception_handlers.py:77`), the in-stream error event
  (`millm/services/inference_service.py:206`) and the readiness probe
  (`millm/api/routes/system/health.py:261-262`).
- `lifespan` (`millm/main.py:277`), beside `disarm_probes_on_startup` (`main.py:379`).
- Admin UI Models page (`admin-ui/src/pages/ModelsPage.tsx:292-296`) and `useModels`
  (`admin-ui/src/hooks/useModels.ts:29-35`).

**Feature 26 dependency (both directions).** 026's runner calls the lease API in process
(026 FTDD §7 "Lease handling"):
- **T-64:** a batch submitted with the caller's `X-miLLM-Lease` runs under that lease. It renews the
  lease by ID and never releases it. Needs `resolve_lease(lease_id)` (model, holder, `expires_at`)
  and `renew_lease(model_id, lease_id, ttl)`.
- **X-01 / T-66:** after a restart the registry is empty. A resumed batch calls
  `acquire_lease(model_id, holder="millm-batch:<id>", ...)`. `ModelNotResidentError` maps to its
  `waiting_reason=model_not_resident`, `ModelLeasedError` to `lease_unavailable`. It runs no row
  until it holds a lease.
- **FR-26.7.7, `POST /v1/batches/{id}/lease`:** attaches a lease the holder took again after a
  restart. 026 validates the header through `resolve_lease` and checks that the lease's `model_id` is
  the batch's model.
- 026 also reads `RequestQueue.holding_count` (introduced here) and supplies the batch backlog to
  `/api/health/detailed` through `register_backlog_provider` (§5.2). Feature 29 ships first (BRD-04
  RSK-09), so these names are fixed here and 026 builds on them.

## 3. Technical Stack

- Python 3.11 (`Dockerfile:7`, `python:3.11-slim`), FastAPI, Pydantic v2, structlog — as every miLLM
  module (PADR §5).
- `secrets.token_urlsafe` and `hashlib.sha256` from the standard library for lease IDs.
  `hmac.compare_digest` for comparison.
- `time.monotonic()` for every expiry decision; `datetime.now(timezone.utc)` for displayed times.
  A wall-clock step must not extend or end a lease.
- A pure ASGI middleware, not Starlette's `BaseHTTPMiddleware`, because the latter wraps streaming
  responses and the chat route streams Server-Sent Events (SSE).
- React 18, TypeScript, TanStack Query and Tailwind in the Admin UI, as today.
- No new dependency.

## 4. Data Design

**No schema change and no migration.** Leases live in process memory (X-01). `models.locked`
(`millm/db/models/model.py:152`) is untouched (C8).

**`LeaseRecord` (internal, frozen dataclass):**

| Field | Type | Notes |
|---|---|---|
| `digest` | `str` | SHA-256 hex of the lease ID; the registry key for lookup by ID |
| `lease_ref` | `str` | First 8 hex characters of `digest`; the only form that appears in logs |
| `model_id` | `int` | Registry key for lookup by model (X-08) |
| `model_name` | `str` | Copied at grant, for display |
| `holder`, `reason` | `str` | Stripped; 1–128 and 1–512 characters |
| `ttl_seconds` | `int` | The TTL most recently granted or renewed |
| `acquired_at`, `renewed_at`, `expires_at` | `datetime` (UTC) | Display |
| `expires_mono` | `float` | Monotonic deadline; the value every check uses |

**Ended-lease memory.** The registry keeps the last 64 ended leases, keyed by digest, with
`end_reason` (`released`, `expired`, `model_unloaded`, `restart`) and `ended_at`. It lets renew and
release answer `409 LEASE_EXPIRED` with a reason instead of `404` for an ID it has seen. It is lost on
restart, deliberately: after a restart every old ID is unknown, and the `404` message says that a
restart ends every lease.

**Validation (in the service, not the schema).** The request schema declares types only. Bounds are
checked in `ModelService.acquire_lease` and `renew_lease`, so a refusal is `400 INVALID_LEASE_REQUEST`
naming the field and the limit (FPRD FR-29.1.2). Pydantic bounds would answer `422` on management
routes (`exception_handlers.py:28-41`) without the limit in a readable message.

**Duration window.** `RequestQueue` keeps the last `QUEUE_DURATION_WINDOW` (50) slot-holding
durations in a `collections.deque`. Nothing persisted.

**Touched cards.** A module-level `set[int]` in `model_loader.py` of torch indices on which a
transformers model was placed. Never shrinks during the process, because a CUDA context outlives the
model.

## 5. API Design

### 5.1 Lease routes (management API, `/api/models`)

| Route | Body / headers | Success | Refusals |
|---|---|---|---|
| `POST /api/models/{model_id}/lease` | `{holder, reason, ttl_seconds?}` | `201`, `LeaseGrantResponse` (the only response carrying `lease_id`) | `400 INVALID_LEASE_REQUEST`; `404 MODEL_NOT_FOUND`; `409 MODEL_NOT_RESIDENT`; `409 MODEL_LEASED`; `503 MODEL_BUSY` (load or unload running) |
| `GET /api/models/{model_id}/lease` | — | `200`, `LeaseStatusResponse` or `null`, plus `last_ended` | `404 MODEL_NOT_FOUND` |
| `POST /api/models/{model_id}/lease/renew` | header `X-miLLM-Lease`; `{ttl_seconds?}` | `200`, `LeaseStatusResponse` | `400`; `404 LEASE_NOT_FOUND`; `409 LEASE_EXPIRED` |
| `DELETE /api/models/{model_id}/lease` | header `X-miLLM-Lease` | `200`, the ended `LeaseStatusResponse` | `404 LEASE_NOT_FOUND`; `409 LEASE_EXPIRED` |

- A lease ID that belongs to a different model than the path is `404 LEASE_NOT_FOUND`, so the route
  reveals nothing about other models.
- `ttl_seconds` omitted means `LEASE_DEFAULT_TTL_SECONDS` (7200) on grant and on renew.
- No approval is involved (P-05). miLLM has no sign-in (BRD-04 §3).
- Responses use the existing `ApiResponse` envelope. Errors use `millm_error_handler`'s management
  format (`exception_handlers.py:125-143`), with holder, reason and `expires_at` in `details`.
  No `error_messages.py` entry is added for `MODEL_LEASED`: `get_user_friendly_message` falls back to
  the exception's own message (`millm/core/error_messages.py:63-66`), which names the holder.

### 5.2 Changes to existing routes

| Route | Change |
|---|---|
| `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings` | Read `X-miLLM-Lease` and `X-miLLM-Load-Policy`. Call `apply_load_policy` before the auto-load block (`chat.py:143-146`, `completions.py:112-115`, `embeddings.py:80-83`); pass `lease_id` to `load_model_and_wait` |
| `POST /api/models/{id}/load`, `/unload` | Read `X-miLLM-Lease`; pass it to the service (`management/models.py:173`, `192`) |
| `GET /api/models`, `GET /api/models/{id}` | `ModelResponse.lease: LeaseSummary \| None` via a new `_with_lease`, called beside `_with_runtime` (`management/models.py:86`) |
| `GET /api/health/detailed` | Typed `inference: InferenceState`; new `lease: LeaseStatusResponse \| None` |
| `GET /api/health/gpus` (new, health router `health.py:22`) | Per-card memory, §5.4 |
| Every `503` | `Retry-After` header, §5.3 |

**`/v1` error codes.** `ERROR_STATUS_MAP` (`errors.py:76-111`) gains `MODEL_LEASED: (409,
"invalid_request_error")` and `MODEL_NOT_RESIDENT: (409, "invalid_request_error")`. The three routes
already catch `MiLLMError` after `ModelLockedError` and answer through `load_refused_error`
(`errors.py:280-302`), which reads the map. So `ModelLeasedError` must **not** subclass
`ModelLockedError`; a test pins the distinct code (`model_leased`, not `model_locked`).

**Header parsing.** `X-miLLM-Load-Policy` accepts `auto` or `refuse`, case-insensitive; anything else
is `400 invalid_parameter` with `param: "X-miLLM-Load-Policy"`. `X-miLLM-Lease` is passed through as an
opaque string; the service decides.

**`InferenceState` (typed, stable contract, FR-29.7):**

| Field | Type | Meaning |
|---|---|---|
| `backend`, `cbm_enabled`, `cbm_running` | as today | unchanged |
| `queue_pending` | `int` | interactive requests waiting **plus** holding a slot (unchanged meaning) |
| `queue_max_concurrent`, `queue_max_pending` | `int` | unchanged |
| `in_flight` | `int \| null` | slots held now, idle cache release included (T-90); `null` while `cbm_running` |
| `queue_waiting` | `int \| null` | `queue_pending − in_flight` for interactive work; `null` while `cbm_running` |
| `batch_backlog_rows` | `int \| null` | from the registered backlog provider; `null` when none is registered (before Feature 26) |
| `estimated_wait_seconds` | `float \| null` | §7.3; `null` with fewer than 3 samples or while `cbm_running` |
| `error` | `str \| null` | set, and the other new fields `null`, when the block could not be read |

The current `except Exception: pass` (`health.py:369-370`) is replaced: the block is always present.

### 5.3 `Retry-After` (T-88)

One policy function, `retry_after_for(code, details) -> int` in `millm/core/backpressure.py`:

| Code | Value |
|---|---|
| `QUEUE_FULL` | `ceil(estimated_wait_seconds)` clamped to `[1, RETRY_AFTER_MAX_S]` (60); `RETRY_AFTER_QUEUE_DEFAULT_S` (5) without an estimate |
| `MODEL_BUSY`, `MODEL_LOADING` | `RETRY_AFTER_UNLOAD_S` (5) when `details` carries `unloading` or `unloading_model_id` (as `inference_service.py:634` and `model_service.py:850-853` do); else `RETRY_AFTER_LOAD_S` (15) |
| `MODEL_NOT_LOADED` | `RETRY_AFTER_NOT_LOADED_S` (30); the message adds that retrying succeeds only after a model is loaded |
| `INSUFFICIENT_MEMORY` | `RETRY_AFTER_MEMORY_S` (30) |
| `HUB_UNAVAILABLE` | the breaker's remaining recovery time, `ceil(recovery_timeout − (now − last_failure_time))`, at least 1 (`millm/core/resilience.py:36`, `51`, `102`) |
| readiness probe | `RETRY_AFTER_READINESS_S` (5) |
| any other `503` | `RETRY_AFTER_FALLBACK_S` (10), set only by the middleware, which also logs `retry_after_defaulted` |

**Where it is applied.** `create_openai_error` gains `retry_after: int | None`, written as a header.
`millm_error_handler` computes it for every response it sends with status 503, on both API families.
`model_not_loaded_error`, `model_busy_error` and `load_refused_error` pass it. `health.py:261-262`
adds the header to the readiness `JSONResponse`. `_stream_error_event` adds `retry_after` to the error
object when the code maps to 503 on `/v1` (FPRD FR-29.6.5). `openai_exception_handler`
(`errors.py:121`) is not registered (comment at `errors.py:97-101`; `main.py:542-544`) and is left
alone.

**The fallback value is distinct on purpose.** The parametrised test asserts each code's own value
and asserts that `retry_after_defaulted` was never logged. A builder that forgot the header gets 10
seconds and a warning, and fails both assertions.

### 5.4 GPU memory (T-89)

`GET /api/health/gpus` → `GpuMemoryResponse{read_at, cards: [...], reason}`. Per card:

| Field | Source |
|---|---|
| `smi_index`, `uuid`, `name`, `total_mb`, `used_mb`, `free_mb` | `nvidia_smi.query_gpus()` (`millm/ml/nvidia_smi.py:88`) |
| `torch_index` | matched by UUID as `list_gpus` does (`millm/ml/gpu_placement.py:160-183`); `null` when torch cannot see the card |
| `torch_measured` | `true` only when `torch.cuda.is_initialized()` and the torch index is in the touched set |
| `millm_allocated_mb`, `millm_reserved_mb` | `torch.cuda.memory_allocated(i)`, `memory_reserved(i)`; `null` when not measured |
| `engine_memory` | `"not_measured_by_torch"` on cards a resident GGUF model occupies; else `null` |
| `processes`, `processes_reason` | new `nvidia_smi.query_compute_apps()`: `[{pid, used_mb}]` for this card, or `null` with a reason |

**Acceptance 16 comparison (T-89).** torch's reserved figure excludes the CUDA context, which
`nvidia-smi` counts. The acceptance therefore compares, on each card miLLM touched after an unload,
`millm_reserved_mb + cuda_context_mb` with nvidia-smi's per-process `used_memory` for miLLM's
process, within 256 MiB. `cuda_context_mb` is measured once on the node in FTASKS spike 0.2 and
recorded in the review. If the pod cannot see its own process in `--query-compute-apps`, the
per-process figure is read on the node, as BRD-04 acceptance 16 already says ("matches `nvidia-smi`").

**Why the touched set and not "ask torch about every card".** The design must not depend on whether
an allocator-statistics call on an unused device initialises a context there. Reading only cards
miLLM placed weights on makes the guarantee structural (FPRD FR-29.8.4).

### 5.5 Security and performance principles

- The lease ID is never logged, persisted, echoed in a read, or put in a URL path.
- Every lease check is a dictionary lookup under a `threading.Lock` held for microseconds, with no
  `await` inside.
- The GPU endpoint runs `nvidia-smi` in `asyncio.to_thread` (`nvidia_smi.py:66-85` has a 5-second
  timeout) and takes no request-queue slot.

## 6. Component Architecture

```
millm/services/model_lease.py     LeaseRecord, LeaseGrant, EndedLease, LeaseRegistry (singleton),
                                  clear_leases_on_startup()
millm/services/model_service.py   acquire_lease, renew_lease, release_lease, get_lease,
                                  resolve_lease, _refuse_if_leased; lease_id on load/unload/wait
millm/core/errors.py              ModelLeasedError, ModelNotResidentError, LeaseNotFoundError,
                                  LeaseExpiredError, InvalidLeaseRequestError
millm/core/backpressure.py        retry_after_for, estimate_wait_seconds, register_backlog_provider,
                                  backlog_rows
millm/api/retry_after.py          RetryAfterMiddleware (pure ASGI)
millm/api/routes/openai/load_policy.py   parse_load_policy, apply_load_policy
millm/api/routes/management/models.py    lease routes, header pass-through, _with_lease
millm/api/schemas/lease.py        request/response models; LeaseSummary
millm/api/routes/system/health.py InferenceState, lease field, /gpus route
millm/services/gpu_memory.py      read_gpu_memory()
millm/ml/nvidia_smi.py            query_compute_apps()
millm/ml/model_loader.py          touched-card set, recorded in LoadedModelState.set()
millm/services/request_queue.py   holding_count, record of durations, median_hold_seconds
millm/main.py                     clear_leases_on_startup() call; RetryAfterMiddleware registration
admin-ui: types/api.ts (LeaseSummary), components/models/LeaseBadge.tsx, ModelsPage.tsx,
          ModelDetailsModal.tsx, LoadedModelCard.tsx, hooks/useModels.ts
```

**Separation of concerns.** The registry knows nothing about models beyond IDs and a residency
callback. `ModelService` owns the rules (residency, load in progress, the lift). Routes do headers and
HTTP. The backpressure module owns numbers, not responses.

**Reusability.** `_refuse_if_leased` is the only enforcement function. `apply_load_policy` is the only
refuse-policy function. `retry_after_for` is the only value function. Each has exactly the callers
listed in §2, asserted by AST call tests (FTID §8).

## 7. State Management

### 7.1 Lease lifecycle

`granted → (renewed)* → ended{released | expired | model_unloaded | restart}`.

- **Grant** requires: the model is the loader's resident model, the row is `LOADED`, the loader is not
  unloading, no load is in progress (`_loading_model_id is None`), and no live lease on that model.
- **Expiry** is lazy. Every read compares `time.monotonic()` with `expires_mono`. An expired record is
  moved to ended memory on the read that notices it, and logged `lease_expired` once.
- **Self-healing read (FPRD FR-29.1.9).** `current(model_id)` also ends a lease whose model is no
  longer the loader's resident model, reason `model_unloaded`. This covers a forced unload after a
  timeout (`model_service.py:1281-1288`) and any path that empties the loader without passing through
  `unload_model`'s success branch.
- **Unload success** ends the lease explicitly, beside the auto-unlock (`model_service.py:1306-1312`).
- **Restart.** The registry starts empty in a new process. `clear_leases_on_startup()` still runs
  from `lifespan` and logs the count it cleared. In production the count is always 0. The call exists
  so the reconciliation is explicit, testable, and survives a future change that makes leases
  durable. The test fills the registry, runs the function for real and asserts it is empty; a second
  test asserts `lifespan` calls it.

### 7.2 The lift

`_refuse_if_leased(operation, target_model_id, lease_id)`:
1. `lease = registry.current(resident_model_id)`. No lease → return.
2. `lease_id` given and `sha256(lease_id) == lease.digest` (constant-time) → return.
3. Otherwise raise `ModelLeasedError(holder, reason, expires_at, leased_model, operation, target)`.

A wrong lease ID on a request that needed no lift is logged `lease_header_unmatched` at warning and
ignored (FPRD FR-29.3.3).

### 7.3 Queue state and the estimate

`RequestQueue.acquire` increments `_holding` after the semaphore is taken and decrements it in the
`finally` before release, alongside the existing `_pending` handling
(`request_queue.py:147-169`). The time between those two points is appended to the duration window.

```
estimated_wait_seconds = median(window) × (queue_waiting + in_flight) / max_concurrent
```

`null` when the window holds fewer than 3 samples or CBM is running. The batch backlog is not added:
interactive requests go first at every chunk boundary (026 FTDD §7, constraint 2). The formula is
printed in the manual.

### 7.4 Side effects

Lease events are logged only. There is no Socket.IO event; the Admin UI polls (§9). Nothing writes to
the database.

### 7.5 Admin UI state

`useModels` keeps its query. Its `refetchInterval` (`useModels.ts:29-35`) returns 10,000 ms when any
model carries a lease, in addition to the existing 2,000 ms during downloads and loads. The badge
computes "expires in" from `expires_at` on a 1-second local timer, and hides itself when the time is
past, without waiting for the next poll (FPRD FR-29.5.4).

## 8. Security Considerations

- **No authentication** (BRD-04 §3, decision 7). The lease ID is a bearer secret with 192 bits of
  entropy (`token_urlsafe(24)`).
- **No approval** to take or release a lease (P-05). Loading stays gated on miStudio's side (C7).
- **Never disclosed after grant.** Stored as a digest. Logs carry `lease_ref` (8 hex characters of the
  digest), never the ID. Header, not path, so uvicorn's access log never records it.
- **Information hiding.** A lease ID used against another model's route answers `404`, like an unknown
  one.
- **Input bounds.** `holder` ≤ 128 characters, `reason` ≤ 512, both stripped and non-empty; TTL
  1–7200.
- **Denial of service.** A lease can block model changes for at most 2 hours without renewal (RSK-04).
  There is no force-release (T-86). The holder and expiry are visible to the operator.

## 9. Performance & Scalability

- A lease check is a dictionary lookup; the `/v1` path adds no database round trip (FPRD §8).
- `/api/health/detailed` gains no `nvidia-smi` call; the GPU read is its own endpoint.
- The duration window is 50 floats; the median is computed on read.
- The Admin UI polls models every 10 seconds only while a lease exists.
- One process, one event loop (`Dockerfile:124`, uvicorn with no `--workers`), so an in-memory
  registry is the authoritative one. **If miLLM ever runs several workers, the registry must move to
  shared storage** — recorded in §12.

## 10. Testing Strategy

- **Unit, registry:** grant, conflict, renew (expiry from now), release, lazy expiry with an injected
  monotonic clock, ended-lease reasons, self-healing on a residency change, `clear`.
- **Unit, service:** residency, load in progress, unloading, bounds, the lift, `locked` and lease
  together (lease first).
- **Enforcement, through the real service:** each of the five entry points refuses under a foreign
  lease and proceeds with the right header, asserting the payload (holder, expiry) and that the load
  call count stays zero on refusal.
- **Reachability:** lease routes and `/api/health/gpus` in `app.openapi()["paths"]` (memory: `app.routes`
  is not a route list); the middleware present in the built app's middleware stack.
- **Call assertions by AST** (memory `source-scrape-guards-i-keep-writing`): `_refuse_if_leased` called
  in `load_model`, `unload_model`, `load_model_and_wait`; `apply_load_policy` called in each of the
  three routes before `load_model_and_wait`; `clear_leases_on_startup` called in `lifespan`.
- **Startup reconciliation:** the named function run for real against a filled registry, plus the AST
  call test. Shape copied from `tests/unit/services/test_probe_survives_no_restart.py:187-193`.
- **`Retry-After`:** parametrised over every producible code; the `retry_after_defaulted` log never
  fires for them; a synthetic bare-503 route gets the fallback.
- **Queue state:** with a fake slow operation, one holding and two waiting give `in_flight` 1 and
  `queue_waiting` 2; CBM running gives `null`s.
- **GPU:** parser fixtures; `nvidia-smi` absent; untouched card never calls torch (a mock that fails
  on `memory_reserved` for that index); GGUF resident card labelled.
- **Admin UI (Vitest):** badge text, expiry hiding, no lease ID in the DOM.
- **Fixtures must disagree with the defect:** two different holders; a requested model different from
  the resident one; a clock that moves past expiry; two cards, one touched and one not.
- **Mutation controls:** one per lease-enforcement path and per wiring line (FTID §8.4). Each is run,
  shown red, restored, and the restore verified by `git diff` before the next.

## 11. Deployment & DevOps

- **No migration, no new container, no Kubernetes manifest change.** New settings have code defaults
  and are documented in `.env.example` and `manual/docs/reference/configuration.md`.
- **Settings:** `LEASE_DEFAULT_TTL_SECONDS` 7200, `LEASE_MAX_TTL_SECONDS` 7200,
  `LEASE_HOLDER_MAX_CHARS` 128, `LEASE_REASON_MAX_CHARS` 512, `LEASE_ENDED_MEMORY` 64,
  `QUEUE_DURATION_WINDOW` 50, and the eight `RETRY_AFTER_*` values in §5.3.
- **Monitoring:** structlog events `lease_granted`, `lease_renewed`, `lease_released`,
  `lease_expired`, `lease_ended`, `lease_refused`, `lease_header_unmatched`, `leases_cleared_on_startup`,
  `retry_after_defaulted`.
- **Contract:** `docs/mcp-contract.md` gains the lease routes, headers, codes, health fields and the
  GPU route in the next additive version, so miStudio 034 FR-18 can build its tools. The public mirror
  strips `docs/` except `schemas/`, so no test may require that file on the mirror (see the miStudio
  2026-10-02 lesson recorded in its `CLAUDE.md`).
- **Rollout:** a deploy restarts the pod, which ends every lease (X-01). Holders see `404
  LEASE_NOT_FOUND` on renew, with the message naming the restart, and re-acquire.
- **Rollback:** revert the image. No data to migrate back.

## 12. Risk Assessment

| Risk | Mitigation |
|---|---|
| A new load path bypasses the lease | The check lives in the service; AST call tests on three methods; mutation controls |
| Check-then-claim race between a lease grant and a load | Both are synchronous between their check and their claim; grant refuses while `_loading_model_id` is set |
| A forced unload leaves a lease on a model that is gone | Self-healing read ends it (§7.1) |
| A future multi-worker deployment splits the registry | Recorded here and in FTID §9; the registry is one module to replace |
| The fallback hides a forgotten builder | Distinct value plus a warning, both asserted absent for known codes |
| `in_flight` counts the idle cache release | Documented (T-90); it does block a new request |
| Estimate mistaken for a promise | Named an estimate; `null` without data |
| `nvidia-smi --query-compute-apps` sees no PIDs in the pod | `processes: null` with a reason; acceptance 16 reads the node (T-89) |
| 026 assumes names this design changes | Names in §2 are fixed here; 026's FTASKS gate on this feature |

**Alternatives considered.**
- *Widen `locked` into a lease* — rejected by PADR §10 and C8.
- *Persist leases in PostgreSQL* — rejected: X-01 ends them at restart, so a table adds a
  reconciliation and a migration for no observable gain.
- *Lease check in each route* — rejected: five sites today, more later.
- *`Retry-After` computed only in the middleware* — rejected: the middleware cannot see the error
  code without buffering the body; it is the safety net, not the policy.
- *Ask torch about every card* — rejected: risks creating the context the read exists to report.

**Complexity:** medium. The load path's race history (`model_service.py:824-837`) is the main risk.

## 13. Development Phases

| Phase | Content | Depends on | Milestone |
|---|---|---|---|
| 0 | Spikes: re-verify lines at HEAD; in-pod `--query-compute-apps` and CUDA context size | — | Spike record |
| 1 | Errors, settings, error map, `backpressure.py` value policy | 0 | Unit green |
| 2 | `LeaseRegistry`, service lease API | 1 | Registry and service tests green |
| 3 | Enforcement in `load_model`, `unload_model`, `load_model_and_wait`; unload ends lease | 2 | Enforcement tests + controls red-on-mutation |
| 4 | Startup clear and self-healing read | 2 | Reconciliation tests + control |
| 5 | Lease routes, headers on `/v1` and management, refuse policy, `ModelResponse.lease` | 3 | Route and reachability tests |
| 6 | `Retry-After` builders, middleware, in-stream field | 1 | Parametrised 503 test |
| 7 | Queue counters, typed health block, lease field, backlog provider | 2, 6 | Contract test |
| 8 | GPU endpoint, touched set, compute apps | 0 | GPU tests |
| 9 | Admin UI badge and polling | 5 | Vitest green |
| 10 | Manual, error codes, configuration, MCP contract | 5–8 | Docs build |
| 11 | Feature acceptance, hardware, mutation record | all | Review record |

Estimate: backend 3–4 days, Admin UI 1 day, hardware half a day.

## 14. Decisions from Clarifying Questions

Clarifying rounds were waived. Each decision cites its source.

| # | Question | Decision | Source |
|---|---|---|---|
| TD1 | Lease beside or instead of `locked`? | Beside; `locked` keeps working unchanged | C8 |
| TD2 | Where do leases live? | Process memory; no table | X-01 |
| TD3 | One lease per server or per model? | Registry keyed by model; at most one live because only the resident model is leasable | X-08; T-85 |
| TD4 | Approval to take or release? | None | P-05 |
| TD5 | Default and maximum TTL? | 7200 seconds each, renewable | Feature-PRD decisions "Default TTL"; checkpoint technical default |
| TD6 | Lease on a non-resident model; across a swap? | Refused; ends with residency | T-85 |
| TD7 | miStudio workers take the lease? | No | T-84 |
| TD8 | Force-release? | No | T-86 |
| TD9 | `/v1/models` under a lease? | Unchanged | T-87 |
| TD10 | Restart and a resumed batch? | Registry cleared; batch re-acquires or waits | X-01; T-66; 026 FTDD §7 |
| TD11 | A batch under its caller's lease? | Supported: `resolve_lease` and `renew_lease` by ID; the batch never releases it | T-64; 026 FR-26.7.5, FR-26.7.7 |
| TD12 | Where is the lease ID carried? | `X-miLLM-Lease` header on every route, including renew and release | FPRD FR-29.1.6; access logs record paths |
| TD13 | Lease route shapes | §5.1 | FPRD FR-29.1.7 left them to this design; miStudio 034 FR-18 waits on them |
| TD14 | TTL validation location | Service, `400 INVALID_LEASE_REQUEST` | FPRD FR-29.1.2; management validation is `422` (`exception_handlers.py:28-41`) |
| TD15 | `Retry-After` values | §5.3 | T-88 |
| TD16 | `Retry-After` mechanism | Builders set it; ASGI middleware falls back with a distinct value and a warning | FPRD FR-29.6.2 |
| TD17 | Body `retry_after` on HTTP errors | No; in-stream only | FPRD v1.1 FR-29.6.4, FR-29.6.5 |
| TD18 | `in_flight` under CBM; idle release | `null` under CBM; idle release counts | T-90; `cbm_backend.py:29` exposes no active count |
| TD19 | Backlog in the estimate | Not added | 026 FTDD §7 constraint 2 |
| TD20 | GPU route path | `GET /api/health/gpus` on the existing health router | FPRD FR-29.8.1 left it to this design |
| TD21 | Acceptance 16 comparison | reserved + measured context vs per-process `used_memory`, 256 MiB | T-89 |
| TD22 | Which cards torch is read on | Cards a transformers model was placed on | FPRD FR-29.8.4 |
| TD23 | Socket event for lease changes | None; poll | No consumer needs push; R-04.42 asks for visibility |
| TD24 | `MODEL_LOADING` producer | The refuse policy's "model loading now" answer | FPRD FR-29.4.3, FR-29.6.6 |

**Open items:** none for the operator. One technical unknown is closed by FTASKS spike 0.2: whether
the pod sees its own process in `nvidia-smi --query-compute-apps`, and the CUDA context size used in
acceptance 16.
