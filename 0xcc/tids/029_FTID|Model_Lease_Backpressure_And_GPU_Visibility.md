# Technical Implementation: Model Lease, Backpressure and GPU Visibility

## miLLM Feature 29

**Version:** 1.0 · **Created:** 2026-10-06 · **Status:** Planned
**Inputs:** `029_FPRD|Model_Lease_Backpressure_And_GPU_Visibility.md` v1.1 ·
`029_FTDD|Model_Lease_Backpressure_And_GPU_Visibility.md` v1.0 · PADR v1.5 §5 and §10
**Consumer contract:** `026_FTDD|Batch_API.md` §2 and §7 (in-process lease API, `holding_count`,
backlog provider)

Code references are to miLLM at `7aa659c`, verified 2026-10-06. **Re-verify every line before
editing** (FTASKS 0.1): sibling features 025–028 and 030 edit the same route files. Clarifying rounds
were waived; decisions are in §15.

---

## 1. Implementation Overview

Four changes, each with one owner function:

| Change | Owner | Callers |
|---|---|---|
| Lease rules | `ModelService._refuse_if_leased` | `load_model`, `unload_model`, `load_model_and_wait` |
| Refuse-load policy | `load_policy.apply_load_policy` | the three `/v1` routes |
| `Retry-After` value | `backpressure.retry_after_for` | error builders, handler, readiness, stream event, middleware |
| Startup reconciliation | `model_lease.clear_leases_on_startup` | `lifespan` |

**Principles.**
- **One enforcement function, asserted by its calls.** A guard spread over routes is a guard a new
  route forgets. Tests walk the AST for the **call**, never search source text (memory
  `source-scrape-guards-i-keep-writing`).
- **No `await` between a check and its claim.** The load path's race history
  (`millm/services/model_service.py:824-837`) is why.
- **Unmeasured is `null`, never `0`.** Backlog before Feature 26, `in_flight` under continuous
  batching (CBM), torch memory on an untouched card.
- **The lease ID exists in exactly two places:** the grant response and the caller's header. The
  registry holds its digest.

**Integration points.** `ModelService` (`model_service.py:782`, `1194`, `1442`), `LoadedModelState`
(`millm/ml/model_loader.py:277`), `RequestQueue` (`millm/services/request_queue.py:37`), the error
builders (`millm/api/routes/openai/errors.py`), `millm_error_handler`
(`millm/api/exception_handlers.py:77`), `_stream_error_event`
(`millm/services/inference_service.py:206`), the health router (`millm/api/routes/system/health.py:22`),
`lifespan` (`millm/main.py:277`) and `create_app` (`main.py:529`, `542-547`).

## 2. File Structure and Organization

**New files:**

| File | Contents |
|---|---|
| `millm/services/model_lease.py` | `LeaseRecord`, `LeaseGrant`, `EndedLease`, `LeaseRegistry`, `clear_leases_on_startup` |
| `millm/core/backpressure.py` | `retry_after_for`, `estimate_wait_seconds`, `register_backlog_provider`, `backlog_rows` |
| `millm/api/retry_after.py` | `RetryAfterMiddleware` |
| `millm/api/routes/openai/load_policy.py` | `parse_load_policy`, `apply_load_policy` |
| `millm/api/schemas/lease.py` | `LeaseCreateRequest`, `LeaseRenewRequest`, `LeaseGrantResponse`, `LeaseStatusResponse`, `LeaseSummary`, `EndedLeaseResponse` |
| `millm/services/gpu_memory.py` | `read_gpu_memory()` |
| `admin-ui/src/components/models/LeaseBadge.tsx` | badge |
| `tests/unit/services/test_model_lease.py` | registry |
| `tests/unit/services/test_lease_enforcement.py` | service rules, enforcement, AST call tests |
| `tests/unit/api/test_lease_routes.py` | routes, headers, reachability |
| `tests/unit/api/test_load_policy.py` | refuse policy on three routes |
| `tests/unit/api/test_retry_after.py` | every 503, middleware fallback |
| `tests/unit/api/test_health_queue_state.py` | typed block, lease field, contract field set |
| `tests/unit/api/test_gpu_memory_route.py` and `tests/unit/services/test_gpu_memory.py` | GPU read |
| `tests/unit/services/test_request_queue_holding.py` | holding counter, durations |
| `admin-ui/src/components/models/__tests__/LeaseBadge.test.tsx` | badge |

**Modified files:** `millm/services/model_service.py`, `millm/core/errors.py`,
`millm/core/config.py`, `.env.example`, `millm/api/routes/openai/errors.py`,
`millm/api/exception_handlers.py`, `millm/api/routes/openai/{chat,completions,embeddings}.py`,
`millm/api/routes/management/models.py`, `millm/api/schemas/model.py`,
`millm/api/routes/system/health.py`, `millm/services/request_queue.py`,
`millm/services/inference_service.py` (stream event only), `millm/ml/nvidia_smi.py`,
`millm/ml/model_loader.py`, `millm/main.py`, `tests/unit/test_startup_reset.py`,
`admin-ui/src/types/api.ts`, `admin-ui/src/pages/ModelsPage.tsx`,
`admin-ui/src/components/models/{ModelDetailsModal,LoadedModelCard,index}.tsx|ts`,
`admin-ui/src/hooks/useModels.ts`, `docs/mcp-contract.md`, `manual/docs/api/models.md`,
`manual/docs/api/management-api.md`, `manual/docs/reference/error-codes.md`,
`manual/docs/reference/configuration.md`, `manual/docs/features/model-management.md`.

**Import pattern.** `model_lease.py` imports nothing from `model_service` (no cycle); it takes a
`resident_model_id: Callable[[], int | None]` at construction. `backpressure.py` imports settings at
module level and the inference service lazily inside functions, as `unload_model` already does
(`model_service.py:1238-1240`).

## 3. Component Implementation Hints

### 3.1 `LeaseRegistry`

```python
class LeaseRegistry:
    """One live lease per model (X-08). Process memory only (X-01). Holds DIGESTS, never IDs."""
    _instance = None                     # singleton, like LoadedModelState (model_loader.py:288-295)

    def grant(self, model_id, model_name, holder, reason, ttl_s) -> LeaseGrant
    def renew(self, model_id, lease_id, ttl_s) -> LeaseRecord     # LeaseNotFoundError / LeaseExpiredError
    def release(self, model_id, lease_id) -> EndedLease
    def resolve(self, lease_id) -> LeaseRecord | None             # live only; for Feature 26
    def current(self, model_id) -> LeaseRecord | None             # lazy expiry + residency self-heal
    def matches(self, record, lease_id) -> bool                   # hmac.compare_digest on digests
    def end_for_model(self, model_id, reason) -> EndedLease | None
    def last_ended(self, model_id) -> EndedLease | None
    def clear(self, reason="restart") -> int
```

- A `threading.Lock` guards every method; no method awaits.
- The clock is injectable (`monotonic=time.monotonic`, `now=lambda: datetime.now(timezone.utc)`), so
  tests move time without sleeping.
- `current()` ends a record whose `expires_mono <= monotonic()` (reason `expired`) or whose
  `model_id != resident_model_id()` (reason `model_unloaded`), logging once. Expiry is `<=`: at the
  deadline the lease is gone (mutation control M5).

### 3.2 `ModelService` additions

```python
async def acquire_lease(self, model_id, holder, reason, ttl_seconds=None) -> LeaseGrant
async def renew_lease(self, model_id, lease_id, ttl_seconds=None) -> LeaseRecord
async def release_lease(self, model_id, lease_id) -> EndedLease
async def get_lease(self, model_id) -> tuple[LeaseRecord | None, EndedLease | None]
def resolve_lease(self, lease_id) -> LeaseRecord | None
def _refuse_if_leased(self, operation: str, target_model_id: int, lease_id: str | None) -> None
```

`acquire_lease` order: validate inputs (`InvalidLeaseRequestError`) → `get_model` (404) → if
`self._loading_model_id is not None` or `self.loader.is_unloading` → `ModelBusyError` with
`unloading` detail as appropriate → if `self.loader.loaded_model_id != model_id` or the row is not
`LOADED` → `ModelNotResidentError` naming the resident model → `registry.grant(...)`. No `await`
after the residency checks and before `grant`. `get_model` is the only await, and it comes first.

### 3.3 Enforcement call sites (exact placement)

| Method | Insert | Before |
|---|---|---|
| `load_model` | `self._refuse_if_leased("load", model_id, lease_id)` | the slot check `if self._loading_model_id is not None` (`model_service.py:832`) — no await in between |
| `load_model` internal unload | `await self.unload_model(current_model_id, lease_id=lease_id)` | replaces `model_service.py:892` |
| `unload_model` | `self._refuse_if_leased("unload", model_id, lease_id)` | the `is_unloading` check (`model_service.py:1218`), after the not-loaded check (`1212-1216`) |
| `unload_model` success | `self._leases.end_for_model(model_id, "model_unloaded")` | the auto-unlock update (`model_service.py:1306-1312`) |
| `load_model_and_wait` | `self._refuse_if_leased("auto_load", model_id, lease_id)`, and pass `lease_id` to `load_model` (`model_service.py:1483`) | the `locked` check (`model_service.py:1472-1479`), after the early return (`1468-1470`) |

Signatures gain a keyword-only `lease_id: str | None = None`, so every existing caller keeps working
(`main.py:87` startup auto-load passes none; no lease can exist that early).

### 3.4 `apply_load_policy`

```python
def parse_load_policy(raw: str | None) -> Literal["auto", "refuse"]   # 400 otherwise
async def apply_load_policy(policy, model_row, inference, service) -> JSONResponse | None
```

Returns `None` when the request may continue: policy `auto`, or the model is resident. Under
`refuse` with the model not resident: if `model_row.status == LOADING` or
`service._loading_model_id == model_row.id` → `503 model_loading` with `Retry-After`; else
`409 model_not_resident` naming requested and resident models, plus the resident model's lease
summary when one exists (FPRD FR-29.4.4). Called in each route after the existing pre-load refusals
(not found, embedding-only, GGUF) and immediately before the auto-load block (`chat.py:143-146`,
`completions.py:112-115`, `embeddings.py:80-83`).

### 3.5 `RetryAfterMiddleware`

```python
class RetryAfterMiddleware:
    def __init__(self, app): self.app = app
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http": return await self.app(scope, receive, send)
        async def wrapped(message):
            if message["type"] == "http.response.start" and message["status"] == 503:
                headers = list(message.get("headers", []))
                if not any(k.lower() == b"retry-after" for k, _ in headers):
                    headers.append((b"retry-after", str(settings.RETRY_AFTER_FALLBACK_S).encode()))
                    logger.warning("retry_after_defaulted", path=scope.get("path"))
                    message = {**message, "headers": headers}
            await send(message)
        await self.app(scope, receive, wrapped)
```

Registered in `create_app` with `app.add_middleware(RetryAfterMiddleware)` next to CORS
(`main.py:529`). Starlette runs the exception handlers inside the middleware stack, so handler
responses pass through it.

### 3.6 `RequestQueue`

Add `self._holding = 0` and `self._durations = deque(maxlen=settings.QUEUE_DURATION_WINDOW)`. In
`acquire`, after `acquired = True` (`request_queue.py:154`): `self._holding += 1; started =
time.monotonic()`. In the `finally` (`request_queue.py:159-169`), when `acquired`: `self._holding -=
1; self._durations.append(time.monotonic() - started)` before releasing the semaphore. Properties
`holding_count` and `median_hold_seconds() -> float | None` (`None` below 3 samples). `pending_count`
is unchanged. Feature 26 later adds background counters beside these (026 FTDD §7).

### 3.7 Touched cards

In `model_loader.py`, module level: `_TORCH_TOUCHED_INDICES: set[int] = set()` and
`def torch_touched_indices() -> frozenset[int]`. In `LoadedModelState.set()` (`model_loader.py:333`),
when `model.engine == ENGINE_TRANSFORMERS` (`model_loader.py:146`), add `model.gpu_indices`. Never
removed.

## 4. Database Implementation Approach

**N/A — no schema change.** Leases are in memory (X-01). `models.locked` is untouched (C8).
`STALE_STATE_RESETS` (`main.py:134`) is a list of SQL statements and gets no entry; the lease
reconciliation is the named function `clear_leases_on_startup`, called from `lifespan` beside
`disarm_probes_on_startup` (`main.py:379`), in the same shape and for the same reason that function's
docstring records (`main.py:209-230`).

## 5. API Implementation Strategy

**Lease routes** go in `millm/api/routes/management/models.py`, on the existing router, so no new
`include_router` line is needed in `millm/api/routes/__init__.py:35`. Order them before any route with
a broader path match. Each takes `x_millm_lease: Annotated[str | None, Header(alias="X-miLLM-Lease")]`
where needed.

```python
@router.post("/{model_id}/lease", status_code=201, response_model=ApiResponse[LeaseGrantResponse])
async def acquire_lease(model_id: ModelId, body: LeaseCreateRequest, service: ModelServiceDep): ...
@router.get("/{model_id}/lease", response_model=ApiResponse[LeaseStatusEnvelope])
@router.post("/{model_id}/lease/renew", response_model=ApiResponse[LeaseStatusResponse])
@router.delete("/{model_id}/lease", response_model=ApiResponse[EndedLeaseResponse])
```

`LeaseCreateRequest`: `holder: str`, `reason: str`, `ttl_seconds: int | None = None`, `extra="forbid"`.
No `Field` bounds; the service validates and raises `InvalidLeaseRequestError` (`400`).

**Header pass-through on load and unload** (`management/models.py:159-174`, `183-193`): add the header
parameter and `lease_id=x_millm_lease`.

**`/v1` routes.** Add both header parameters to the three route functions (`chat.py:101`,
`completions.py:51`, `embeddings.py:47`). Pass `lease_id` to `load_model_and_wait`. `ModelLeasedError`
reaches the existing `except MiLLMError as exc: return load_refused_error(request.model, exc)`; no new
`except` clause is needed, and none may catch it as `ModelLockedError`.

**Error classes** (`millm/core/errors.py`, beside `ModelLockedError` at `183-187`):

```python
class ModelLeasedError(MiLLMError):        code = "MODEL_LEASED";          status_code = 409
class ModelNotResidentError(MiLLMError):   code = "MODEL_NOT_RESIDENT";    status_code = 409
class LeaseNotFoundError(MiLLMError):      code = "LEASE_NOT_FOUND";       status_code = 404
class LeaseExpiredError(MiLLMError):       code = "LEASE_EXPIRED";         status_code = 409
class InvalidLeaseRequestError(MiLLMError):code = "INVALID_LEASE_REQUEST"; status_code = 400
```

`ModelLeasedError.__init__(record, operation, target_model_id)` builds the message
`"Model '<name>' is leased by '<holder>' until <expires_at> (<reason>); <operation> of model <target> refused."`
and `details` `{holder, reason, expires_at, leased_model_id, leased_model_name, operation,
target_model_id}`. Never the digest or `lease_ref`.

**`ERROR_STATUS_MAP`** (`errors.py:76-111`): add `MODEL_LEASED`, `MODEL_NOT_RESIDENT` (both
`(409, "invalid_request_error")`). Add a builder `model_not_resident_error(requested, resident,
lease)` beside `model_locked_error` (`errors.py:234-243`).

**`Retry-After` in builders.** `create_openai_error(..., retry_after: int | None = None)` returns
`JSONResponse(..., headers={"Retry-After": str(retry_after)} if retry_after else None)`.
`model_not_loaded_error` (`errors.py:150-157`), `model_busy_error` (`305-319`) and
`load_refused_error` (`280-302`, when the mapped status is 503) pass
`retry_after_for(code, details)`. In `millm_error_handler` (`exception_handlers.py:77`), compute the
status as today; if it is 503, pass `retry_after` on the OpenAI branch and add the header on the
management `JSONResponse` (`exception_handlers.py:140-143`).

## 6. Frontend Implementation Approach

- **Type** (`admin-ui/src/types/api.ts`, `ModelInfo` at line 19): `lease?: LeaseSummary | null` with
  `LeaseSummary = { holder: string; reason: string; acquired_at: string; expires_at: string;
  seconds_remaining: number }`. No `lease_id` field exists in any frontend type.
- **`LeaseBadge`** (`components/models/LeaseBadge.tsx`): props `{ lease: LeaseSummary }`. Renders a
  `lucide-react` `KeyRound` icon and text "Leased by {holder} · expires in {h}h {m}m", tooltip with
  reason and the exact local expiry time. A `useEffect` 1-second interval recomputes the remainder
  from `expires_at` and returns `null` once it is past. `aria-label` states holder and expiry.
  Amber (`text-amber-400`) so it never reads as the yellow steering lock (`ModelsPage.tsx:293-296`).
- **Placement:** `ModelsPage.tsx` beside the lock indicator (`292-296`); `ModelDetailsModal.tsx` a
  "Lease" row; `LoadedModelCard.tsx` under the model name. Export from `components/models/index.ts`.
- **Polling** (`useModels.ts:29-35`): return 10,000 when any model has `lease`, keep 2,000 during
  downloads and loads, else `false`.
- **No mutation**, no lease action (T-86).

## 7. Business Logic Implementation Hints

- **Grant precondition order** is in §3.2; refusals are ordered so the most specific reason wins.
- **The lift** is a constant-time digest comparison (`hmac.compare_digest`).
- **Renew** sets `expires_mono = monotonic() + ttl` and `expires_at = now() + ttl`, not old + ttl.
- **`_refuse_if_leased` reads `self.loader.loaded_model_id`** for the lease to check, not the target:
  the lease protects the resident model, whatever the request asks to load.
- **Estimate** (`backpressure.estimate_wait_seconds(queue, cbm_running)`):
  `median * (waiting + holding) / max_concurrent`, where `waiting = pending_count − holding_count`,
  clamped at 0; `None` below 3 samples or when `cbm_running`.
- **Backlog provider:** `register_backlog_provider(fn: Callable[[], int])` stores one callable;
  `backlog_rows()` returns `None` when none is registered and logs and returns `None` if it raises.
- **GPU read** (`read_gpu_memory()`, synchronous, run with `asyncio.to_thread`): `query_gpus()`; if
  empty, `{"cards": [], "reason": "nvidia-smi unavailable"}`. Map UUIDs to torch indices with
  `gpu_placement._torch_index_by_uuid()` (`gpu_placement.py:160`). For each card:
  `torch_measured = torch.cuda.is_initialized() and torch_index in torch_touched_indices()`. Only then
  call `memory_allocated` and `memory_reserved`. Attach `query_compute_apps()` rows by UUID.
  `engine_memory` set when the resident model's engine is llama.cpp and the card is in its placement.
- **`query_compute_apps()`**: `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory
  --format=csv,noheader,nounits`, 5-second timeout as `_run_query` (`nvidia_smi.py:66-85`). Returns
  `None` on failure, `[]` when it ran and listed nothing.

## 8. Testing Implementation Approach

### 8.1 Organisation

Backend tests under `tests/unit/{services,api}`; the startup test extends `tests/unit/test_startup_reset.py`;
the admission guard `test_every_request_queue_slot_is_taken_through_admission`
(`tests/unit/services/test_unload_admission.py:451`) must stay green (the queue change adds no new
caller of `acquire`). Admin UI tests beside components.

### 8.2 Patterns

- **Real service, fake loader.** Construct `ModelService` with an in-memory repository fake and a
  `LoadedModelState` whose `_loaded` is a stub `LoadedModel`, as existing service tests do
  (`tests/unit/services/test_model_service.py`). Patch the background `_load_worker` so a load returns
  without a GPU. Assert `_load_worker` call count is 0 on refusal.
- **AST call assertions.** Parse `model_service.py` and find `_refuse_if_leased` calls inside each of
  the three method bodies; parse each route module for `apply_load_policy` before
  `load_model_and_wait` in the same function; parse `main.py`'s `lifespan` for
  `clear_leases_on_startup`. Copy the shape from
  `tests/unit/services/test_probe_survives_no_restart.py:187-193`.
- **Startup reconciliation, for real.** Grant a lease in the registry, call
  `clear_leases_on_startup()`, assert `current()` is `None` and `last_ended().end_reason == "restart"`.
- **Clock.** Inject monotonic and wall clocks; never `sleep`.
- **Log capture.** `structlog.testing.capture_logs` to assert `retry_after_defaulted` absent and
  `lease_*` events present without any lease ID string in any event.
- **Reachability.** Read `app.openapi()["paths"]` for the four lease routes and `/api/health/gpus`
  (memory `app-routes-is-not-a-route-list`). Assert `RetryAfterMiddleware` in `app.user_middleware`.

### 8.3 Fixtures that disagree with the defect

Two holders (`midataworks`, `mistudio-agent`); resident model id 1 and requested model id 2; a lease ID
for model 1 sent to model 2's route; two cards, one touched and one not; a `ModelBusyError` with and
without `unloading_model_id`.

### 8.4 Mutation controls (required; each must turn a test red)

| # | Mutation | Expected red |
|---|---|---|
| M1 | Delete `_refuse_if_leased` call in `load_model` | management load under foreign lease succeeds |
| M2 | Delete it in `unload_model` | management unload under foreign lease succeeds |
| M3 | Delete it in `load_model_and_wait` | `/v1` auto-load refusal becomes `model_locked` or a load |
| M4 | Drop `lease_id=lease_id` from the internal unload (`model_service.py:892`) | holder's own swap refused |
| M5 | Expiry `<=` → `<` | lease honoured at its deadline |
| M6 | `matches` always `True` | wrong lease ID lifts the refusal |
| M7 | Remove `end_for_model` in `unload_model` | lease survives the unload |
| M8 | Remove the residency branch in `current()` | lease survives a forced unload |
| M9 | Delete `clear_leases_on_startup()` call in `lifespan` | AST test red |
| M10 | Make `clear()` a no-op | reconciliation test red |
| M11 | `ModelLeasedError` subclass `ModelLockedError` | `/v1` code becomes `model_locked` |
| M12 | `GET` lease returns the lease ID | no-disclosure test red |
| M13 | Remove `apply_load_policy` call from one route | that route auto-loads under `refuse` |
| M14 | Remove the `Retry-After` from `millm_error_handler` | parametrised test red (fallback value, warning) |
| M15 | Remove `add_middleware(RetryAfterMiddleware)` | bare-503 fallback test red |
| M16 | `backlog_rows()` returns `0` when unregistered | contract test red |
| M17 | Read torch on an untouched card | GPU test red |
| M18 | Grant ignores `_loading_model_id` | grant-during-load test red |

Each control: back up, edit one line, run the affected files, restore, confirm `git diff` is clean and
the line re-greps to its original before the next (memory: two restores once failed silently). A
surviving mutation gets a test, then is re-run as a negative control. Record all in the review.

## 9. Configuration and Environment Strategy

`millm/core/config.py`, after the request-queue block (`config.py:255-256`):

```python
LEASE_DEFAULT_TTL_SECONDS: int = 7200
LEASE_MAX_TTL_SECONDS: int = 7200
LEASE_HOLDER_MAX_CHARS: int = 128
LEASE_REASON_MAX_CHARS: int = 512
LEASE_ENDED_MEMORY: int = 64
QUEUE_DURATION_WINDOW: int = 50
RETRY_AFTER_QUEUE_DEFAULT_S: int = 5
RETRY_AFTER_MAX_S: int = 60
RETRY_AFTER_LOAD_S: int = 15
RETRY_AFTER_UNLOAD_S: int = 5
RETRY_AFTER_NOT_LOADED_S: int = 30
RETRY_AFTER_MEMORY_S: int = 30
RETRY_AFTER_READINESS_S: int = 5
RETRY_AFTER_FALLBACK_S: int = 10
```

A validator refuses `LEASE_DEFAULT_TTL_SECONDS > LEASE_MAX_TTL_SECONDS` at startup. Add each to
`.env.example` with a one-line comment. No feature flag: the lease is inert until someone takes one,
and the header behaviour defaults to today's.

**Single-process assumption.** The image runs uvicorn with no `--workers` (`Dockerfile:124`). State
it in the `model_lease.py` docstring: more than one worker would give each its own registry.

## 10. Integration Strategy

- **Backwards compatibility.** Every new parameter is keyword-only with a `None` default. Absent
  headers reproduce today's behaviour exactly. `/api/health/detailed` keeps every existing field.
- **`locked` (C8).** Untouched: `lock_model`, `set_exclusive_lock`
  (`millm/db/repositories/model_repository.py:197`), the SAE auto-locks
  (`millm/services/sae_service.py:1939`, `2302`) and `/v1/models` filtering
  (`millm/api/routes/openai/models.py:38`, `90`). The lease check precedes the lock check only in
  `load_model_and_wait`.
- **Feature 26.** Exposes `acquire_lease`, `renew_lease`, `release_lease`, `resolve_lease`,
  `RequestQueue.holding_count`, `register_backlog_provider`. 026's FTASKS gate on these names.
- **miStudio 034.** The contract section names the routes, both headers, the four codes, the health
  fields and the GPU route. FR-18's tools and FR-19's header are built against it.
- **miDataworks 005, 007, 009.** No code here; their clients send the headers and read the codes.

## 11. Utilities and Helpers Design

- `model_lease._digest(lease_id) -> str` and `_ref(digest) -> str`; the only places a lease ID is
  hashed.
- `backpressure.retry_after_for(code, details=None) -> int` is pure except for the queue lookup on
  `QUEUE_FULL` and the breaker lookup on `HUB_UNAVAILABLE`, both guarded so a failure returns the
  code's default.
- `lease_summary(record) -> LeaseSummary` is the single serialiser used by `ModelResponse`, the health
  field and the `GET` route (memory `one-list-needs-one-serialiser`). A test compares key sets across
  the three.

## 12. Error Handling and Logging Strategy

| Situation | Response | Log |
|---|---|---|
| Foreign lease on load, unload, auto-load | `409 MODEL_LEASED` / `model_leased` | `lease_refused` (holder, model, operation, target) |
| Refuse policy, not resident | `409 model_not_resident` | info `load_policy_refused` |
| Refuse policy, model loading | `503 model_loading` + `Retry-After` | info |
| Grant on non-resident | `409 MODEL_NOT_RESIDENT` | info |
| Grant during load/unload | `503 MODEL_BUSY` + `Retry-After` | info |
| Grant over a live lease | `409 MODEL_LEASED` | `lease_refused` |
| Bad TTL or text | `400 INVALID_LEASE_REQUEST` naming field and limit | none |
| Unknown lease ID | `404 LEASE_NOT_FOUND`, message: "unknown lease; a restart ends every lease" | info |
| Expired or ended lease ID | `409 LEASE_EXPIRED` with `end_reason` | info |
| Wrong lease header, no lift needed | proceeds | warning `lease_header_unmatched` |
| Startup clear fails | startup continues | error `lease_startup_clear_failed` (never raises, like `main.py:261`) |
| `503` without `Retry-After` reaching the middleware | fallback header | warning `retry_after_defaulted` |
| Health block read fails | block present with `error` | warning |
| `nvidia-smi` fails | `cards: []` + `reason` | debug, as today |

Every `lease_*` event carries `lease_ref`, `holder`, `model_id`; a capture test asserts no event value
equals a lease ID.

## 13. Performance Implementation Hints

- No database access on any lease check; `get_lease` reads the model row only for the name.
- `median_hold_seconds()` sorts at most 50 floats per health read.
- The GPU route is two `nvidia-smi` invocations in one worker thread; it is not on any request path.
- The badge's 1-second timer is per rendered badge; at most one badge is live (one resident model).

## 14. Code Quality and Standards

- Black 100, Ruff, MyPy on `millm/` (PADR §5); `npm run lint` and `npm run typecheck` in `admin-ui`.
- Comments explain why, citing the decision ID (X-01, T-85 …) at the line it governs.
- No `getattr(..., default)` fallbacks on the lease record; read attributes directly so a missing one
  fails loudly.
- Tracked debt recorded in the review, not fixed: the management load and unload ignore `locked`
  (`management/models.py:173`, `192`; FPRD §9, §13).

## 15. Decisions from Clarifying Questions

Clarifying rounds were waived. Each decision cites its source.

| # | Question | Decision | Source |
|---|---|---|---|
| ID1 | Where does the lease code live? | `millm/services/model_lease.py`, rules in `ModelService` | FTDD §6 |
| ID2 | Route file for lease routes | Existing management models router; no new `include_router` | `millm/api/routes/__init__.py:35` |
| ID3 | Lease ID carriage on renew/release | Header, not body or path | FTDD TD12 |
| ID4 | Validation layer | Service raises `400`; schema types only | FTDD TD14 |
| ID5 | Test style for guards | AST call assertions plus behaviour through the real service | memory `source-scrape-guards-i-keep-writing`; FTDD §10 |
| ID6 | Startup reconciliation shape | Named function called from `lifespan`, tested for real and by AST | X-01; `main.py:209-230` precedent |
| ID7 | Middleware type | Pure ASGI | FTDD §3 (streaming) |
| ID8 | Badge colour | Amber, distinct from the steering lock's yellow | FPRD §4 |
| ID9 | Polling rate while leased | 10 seconds | FPRD FR-29.5.4 |
| ID10 | Mutation controls | M1–M18, every lease-enforcement path included | coordinator instruction; global review discipline |
| ID11 | Feature 26 API names | §10 | 026 FTDD §7 |

**Open items:** none for the operator. FTASKS spike 0.2 measures the CUDA context size and in-pod
`--query-compute-apps` visibility for acceptance 16 (T-89).
