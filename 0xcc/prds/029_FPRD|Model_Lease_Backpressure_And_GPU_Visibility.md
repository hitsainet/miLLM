# Feature PRD: Model Lease, Backpressure and GPU Visibility

**Document ID:** 029_FPRD|Model_Lease_Backpressure_And_GPU_Visibility
**Version:** 1.0 (planned)
**Status:** Planned. Feature PRD written 2026-10-06; FTDD, FTID and FTASKS follow.
**Source:** BRD-04 (miLLM — Dataworks Support) §5.10–§5.12: R-04.38–R-04.45.
**PPRD:** Feature 29 (FR-29.1 – FR-29.8), PPRD v1.5 · **PADR:** v1.5 §1 rows "Model lease (v1.5)" and
"Backpressure & GPU visibility (v1.5)"; v1.5 §10 trade-offs "A model lease with holder and expiry vs
extending the `locked` flag" and "`Retry-After` and documented queue fields vs leaving backoff to the
client"
**Binding decisions:** checkpoint decision C8 and the checkpoint technical defaults of 2026-10-06
(`~/app/miDataworks/0xcc/docs/brd-decisions-2026-10-05.md`, "Checkpoint decisions")
**Depends on:** Feature 1 (model load and unload, the existing `locked` flag); Feature 23 (GGUF
auto-load paths). Feature 26 later feeds the batch-backlog field (FR-29.7).
**Co-release:** miStudio BRD-MIS-DATAWORKS-001 lease tools `millm_acquire_lease`,
`millm_renew_lease`, `millm_release_lease`, `millm_lease_status`, and the refuse-load header on its
scoring and generation tools (miStudio file `034_*`, FR-18 and FR-19). Consumers: miDataworks BRD-03
R-03.27 (a label run pins its model), R-03.29 (backpressure) and R-03.64 (jobs take the lease; VRAM
per card).

Code references are to miLLM at `7aa659c` (HEAD, 2026-10-06). Where BRD-04 cites a line that has since
moved, the current line is given.

---

## 1. Feature Overview

**Name:** Model Lease, Backpressure and GPU Visibility.

**What it is:** three additions that let a long job share miLLM safely with interactive users.

1. **Model lease.** A caller takes a lease on the resident model. The lease names a holder, a reason
   and a time to live (TTL). While it is held, nobody else can load, unload or swap the model. A
   request can also opt out of auto-loading with `X-miLLM-Load-Policy: refuse`.
2. **Backpressure.** Every `503` says when to retry, in a `Retry-After` header. The detailed health
   endpoint adds the in-flight count, the batch backlog and an estimated wait.
3. **GPU visibility.** A Representational State Transfer (REST) endpoint reports memory per graphics
   processing unit (GPU) card, including what miLLM itself holds.

**Problem:**
- **Nothing protects a job from a model swap.** Any `/v1` request naming another model loads it on
  demand (`millm/api/routes/openai/chat.py:144-146`, `completions.py:113-115`, `embeddings.py:81-83`).
  The only guard is `locked`, a bare boolean with no holder and no expiry
  (`millm/db/models/model.py:152`). It is set automatically when a sparse autoencoder (SAE) is
  attached (`millm/services/sae_service.py:1939`, `2302`). It refuses only the `/v1` auto-load
  (`millm/services/model_service.py:1472-1479`). `POST /api/models/{id}/load` and `/unload`
  (`millm/api/routes/management/models.py:173`, `192`) never read it.
- **A busy server gives no retry time.** No `Retry-After` is set anywhere in `millm/`. A client must
  guess, and a labelling job either hammers the queue or idles.
- **Queue state is partial and undocumented.** `/api/health/detailed` reports `queue_pending`
  (`millm/api/routes/system/health.py:365-367`). That count includes requests already running, not
  only waiting ones: the queue increments it before taking the semaphore
  (`millm/services/request_queue.py:107-117`, `147-157`). The `inference` block is an untyped dict
  (`health.py:127`) that vanishes silently on any error (`health.py:369`).
- **GPU memory was invisible.** On 2026-10-05 miLLM held 17,396 MiB of the RTX 3090 with no model
  loaded. Only `nvidia-smi` on the node showed it. Per-card figures exist only on the
  `system:metrics` Socket.IO stream (`millm/sockets/progress.py:29-56`) and in the loaded model's
  placement (`health.py:265-279`). No REST endpoint reports them.

**Goals:**
- A long job pins its model, and a crashed holder cannot block model changes for more than its TTL.
- A scoring or generation client can promise it never causes a model swap.
- Every `503` tells the caller when to retry, from one place a new `503` cannot bypass.
- Queue state is a documented, typed contract, and "not measured" is never reported as zero.
- Memory held by miLLM is readable per card over REST, with no shell access to the node.

**Connection to the project:** miLLM serves one model at a time
(`model_service.py:885-892` unloads the resident model before a load). miDataworks label runs,
miStudio's Model Context Protocol (MCP) agents and Open WebUI all share it. This feature makes that
sharing explicit and attributable instead of first-come, first-swapped.

## 2. User Stories & Scenarios

**US-1: a label run pins its model.** miDataworks starts a 25,000-row label run on JEV-9B-decision.
It takes a lease with holder `midataworks`, TTL 7200 seconds and a reason naming the run.
*Acceptance:* the lease ID is returned. While the run holds it, a chat request naming another model
gets `409 MODEL_LEASED`, naming holder and expiry. So does `POST /api/models/{other}/load` without
the lease ID. The run's own requests carry `X-miLLM-Lease` and proceed. (BRD-04 acceptance 14)

**US-2: the holder crashes.** The miDataworks worker dies and never renews.
*Acceptance:* once the TTL passes, the lease is gone. The same load that was refused now succeeds.
No operator action is needed. (BRD-04 acceptance 14; RSK-04)

**US-3: a scorer never swaps.** A miStudio MCP scoring tool sends `X-miLLM-Load-Policy: refuse`
naming a model that is not resident.
*Acceptance:* `409` naming the requested and the resident model. No load starts. Without the header,
the same request auto-loads as today.

**US-4: a busy server says when.** Eleven concurrent requests arrive with `MAX_PENDING_REQUESTS` at
10 (`millm/core/config.py:256`).
*Acceptance:* the overflow request gets `503 queue_full` with a `Retry-After` header in whole seconds.
The OpenAI error envelope and code are unchanged. (BRD-04 acceptance 15)

**US-5: the operator sees who holds the model.** The operator opens the Admin UI Models page.
*Acceptance:* the leased model shows the holder, the reason and the time left. The Dashboard health
data shows the same lease.

**US-6: memory with no model loaded.** After a model is unloaded, a caller reads the per-card
endpoint.
*Acceptance:* each card shows total, used and free memory, and miLLM's own allocated and reserved
memory. The figures match `nvidia-smi` within 256 MiB. (BRD-04 acceptance 16; see Open Question 7)

**Secondary scenarios:**
- **Renewal.** The holder renews before expiry. The expiry moves to now plus the new TTL.
- **Release.** The holder releases on completion, cancellation or failure. The lease ends at once.
- **Holder swaps its own model.** The holder unloads the leased model with its lease ID. The unload
  proceeds, and the lease ends with it (FR-29.3.4; Open Question 3).
- **Steering beside a lease.** An SAE is attached to a leased model. `locked` is set as today. Both
  guards apply independently (C8).

**Edge cases and error scenarios:**

| Situation | Required behaviour |
|---|---|
| Lease requested while another holder's lease is live | `409 MODEL_LEASED`, naming holder and expiry; no new lease |
| Lease requested on a model that is not resident | Refused, naming the resident model (FR-29.1.3; Open Question 3) |
| Lease requested while a load or unload is running | `503 model_busy` with `Retry-After` |
| `ttl_seconds` above 7200, zero or negative | `400`, naming `ttl_seconds` and the limit; never clamped silently |
| Renew or release with a wrong or expired lease ID | `404` for an unknown ID, `409` for an expired one; nothing changes |
| A `/v1` request names the leased model, without the lease ID | Proceeds; a lease blocks swaps, not use |
| A refuse-policy request names a model that is loading now | `503 model_loading` with `Retry-After`; this request starts nothing |
| A `503` raised inside a stream whose `200` is already sent | No header is possible; the in-stream error carries `retry_after` |
| A restart while a lease is held | The lease ends with the resident model (FR-29.1.9) |
| `nvidia-smi` absent or hung | The GPU endpoint returns `200` with `cards: []` and a stated reason, never fabricated zeros |

## 3. Functional Requirements

Each FR-29.x below is the PPRD v1.5 requirement. The numbered items under it refine it into testable
statements. Error responses on `/v1` use the OpenAI error envelope
(`millm/api/routes/openai/errors.py:26-48`). Management routes use the management envelope
(`millm/api/exception_handlers.py:77-130`).

### 3.1 Model lease

**FR-29.1 Take, renew, release and read a lease.** `POST /api/models/{id}/lease` SHALL take
`{holder, ttl_seconds, reason}` and return a lease ID. One lease SHALL exist at a time, renewable and
releasable by its holder, reported by `GET`, and expiring on its own at its TTL. (R-04.38)

- **FR-29.1.1** `holder`, `ttl_seconds` and `reason` are all required. `holder` and `reason` are
  non-empty strings with a maximum length fixed in the FTDD. `holder` is free text: miLLM has no
  sign-in (BRD-04 §3), so it is a label for attribution, not an identity check.
- **FR-29.1.2** `ttl_seconds` is an integer from 1 to `LEASE_MAX_TTL_SECONDS`, default 7200 (checkpoint
  technical default: maximum 2 hours, renewable). A value outside that range is refused with `400`
  naming the field and the limit. It is never clamped.
- **FR-29.1.3** A lease is granted only on the resident model, in state `LOADED`. A request for any
  other model is refused, naming the resident model or "none". Open Question 3 records the
  alternative.
- **FR-29.1.4** One lease exists per server, not per model, because one model is resident at a time.
  A request while another lease is live returns `409 MODEL_LEASED` with holder, reason and expiry.
- **FR-29.1.5** The response returns `lease_id`, `model_id`, `model_name`, `holder`, `reason`,
  `acquired_at`, `expires_at` and `ttl_seconds`. Times are ISO 8601 in UTC.
- **FR-29.1.6** **The lease ID is the only proof of holding.** It is returned once, to the caller that
  took the lease. No read path returns it: not `GET`, not `/api/health/detailed`, not the Admin UI, not
  a log line. Two callers sending the same `holder` string are two different callers.
- **FR-29.1.7** Renew takes the lease ID and a new `ttl_seconds` under FR-29.1.2. The new expiry is now
  plus that TTL, not the old expiry plus it. Release takes the lease ID and ends the lease at once.
  Both refuse an unknown ID with `404` and an expired one with `409`, changing nothing. The route
  shapes are fixed in the FTDD; miStudio 034 FR-18 waits on them.
- **FR-29.1.8** `GET` reports the current lease or none: `model_id`, `model_name`, `holder`, `reason`,
  `acquired_at`, `expires_at` and `seconds_remaining`. An expired lease reads as none.
- **FR-29.1.9** A lease ends when its model stops being resident, whoever unloads it. This includes a
  restart: residency does not survive one (`millm/main.py:134-139` resets `loaded` rows to `ready`),
  so neither does the lease. If lease rows are persisted, `STALE_STATE_RESETS` (`main.py:134`) gains
  an entry for them, and the startup-reset test binds to it.
- **FR-29.1.10** Expiry needs no background task to be correct. Every check compares the current time
  with `expires_at`, so an expired lease never refuses anything.
- **FR-29.1.11** Every grant, renewal, release, expiry and refusal is logged with holder, model and
  reason, never with the lease ID.

**FR-29.2 Refuse others' loads, unloads and swaps.** While a model is leased, a load, unload or swap
by anyone else SHALL be refused with `409 MODEL_LEASED`, naming holder and expiry. This covers every
`/v1` auto-load and `POST /api/models/{id}/load` and `/unload`. (R-04.39)

- **FR-29.2.1** The guarded entry points are the three `/v1` auto-loads (`chat.py:146`,
  `completions.py:115`, `embeddings.py:83`, all through `load_model_and_wait`,
  `model_service.py:1442`), the management load and unload (`management/models.py:173`, `192`), and
  the internal unload that a load performs first (`model_service.py:885-892`).
- **FR-29.2.2** The check lives in `ModelService`, inside `load_model` and `unload_model`, not in the
  routes. A route that forgets it cannot bypass it. The startup auto-load (`main.py:87`) runs before
  any lease can exist.
- **FR-29.2.3** On `/v1` the refusal is HTTP `409` with code `model_leased`, type
  `invalid_request_error`. On management routes it is `409` with code `MODEL_LEASED`. Both bodies name
  holder, reason, `expires_at` and the leased model. `ERROR_STATUS_MAP` (`errors.py:76-111`) gains the
  row.
- **FR-29.2.4** A `/v1` request naming the leased, resident model proceeds without the lease ID. The
  lease guards residency, not use, so interactive chat keeps working during a label run.
- **FR-29.2.5** The lease is checked before `locked`. When both apply, the response is
  `MODEL_LEASED`, since it names a holder and an expiry.
- **FR-29.2.6** A refused load changes nothing: the load slot (`model_service.py:832-837`) is not
  claimed, no unload starts and no row status moves.

**FR-29.3 The holder's requests proceed.** Requests and model operations carrying the holder's
`X-miLLM-Lease: <id>` SHALL proceed. (R-04.40)

- **FR-29.3.1** `X-miLLM-Lease` is read on every `/v1` route and on the management load, unload and
  lease routes.
- **FR-29.3.2** A matching, live lease ID lifts only the lease refusal. Every other check still
  applies: `locked`, a load already running, GPU placement and the load policy (FR-29.4).
- **FR-29.3.3** A wrong or expired lease ID on a request that would be refused is still refused with
  `409 MODEL_LEASED`. A wrong lease ID on a request that needs no lift is ignored, with a logged
  warning.
- **FR-29.3.4** When the holder unloads or swaps the leased model, the operation proceeds and the
  lease ends under FR-29.1.9. The holder takes a new lease on the new model. Open Question 3 asks
  whether the lease should follow the holder's swap instead.

**FR-29.4 Refuse-load policy.** `X-miLLM-Load-Policy: refuse` SHALL turn an auto-load of a
non-resident model into `409`. The default SHALL stay auto-load. (R-04.41)

- **FR-29.4.1** The header is read on `/v1/chat/completions`, `/v1/completions` and
  `/v1/embeddings`. Values are `refuse` and `auto`, case-insensitive. Absent means `auto`, which Open
  WebUI relies on. Any other value is refused with `400` naming the header.
- **FR-29.4.2** Under `refuse`, a request naming a model that is not resident gets `409` with code
  `model_not_resident`, naming the requested model and the resident model or "none". No load starts.
- **FR-29.4.3** Under `refuse`, a request naming the model being loaded right now gets
  `503 model_loading` with `Retry-After`. The model will become resident without this request causing
  anything.
- **FR-29.4.4** Under `refuse`, the lease is not consulted: no load would be attempted. If a lease is
  held, the `409` body still reports it, so the caller sees why the model may not change soon.
- **FR-29.4.5** The checks that already run before an auto-load stay first, unchanged: model not
  found, embedding-only model, and the GGUF refusals (`completions.py:76-94`, `embeddings.py:60-73`).
- **FR-29.4.6** The header has no effect on management routes, which load by explicit request.

**FR-29.5 Lease state is visible.** Lease state SHALL appear in `/api/health/detailed` and on the
Admin UI's model page. (R-04.42)

- **FR-29.5.1** `DetailedHealthResponse` (`health.py:89`) gains a typed `lease` field: the FR-29.1.8
  shape, or `null` when no live lease exists. It never carries the lease ID.
- **FR-29.5.2** The Admin UI Models page (`admin-ui/src/pages/ModelsPage.tsx`) shows a lease badge on
  the leased model, beside the existing lock icon (`ModelsPage.tsx:292-296`). The badge shows holder,
  reason and time remaining. The model details modal shows the same.
- **FR-29.5.3** The model list API (`ModelResponse`) gains the same lease summary, so the page needs no
  second request.
- **FR-29.5.4** The display refreshes at least every 10 seconds while the Models page is open. An
  expired lease disappears without a page reload.
- **FR-29.5.5** The Admin UI offers no lease action in this feature. Open Question 4 asks about an
  operator force-release.

### 3.2 Backpressure

**FR-29.6 `Retry-After` on every `503`.** Every `503` (`QUEUE_FULL`, `MODEL_BUSY`, `MODEL_LOADING`,
`MODEL_NOT_LOADED`, `INSUFFICIENT_MEMORY`) SHALL carry `Retry-After` in seconds, with the OpenAI error
envelope and codes unchanged. (R-04.43)

- **FR-29.6.1** "Every `503`" means every HTTP response miLLM sends with status `503`, on any route.
  Today these come from `ERROR_STATUS_MAP` through `millm_error_handler`
  (`exception_handlers.py:77-130`), from `model_not_loaded_error` (`errors.py:150-157`),
  `model_busy_error` (`errors.py:305-319`) and `load_refused_error` (`errors.py:280-302`), from
  `HubUnavailableError` on management routes (`millm/services/cluster_hub_service.py:66-70`), and from
  the readiness probe (`health.py:261`).
- **FR-29.6.2** The header is added at one choke point that every `503` passes through. A test sends
  each code that can be produced and asserts the header. A second test fails if a new `503` site
  bypasses the choke point.
- **FR-29.6.3** The value is a whole number of seconds, at least 1, as HTTP's `delay-seconds` form
  requires. Each code has a stated rule, fixed in the FTDD:
  - `QUEUE_FULL`: from the estimated wait (FR-29.7.4), rounded up.
  - `MODEL_BUSY` and `MODEL_LOADING`: from load progress where known, else a configured default.
  - `MODEL_NOT_LOADED` and `INSUFFICIENT_MEMORY`: a configured default.
  - `HUB_UNAVAILABLE`: the circuit breaker's remaining recovery time (`cluster_hub_service.py:57`).
  - The readiness probe: a configured default.
- **FR-29.6.4** Status codes, error types, codes and messages stay as they are. Only the header is
  added. The error body also carries `retry_after` with the same value, so a client that reads only
  bodies sees it.
- **FR-29.6.5** A refusal yielded inside a stream whose `200` is already committed
  (`inference_service.py:673-697`, `raise_refusal=False`) cannot carry a header. Its in-stream error
  carries `retry_after` instead.
- **FR-29.6.6** `MODEL_LOADING` is mapped (`errors.py:79`) and raised by nothing in `millm/`. FR-29.4.3
  gives it a producer. If the FTDD chooses otherwise, it states that the code is unreachable.

**FR-29.7 Documented queue state.** `/api/health/detailed` SHALL add the in-flight count, the batch
backlog in rows and an estimated wait beside the existing queue depth, documented as a stable
contract. (R-04.44)

- **FR-29.7.1** The `inference` block becomes a typed model, replacing `dict[str, Any]`
  (`health.py:127`). A failure to read it reports the block with an `error` field, never by omitting
  it silently (`health.py:369`).
- **FR-29.7.2** Existing fields keep their names and meanings: `backend`, `cbm_enabled`,
  `cbm_running`, `queue_pending`, `queue_max_concurrent`, `queue_max_pending`. `queue_pending` is
  documented as it is: requests waiting **plus** running.
- **FR-29.7.3** New fields:
  - `in_flight`: requests holding a request-queue slot now. The idle cache release also takes a slot
    (`inference_service.py:756`); the FTDD decides whether it counts, and the documentation says.
  - `queue_waiting`: `queue_pending` minus `in_flight`.
  - `batch_backlog_rows`: rows not yet run across in-progress batches. `null` until Feature 26 ships,
    meaning "no batch API", never `0`.
  - `estimated_wait_seconds`: an estimate of how long a request arriving now waits for a slot.
    `null` when there is no measurement to base it on.
- **FR-29.7.4** The estimate is computed from measured recent request durations and the work ahead,
  including the batch backlog once Feature 26 exists. The formula is fixed in the FTDD and stated in
  the documentation. The field name and the documentation both call it an estimate.
- **FR-29.7.5** When continuous batching (CBM) is running, requests it serves hold no queue slot
  (`inference_service.py:745-747`). `in_flight` then either includes CBM's active requests or is
  `null`, never a count that silently leaves them out.
- **FR-29.7.6** The contract is documented in `manual/docs/api/management-api.md` and in
  `docs/mcp-contract.md`, whose `GET /api/health/detailed` row (`docs/mcp-contract.md:92`) gains the
  new fields. A schema test pins the field set, so removing a field turns a test red.

### 3.3 GPU visibility

**FR-29.8 Per-card memory over REST.** A REST endpoint SHALL return, per card: index, universally
unique identifier (UUID), name, total, used and free memory in MiB, and miLLM's own allocated and
reserved memory on that card. (R-04.45)

- **FR-29.8.1** The endpoint is a `GET` under `/api/`. Its path is fixed in the FTDD. It returns
  `cards: [...]` and `read_at`.
- **FR-29.8.2** Total, used and free memory come from `nvidia-smi`, through the existing reader
  (`millm/ml/nvidia_smi.py:88`). Index is torch's index, matched by UUID as `list_gpus` does
  (`millm/ml/gpu_placement.py:186-222`). A card `nvidia-smi` reports and torch cannot see is listed
  with `torch_index: null`, not dropped.
- **FR-29.8.3** `millm_allocated_mb` and `millm_reserved_mb` come from torch's allocator on that card.
- **FR-29.8.4** **Reading a card never creates a CUDA context on it.** A context costs memory that
  other tenants of the node place against (`millm/ml/model_loader.py:220-226`). On a card where torch
  holds no context, both miLLM fields are `0` with `torch_context: false`, which is a true reading,
  not an estimate.
- **FR-29.8.5** llama.cpp memory is not torch's (`model_loader.py:343-345`). When a GGUF model is
  resident, each card it occupies reports `engine_memory: "not_measured_by_torch"`. The torch fields
  stay true for what torch holds. The FTDD decides whether a per-process read can fill the gap
  (Open Question 7).
- **FR-29.8.6** The `nvidia-smi` read runs in a worker thread. It can take up to its 5-second timeout
  (`nvidia_smi.py:66-85`), and called inline it would stall the event loop, as the load pre-check once
  did (`model_service.py:866-871`).
- **FR-29.8.7** When `nvidia-smi` is absent or fails, the endpoint returns `200` with `cards: []` and
  `reason`. It never reports zeros as a reading.
- **FR-29.8.8** The endpoint takes no request-queue slot and is safe to poll during a generation.

### 3.4 Coverage of BRD-04 requirements

| BRD-04 | Topic | PPRD FR | Refined in this PRD | Status |
|---|---|---|---|---|
| R-04.38 | Lease: take, renew, release, read, TTL | FR-29.1 | FR-29.1.1 – FR-29.1.11 | covered (scope: Open Question 3) |
| R-04.39 | `409 MODEL_LEASED` for others' loads and swaps | FR-29.2 | FR-29.2.1 – FR-29.2.6 | covered |
| R-04.40 | `X-miLLM-Lease` lets the holder through | FR-29.3 | FR-29.3.1 – FR-29.3.4 | covered |
| R-04.41 | `X-miLLM-Load-Policy: refuse` | FR-29.4 | FR-29.4.1 – FR-29.4.6 | covered |
| R-04.42 | Lease state in health and Admin UI | FR-29.5 | FR-29.5.1 – FR-29.5.5 | covered |
| R-04.43 | `Retry-After` on every `503` | FR-29.6 | FR-29.6.1 – FR-29.6.6 | covered |
| R-04.44 | In-flight, backlog, estimated wait | FR-29.7 | FR-29.7.1 – FR-29.7.6 | covered (backlog fed by Feature 26) |
| R-04.45 | Per-card memory endpoint | FR-29.8 | FR-29.8.1 – FR-29.8.8 | covered (GGUF gap: Open Question 7) |

Totals: 8 of 8 owned requirements covered, 8 PPRD FRs refined into 52 testable items. No BRD-04
requirement outside §5.10–§5.12 is claimed here. R-04.22 (a batch takes the lease) is Feature 26's and
uses FR-29.1–FR-29.3.

## 4. User Experience Requirements

- **Models page.** A lease badge beside the lock icon (`ModelsPage.tsx:292-296`), in the existing
  Tailwind dark theme and `lucide-react` icon set. Text: holder, reason, and "expires in 1h 12m". The
  exact expiry time is in the tooltip. The lock icon's tooltip keeps "Locked for steering", so the two
  guards are never confused.
- **Model details modal.** A "Lease" row with the same facts.
- **Dashboard.** No new card. The health data the Dashboard reads carries the lease.
- **No lease controls.** The UI displays; it does not take, renew or release (FR-29.5.5).
- **Accessibility.** The badge has a text label, not colour alone, and an `aria-label` stating holder
  and expiry.
- **The HTTP contract is the main experience.** Every refusal names its reason and the next step:
  the holder and expiry for `MODEL_LEASED`, the resident model for `model_not_resident`, and seconds
  for `Retry-After`.

## 5. Data Requirements

- **Lease record:** `lease_id` (random, unguessable), `model_id`, `holder`, `reason`, `acquired_at`,
  `expires_at`, `ttl_seconds`, `released_at`, `end_reason` (`released`, `expired`, `model_unloaded`,
  `restart`). Whether it lives in process memory or a table is an FTDD choice. Either way FR-29.1.9
  holds: a restart ends it.
- **If persisted:** a migration in `alembic/`, an entry in `STALE_STATE_RESETS` (`main.py:134`), and
  the lease ID stored as a hash, since it is the only proof of holding (FR-29.1.6).
- **`models.locked`** is unchanged (C8). No column is added to it or reinterpreted.
- **Request-duration samples** for the estimated wait: a bounded in-memory window. Nothing persisted.
- **Configuration** (`millm/core/config.py`): `LEASE_MAX_TTL_SECONDS` (default 7200) and the
  `Retry-After` defaults per code. Each documented in `manual/docs/reference/configuration.md`.

## 6. Technical Constraints

- **C8 (binding):** the lease sits beside `locked`. Callers migrate to it. `locked` is retired in a
  later increment, not here.
- **PADR v1.5 §10:** a lease with holder and expiry rather than widening `locked`, because the steering
  path and `/v1/models` already read that flag (`millm/api/routes/openai/models.py:38`, `90`).
- **Single resident model** (BRD-04 §3; `model_service.py:885-892`). One lease per server follows.
- **`MAX_CONCURRENT_REQUESTS` is 1** (`config.py:255`). Per-request steering, sensing and probes rely
  on serial execution through `_admit()` (`inference_service.py:649`). Nothing here changes admission.
- **No sign-in** (BRD-04 §3, decision 7). The lease ID is a bearer secret. `holder` is a label.
- **No CUDA context created by a read** (FR-29.8.4), matching the existing placement code.
- **Python 3.11, FastAPI, SQLAlchemy async, Pydantic v2, React 18 with TypeScript and Zustand**
  (PADR §5).

## 7. API/Integration Specifications

| Method and path | Purpose | Notes |
|---|---|---|
| `POST /api/models/{id}/lease` | Take a lease | Body `{holder, ttl_seconds, reason}`; returns FR-29.1.5 |
| Renew, release (paths in FTDD) | Renew or release | Lease ID required; FR-29.1.7 |
| `GET` lease (path in FTDD) | Read the current lease | Never returns the lease ID |
| `GET /api/health/detailed` | Health, lease, queue state | Adds `lease`, typed `inference` with new fields |
| `GET /api/...` GPU (path in FTDD) | Per-card memory | FR-29.8 |

**Request headers read:** `X-miLLM-Lease` (FR-29.3), `X-miLLM-Load-Policy` (FR-29.4).

**Response headers written:** `Retry-After` on every `503` (FR-29.6).

**New error codes:** `MODEL_LEASED` (`409`), `model_not_resident` (`409`, `/v1` only). `MODEL_LOADING`
gains a producer (FR-29.6.6).

**Consumers:**
- miStudio 034 FR-18 calls the lease routes with the agent identity as `holder` and sends
  `X-miLLM-Lease` when given. FR-19 sends `X-miLLM-Load-Policy: refuse` on every scoring and
  generation request. 034 passes `MODEL_LEASED` and `Retry-After` to the agent unchanged.
- miDataworks 005 FR-005.37 takes, renews and releases the lease around a label run.
  FR-005.38 refuses to start when the wrong model is resident, which matches FR-29.1.3.
  FR-005.46 honours `Retry-After`.
- Feature 26 (batch API) takes the lease for a whole batch (R-04.22) and feeds `batch_backlog_rows`.

## 8. Non-Functional Requirements

- **Overhead.** The lease check adds no database round trip to a `/v1` request that names the resident
  model. It is an in-memory comparison.
- **Correctness under concurrency.** Two simultaneous lease requests produce exactly one lease. A
  lease request and a load racing each other produce one consistent outcome. The check and the claim
  happen with no `await` between them, as the load slot already does (`model_service.py:824-837`).
- **Expiry precision.** A lease is honoured until `expires_at` and refused nothing after it, measured
  by the server's clock, to within one second.
- **Health endpoint cost.** `/api/health/detailed` stays free of `nvidia-smi`. The GPU read lives on
  its own endpoint (FR-29.8).
- **Privacy.** No lease ID in any log, health response, UI or error body (FR-29.1.6).

## 9. Feature Boundaries (Non-Goals)

- **Retiring `locked`.** Deferred by C8 to a later increment.
- **Making the management load honour `locked`.** Today it does not (`management/models.py:173` calls
  `load_model`, which never reads the flag). This is a pre-existing gap in the code this feature
  touches. It is recorded as tracked debt (section 13), not fixed, because C8 leaves `locked`'s
  meaning alone until it is retired.
- **Co-residency** of several models, and a GPU scheduler shared with miStudio's workers (BRD-04 §4,
  §7). Open Question 2 asks whether miStudio's workers take the lease.
- **Authentication** (decision 7).
- **MCP tools** for the lease, the GPU endpoint or the queue fields. miStudio owns them (034 FR-18; its
  §9 declines tools for R-04.44 and R-04.45).
- **Changing `/v1/models`** for a lease. A locked model hides the others there; a lease does not
  (Open Question 5).
- **Changing management status codes.** `MODEL_BUSY` stays `409` and `INSUFFICIENT_MEMORY` stays
  `507` on management routes (`millm/core/errors.py:178`, `199`). They are not `503`, so FR-29.6 does
  not touch them.
- **Raising `MAX_CONCURRENT_REQUESTS`** or re-enabling continuous batching in production.

## 10. Dependencies

- **Feature 1:** `ModelService.load_model`, `unload_model`, `load_model_and_wait`
  (`model_service.py:782`, `1194`, `1442`), `LoadedModelState` (`model_loader.py:277`).
- **Feature 23:** the GGUF refusals before auto-load, which stay first (FR-29.4.5).
- **Feature 26:** feeds `batch_backlog_rows` (026 FR-26.4.7) and takes the lease (026 FR-26.7).
  Feature 29 ships first (RSK-09). Its restart behaviour must match FR-29.1.9 (Open Question 9).
- **Feature 25:** none required. Feature 25's strict mode applies to the lease routes' bodies only if
  they are `/v1`, and they are not.
- **Libraries:** none new.
- **Infrastructure:** `nvidia-smi` in the miLLM image, as today.
- **Timeline:** first in the BRD-04 build order with Feature 25 (BRD-04 RSK-09). miDataworks 005
  cannot ship its miLLM label runs without it (005 §10, "Sequencing").

## 11. Success Criteria

Each wiring item is accepted only by a test that fails when its registration or call line is
removed, asserting payload and call count (FR-20.3).

1. **Lease (hardware; BRD-04 acceptance 14).** With a lease held by `midataworks`, a chat request
   naming another model gets `409 MODEL_LEASED`. So does `POST /api/models/{other}/load` without the
   lease ID. After the TTL passes without renewal, the same load succeeds.
2. **Holder passes.** The same load with the right `X-miLLM-Lease` proceeds, and the lease ends with
   the unloaded model.
3. **Refuse policy.** A `refuse` request naming a non-resident model gets `409 model_not_resident` and
   no load starts, asserted by the load call count staying zero.
4. **Backpressure (BRD-04 acceptance 15).** Eleven concurrent requests produce a `503 queue_full`
   carrying `Retry-After`, an integer of at least 1.
5. **Every `503`.** Each producible `503` code carries `Retry-After`, asserted per code.
6. **Queue contract.** `/api/health/detailed` returns `in_flight`, `queue_waiting`,
   `batch_backlog_rows` (`null` before Feature 26) and `estimated_wait_seconds`. With one request
   running and two waiting, `in_flight` is 1 and `queue_waiting` is 2.
7. **GPU visibility (hardware; BRD-04 acceptance 16).** After a model is unloaded, the per-card
   endpoint shows miLLM's reserved memory on each card, matching `nvidia-smi` within 256 MiB under the
   comparison Open Question 7 fixes. No CUDA context appears on a card miLLM had not used.
8. **Lease visible.** The Models page and `/api/health/detailed` show holder, reason and expiry, and
   no lease ID appears in either.

## 12. Testing Requirements

- **Unit (lease):** grant, conflict, renew (new expiry from now), release, expiry by clock, wrong ID,
  expired ID, TTL bounds (1, 7200, 0, 7201), one lease under concurrent requests, lease ends on unload
  and on startup reset.
- **Unit (guard placement):** the lease check is asserted inside `load_model` and `unload_model` by
  walking the abstract syntax tree (AST) for the **call**, not by searching source text.
- **Reachability:** each new route present in `app.openapi()["paths"]`, not in `app.routes`; each
  guarded entry point refuses under a foreign lease, with the refusal payload asserted.
- **Integration:** the three `/v1` auto-loads and the two management routes each return `409
  MODEL_LEASED` under a foreign lease, and proceed with the right header.
- **Retry-After:** a parametrised test over every producible `503` code, plus a guard that fails when a
  new `503` site bypasses the choke point.
- **Health contract:** a schema test pinning the `inference` and `lease` field sets.
- **GPU endpoint:** parser fixtures for one and two cards; `nvidia-smi` absent; a card torch cannot
  see; a test that the read creates no CUDA context (a mocked torch that fails on context creation).
- **Admin UI (Vitest):** badge renders holder, reason and remaining time; expired lease disappears;
  no lease ID in the DOM.
- **Mutation controls (required).** At minimum: delete the lease check in `load_model`; delete it in
  `unload_model`; make expiry compare `>` instead of `>=`; return the lease ID from `GET`; drop
  `Retry-After` at the choke point; report `batch_backlog_rows` as `0`; read torch memory on a card
  with no context. Each must turn a test red. A surviving mutation gets a test, then the mutation is
  re-run as a negative control and recorded.
- **Fixtures must disagree with the defect.** Lease tests use a clock that can move past expiry, two
  different holders, and a resident model that differs from the requested one.
- **Hardware:** success criteria 1, 7 on the GPU node.

## 13. Implementation Considerations

- **Complexity:** medium. The lease touches the load path, which has a long history of races
  (`model_service.py:824-837`, `1482-1494`).
- **Recommended order:** (1) `Retry-After` choke point and queue fields; (2) lease service and guard in
  `ModelService`; (3) routes and headers; (4) refuse policy; (5) GPU endpoint; (6) Admin UI.
- **Risk: a guard in the routes only.** Five call sites today, and more later. FR-29.2.2 puts it in the
  service.
- **Risk: `queue_pending` misread.** Its name suggests waiting only; it counts running too. FR-29.7.2
  documents it rather than renaming a field consumers already read.
- **Risk: an estimate that looks exact.** FR-29.7.4 names it an estimate and returns `null` without
  data.
- **Risk: the lease ID leaks through a read.** FR-29.1.6 plus a mutation control.
- **Tracked debt (recorded, not fixed):** the management load and unload ignore `locked`
  (`management/models.py:173`, `192`; `model_service.py:782`). A steering model can be evicted from
  the Admin UI or by miStudio's `millm_load_model`. Retiring `locked` in favour of the lease (C8) is
  the intended fix.
- **Estimate:** backend 3–4 days, Admin UI 1 day, hardware acceptance half a day.

## 14. Open Questions

Batched for the operator. Each carries the default this PRD assumes until answered.

1. **May an agent take a model lease without approval?** (miStudio 034, open question 4.) Acquiring
   loads nothing, so load approval does not gate it. But an agent's lease blocks the operator's own
   loads for up to 2 hours. miLLM has no sign-in and no approval mechanism, so this is decided on the
   miStudio side. *miStudio's recommendation:* no approval; the holder is named in miLLM's refusal and
   lease state. *Default here:* miLLM grants any well-formed request.
2. **Should miStudio's GPU workers take the lease too?** (BRD-04 §9 question 6, second part.) It edges
   into co-residency, which BRD-04 puts out of scope. *Default:* no; the lease covers miLLM's resident
   model only.
3. **Lease scope across a swap.** May a lease be taken on a model that is not resident, as a
   reservation? And when the holder swaps models with its lease ID, should the lease follow to the new
   model? *Default:* resident model only (FR-29.1.3); the lease ends when its model leaves residency
   (FR-29.1.9, FR-29.3.4), and the holder takes a new one. Both consumers lease an already-resident
   model (miDataworks 005 FR-005.37–FR-005.38; miStudio 034 FR-18 "acquiring a lease loads nothing").
4. **Operator force-release.** Should the operator be able to end someone else's lease from the Admin
   UI, as a break-glass? BRD-04 names only the holder as releasing (R-04.38). *Default:* no; the TTL of
   at most 2 hours bounds a stale lease (RSK-04).
5. **`/v1/models` under a lease.** A locked model is the only one `/v1/models` lists
   (`models.py:38`, `90`). Should a leased model behave the same, so Open WebUI's picker does not offer
   models that will be refused? *Default:* no change; the refusal names holder and expiry.
6. **`Retry-After` values** (technical, resolved in the FTDD). The estimate formula, the per-code
   defaults, and what `MODEL_NOT_LOADED` should say, since retrying never helps until someone loads a
   model.
7. **What acceptance 16 compares against** (technical, resolved in the FTDD). torch's reserved figure
   excludes the CUDA context itself, which `nvidia-smi` counts as used. The FTDD states whether "miLLM's
   reserved memory matches `nvidia-smi` within 256 MiB" compares torch-reserved with the process's
   per-process figure from `nvidia-smi`, or something else, and whether a per-process read works inside
   the pod's process namespace. That read is also the only way to measure llama.cpp memory
   (FR-29.8.5).
8. **`in_flight` under CBM and the idle cache release** (technical, resolved in the FTDD;
   FR-29.7.3, FR-29.7.5).
9. **A batch's lease across a restart** (cross-feature, with Feature 26). Feature 26's coverage table
   maps acceptance 8 to "lease held across restart" (026 FPRD §3, R-04.22 row). Under FR-29.1.9 a
   restart ends every lease, because the model is no longer resident. *Default:* the resumed batch
   re-takes the lease once its model is resident again, and waits otherwise. The two FTDDs must agree
   on one reading.

## 15. Decisions from Clarifying Questions

Clarifying rounds were waived. Each question is pre-answered from a cited source. "Derived" marks an
answer that follows from a cited source rather than stating it. Questions with no source are Open
Questions above.

| # | Question | Answer | Source |
|---|---|---|---|
| D1 | Does the lease replace `locked`? | No. Beside it; callers migrate; `locked` retired later | Checkpoint C8; closes BRD-04 §9 question 1 |
| D2 | Maximum TTL? | 7200 seconds, renewable | Checkpoint technical default; closes BRD-04 §9 question 6, first part |
| D3 | TTL above the maximum? | Refused with `400`, never clamped | Derived: BRD-04 §2 "every request field is honoured or refused" |
| D4 | One lease per server or per model? | Per server | R-04.38 "one lease exists at a time"; one resident model (BRD-04 §3) |
| D5 | Does a lease block use of the leased model by others? | No; only load, unload and swap | R-04.39; R-04.19 (interactive chat interleaves) |
| D6 | Is the lease ID returned by reads? | Never | Derived: R-04.40 (the ID is what lets the holder through); no sign-in (BRD-04 §3) |
| D7 | Does a lease survive a restart? | No | Derived: residency does not survive (`main.py:134-139`); consumers resume on a lost lease (miDataworks 005 FR-005.39) |
| D8 | Where does the lease check live? | In `ModelService.load_model` and `unload_model` | Derived: R-04.39 covers five entry points; one guard cannot be forgotten by a route |
| D9 | Lease or `locked` first when both apply? | Lease first | Derived: R-04.39 requires naming holder and expiry |
| D10 | Default load policy? | Auto-load | R-04.41 (Open WebUI relies on it) |
| D11 | Refuse policy and a model loading now? | `503 model_loading` with `Retry-After` | Derived: R-04.41 forbids causing a load, not waiting for one; R-04.43 |
| D12 | Which `503`s get `Retry-After`? | Every `503` response on any route | R-04.43 "every 503"; PADR v1.5 §10 |
| D13 | Management `409 MODEL_BUSY` and `507`? | Unchanged | R-04.43 lists `503`s and keeps codes "as they are" |
| D14 | Batch backlog before Feature 26? | `null`, not `0` | Derived: R-04.44 "stable contract"; an unmeasured value is not a zero |
| D15 | Rename `queue_pending`? | No; document it as waiting plus running | R-04.44 "already in `/api/health/detailed`"; `request_queue.py:107-117` |
| D16 | Can the GPU read create a CUDA context? | No | `model_loader.py:220-226`; `gpu_placement.py:186-194` |
| D17 | GGUF memory per card? | Stated as not measured by torch | `model_loader.py:343-345`; Open Question 7 |
| D18 | Lease controls in the Admin UI? | Display only | R-04.42 "lease state appears"; Open Question 4 |
| D19 | MCP lease tools? | miStudio's (034 FR-18) | BRD-04 §4 (MCP tools out of scope) |
| D20 | Does the management load start honouring `locked`? | No; recorded as debt | C8 (no change to `locked` before retirement) |
