# miLLM ↔ Unified MCP Server Contract

**Status:** Normative for miLLM Feature 9 (Unified MCP), Feature 15 (Circuit Edge Sensing / circuit MCP surface), Feature 19 (Concurrent Circuit Serving) and Feature 24 (Probe Monitor Runtime), the model-lease / backpressure surface of Feature 29, the stateless probe scoring and per-request activations of Feature 27, and the Batch API of Feature 26. **Version:** 1.11 (2026-10-07)
**Consumer:** the unified MCP server that ships in the miStudio repo
(`backend/src/mcp_server/`), exposing `millm_runtime` / `millm_clusters` /
`millm_sensing` / `millm_circuits` / `millm_probes` tool categories against a miLLM
deployment.

## 1. Versioning rule

This contract is **additive-only**: miLLM may add endpoints, response fields, and
error codes; it must not rename or remove anything listed here, change field
types, or change status-code semantics without a new contract version. The MCP
server must tolerate unknown fields everywhere.

**v1.11 (2026-10-07)** is a strict additive superset of v1.10: it adds the Batch API (§4f) —
`/v1/files` and `/v1/batches` routes in OpenAI's shapes, the miLLM `POST /v1/batches/{id}/lease`
extension, the `completion_window` hours extension, the `millm` extension objects on the batch
object and on every output line, the `application/jsonl` results media type, the `batch:progress`
Socket.IO event, and the codes `INVALID_BATCH_REQUEST`, `BATCH_FILE_LIMIT`, `FILE_NOT_FOUND`,
`FILE_EXPIRED`, `FILE_DELETED`, `FILE_IN_USE`, `BATCH_NOT_FOUND`, `BATCH_STATE_CONFLICT`.
`GET /api/health/detailed`'s `inference.batch_backlog_rows` now carries a number (it was always
`null` before a batch API existed). **No tool is added here**: the batch tools are miStudio's
(034 phase 6, `millm_upload_batch_file`, `millm_submit_batch`, `millm_batch_status`,
`millm_cancel_batch`, `millm_batch_results`) to build against §4f. Nothing was renamed, removed or
re-typed.

**v1.10 (2026-10-06)** is a strict additive superset of v1.9: it adds `POST /api/probes/score`
(stateless probe scoring, §4 `millm_probes`), the `INVALID_PROBE_SCORE_REQUEST` and
`SAE_ACTIVATIONS_REFUSED` codes, the `return_sae_activations` request field and the optional
`millm` response object on `/v1/chat/completions` and `/v1/completions`, `X-miLLM-Probe-Verdicts`
on `/v1/completions` (FR-24.7 promised it on both routes; only chat sent it), two optional keys
(`model`, `checked_at`) in a stored parity report, and the not-scored reasons `batched_request`
(batched chat), `continuous_batching` (now on CBM chat and text, not only CBM streaming) and
`engine_unsupported` (llama.cpp). It also STATES the verdict boundary in §4d — `score >= threshold`
— which the runtime has applied since 2026-10-03. **No tool is added here**: the scoring tool is
miStudio's (034 `millm_score_probes`) to build against this section. Nothing was renamed, removed
or re-typed; a response that did not ask for activations carries no `millm` key.

**v1.9 (2026-10-06)** is a strict additive superset of v1.8: it adds the model-lease routes
(§4e), the `X-miLLM-Lease` and `X-miLLM-Load-Policy` request headers, the `MODEL_LEASED`,
`MODEL_NOT_RESIDENT`, `LEASE_NOT_FOUND`, `LEASE_EXPIRED` and `INVALID_LEASE_REQUEST` codes,
a `Retry-After` header on every `503`, the typed `inference` block and the `lease` field on
`GET /api/health/detailed`, and `GET /api/health/gpus`. **No tool is added here**: the lease
tools and the refuse-load header on scoring and generation tools are miStudio's to build
(miStudio `034` FR-18, FR-19) against this section. Nothing was renamed, removed or re-typed;
`inference` keeps every field it had, and absent headers reproduce v1.8 behaviour exactly.

**v1.8 (2026-10-02)** is a strict additive superset of v1.7: it adds
`POST /api/probes/{probe_id}/recalibrate`, the `millm_recalibrate_probe` tool, the
`PROBE_RECALIBRATION_MISMATCH` and `PROBE_THRESHOLD_UNCALIBRATED` error codes, and
`threshold_revision` on `_probe_summary`, each armed-probe `status()` entry and every
`probe_events` row. Nothing was renamed, removed or re-typed.

⚠ **AND IT CHANGES NO EXISTING SEMANTICS, WHICH IS THE PART WORTH STATING.** `on_conflict` is
still `rename|fail` with no `replace`; a probe's weights, layer, scope, rule and basis still
cannot change in place under a `probe_id`. What v1.8 permits is moving the DECISION BAR, which
each event already records for itself — see the bar-vs-detector boundary in §4d. A v1.7 client
that ignores the new route is unaffected, and a probe nobody recalibrates stays at
`threshold_revision` 1 forever.

**v1.6 (2026-09-27)** is a strict additive superset of v1.5.1: it adds the
`millm_probes` tool category (§4d), the `/api/probes/*` endpoints, the probe error
codes (§5), and the probe evidence-rung rule (§4d-bis). No earlier endpoint, field,
type, or error code changed. A v1.5 client that ignores the probe surface is
unaffected.

⚠ **This version is a CO-RELEASE.** The tools live in miStudio
(`backend/src/mcp_server/tools/millm_probes.py`) and the routes live here; neither
half is shippable alone, and this repo's history records what happens when that is
got wrong — sixteen `millm_circuit_*` tools were fully implemented, unit-tested and
documented in this file while never registered with the server, so the suite was
green and the contract said ✅ while no agent could call the feature. The guard is
`tests/unit/test_mcp_tool_paths_are_real.py` under
`MILLM_REQUIRE_CROSS_REPO_CHECKS=1`: it reads miStudio's tool module and requires
every path it calls to be a route this app actually serves.

**v1.1 (2026-07-20)** is a strict additive superset of v1.0 (Circuit Runtime,
BRD-MILLM-CIRCUITS-001): it adds the `millm_circuits` tool category (§4), the
`/api/circuits/*` endpoints and circuit edge-sensing routes, the circuit error
codes (§5), and the rung-vocabulary rule (§4a). No v1.0 endpoint, field, or
error code changed. A v1.0 client that ignores the circuit surface is unaffected.

**v1.3 (2026-07-20)** is a strict additive superset of v1.2: the circuit-sensing
status payload gains `requests_sensed`, `requests_truncated`, and `ws_throttled`
(§4a-quinquies). No endpoint, field, type, or error code was removed or changed.

**v1.2 (2026-07-20)** is a strict additive superset of v1.1: it tightens the
*meaning* of `reapplied`/`superseded` on the intensity route (§4a-ter, Feature
16 — the values are now truthful rather than unconditional) and adds
`truncated_layers` to the circuit-sensing status payload (§4a-quater, Feature
17). No endpoint, field name, type, or error code was removed or changed. A
v1.1 client keeps working; one that reads `reapplied` gets a more accurate
answer than before.

## 2. Response envelope

Every management endpoint (everything under `/api/`) returns:

```json
{ "success": true,  "data": <payload>, "error": null }
{ "success": false, "data": null, "error": { "code": "UPPER_SNAKE", "message": "…", "details": { } } }
```

- **The envelope is authoritative for machine handling** — unwrap in the
  client, never in tools. HTTP status *usually* mirrors the error class
  (400/404/409/422/503), but NOT always: the cluster import route returns
  some refusals as **HTTP 200 with `success: false`** (`PAYLOAD_TOO_LARGE`,
  `UNKNOWN_KIND`, contract `VALIDATION_ERROR`, `NO_ACTIVE_CLUSTER`) — house
  style for handler-level failures. Never branch on status alone.
- Non-envelope endpoints: `GET /api/clusters/{id}/export` returns the RAW
  portable cluster-definition document (the response *is* the artifact);
  `GET /api/health`, `/api/health/detailed`, and `/api/health/ready` return
  bare DTOs (no envelope); `/v1/*` endpoints speak the OpenAI error shape.
- FastAPI *request-validation* failures on management routes (bad query/body
  types, out-of-range values) return the default 422
  `{"detail": [...]}` shape — no envelope, no `error.code`. Clients should
  validate tool arguments before calling.

## 3. Health-gate contract

| Endpoint | Purpose | Notes |
|---|---|---|
| `GET /api/health` | **Gate hot path.** Cheap liveness: `{status, version, timestamp, uptime_seconds}` | No DB read. Poll ≤ 1/10 s (gate TTL). 3 s timeout recommended |
| `GET /api/health/detailed` | One-call status for `millm_status` | Includes `model_loaded`, `model_name`, `sae_attached`, `sae_id`, `inference`, `active_profile` and (v1.9) `lease`. `inference` is a typed, always-present block (v1.9): `backend`, `cbm_enabled`, `cbm_running`, `queue_pending` (waiting **plus** holding), `queue_max_concurrent`, `queue_max_pending`, and new `in_flight`, `queue_waiting`, `batch_backlog_rows`, `estimated_wait_seconds`, `error` — each new count `null` when unmeasured (under continuous batching; backlog before a batch API), never `0` |
| `GET /api/health/gpus` (v1.9) | Per-card memory | `{read_at, cards[], reason}`; per card `smi_index`, `uuid`, `name`, `total_mb`, `used_mb`, `free_mb`, `torch_index` (null when torch cannot see it), `torch_measured`, `millm_allocated_mb`/`millm_reserved_mb` (null unless miLLM placed a transformers model on the card), `engine_memory` (`"not_measured_by_torch"` for a resident GGUF card), `processes[{pid, used_mb}]`/`processes_reason`. `cards: []` + `reason` without nvidia-smi. No tool consumes it (miStudio 034 §9) |

**`active_profile`** (added for this contract):
`{id, name, source_kind: "manual"|"cluster", intensity, sensing_enabled} | null`.

**Gate semantics:** available ⇔ **2xx AND `status != "unhealthy"`** —
`degraded` IS available (miLLM with no model loaded must still accept cluster
imports and report status). Connection failure, timeout, non-2xx (including
3xx redirects, which the gate does not follow), or a 2xx body reporting
`status: "unhealthy"` (reserved; today's liveness endpoint only ever reports
healthy) mark the product unavailable; tools then return a structured
`{"unavailable": "millm", "reason": …}` result and are **never unregistered**
(MCP clients cache tool lists).

## 4. Endpoint inventory consumed by the MCP tools

### `millm_runtime`
| Tool | Endpoint |
|---|---|
| `millm_status` | `GET /api/health/detailed` |
| `millm_list_profiles` | `GET /api/profiles` |
| `millm_activate_profile` | `POST /api/profiles/{id}/activate` |
| `millm_deactivate_profile` | `POST /api/profiles/{id}/deactivate` |
| `millm_set_intensity` | `PUT /api/clusters/active/intensity` (`{intensity, reapply}`) |

### `millm_clusters`
| Tool | Endpoint |
|---|---|
| `millm_list_clusters` | `GET /api/clusters` |
| `millm_import_cluster` (inline) | `POST /api/clusters/import?activate=&on_conflict=` (body = raw v1 document; `on_conflict`: `rename` (default) \| `fail`) |
| `millm_import_cluster` (hub) | `POST /api/clusters/hub/import` (`{repo_id, filename, revision?, activate?, on_conflict?}`) |
| `millm_hub_search` | `GET /api/clusters/hub/search?q=&base_model=&limit=` |
| `millm_activate_cluster` | `POST /api/clusters/{id}/activate` |
| `millm_deactivate_cluster` | `POST /api/clusters/{id}/deactivate` |
| `millm_export_cluster` | `GET /api/clusters/{id}/export` (raw document — no envelope) |

### `millm_sensing`
| Tool | Endpoint |
|---|---|
| `millm_sensing_status` | `GET /api/sensing/status` |
| `millm_sensing_events` | `GET /api/sensing/events?profile_id=&limit=&since=` (list rows include context fields) |
| `millm_sensing_enable` / `_disable` | `POST /api/sensing/{profile_id}/enable` / `/disable` |
| `millm_sensing_config` | `PUT /api/sensing/{profile_id}/config` (`{min_k}`; null restores the all-sensable default) |

### `millm_circuits` (v1.5 — Circuit Runtime + Concurrent Serving + MCP surface)

> ✅ **STATUS CORRECTION — RESOLVED (2026-07-21, Feature 20).**
>
> **Kept, not deleted.** This is the record of how a contract table read as a
> shipped tool surface for an entire increment, and deleting it would erase the
> only evidence that it did.
>
> **What was wrong (2026-07-20):** the `F13 ✅` / `F15 ✅` marks described the
> **REST endpoints**, which shipped and were tested. They did NOT mean an MCP
> tool was registered — miStudio's server registered three miLLM categories,
> and there was no `millm_circuits` module at all. Every circuit tool named
> here was uncallable by an agent, and nothing said so.
>
> **What fixed it:** Feature 20 ships `millm_circuits` (16 tools) and, more
> importantly, the REACHABILITY ASSURANCE that would have caught this:
> registry, built-server and per-tool CALLER assertions, each proven by a
> mutation that turns them red. A mark on this table is now backed by a test
> that fails when the wiring is removed.
>
> **Marks below are three-state:** `REST ✅ · MCP ✅` (endpoint ships AND a tool
> is registered and caller-asserted), `REST ✅ · MCP not registered` (endpoint
> ships, no tool — an agent must call REST directly), `not served`.

**REST endpoints implemented (Features 13 + 15).** The HUB rows below remain
reserved and are NOT served — calls to them 404 today. They are listed so the
tool surface stays stable; do not register them against a deployment that has
not shipped them.

Edge sensing (Feature 15) shipped under the prefix **`/api/circuit-sensing`**,
not the `/api/circuits/…/sensing` paths this table originally reserved: it is
its own resource with its own retention and event store rather than a
sub-collection of a circuit, and the flat prefix matches `/api/sensing`.

| Tool | Endpoint | Status |
|---|---|---|
| `millm_circuit_status` | `GET /api/circuits/active` (active circuit + attached-SAE set + rung; `null` when none) | REST ✅ · MCP ✅ |
| `millm_list_circuits` | `GET /api/circuits?promoted=&min_rung=&limit=&offset=` (slim rows carry `rung`, `rung_language`, layers, edge_count) | REST ✅ · MCP ✅ |
| `millm_import_circuit` (inline) | `POST /api/circuits/import?on_conflict=` (body = raw circuit-definition document; its `kind` is bare `mistudio.circuit-definition` — the `/v1` names the schema VERSION, never the kind). Import does NOT activate — call `/{id}/activate?acknowledge_unvalidated=` separately, so the evidence gate is always an explicit step. | REST ✅ · MCP ✅ |
| _(hub import — deliberately NO tool, EC-20.5: a circuit references several SAEs by id, and importing one from a remote pack without checking those references serves it against the wrong feature basis)_ | `POST /api/circuits/hub/import` (`{repo_id, filename, revision?, activate?, on_conflict?, acknowledge_unvalidated?}`) | **F15 — not served** |
| _(hub search — deliberately no tool, same reason)_ | `GET /api/circuits/hub/search?q=&base_model=&limit=` (tag `mistudio-circuit-definition`) | **F15 — not served** |
| `millm_activate_circuit` | `POST /api/circuits/{id}/activate?acknowledge_unvalidated=` (fully serveable, or slice-fallback when the SAE set is incomplete) | REST ✅ · MCP ✅ |
| `millm_deactivate_circuit` | `POST /api/circuits/{id}/deactivate` | REST ✅ · MCP ✅ |
| `millm_export_circuit` | `GET /api/circuits/{id}/export` (raw circuit document — no envelope) | REST ✅ · MCP ✅ |
| `millm_set_circuit_intensity` | `PUT /api/circuits/active/intensity` (`{intensity, reapply, acknowledge_unvalidated}`; one global λ scales all layers. `acknowledge_unvalidated` re-passes the rung-2 gate on every dial — see §4a) | REST ✅ · MCP ✅ |
| `millm_circuit_sensing_status` | `GET /api/circuit-sensing/status` (armed state, layers, **`sensable_edges` + `unsensable_edges[{edge_key,reason,detail}]`**, `max_token_lag`, overhead, **`truncated_layers[]` (v1.2)**, **`requests_sensed`/`requests_truncated`/`ws_throttled` (v1.3)**, `enabled_circuits`) | REST ✅ · MCP ✅ |
| `millm_circuit_sensing_events` | `GET /api/circuit-sensing/events?circuit_id=&edge_key=&limit=&since=` (rows carry nested `up`/`down` `{layer,feature_idx,pos,act}`, `token_lag`, ±K `context_parts`, `edge_rung` + `edge_rung_language`) | REST ✅ · MCP ✅ |
| `millm_circuit_sensing_enable` | `POST /api/circuit-sensing/{circuit_id}/enable` (off by default, opt-in) | REST ✅ · MCP ✅ |
| `millm_circuit_sensing_disable` | `POST /api/circuit-sensing/{circuit_id}/disable` (recorded events are kept) | REST ✅ · MCP ✅ |
| `millm_circuit_sensing_event` | `GET /api/circuit-sensing/events/{event_id}` (one observation with its context window) | REST ✅ · MCP ✅ |
| `millm_circuit_sensing_clear` | `DELETE /api/circuit-sensing/events?circuit_id=` | REST ✅ · MCP ✅ |
| `millm_circuit_claims` | `GET /api/circuits/claims` (layer → claimant, `composed` flagged; the unit of contention is the LAYER) | REST ✅ · MCP ✅ |
| `millm_release_circuit_claims` | `POST /api/circuits/claims/release?circuit_id=` (**recovery** — release ONE circuit's stuck claims; scoped deliberately, there is no "release everything") | REST ✅ · MCP ✅ |
| `millm_delete_circuit` | `DELETE /api/circuits/{id}` (deactivates first if serving) | REST ✅ · MCP ✅ |

### `millm_probes` (v1.8 — Feature 24, Probe Monitor Runtime; v1.8 adds `millm_recalibrate_probe`, which moves a bar in place)

> **STATUS: routes served, tools registered.** Both halves shipped 2026-09-27. The
> three-state convention this file uses elsewhere applies: *served* means a path in
> `app.openapi()["paths"]`, *registered* means present in the MCP server's live
> registry, and *documented* means listed here. All three must hold. A row is not
> ✅ on two of them.
>
> ⚠ Five of these routes did not exist when phase 7 of Feature 024 was marked done,
> and the reachability test passed anyway because it asserted a **subset** of paths.
> The arming service, the identity gate, the parity engine, the k-sparse slice and the
> Hub service therefore had no production caller at all. The set — not a count — is now
> asserted against the FPRD.

| Tool | Endpoint |
|---|---|
| `millm_import_probe` | `POST /api/probes/import?on_conflict=rename\|fail` (body = the `mistudio.probe-definition/v1` document) **or** `POST /api/probes/hub/import` (`{repo_id, filename, revision?, on_conflict?}`) |
| `millm_list_probes` | `GET /api/probes?armed=` |
| `millm_arm_probe` | `POST /api/probes/{probe_id}/arm` (`{acknowledge_below_rung2?: bool, reason?: str, windows?: list[str]\|null}`) |
| `millm_recalibrate_probe` | `POST /api/probes/{probe_id}/recalibrate` (`{decision, mistudio_probe_id, mistudio_run_id?, calibration_id?, reason?}`) — moves the DECISION BAR of an already-imported, possibly armed probe. The body is `extra="forbid"` and therefore cannot carry a detector; a cut that cannot be matched to the stored `provenance.probe_id` is refused. See the bar-vs-detector boundary above |
| `millm_disarm_probe` | `POST /api/probes/{probe_id}/disarm` |
| `millm_probe_status` | `GET /api/probes/status` |
| `millm_probe_events` | `GET /api/probes/events?probe_id=&request_id=&limit=` |

**Stateless scoring (v1.10 — Feature 27).** `POST /api/probes/score`
(`{probe_ids?, inputs: [{token_ids, prompt_tokens?} | {messages} | {text}], windows?,
return_token_ids?}`, at most `PROBE_SCORE_MAX_INPUTS` (64) inputs and `PROBE_SCORE_MAX_PROBES` (8)
probes) asks imported probes — armed or not — about stored inputs and **persists nothing**: no
`probe_events` row, no runtime request context, no change to the armed set or a stored parity
report. Each input runs in its own admission slot with every SAE suppressed, one input per forward,
never packed. The result per input, probe and window carries `score`, `threshold`, `verdict`
(`true`/`false`/`null` — `null` means the probe said nothing and is never `false`), `rung`,
`rung_language` (verbatim), `provisional`, `threshold_revision`, `n_scored_tokens` and
`not_scored_reason`; each probe carries its stored parity status (`passed`/`failed`/`never_run`,
and `checked_against` — `"unknown"` for a report written before v1.10). With `probe_ids` given, any
mismatch refuses the request with arming's own error; omitted, mismatched probes are listed in
`skipped` with their code. A model change mid-request fails the remaining inputs with
`MODEL_CHANGED` and keeps the earlier results. ⚠ `text` inputs are refused with
`INVALID_PROBE_SCORE_REQUEST` until the one-user-turn render has reproduced a miStudio AUROC on
hardware (T-49). Consumed by miStudio's 034 scoring tool; no tool is registered in this contract
version. It is not a generation endpoint and carries no `X-miLLM-Steering` header.

Also served, and deliberately **not** exposed as tools:
`GET /api/probes/{probe_id}` (carries the whole definition — large, and an agent that
wants it can read the file it imported), `POST /api/probes/{probe_id}/parity` (an
operator action whose report is already on the list row), `GET /api/probes/hub/search`
and `GET /api/probes/hub/{repo_id:path}/definitions` (browsing is a human activity;
an agent importing from the Hub already knows the repo and filename), and
`DELETE /api/probes/events`.

**`on_conflict` is `rename|fail`. There is no `replace`, and this is normative.**
Overwriting a definition in place while its probe is armed would change the detector
underneath a running monitor while every event before and after kept the same
`probe_id` — the history would describe two different detectors as one. Re-importing
a rebuilt probe is **disarm → delete → import**.

> **MOVING A BAR IS NOT REPLACING A DETECTOR.**
>
> A probe definition carries two kinds of fact. The DETECTOR is everything that determines what
> number the probe produces: `head.weights`, `bias`, `norm_mean`, `norm_std`, `attention_query`,
> `read.layer`, `read.hook_point`, `scope`, `basis`, the `sae` block and its `feature_indices`,
> `aggregation.rule` and its `params`, `model`, and the `evidence` that says what the number is
> evidence of. The BAR is everything that determines only where that number is cut:
> `decision.threshold`, `target_fpr`, `realised_fpr`, `threshold_source`, `calibration`,
> `windows` and `length_bands`.
>
> `replace` was refused because it replaces the first kind, and the objection above stands exactly
> as written. It turns on a specific property of `probe_events`: the row records a `score` whose
> MEANING comes from the detector, and nothing on the row records which detector produced it.
> Change the weights and event #1's `score = 2.9` and event #900's `score = 2.9` are measurements
> of different quantities under one id, with nothing to tell them apart.
>
> A moved bar is not that, for one concrete reason: the event row ALREADY records the bar it was
> judged against, per verdict, at judgement time — including the length-band override — and nothing
> joins an event back to `probes.threshold`. After a re-cut, event #1 still says it was judged at
> 2.8786 and event #900 says 2.4011; both are true and both remain comparable, because the score
> beneath each was produced by the same weights at the same layer under the same scope with the
> same rule. The score is the measurement; the bar is the line drawn across it.
>
> THE RULE: a probe's identity is everything that determines its SCORE; its bar is everything that
> only determines the CUT. The first may never change in place under a probe id. The second may,
> through `POST /api/probes/{probe_id}/recalibrate`, which is `extra="forbid"` and therefore
> structurally incapable of carrying a detector, refuses any cut it cannot match to
> `provenance.probe_id`, never stores the incoming object as the definition, and refuses a
> threshold with no budget and no named source. `on_conflict` remains `rename|fail`.

**`GET /api/probes/events` list rows carry NO context text.** `context_text` and
`context_token_ids` are the decoded window around a firing position, i.e. the user's
words, and they are served only by `GET /api/probes/events/{event_id}` — which no
v1.6 tool consumes. An agent that needs one asks for it explicitly; it does not
receive a feed of prompts.

**Windows, and why a request now yields several verdicts (v1.7).** A probe scores the
tokens in a window and takes the **mean**. Scored over the whole request that mean mixes
the person's prompt with the model's reply, and the reply is long and low-scoring, so the
same conversation scores differently depending on how much the model happened to say.
Measured 2026-09-30: one sentence scored **+9.86 (fires)** against an 8-token reply and
**−0.32 (silent)** against a full one.

`windows` selects which slices the same weights are read over — any of `all`, `prompt`,
`response`, `last_user`. `null` (or an absent key) means `all`, `prompt` and `response`, plus
`last_user` **only when the definition carries a bar for it** (`decision.windows.last_user`) — a
probe exported before that window existed would otherwise report it against a global bar cut over
a different span. An explicit `[]` means the probe's own scope alone, which is what it did before
v1.7. Each window yields **its own verdict and its own event**, so one request can produce four.

⚠ **`prompt` IS EVERY TURN BEFORE THE REPLY; `last_user` IS THE NEWEST USER MESSAGE ALONE
(2026-10-04).** A client that resends the conversation (Open WebUI, LibreChat, most agent
frameworks) puts every earlier turn, the system prompt and any retrieved documents into `prompt`,
so a high-stakes earlier turn keeps firing on every later one. `last_user` reads only the newest
user message — its header, content and end-of-turn, the span miStudio calibrated it on, found by
rendering the conversation's prefixes with the same chat template. A request with no user turn, a
raw-text completion, or a template that rewrites earlier turns reports that window not scored, with
the reason (`no_user_turn`, `text_completion_has_no_user_turn`, `last_user_span_unresolved`,
`no_chat_template`, `no_user_header`), never a guessed span. The span starts at the newest user
message's OWN role header: a BOS or a system block the template injects in front of a first message
(Llama-3.1 adds ~25 tokens even when no system message was sent) is not part of it. Both repos are
held to `docs/schemas/last-user-span-cases.json`, byte-identical in each.

⚠ **SOME TEMPLATES CANNOT YIELD THIS WINDOW, AND SAY SO RATHER THAN GUESS.** The span is found by
rendering the conversation's prefixes, which requires that rendering `messages[:i+1]` extends
rendering `messages[:i]`. A template that renders an earlier turn differently once a later one
exists — one that strips reasoning from previous assistant turns, as some reasoning-model templates
do — fails that check on a multi-turn request and reports `last_user_span_unresolved`. Single-turn
requests are unaffected. A template that is not prefix-stable even for one message (measured:
Phi-4-mini-instruct) yields no `last_user` window at all, in either repo.

⚠ **A WINDOW'S OWN `length_bands` (`decision.windows[w].length_bands`) REFINE ITS OWN BAR**, cut
from that window's negatives and token counts, and take precedence over `decision.length_bands`.
A recalibrate push refreshes the bars of the windows a probe is ARMED with; a window it was not
armed with (a `last_user` bar pushed to a probe armed before it had one) is read only after a
re-arm.

⚠ **`windows` IS NOT `scope`, AND DOES NOT CHANGE IT.** `scope` is the probe's identity:
what it was trained on, what its threshold was cut under, and the only thing the parity gate
can verify. A probe whose CONTRACT scope is `prompt` or `response` is still refused at arm
time on reproducibility grounds, exactly as in v1.6.

⚠ **A WINDOW IS `provisional` WHEN IT HAS NO BAR OF ITS OWN, OR ITS WEIGHTS NEVER SAW ITS
TOKENS.** A threshold is the `(1 - target_fpr)` quantile of negatives aggregated under **one**
window; read over a different one the same number no longer names the same false-positive rate,
so a window without its own bar (`decision.windows`) is provisional. A window WITH its own bar is
not — except `response`, which stays provisional whatever bar it carries: miStudio's training
corpus is prose wrapped as a single user turn, so weights fitted there have never seen a model
reply. A provisional verdict still fires, by operator decision, and is a **ranking, not a rate**.

⚠ **`decision.length_bands` APPLY ONLY TO THE PROBE'S OWN SCOPE, AND THERE ONLY WHERE THEY WERE
CUT FROM THE SAME PASS AS THAT WINDOW'S BAR.** They are cut with the global bar. Every other window
is judged against its own bar (or the global one), never a band. On a non-`all` probe the own
window's bar (`decision.windows`, cut under miStudio's `input`) and the bands (cut under `user`)
come from different passes, so a window with its own bar is judged against that bar and the bands
apply only when it has none. Until 2026-10-03 the runtime applied the bands to every window.

**What a probe verdict is, on the `/v1` side.** Non-streaming responses carry
`X-miLLM-Probe-Verdicts` (RFC 8941 structured field); streaming responses carry a
final chunk with `choices: []` and a `millm_probe_verdicts` extension before
`[DONE]`. Both are additive: a v1.5 client ignores the header and skips the
empty-choices chunk, which is what the OpenAI SDK and Open WebUI both already do.

Since v1.7 each header member carries `window=<all|prompt|response|last_user>`, and
`provisional=?1` where it applies — both as **parameters**, so a v1.6 consumer reading the
name and the score is unaffected. The parameters cannot be omitted: one probe now emits
several members and they would otherwise share an identical name token. The streaming
payload gained `probe_id`, `window` and `provisional` for the same reason; it carried no
probe id at all before.

**The verdict boundary (v1.10 states it; the runtime has applied it since 2026-10-03).** A verdict
fires when `score >= threshold` — a score exactly on the bar FIRES, on every miLLM surface: the
live header, the streaming chunk, the event row and `POST /api/probes/score`. The producer cuts
the bar at a negative's score and counts that negative as admitted, so `>` would admit one negative
fewer than the definition's `realised_fpr`. The comparison lives in one function
(`ProbeRequestContext._verdict_for`), and a guard keeps it the only one.

### 4d-bis. Probe evidence-rung rule (v1.6 — Feature 24)

A probe's rung is **a number and the server's words for it, together**. The MCP client
must surface `rung_language` verbatim and must not compose its own phrase from `rung`.
miStudio owns this vocabulary; miLLM mirrors it; a third rendering in the MCP layer is
free to drift, and the thing most likely to drift is a detector's language rising above
its evidence.

| rung | `rung_language` |
|---|---|
| 0 | `trained` |
| 1 | `detects on held-out data` |
| 2 | `detects on unseen tasks` |
| 3 | `detects on unseen tasks, compared with a judge` |

⚠ **Rung 3 is "compared with", not "beats".** A judge scored the same data; it may have
won, and on miStudio's own reference run it did — the judge averaged 0.8744 AUROC against
the 1.2B probe's 0.7938, on all five sets. An agent that reads rung 3 as "better than a
judge" will over-trust it. The wording is load-bearing and must not be paraphrased.

**The rung is the highest PASSED, not a chain**, and `millm_arm_probe` on a probe below
rung 2 returns `UNVALIDATED_PROBE` (200 + envelope) unless
`acknowledge_below_rung2=true`. That acknowledgement is stored separately from the one
inside the definition: the agent or person who exported a weak probe and the one arming
it against live traffic are not necessarily the same, and only the second is choosing to
monitor with it.

**A probe records; it does not act.** No tool in this category stops, re-routes or
alters a generation, and none will without a new contract version. A verdict is evidence
for a downstream reader, not a gate. Nor is a verdict a cause: a probe detects, it does
not explain.

### 4a. Circuit evidence-rung rule (v1.1)
Every circuit and edge field carries `rung` (0–3 int) and `rung_language`
(server-rendered phrase), mirrored VERBATIM from miStudio's evidence ladder:
`0 → "associated"`, `1 → "suggested (attribution-supported)"`,
`2 → "causally validated (edge)"`, `3 → "faithfulness-tested (circuit)"`.
The circuit's rung is the MIN over its edges (empty → 0). The word **"causal"
must never appear for a rung below 2** — clients surface `rung_language`
verbatim, never re-phrase. Activating a circuit whose rung < 2 requires
`acknowledge_unvalidated=true`; without it the route refuses with
`UNVALIDATED_CIRCUIT` (200 + `success:false`, house style).

### 4a-bis. What an edge observation is — and is not (v1.1 — Feature 15)

An edge sensing row records that an edge's UPSTREAM member fired and its
DOWNSTREAM partner then fired within the lag window, in the authored
direction. Three cases deliberately produce NO row: a lone upstream fire, a
reversed pair, and a same-position co-fire (simultaneous firing is
co-activation, not a sequence — reporting it as up→down would assert an
ordering never observed).

**An observation is not validation.** It is co-activation evidence in the
authored direction; it never raises an edge's rung, and clients must not
present a high observation count as evidence of causality. Each row stores
`edge_rung_language` AS OF THE MOMENT OF OBSERVATION, so a later
re-validation in miStudio cannot retroactively upgrade old rows — clients
render the stored phrase, never today's.

**Absence of rows is not absence of firing.** `unsensable_edges` on the status
route lists every edge that could not be watched, with a reason
(`layer_not_attached` — common under slice-fallback, which serves one layer;
`no_activation_threshold`; `endpoint_not_a_feature` for a cluster-supernode
endpoint). A client that shows an empty event list without also surfacing
these is presenting absence of observation as evidence of absence.

Exclusions a client should expect: an armed circuit forces serial routing
(batched rows cannot be attributed to a request), speculative decoding is
skipped entirely (absolute positions diverge), and a request is capped at 20
observations with a `truncated` flag.

### 4a-ter. `reapplied` is authoritative (v1.2 — Feature 16)

`PUT /api/circuits/active/intensity` returns `reapplied` and `superseded`.
**`reapplied: true` means the value is LIVE**, not merely that a steering call
was made. It was previously unconditional, so an operator whose change was
overwritten by an in-flight request's steering restore still saw `true`.

- `reapplied: false, superseded: true` — **another authoritative write landed
  after yours** (a different operator, an activation, an attach/detach), so
  your value is no longer live. Re-issue it if you still want it.
- `reapplied: false, superseded: false` — it was never pushed to the model at
  all; the accompanying warning says why (typically a slice-fallback circuit,
  whose backing cluster profile owns its own intensity, or an apply that
  raised after the intensity had already been recorded).

A concurrent operator change WINS over an in-flight request: the request's
restore is skipped rather than overwriting the newer authoritative write. So a
per-request dial can no longer be the cause of `superseded` — only another
authoritative writer can.

### 4a-quater. `truncated_layers` names the incomplete layer (v1.2 — Feature 17)

`GET /api/circuit-sensing/status` adds `truncated_layers: int[]` — the layers
that dropped events in the last drained request. Additive; a client that
ignores it is unaffected.

**An empty list is a positive claim**, not an absence of information: every
armed layer reported completely. That is a different statement from "no events
were observed", and the distinction is why this names layers rather than being
a boolean. Previously the runtime knew only that *something* had truncated, so
a layer that observed everything was indistinguishable from one that dropped
events, and the honest reading of any empty result was "maybe".

An agent must therefore not report a circuit as quiet when the layer it cares
about appears in `truncated_layers` — the correct statement is that the
observation is incomplete for that layer. Truncation is a load-shedding
outcome, never evidence about the circuit, and (per §4a) it must never move an
edge's rung or soften the rung language.

### 4a-quinquies. Telling "quiet" apart from "broken" (v1.3 — Feature 17)

`GET /api/circuit-sensing/status` gains three counters. Each exists because two
very different situations produced identical readings.

| Field | Zero means | Non-zero means |
|---|---|---|
| `requests_sensed` | **No request has reached sensing at all** — a wiring or skip condition, NOT quiet traffic. Check `paused_reason`. | that many boundaries were observed |
| `requests_truncated` | no request has ever lost data since arming | the circuit HAS lost data, even if `truncated_layers` is now empty |
| `ws_throttled` | nothing was declined by the live-panel cap | the panel is showing a SAMPLE of a busy request — not a fault |

**`ws_throttled` is not loss.** Throttled events are persisted and readable
through the events API; `ws_dropped` is the field that indicates real delivery
failure. An agent must not report throttling as missing data.

**`requests_truncated` outlives `truncated_layers`.** The latter describes only
the last drained request and is superseded when the next one begins, so a rare
truncation can be missed by a poll. Read the pair: `truncated_layers` says which
layers to distrust right now, `requests_truncated` says whether this circuit has
ever lost data.

As in §4a-quater, none of these move an edge's rung or soften rung language.
Truncation and throttling are load-shedding outcomes, never evidence.

### 4a-sexies. Layer contention and composition (v1.5 — Feature 19)

Several circuits may serve AT ONCE, provided their claim sets are disjoint. The
unit of contention is the **LAYER**, not the feature, because steering composes
additively into one per-layer dict:

```
modified = original + Σ(strength_i × W_dec[i])
```

Two circuits steering DIFFERENT features on the same layer still contend — both
contribute to the same residual-stream sum, and nothing bounds that sum (the
±200 clamp bounds each member individually).

**New endpoints (additive):**

| Endpoint | Purpose |
|---|---|
| `GET /api/circuits/claims` | `[{layer, circuit_id, circuit_name, composed, steering_keys}]` — who holds which layer. Not answerable from the circuit list: two circuits can both be active while contending for nothing. |
| `POST /api/circuits/{id}/activate?allow_layer_overlap=true` | Compose onto a held layer. Refused by default. |
| `POST /api/circuits/claims/release?circuit_id=…` | **Recovery.** Release one circuit's stuck claims. Scoped to a single circuit; there is deliberately no "release everything". |

**When a refusal names a circuit that is not running.** Claims are released on
deactivation and on restart, but a failure in either path can leave one live.
The symptom is a `CIRCUIT_LAYER_CONTENTION` refusal whose `incumbent` is a
circuit that `GET /api/circuits/active` does not list. An agent's correct move
is `POST /api/circuits/claims/release` for that circuit id — NOT retrying the
activation, and NOT `allow_layer_overlap=true`, which would compose against a
circuit that is not there.

The response carries `warnings[]`: an unknown `circuit_id` and a circuit that
held no claims read differently, because the remedies differ. A circuit that is
still ACTIVE warns that it is now steering layers it does not hold — releasing
a live circuit's claims is a recovery action, not routine.

**New refusal code — `CIRCUIT_LAYER_CONTENTION`** (200 + `success:false`, house
style: the operation does not apply, nothing is missing).

Two shapes, and an agent MUST distinguish them by `details.overridable`:

| `overridable` | Meaning | Agent behaviour |
|---|---|---|
| `true` | **Contention** — same layer, different features. | May retry with `allow_layer_overlap=true`, but only on explicit human instruction. `details.measured_hazard` MUST be surfaced to the human first. |
| `false` | **Collision** — same `(layer, feature_idx)`. One strength would silently overwrite the other and the served value would belong to neither author. | **Never retry.** `details.override_param` is ABSENT precisely so it cannot be guessed. Report `details.colliding_keys` and stop. |

`details.measured_hazard` carries the close-out measurement behind the default
refusal, including its own caveat:

```json
{ "source": "GPU close-out 2026-07-20, LFM2.5-1.2B-Instruct",
  "one_layer_at_strength_5": "coherent, indistinguishable from baseline",
  "two_layers_at_strength_5": "degenerate output (repeated tokens)",
  "note": "one model, one fixture — indicative, not exhaustive" }
```

An agent must relay the `note` with the finding. Presenting the measurement as
more than one model and one fixture is the same overclaim the evidence ladder
exists to prevent.

**Composition SUPPRESSES the rung header.** While any served layer is composed,
`X-miLLM-Circuit-Rung` is OMITTED — the rung describes ONE circuit's evidence,
and when two circuits sum on a layer no single rung describes what the user
received. An agent must not substitute either circuit's rung for the missing
header, and must not describe a composed response as carrying that circuit's
evidence. This is the same rule that already omits the header for
slice-fallback.

`GET /api/circuits/{id}/activate` responses carry `composed_layers` and
`allowed_layer_overlap` when an override was used, so the acceptance is
recorded in the response and not only in the server log.

**Several circuits serving.** `GET /api/circuits/active` returns a LIST
(`?single=true` keeps the pre-v1.4 single-object shape and UNDER-REPORTS when
several serve). While more than one circuit serves in `full` mode, the
per-request dial and the rung header are BOTH suppressed — no single circuit's
dial or evidence describes the response. An agent must not substitute either
circuit's rung, and must not report "the active circuit" as singular.

**A slice-fallback serve is single-active.** A slice is steered by a cluster
profile, and only one profile can be active, so activating a second
slice-fallback circuit is refused with `reason:
"slice_fallback_is_single_active"`. That refusal is NOT overridable by
`allow_layer_overlap`; the remedy is to deactivate the named circuits or attach
the missing SAEs so the circuit can serve in full.

**`CIRCUIT_ALLOW_CONCURRENT`** gates the whole capability and defaults FALSE for
one release. With it off, a contention refusal names configuration as the
reason and CANNOT be overridden — an agent that retries with
`allow_layer_overlap=true` will be refused identically. It must report the
configuration requirement rather than looping.

### 4b. Circuit per-request dial (v1.1 — Feature 14)

`POST /v1/chat/completions` accepts the miLLM extension field
`steering_intensity` (`"off" | "min" | "max"`, or a numeric λ). When a circuit
is serving in `full` mode, one λ scales EVERY layer together, each member
through its own layer's SAE, for that request only. Two rules differ from the
cluster dial and clients must not assume the cluster semantics:

- **Both ends clamp.** Numeric λ is clamped into the circuit's declared
  `budget.intensity_range` intersected with the configured envelope — the floor
  as well as the ceiling. `0.1` against an authored `[0.5, 1.5]` resolves to
  `0.5`. Only an exact `0`/`"off"` is honored below the floor.
- **The default floor is `0.0`, not `0.5`.** Circuits use
  `CIRCUIT_INTENSITY_MIN` where clusters use `CLUSTER_INTENSITY_MIN`. A circuit
  whose document declares no `intensity_range` therefore makes `"min"` and
  `"off"` the same request — clients that offer a `min` control MUST disclose
  this rather than implying a non-zero bound.

Members are re-derived from their AUTHORED strengths, so the dial is absolute
rather than compounding on the circuit's stored intensity. Each member clamps
to ±200 at apply time, so at a high λ relative proportions can compress.
A circuit in `slice_fallback` is dialled through its backing cluster profile
and therefore follows the CLUSTER rules above, including the `0.5` floor.

Responses carry `X-miLLM-Circuit-Rung` in RFC 8941 structured form when — and
only when — a circuit is genuinely steering that response:

```
X-miLLM-Circuit-Rung: 2; language="causally validated (edge)"
```

The header is OMITTED for no active circuit, a slice-fallback serve, an
unparseable definition, or no SAE attached on any member layer. **Its absence
never means rung 0**; it means "no circuit-attributable steering here". Clients
MUST NOT derive evidence language from whether a circuit row is `is_active` —
`GET /api/circuits/active` carries a `steering: bool|null` field giving the
server's own verdict, and that (or this header) is the only correct source.
`null` means the server did not evaluate it (older build), not "not steering".

### 4c. Circuit slice-fallback (v1.1)
When not all of a circuit's referenced SAEs are attached, activation degrades
to the per-layer `cluster-definition/v1` slice (a valid v1 cluster document;
the partial-rendering marker rides in the slice name + `provenance.source_note`).
`GET /api/circuits/active` reports `serving_mode: "full" | "slice_fallback"` and,
in fallback, the bound layer(s) — a slice is never presented as the whole circuit.

Notes:
- **Member `meta` (contract rev 2026-07-17):** each member may carry an
  optional, extensible `meta` object — display/reference data only
  (description, category, label_source, interpretability, mean_activation,
  top_tokens, signature, example{text,span}, neuronpedia URL). ALL fields
  optional; unknown keys MUST be preserved (producers may add more); nothing
  in `meta` is ever load-bearing for steering math. `member.label` is
  populated by miStudio's export enrichment.
- **Member sign rule:** a NEGATIVE `strength` is already directional (the
  `sign` field is redundant there); a non-negative `strength` takes its
  direction from `sign`. Consumers must NOT blindly multiply — miStudio
  exports signed strengths with a derived sign, and multiplying
  double-negates suppressions into amplifications.
- `GET /api/clusters` summaries include `members`: `[feature_idx,
  label|null, strength]` triples (first 20) for tile/chip display.
- `repo_id` contains a slash; miLLM's hub routes declare `{repo_id:path}` —
  clients must NOT URL-encode the slash.
- Cluster activation enforces the declared-feature-space gate server-side
  (422 with a human-readable reason); imports of incompatible definitions
  succeed as **unbound** and refuse only at activation.
- Payload caps: import documents ≤ 1 MB, ≤ 20 members/definition,
  ≤ 50 definitions/bundle.

### 4e. Model lease and refuse-load policy (v1.9 — Feature 29)

No tool in this repo or the MCP server consumes these yet; miStudio `034` FR-18 builds
`millm_acquire_lease`, `millm_renew_lease`, `millm_release_lease` and `millm_lease_status` on
them, and FR-19 sends `X-miLLM-Load-Policy: refuse` on every scoring and generation call.

| Endpoint | Body / headers | Success | Refusals |
|---|---|---|---|
| `POST /api/models/{id}/lease` | `{holder, reason, ttl_seconds?}` | `201`, `{lease_id, model_id, model_name, holder, reason, acquired_at, renewed_at, expires_at, ttl_seconds, seconds_remaining}` — the ONLY response carrying `lease_id` | `400 INVALID_LEASE_REQUEST`, `404 MODEL_NOT_FOUND`, `409 MODEL_NOT_RESIDENT`, `409 MODEL_LEASED`, `409 MODEL_BUSY` |
| `GET /api/models/{id}/lease` | — | `200`, `{lease: <the lease without lease_id> \| null, last_ended: {…, end_reason, ended_at} \| null}` | `404 MODEL_NOT_FOUND` |
| `POST /api/models/{id}/lease/renew` | header `X-miLLM-Lease`; `{ttl_seconds?}` | `200`, the lease (new expiry = now + TTL) | `400`, `404 LEASE_NOT_FOUND`, `409 LEASE_EXPIRED` |
| `DELETE /api/models/{id}/lease` | header `X-miLLM-Lease` | `200`, the ended lease (`end_reason: released`) | `404 LEASE_NOT_FOUND`, `409 LEASE_EXPIRED` |

- `ttl_seconds`: integer 1–7200, default 7200; outside the range `400`, never clamped.
  `holder` 1–128 chars, `reason` 1–512, both required free text (no approval step).
- Only the resident, `LOADED` model can be leased; one live lease at a time. A lease ends on
  expiry, release, the model stopping being resident (`model_unloaded`) and **every restart**
  (`restart`: renew then answers `404 LEASE_NOT_FOUND`, "unknown lease; a restart ends every
  lease"). A lease ID for another model's path is `404`.
- `POST /api/models/{id}/load` and `/unload` read `X-miLLM-Lease`. Under a foreign lease they are
  `409 MODEL_LEASED` with `details {holder, reason, expires_at, leased_model_id,
  leased_model_name, operation, target_model_id}`; with the holder's ID they proceed and the lease
  ends with the unloaded model. They ignore `X-miLLM-Load-Policy`.
- `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` read both headers. Under a
  foreign lease an auto-load is `409 model_leased` (`invalid_request_error`), never
  `model_locked`. `X-miLLM-Load-Policy: refuse` (case-insensitive; default `auto`) answers a
  non-resident model with `409 model_not_resident` (naming requested and resident model, and the
  resident model's lease) and a model being loaded with `503 model_loading` + `Retry-After`;
  nothing loads. Any other value is `400 invalid_parameter`, `param: "X-miLLM-Load-Policy"`.
- **`Retry-After` (v1.9):** every `503` on any route carries it, in whole seconds ≥ 1, with the
  envelope and code unchanged. A client passes it to the agent unchanged. A refusal inside a
  committed stream carries `retry_after` in its error event instead.

### 4f. Batch API (v1.11 — Feature 26)

A durable JSONL job: upload a file, create a batch, poll it, cancel it, download results. Every
row runs through the same admission slot and the same service code as the synchronous endpoint;
a batch NEVER loads, unloads or swaps a model. No authentication, like every miLLM route. Every
route accepts `X-miLLM-Load-Policy` and `X-miLLM-Lease` and is not broken by them.

| Endpoint | Body / headers | Success | Refusals |
|---|---|---|---|
| `POST /v1/files` | multipart: `file` (JSONL), `purpose: "batch"` | `200`, OpenAI file object `{id, object: "file", bytes, created_at, expires_at, filename, purpose, status, status_details}` | `400 invalid_batch_request` (`param: purpose` / `file`), `400 batch_file_limit` (names the limit and the MEASURED value; nothing stored) |
| `GET /v1/files` | `?purpose&limit(1–10000, 100)&after&order(desc)` | `200`, `{object: "list", data, first_id, last_id, has_more}`, newest first | — |
| `GET /v1/files/{id}` | — | `200`, file object (`status` `processed`, `deleted` or `expired`) | `404 file_not_found` |
| `GET /v1/files/{id}/content` | — | `200`, the bytes, **`Content-Type: application/jsonl`** | `404 file_not_found`, `404 file_expired`, `404 file_deleted` |
| `DELETE /v1/files/{id}` | — | `200`, `{id, object: "file", deleted: true}` | `404`, `409 file_in_use` (names the non-terminal batch) |
| `POST /v1/batches` | `{input_file_id, endpoint, completion_window, metadata?, output_expires_after?: {anchor: "created_at", seconds}, pack?}`; optional `X-miLLM-Lease`, `X-miLLM-Strict` | `200`, batch object in `validating` | `404 file_not_found`; `400 invalid_batch_request` (`param` `endpoint`, `completion_window`, `output_expires_after.seconds`, `input_file_id`); `409 model_leased` (a live lease the request does not present); `400 unused_fields_refused` under strict |
| `GET /v1/batches` | `?limit(1–100, 20)&after` | `200`, list, newest first | — |
| `GET /v1/batches/{id}` | — | `200`, batch object, counts current to the last recorded chunk | `404 batch_not_found` |
| `POST /v1/batches/{id}/cancel` | — | `200`, batch object in `cancelling` (idempotent while cancelling) | `404`, `409 batch_state_conflict` (names the status) |
| `POST /v1/batches/{id}/lease` | header `X-miLLM-Lease` (required) | `200`, batch object, `millm.lease_mode: "caller"` | `400 invalid_lease_request`, `404 lease_not_found` (unknown, or a lease on another model), `409 batch_state_conflict` (terminal) |

- **Endpoints** a batch may name: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`,
  `/api/probes/score` — accepted only while the route is served (derived from the live OpenAPI
  document). A probe-score line's body has no `model`; its batch's model is the resident model.
- **Input line:** `{custom_id (≤512, unique), method: "POST", url: <the batch endpoint>, body}`.
  Every line is validated before any row runs, **strict whatever the headers said**; an invalid
  line becomes an error-file line `{id, custom_id, line, response: null, error: {code, message}}`
  and the batch's `errors.data` lists the first 100 as `{code, line, message, param}`.
- **Statuses** (exactly OpenAI's): `validating → in_progress | failed | cancelling`;
  `in_progress → finalizing | cancelling | expired | failed`; `finalizing → completed | failed`;
  `cancelling → cancelled`. A file with no valid line, lines naming two models, a non-resident
  model, or a foreign lease at the move to `in_progress` end `failed` with `errors` set.
- **`completion_window`:** `"24h"`, or whole hours `"<N>h"` from 1 to 168 (miLLM extension;
  `millm.completion_window_extension: true`). `expires_at = created_at + window`; unrun rows of an
  expired batch are error lines `batch_expired`, of a cancelled one `batch_cancelled`.
- **Limits:** 50,000 lines, 209,715,200 bytes per file (OpenAI's), 1 MiB per line.
- **Retention:** every batch file expires 30 days after creation (outputs: `output_expires_after`,
  3,600–2,592,000 s). Files a non-terminal batch references are never pruned.
- **Batch object extension:** `millm: {pack, waiting_reason (queued | model_not_resident |
  lease_unavailable | null), lease_mode (own | caller | null), completion_window_extension,
  output_expires_after}`.
- **Output line:** `{id, custom_id, response: {status_code, request_id, body, millm: {packed,
  headers}}, error: null}`, in input-line order. `body` is what the synchronous endpoint returns;
  `millm.headers` holds every `X-miLLM-*` value the synchronous route would set. A row failing on
  its own merits is an error-file line carrying the synchronous status and body.
- **Packing:** scoring rows (chat/text completions in scoring mode) run in right-padded packs
  only when `pack: true` (default from `BATCH_PACK_DEFAULT`, **false**). Generation and probe-score
  rows always run singly; embedding rows run singly. **Measured 2026-10-07 on JEV-9B-decision (bfloat16), RTX 3090, 10,000 scoring rows: single 23.2 rows/s, packed 63.7 rows/s (2.7x); 57 of 10,000 rows (0.57%) changed their top token, max |logprob diff| 0.349** — so packing is off unless a batch
  asks for it, and an unpacked row equals the synchronous endpoint.
- **Leases:** a batch takes its own lease (`holder: "millm-batch:<id>"`), renews it and releases it
  at any terminal status. A batch created with a valid `X-miLLM-Lease` — or handed one through
  `/lease` — runs under the caller's lease, renews it and never releases it. A restart ends every
  lease: a resumed batch re-acquires one or waits, running no row, until `expires_at`.
- **Progress:** Socket.IO `batch:progress` `{id, status, request_counts}` on every transition and
  after recorded chunks (throttled to one per second per batch). `GET /api/health/detailed`
  `inference.batch_backlog_rows` = rows not yet run across active batches.

## 5. Error codes the MCP client must map

`VALIDATION_ERROR` (422), `PROFILE_NOT_FOUND` (404), `MODEL_NOT_LOADED`
(**400** on management routes; the 503 mapping applies only to `/v1/*`),
`SAE_NOT_ATTACHED` (409/400 family), `PAYLOAD_TOO_LARGE` (200+envelope),
`UNKNOWN_KIND` (200+envelope), `HUB_UNAVAILABLE` (503, circuit open),
`NO_ACTIVE_CLUSTER` (200+envelope), `INTERNAL_ERROR` (500).
(`SENSING_EVENT_NOT_FOUND` (404) exists on the event-detail route, which no
MCP tool currently consumes.) Unknown codes: surface `error.message`
verbatim — messages are written to be user/agent-safe.

**v1.1 circuit codes:** `CIRCUIT_NOT_FOUND` (404), `SAE_SET_INCOMPLETE`
(422 — a referenced SAE is not attached; carries the offending
`{feature_idx, layer, sae_id}` list; activation degrades to slice-fallback),
`INCOMPATIBLE_FEATURE_SPACE` (422 — a referenced SAE's feature space does not
match the attached SAE at that layer), `UNVALIDATED_CIRCUIT` (200+envelope —
rung < 2 activation without `acknowledge_unvalidated=true`), `NO_ACTIVE_CIRCUIT`
(200+envelope — intensity/sensing call with no active circuit),
`AMBIGUOUS_ACTIVE_CIRCUIT` (200+envelope — the `PUT /api/circuits/active/intensity`
dial while more than one circuit is serving, so there is no single "active
circuit"; carries `details.active_circuits[{id, name}]`. Deactivate all but one,
or dial the layers through the owning cluster). Reused as-is:
`UNKNOWN_KIND`, `PAYLOAD_TOO_LARGE`, `HUB_UNAVAILABLE`. `CIRCUIT_SENSING_EVENT_NOT_FOUND` (404 — an edge sensing event id that does not exist; F15).

**v1.6 probe codes:** `PROBE_NOT_FOUND` (404), `PROBE_MODEL_MISMATCH` (409 — the
definition was fitted on a different model than the one loaded; `details.mismatches`
names **every** differing field, not the first, so an agent can tell "wrong model
loaded" from "wrong probe imported"), `PROBE_PARITY_FAILED` (409 — this build does not
reproduce the scores miStudio recorded for the definition's test vectors;
`details.max_abs_diff` may be **null**, meaning no vector could be compared at all,
which is not "zero off"), `UNVALIDATED_PROBE` (200+envelope — arming below rung 2
without `acknowledge_below_rung2=true`; carries `rung`, `rung_language` and
`next_step`), `PROBE_LIMIT` (409 — `PROBE_MAX_ARMED` already armed),
`PROBE_SAE_MISSING` (409 — a k-sparse probe's SAE is not downloaded here; names the
repo and path), `PROBE_SAE_MISMATCH` (409 — the downloaded SAE is not the one the probe
was fitted against), `PROBE_NO_MODEL_LOADED` (409 — nothing loaded, or its width and
depth are unreadable; **never defaulted**, because a fabricated `d_model` compares
cleanly against a definition and means nothing), `PROBE_HOOK_UNSUPPORTED` (409 — the
loaded runtime, e.g. llama.cpp, exposes no module tree to hook). Reused as-is:
`UNKNOWN_KIND`, `PAYLOAD_TOO_LARGE`, `HUB_UNAVAILABLE`, `VALIDATION_ERROR`.

**v1.9 lease codes:** `MODEL_LEASED` (409 — another holder leases the resident model; `details`
name holder, reason, `expires_at`; on `/v1` `model_leased`), `MODEL_NOT_RESIDENT` (409 — a lease
on a model that is not resident; on `/v1` `model_not_resident` answers the refuse-load policy),
`LEASE_NOT_FOUND` (404 — unknown ID, another model's ID, or any ID after a restart),
`LEASE_EXPIRED` (409 — `details.end_reason`), `INVALID_LEASE_REQUEST` (400 — `details.param`
names the field and limit). `MODEL_LOADING` (503, `/v1`) now has a producer: the refuse-load
policy for a model being loaded. Every `503` carries `Retry-After`.

**v1.10 codes (Feature 27):** `INVALID_PROBE_SCORE_REQUEST` (400 — a scoring request refused as a
whole before any forward: no or too many inputs, too many probes, two input kinds in one input,
`prompt_tokens` past the input, a token id outside the vocabulary, every probe skipped, or `text`
before its render is verified; `details.param` names the field) and `SAE_ACTIVATIONS_REFUSED` (400,
`/v1` `sae_activations_refused` — `n > 1`, `extra_messages`, several prompts, an ambiguous SAE, a
feature index past the SAE's width, `top_k` or the worst-case entry count over its cap). Per-input
`MODEL_CHANGED` and `TOKENIZATION_FAILED` are data inside a 200, never a status.

**v1.11 codes (Feature 26):** `INVALID_BATCH_REQUEST` (400 — `details.param` names the field),
`BATCH_FILE_LIMIT` (400 — names the limit and the measured value), `FILE_NOT_FOUND` (404),
`FILE_EXPIRED` (404 — content pruned by retention; the record remains), `FILE_DELETED` (404),
`FILE_IN_USE` (409 — `details.batch_id`), `BATCH_NOT_FOUND` (404), `BATCH_STATE_CONFLICT` (409 —
`details.status`). All are lowercased on `/v1` (e.g. `file_in_use`).

⚠ **`UNVALIDATED_PROBE` and `PROBE_MODEL_MISMATCH` must not be collapsed into one
"arming failed".** They call for opposite actions — the first is resolved by asserting
intent, the second can never be resolved by retrying.

## 6. Auth posture & deployment guidance (Task 1.3)

miLLM's management API is **unauthenticated by design** in the current
release. The supported topology is **same-network-segment deployment**: the
MCP server and miLLM must be reachable only on a trusted segment (cluster
namespace / LAN); do not expose `/api/*` to untrusted networks. The MCP
server's own bearer-token auth protects the agent-facing surface; it does
NOT add auth to miLLM. If a future miLLM release adds a management bearer
token, it will arrive as an additive `Authorization` requirement announced in
a new contract version; the client should already send a configurable
optional bearer header to be forward-compatible.

Deployment wiring (miStudio side): set `MILLM_API_URL` (e.g.
`http://millm-backend.millm.svc.cluster.local:8000`) and opt in via
`MCP_TOOL_CATEGORIES=...,millm_runtime,millm_clusters,millm_sensing,millm_circuits,millm_probes`.
With `MILLM_API_URL` unset, the millm_* categories are skipped at registration
(logged once) — miStudio-only deployments are unaffected.

⚠ **`millm_probes` is NOT in `DEFAULT_CATEGORIES`, and registering it in code is
necessary and not sufficient.** An explicit `MCP_TOOL_CATEGORIES` in the k8s manifest
or compose file overrides the default list, so a category absent from that variable is
absent from the deployment however thoroughly it is registered and tested. That is four
layers — the module, `MILLM_CATEGORY_MODULES`, `VALID_CATEGORIES`, and the manifest —
and this estate's sixteen-unregistered-tools failure lived in the fourth.

## 7. Cross-product agent flow (reference)

```
miStudio: export_cluster_definition(profile_id)     → v1 document
miLLM:    millm_import_cluster(definition=…, activate=true)
miLLM:    millm_set_intensity(1.2)
miLLM:    millm_sensing_enable(profile_id)
miLLM:    millm_sensing_events(profile_id, limit=20)
```

Circuit flow (v1.5) — discover/validate in miStudio, serve/sense in miLLM.

**F20 R2-11: this was not runnable.** It showed
`millm_import_circuit(definition=…, activate=true, acknowledge_unvalidated=…)`
— two arguments the tool does not take — so an agent following the reference
flow failed on line one with unexpected keyword arguments. It also contradicted
this section's own §4 row, which says import does NOT activate.

```
miStudio: export circuit          → circuit-definition/v1 (multi-SAE, edges, rungs)

miLLM:    millm_import_circuit(definition=…, on_conflict="rename")
          → import NEVER activates: the evidence gate is always a separate,
            explicit step

miLLM:    millm_list_circuits()   → read `rung` / `rung_language` BEFORE serving.
                                    Activation is gated at rung 2.

miLLM:    millm_activate_circuit(circuit_id)
          ├─ UNVALIDATED_CIRCUIT      → rung < 2. Report the phrase; re-send with
          │                             acknowledge_unvalidated=true ONLY on
          │                             explicit human instruction.
          ├─ CIRCUIT_LAYER_CONTENTION → read details.overridable:
          │    overridable=true       → surface details.measured_hazard INCLUDING
          │                             its note, then optionally re-send with
          │                             allow_layer_overlap=true
          │    overridable=false      → same feature on the same layer. NEVER
          │                             retry; report details.colliding_keys.
          │    incumbent not in millm_circuit_status()
          │                          → its claim is STUCK:
          │                             millm_release_circuit_claims(incumbent_id)
          │                             NOT a retry, NOT the override.
          └─ ok → serving_mode "full", or "slice_fallback" when the SAE set is
                  incomplete (a PER-LAYER PROJECTION — the circuit's own rung
                  does not describe it)

miLLM:    millm_circuit_status()   → a LIST. While several circuits serve, the
                                     dial and the rung header are BOTH
                                     suppressed.

miLLM:    millm_set_circuit_intensity(1.2)
          ├─ NO_ACTIVE_CIRCUIT        → activate something first
          ├─ AMBIGUOUS_ACTIVE_CIRCUIT → several serving; deactivate all but one
          └─ rung < 2                 → pass acknowledge_unvalidated=true again;
                                        the gate re-applies on every dial

miLLM:    millm_circuit_sensing_enable(circuit_id)
miLLM:    millm_circuit_sensing_status()   → if no events: this says WHETHER the
                                             circuit is armed and which edges are
                                             unsensable, with reasons. Do not poll
                                             events waiting for what cannot fire.
miLLM:    millm_circuit_sensing_events(circuit_id, limit=20)
          → OBSERVATIONS. A row is a correlation on live traffic; it NEVER
            raises a rung.

cleanup:  millm_circuit_sensing_clear(circuit_id="…")  ← scope is REQUIRED
          millm_deactivate_circuit(circuit_id)         ← releases its claims
```
