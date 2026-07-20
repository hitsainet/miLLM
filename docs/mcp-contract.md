# miLLM ↔ Unified MCP Server Contract

**Status:** Normative for miLLM Feature 9 (Unified MCP) and Feature 15 (Circuit Edge Sensing / circuit MCP surface). **Version:** 1.2 (2026-07-20)
**Consumer:** the unified MCP server that ships in the miStudio repo
(`backend/src/mcp_server/`), exposing `millm_runtime` / `millm_clusters` /
`millm_sensing` / `millm_circuits` tool categories against a miLLM deployment.

## 1. Versioning rule

This contract is **additive-only**: miLLM may add endpoints, response fields, and
error codes; it must not rename or remove anything listed here, change field
types, or change status-code semantics without a new contract version. The MCP
server must tolerate unknown fields everywhere.

**v1.1 (2026-07-20)** is a strict additive superset of v1.0 (Circuit Runtime,
BRD-MILLM-CIRCUITS-001): it adds the `millm_circuits` tool category (§4), the
`/api/circuits/*` endpoints and circuit edge-sensing routes, the circuit error
codes (§5), and the rung-vocabulary rule (§4a). No v1.0 endpoint, field, or
error code changed. A v1.0 client that ignores the circuit surface is unaffected.

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
| `GET /api/health/detailed` | One-call status for `millm_status` | Includes `model_loaded`, `model_name`, `sae_attached`, `sae_id`, `inference`, and `active_profile` |

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

### `millm_circuits` (v1.1 — Circuit Runtime)

> ⚠️ **STATUS CORRECTION (2026-07-20).** The `F13 ✅` / `F15 ✅` marks below
> describe the **REST endpoints**, which do ship and are covered by tests.
> They do **NOT** mean an MCP tool is registered. miStudio's MCP server
> registers exactly three miLLM categories — `millm_runtime`,
> `millm_clusters`, `millm_sensing` (`backend/src/mcp_server/tools/__init__.py`
> `MILLM_CATEGORY_MODULES`). **There is no `millm_circuits` module**, so none
> of the circuit tools named here are callable by an agent today. An agent
> reaching a circuit must call the REST route directly.
>
> This table read as a shipped tool surface for an entire increment. Wiring
> the category is tracked in BRD-MILLM-CIRCUITS-002.

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
| `millm_circuit_status` | `GET /api/circuits/active` (active circuit + attached-SAE set + rung; `null` when none) | REST ✅ · MCP not registered |
| `millm_list_circuits` | `GET /api/circuits?promoted=&min_rung=&limit=&offset=` (slim rows carry `rung`, `rung_language`, layers, edge_count) | REST ✅ · MCP not registered |
| `millm_import_circuit` (inline) | `POST /api/circuits/import?on_conflict=` (body = raw `mistudio.circuit-definition/v1` document). Import does NOT activate — call `/{id}/activate?acknowledge_unvalidated=` separately, so the evidence gate is always an explicit step. | REST ✅ · MCP not registered |
| `millm_import_circuit` (hub) | `POST /api/circuits/hub/import` (`{repo_id, filename, revision?, activate?, on_conflict?, acknowledge_unvalidated?}`) | **F15 — not served** |
| `millm_circuit_hub_search` | `GET /api/circuits/hub/search?q=&base_model=&limit=` (tag `mistudio-circuit-definition`) | **F15 — not served** |
| `millm_activate_circuit` | `POST /api/circuits/{id}/activate?acknowledge_unvalidated=` (fully serveable, or slice-fallback when the SAE set is incomplete) | REST ✅ · MCP not registered |
| `millm_deactivate_circuit` | `POST /api/circuits/{id}/deactivate` | REST ✅ · MCP not registered |
| `millm_export_circuit` | `GET /api/circuits/{id}/export` (raw circuit document — no envelope) | REST ✅ · MCP not registered |
| `millm_set_circuit_intensity` | `PUT /api/circuits/active/intensity` (`{intensity, reapply}`; one global λ scales all layers) | REST ✅ · MCP not registered |
| `millm_circuit_sensing_status` | `GET /api/circuit-sensing/status` (armed state, layers, **`sensable_edges` + `unsensable_edges[{edge_key,reason,detail}]`**, `max_token_lag`, overhead, **`truncated_layers[]` (v1.2)**, `enabled_circuits`) | REST ✅ · MCP not registered |
| `millm_circuit_sensing_events` | `GET /api/circuit-sensing/events?circuit_id=&edge_key=&limit=&since=` (rows carry nested `up`/`down` `{layer,feature_idx,pos,act}`, `token_lag`, ±K `context_parts`, `edge_rung` + `edge_rung_language`) | REST ✅ · MCP not registered |
| `millm_circuit_sensing_enable` / `_disable` | `POST /api/circuit-sensing/{circuit_id}/enable` / `/disable` (off by default, opt-in) | REST ✅ · MCP not registered |
| `millm_circuit_sensing_event` | `GET /api/circuit-sensing/events/{event_id}` (one observation with its context window) | REST ✅ · MCP not registered |
| `millm_circuit_sensing_clear` | `DELETE /api/circuit-sensing/events?circuit_id=` | REST ✅ · MCP not registered |

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
(200+envelope — intensity/sensing call with no active circuit). Reused as-is:
`UNKNOWN_KIND`, `PAYLOAD_TOO_LARGE`, `HUB_UNAVAILABLE`. `CIRCUIT_SENSING_EVENT_NOT_FOUND` (404 — an edge sensing event id that does not exist; F15).

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
`MCP_TOOL_CATEGORIES=...,millm_runtime,millm_clusters,millm_sensing`. With
`MILLM_API_URL` unset, the millm_* categories are skipped at registration
(logged once) — miStudio-only deployments are unaffected.

## 7. Cross-product agent flow (reference)

```
miStudio: export_cluster_definition(profile_id)     → v1 document
miLLM:    millm_import_cluster(definition=…, activate=true)
miLLM:    millm_set_intensity(1.2)
miLLM:    millm_sensing_enable(profile_id)
miLLM:    millm_sensing_events(profile_id, limit=20)
```

Circuit flow (v1.1) — discover/validate in miStudio, serve/sense in miLLM:

```
miStudio: export circuit               → circuit-definition/v1 (multi-SAE, edges, rungs)
miLLM:    millm_import_circuit(definition=…, activate=true,
                               acknowledge_unvalidated=<true if rung<2>)
          → serving_mode "full" (all SAEs attached) or "slice_fallback"
miLLM:    millm_set_circuit_intensity(1.2)          → one λ scales all layers
miLLM:    millm_circuit_sensing_enable(circuit_id)  → watch EDGES fire (up→down)
miLLM:    millm_circuit_sensing_events(circuit_id, limit=20)
```
