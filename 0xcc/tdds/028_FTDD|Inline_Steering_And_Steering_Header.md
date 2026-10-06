# Technical Design: Inline Steering and Steering-State Header
## miLLM Feature 28

**Version:** 1.0 · **Created:** 2026-10-06 · **Status:** Planned
**References:** `0xcc/prds/028_FPRD|Inline_Steering_And_Steering_Header.md` v1.1 (FR-28.1 – FR-28.4) ·
BRD-04 §5.8 (R-04.31 – R-04.34) · PPRD v1.5 Feature 28 · PADR v1.5 §10 · operator decisions of
2026-10-06 (T-78 – T-83, P-22, X-07, X-09; `~/app/miDataworks/0xcc/docs/brd-decisions-2026-10-05.md`
and `fprd-open-questions-2026-10-06.md`)
**Consumers that pin this document:** miDataworks 007 FTDD §6.2 (hash and header parser,
`~/app/miDataworks/0xcc/tdds/007_FTDD|Synthetic_Generation_And_Steered_Pairs.md`), miStudio 034
FR-21 (`X-miLLM-Steering` verbatim), miLLM Feature 26 FR-26.10.1 (batch output lines).

Code references are to miLLM at `f5c71b6` (HEAD, 2026-10-06). Every `path:line` was re-read at that
commit.

Terms: SAE = sparse autoencoder; CBM = continuous batching manager; SSE = Server-Sent Events;
RFC 8941 = the IETF standard for HTTP structured field values; GGUF = llama.cpp's model file format.

---

## 1. Executive Summary

The business goal is one sentence: every generation answer says exactly how it was steered, and a
caller can steer one request with any feature set without touching global state.

| Area | Decision | Rationale |
|---|---|---|
| Inline apply | `_apply_inline_steering`, beside `_apply_request_steering`, returning the existing per-layer saved shape | `_restore_request_profile` restores it unchanged, epoch guard included |
| Others under inline | `enable_steering(False)` on every other attached entry | T-79; disabling, not suppressing, keeps sensing and monitoring (T-80) |
| Explicit unsteered | `enable_steering(False)` on every attached entry | R-04.32; T-80 |
| Report source | An in-memory **snapshot of what the hooks read** — each entry's live values, its enabled flag, the epoch — taken inside the slot after generation and before restore | The header must describe what ran, not what was asked (PADR v1.5 §10) |
| Report labelling | A reader labels each steered entry by matching its values against the request record, the serving circuit, and the active profile; otherwise `manual` | A label is earned by equal values, never assumed from a flag |
| Hash | SHA-256 of a line-based canonical form; strengths as binary64 bit patterns | Byte-exact across languages; X-07 makes it canonical for the suite |
| Header | RFC 8941 list; a fixed parameter set per kind; serialised by one function | miDataworks parses it with an RFC 8941 library (007 FTDD §6.2) |
| Streaming | One terminal chunk `{"choices": [], "millm_steering": "<header value>"}` before `[DONE]` | Copies the probe verdict chunk clients already tolerate |
| Pre-load refusal | Non-empty `steering` on a non-resident model refused before auto-load | T-82; no load path re-attaches an SAE |
| First-SAE profile defect | Not changed; made observable | BRD-04 scope; FPRD Open Question 1 |

## 2. System Architecture

### 2.1 Request flow (serial chat; the other paths in §2.3)

```
POST /v1/chat/completions
  route (chat.py)
    reset_steering_memo()                      # also clears the new report and record
    pre-load checks: steering on GGUF row → 400; non-empty steering + model not resident → 400
    auto-load (unchanged)
    await inference.create_chat_completion(request)
      async with _admit():
        epoch_at_admission ← AttachedSAEState().steering_epoch      # recorded in _REQUEST_STEERING
        saved ← _apply_inline_steering(...)    | _apply_explicit_unsteered(...)
              | _apply_request_steering(profile, dial)   (unchanged)
        try: generate
        finally:
          snapshot ← SteeringSnapshot.capture(AttachedSAEState())   # what the hooks read
          _restore_request_profile(saved)                           # unchanged
      report ← await SteeringStateReader.describe(snapshot, record) # DB reads, after the slot
      _STEERING_REPORT.set(report)
    response.headers["X-miLLM-Steering"] = get_steering_report().header
```

### 2.2 Components

```
millm/core/steering_state.py          NEW, pure (no torch, no DB)
  SteeringItem (kind + fields) · canonical_set_form() · steering_set_hash()
  serialize_steering_header() · encode_name() · format_intensity()
millm/services/steering_report.py     NEW
  SteeringSnapshot.capture(state) → per-entry {sae_id, layer, values≠0, enabled}, epoch
  RequestSteeringRecord              what this request applied (kind, target, applied set, clamped, λ)
  SteeringStateReader.describe(snapshot, record, epoch_at_admission) → SteeringReport
millm/services/inference_service.py   MODIFIED
  _REQUEST_STEERING, _STEERING_REPORT contextvars; get_steering_report()
  _apply_inline_steering · _apply_explicit_unsteered · _finish_request_steering
  _has_steering_override reads `steering`
  create_text_completion gains the apply/finish block
  stream_chat_completion emits the terminal steering chunk
  _refuse_unsupported_llamacpp_request refuses `steering`
millm/api/schemas/openai.py           MODIFIED  InlineSteering, InlineSteeringFeature; fields on both requests
millm/api/routes/openai/chat.py       MODIFIED  pre-load refusals; header after generation
millm/api/routes/openai/completions.py MODIFIED  reset memo; pre-load refusals; header after generation
```

### 2.3 Every generation path and where its report comes from

| Path | Entry point | Snapshot taken | Request record |
|---|---|---|---|
| Serial chat | `create_chat_completion` (`inference_service.py:3574`) | in the `finally` before restore (`:3734`) | yes |
| Batched chat | `_create_batched_chat_completion` (`:3393`) | before restore (`:3506`) | yes |
| Streaming chat | `stream_chat_completion` (`:4288`) | **before the final chunk** (`:4650-4653`), beside `_probe_finish`; the restore in `finally` (`:4762`) runs after the stream has closed, too late to report | yes |
| Text completion | `create_text_completion` (`:4780`) | new block inside `_admit()` (`:4828`) | yes |
| Text scoring | `_score_text_completion` (`:4930`) | none; constant `none` (runs under `_unsteered`, `:1176`) | no |
| Chat scoring (Feature 25) | `_score_prompts`, the loop Feature 25 extracts from `_score_text_completion` and shares with text scoring (025 FTDD §1) | none; constant `none` | no |
| CBM chat / text | `_cbm_chat_completion` (`:5230`), `_cbm_text_completion` (`:5418`) | after the CBM returns | no (inline never reaches CBM, FR-28.1.8) |
| llama.cpp chat / text | `_llamacpp_chat_completion` (`:3960`), `_llamacpp_text_completion` (`:4221`) | after generation; entries are expected empty | no (steering refused, `:3885-3890`) |

The CBM and llama.cpp rows read global state that can move during generation. The `changed` flag
(FR-28.3.6) covers that, from an epoch read at entry.

### 2.4 Integration points

- **F16 epoch.** Inline and unsteered applies are not authoritative and never bump it
  (FR-28.1.10). Every authoritative writer already bumps it with a reason
  (`sae_service.py:882`, `:1210`, `:2317`, `:2510`, `:2566`, `:2588`, `:2606`, `:2624`;
  `profile_service.py:500`, `:566`; `circuit_service.py:483`, `:1236`, `:1446`).
- **F18 circuit plan.** The circuit label uses `_steering_circuit()` (`inference_service.py:1513`)
  and `CircuitSteeringEngine.plan_for` (`:1554`), the single "what is steering" predicate.
- **F24 verdict chunk.** Same shape and position (`inference_service.py:2586-2611`).
- **Feature 25.** `steering` moves from refused to honoured on transformers chat and completions
  (FR-25.3.7); scoring still refuses it (FR-25.7.2).
- **Feature 26.** Calls `steering_report_for_row()` (§6.4) per output line.

## 3. Technical Stack

- Python 3.11, FastAPI, Pydantic v2, structlog: unchanged.
- `hashlib.sha256` and `struct.pack(">d", x)` from the standard library for the hash.
- **No RFC 8941 library in miLLM.** The header is produced by one ~60-line serialiser covering only
  the types used here (token, string, integer, boolean). Its output is verified against an
  independent parser in a test. `http-sfv` is already the consumer's choice (007 FTDD §3), so
  the test parses miLLM's output with it as a dev-only dependency. That catches a grammar slip that
  a self-written round trip would agree with.
- No new runtime dependency, no migration, no configuration flag.

## 4. Data Design

No database change. All state is in memory, per request.

- **`RequestSteeringRecord`** (contextvar `_REQUEST_STEERING`, reset per request):
  `kind` (`inline` | `unsteered` | `profile` | `dial` | `none`), `sae_id`, `layer`, `applied`
  (`dict[int, float]`, non-zero, post-clamp), `clamped` (int), `profile_name`, `intensity`,
  `epoch_at_admission`.
- **`SteeringSnapshot`**: `epoch` and, per attached entry, `sae_id`, `layer`, `enabled`, and the
  live values with zeros removed. Values are copied (`get_steering_values` returns a copy,
  `sae_wrapper.py:445-447`).
- **`SteeringReport`** (contextvar `_STEERING_REPORT`): `items: list[SteeringItem]`, `header: str`.

**Validation (schema layer, before admission):**
- `features[].index` int ≥ 0; `strength` a finite float; booleans refused (as `openai.py:182-188`).
- Duplicate indices refused.
- `sae_id` with an empty `features` list refused.
- `steering` with `profile`, or with `steering_intensity`, refused (FR-28.2.1; T-78).

**Validation (inside the slot, against live state):** SAE selection (FR-28.1.3/4), index range
(FR-28.1.6), clamp (FR-28.1.7). All before any mutation.

## 5. API Design

### 5.1 Request

```json
"steering": {"sae_id": "<attached SAE id, optional>", "features": [{"index": 1234, "strength": 8.0}]}
"steering": {"features": []}
```

Errors use the existing OpenAI error envelope (`millm/api/routes/openai/errors.py:121`):

| Condition | Error | Status | Where |
|---|---|---|---|
| schema conflicts (§4) | `invalid_request_error`, `param` names the field | 400 | Pydantic |
| GGUF model row with `steering` | `EngineUnsupportedError` | 400 | route, before auto-load |
| non-empty set, model not resident | `SAE_NOT_ATTACHED` | 400 | route, before auto-load (T-82) |
| non-empty set, no SAE / unknown `sae_id` | `SAE_NOT_ATTACHED` | 400 | inside slot |
| `sae_id` omitted with ≥ 2 entries, or one `sae_id` at ≥ 2 layers | `invalid_request_error`, naming each `(sae_id, layer)` | 400 | inside slot |
| index out of range | `INVALID_FEATURE_INDEX` | 400 | inside slot |

A streaming request must not commit a 200 before these are known. The route therefore runs the
in-slot checks once as a dry run before returning the `StreamingResponse`, as it already does for
a bad profile name (`chat.py:232-237`). State can still change between the check and admission; the
in-slot check is authoritative, and a stream that fails there ends with the existing error chunk.

### 5.2 The `X-miLLM-Steering` header (normative)

The field value is an RFC 8941 **List** (§3.1), serialised per RFC 8941 §4.1. Each member is an
**Item** whose bare item is a **Token** naming the kind, followed by Parameters.

```
X-miLLM-Steering = sf-list                              ; RFC 8941 §3.1
member           = kind *( ";" param )
kind             = "none" / "unknown" / "profile" / "inline" / "manual" / "circuit"
```

| Kind | Parameters, in this order | Notes |
|---|---|---|
| `none` | `changed`? | Only member when present |
| `unknown` | `reason`?, `changed`? | Only member when present |
| `profile` | `name`, `source`, `intensity`, `sae`, `layer`, `features`, `hash`, `clamped`?, `changed`? | |
| `inline` | `sae`, `layer`, `features`, `hash`, `clamped`?, `changed`? | |
| `manual` | `sae`, `layer`, `features`, `hash`, `changed`? | |
| `circuit` | `id`, `intensity`, `composed`?, `changed`? | One member per circuit, not per layer |

`?` = present only when it applies.

| Parameter | RFC 8941 type | Value |
|---|---|---|
| `sae` | String | the attached SAE's ID |
| `layer` | Integer | the entry's layer |
| `features` | Integer | count of non-zero applied features |
| `hash` | String | `sha256:` + 64 lowercase hex digits (§5.3) |
| `clamped` | Integer ≥ 1 | features whose value `clamp_steering` changed; omitted when 0 |
| `name` | String | the profile name, percent-encoded (below) |
| `source` | Token | `request` (the request's `profile` field) or `active` (the globally active profile) |
| `intensity` | String | the effective λ as the shortest decimal that round-trips to the binary64 value (Python `repr(float)`); compare by parsing to a float, never as text |
| `id` | String | the circuit ID |
| `composed` | Boolean, true | a served layer carries more than one circuit (F19) |
| `changed` | Boolean, true | the steering epoch moved during the request (FR-28.3.6); on every member when set |
| `reason` | Token | `claims_unreadable`, `profile_unreadable`, `llamacpp_entries`, or `read_failed` |

Rules:
- **Booleans** are serialised bare (`changed`, not `changed=?1`), as RFC 8941 §4.1.1.2 requires
  for a true boolean parameter.
- **Why `intensity` is a String:** an RFC 8941 Decimal has at most three fractional digits
  (§3.3.2). λ = 0.4375 would arrive as 0.438 and fail an exact comparison.
- **Percent-encoding of `name`:** RFC 8941 Strings are printable ASCII only. Each UTF-8 byte outside
  `%x20-7E`, and each of `%`, `"` and `\`, is written as `%XX`, uppercase hex. The consumer decodes.
  RFC 9651 Display Strings are not used, because the consumer's library is RFC 8941 only.
- **Member order:** circuits first by `id`, then SAE items by `layer`, then `sae`.
- **Hash scope:** the hash covers SAE and features only. `layer` travels beside it, so a client that
  knows only what it sent can still verify.

Examples (TV-1 is in §5.3):

```
X-miLLM-Steering: none
X-miLLM-Steering: inline;sae="LiquidAI--LFM2.5-1.2B-Instruct-sae--layer_11";layer=11;features=1;hash="sha256:a4eae730e5105f422b93abeece7e03bda3c29b096c07aa9cee6f247e12844105"
X-miLLM-Steering: profile;name="humor";source=active;intensity="1.0";sae="LiquidAI--LFM2.5-1.2B-Instruct-sae--layer_11";layer=11;features=12;hash="sha256:…"
X-miLLM-Steering: circuit;id="crc_124fd83d1f2a";intensity="0.5";changed
X-miLLM-Steering: unknown;reason=claims_unreadable
```

**Streaming.** No `X-miLLM-Steering` header is sent before the body. The stream ends with:

```
data: {"id": "...", "object": "chat.completion.chunk", "created": 0, "model": "...", "choices": [], "millm_steering": "<the exact header value>"}
```

It comes after the final content chunk and the probe verdict chunk, and before `data: [DONE]`. It is
always emitted. Its absence means a server that predates this feature.

### 5.3 The steering-set hash (normative, canonical across the suite: X-07)

**Applied set.** For an SAE entry, the applied set is the map `index → strength` the hook reads,
with every zero strength removed. `-0.0` is a zero. For inline steering that is
`clamp_steering(strength)` for each requested feature. For a profile it is
`clamp_steering(float(stored) * λ)`, computed in binary64 exactly as both apply paths do
(`inference_service.py:2114`; `profile_service.py:444-447`).

**Canonical form.** A sequence of lines, each ended by one LF (`0x0A`), the last included,
encoded as UTF-8:

```
line 1   millm.steering-set/v1
line 2   sae=<sae_id>
line 3+  <index>:<bits>        one per applied feature, ascending index
```

- `<index>`: base-10, no sign, no leading zeros (`0` is `0`).
- `<bits>`: the strength's IEEE-754 binary64 value, big-endian, as 16 lowercase hex digits
  (Python `struct.pack(">d", s).hex()`; JavaScript `DataView.setFloat64(0, s)` then hex).
- No spaces anywhere. No byte-order mark.

**Hash.** `"sha256:" + lowercase_hex(SHA-256(canonical_form_bytes))`.

**Why bit patterns and not decimal text:** shortest-decimal formatting differs between languages in
its exponent form (`1e-05` against `1e-5`). A bit pattern has one spelling. JSON input `0.1`
parses to the same binary64 in every conforming parser.

**Published test vectors** (also in the API reference, and pinned in
`tests/unit/core/test_steering_state.py`):

| ID | SAE ID | Input features | Canonical form (`\n` = LF) | Hash |
|---|---|---|---|---|
| TV-1 | `LiquidAI--LFM2.5-1.2B-Instruct-sae--layer_11` | `[{1234, 8.0}]` | `millm.steering-set/v1\nsae=LiquidAI--LFM2.5-1.2B-Instruct-sae--layer_11\n1234:4020000000000000\n` | `sha256:a4eae730e5105f422b93abeece7e03bda3c29b096c07aa9cee6f247e12844105` |
| TV-2 | same | `[{1234, -8.0}]` | `…\n1234:c020000000000000\n` | `sha256:cc6c48faa720096e5c65fc1b2afab1b9b87659fb4465f20db421a2ed7fed78a7` |
| TV-3 | `jbloom--gemma-2-2b-res-jb--layer_20--width_16k--average_l0_71` | `[{77, -2.5}, {5, 0.1}, {900, 0.0}, {12, -0.0}]` | `millm.steering-set/v1\nsae=jbloom--gemma-2-2b-res-jb--layer_20--width_16k--average_l0_71\n5:3fb999999999999a\n77:c004000000000000\n` | `sha256:b843912201c5c18f872976e35af288e5f484d81e31a03a1dc16c3b9dbd82e710` |
| TV-4 | same as TV-3 | `[{3, 500.0}]` (clamped to 200.0) | `…\n3:4069000000000000\n` | `sha256:3f51779e6e33ee248acf6f520cb9475b4ef62f68275060bb69d2e228033b78df` |

TV-1 and TV-2 are the two sides of a P-22 steered pair: one SAE feature index, opposite strengths.
TV-3 exercises ordering, a fraction, and dropped zeros. TV-4 shows that the hash describes the
applied value: a client hashing the 500.0 it sent gets a different hash, and the header's
`clamped=1` says why.

### 5.4 Security and performance principles

- No prompt text and no raw strengths are echoed; only counts and a hash.
- No new authentication; miLLM is LAN-only (BRD-04 §3).
- Building the report holds no slot on the non-streaming paths: the snapshot is a memory copy taken
  inside the slot, and database reads happen after release. The streaming path must describe before
  its final chunk, still inside the slot, so it pays at most one active-profile read there.

## 6. Component Architecture

### 6.1 `millm/core/steering_state.py` (pure)

Holds the hash, the canonical form, the header serialiser and the item dataclasses. It has no
imports from `millm.services`, so it can be tested in microseconds and copied by a consumer.

### 6.2 `millm/services/steering_report.py`

- `SteeringSnapshot.capture(state) -> SteeringSnapshot`: synchronous; reads `entries()`
  (`sae_service.py:544-547`), each `is_steering_enabled` and `get_steering_values()`, and
  `steering_epoch` (`sae_service.py:475`). Never raises; on error returns a snapshot marked
  `read_failed`.
- `SteeringStateReader.describe(snapshot, record) -> SteeringReport`, async. For each entry that
  is enabled and has a non-empty applied set:
  1. **Request record matches** (same `(sae_id, layer)` and equal applied set) → `inline`, or
     `profile; source=request`.
  2. **Serving circuit covers the layer** (`_steering_circuit()` and its plan) → grouped into one
     `circuit` item, λ from the record when the request dialled it, else the circuit row's
     `intensity` (`millm/db/models/circuit.py:66`). `composed` from a strict variant of
     `_any_layer_composed` (`inference_service.py:1568`); an unreadable claims table makes the whole
     report `unknown;reason=claims_unreadable`. The rung echo keeps its fail-open choice; this
     header is an honesty statement and fails closed.
  3. **Active profile's applied set equals the entry's** (`ProfileRepository.get_active()`) →
     `profile; source=active`.
  4. Otherwise → `manual`.
  If nothing is steered → `none`. If the epoch at capture differs from `epoch_at_admission`, every
  item carries `changed`. A `profile` item whose profile records an `sae_id` other than the entry's
  logs `profile_sae_mismatch` (FPRD §13).
- Any exception inside `describe` yields `unknown;reason=read_failed` and a warning. It never
  propagates (FR-28.3.7).

### 6.3 Inference-service seams

- `_apply_inline_steering(steering, request_id) -> dict`: select the entry, validate, clamp, save
  every attached entry's `values` and `enabled`, then clear and set the target and disable the
  others. It returns `{"layers": [...], "epoch": ..., "request_id": ..., "kind": "inline"}`.
- `_apply_explicit_unsteered(request_id) -> dict | None`: the same saved shape; disables every
  entry; returns `None` when nothing is attached.
- `_restore_request_profile` branches on `saved.get("layers") is not None` instead of
  `saved.get("circuit")` (`inference_service.py:2206`), so the per-layer restore serves circuit,
  inline and unsteered shapes alike. `path` in the skip log reads `saved.get("kind")`.
- `_finish_request_steering(saved) -> SteeringSnapshot`: capture, then restore. It replaces the bare
  restore call at the non-streaming generation sites, so they cannot restore without capturing. The
  streaming path captures before its final chunk and keeps its two restores (setup-error `:4450`,
  normal `:4762`) unchanged.
- `_has_steering_override` adds `getattr(request, "steering", None) is not None`.

### 6.4 Route and batch seam

- `get_steering_report() -> SteeringReport | None` reads the contextvar after the awaited service
  call, the pattern `get_probe_verdicts()` uses (`inference_service.py:328-330`).
- `steering_report_for_row(...)` returns the same header string for Feature 26.
- Header, stream chunk and batch line all call `serialize_steering_header` once.

## 7. State Management

- **Global state** is `AttachedSAEState` (process singleton). This feature mutates it only inside
  `_admit()` and restores it in `finally`. `MAX_CONCURRENT_REQUESTS` stays 1
  (`millm/core/config.py:255`).
- **Request state** lives in contextvars. The InferenceService is a process singleton, so it cannot
  hold per-request state (`inference_service.py:270-277`). `reset_steering_memo()`
  (`inference_service.py:333-349`) also resets `_REQUEST_STEERING` and `_STEERING_REPORT`.
  `completions.py` gains the reset call it lacks today, since text completions now read
  `_steering_circuit()`.
- **Streaming.** `StreamingResponse` iterates the generator in another task, so the route cannot
  read the contextvar. The generator emits the chunk itself, as it does for verdicts.
- **No caching.** The circuit memo `_STEERING_CIRCUIT_MEMO` is reused within one request only.

## 8. Security Considerations

- Inputs are validated at the schema (types, finiteness, duplicates, conflicts) and against live
  state (attachment, index range) before any mutation.
- Strength magnitude is bounded by `clamp_steering` (`steering_range.py:11-16`), never by trusting
  the client.
- The header carries operator-chosen profile names, percent-encoded so a name cannot inject header
  syntax.
- No authentication change. BRD-04 §7 excludes it.

## 9. Performance & Scalability

- Inline apply cost equals a profile apply: one `_rebuild_steering_delta` per touched entry
  (`sae_wrapper.py:472-497`).
- The snapshot is a dictionary copy per attached entry, about 2–5 entries in practice.
- `describe` makes at most one active-profile read and reuses the memoised circuit read. Target:
  under 5 ms at p95 on the serial path; the overhead test in §10 measures it.
- No change to throughput: one request in flight, as before.

## 10. Testing Strategy

| Level | What | Where |
|---|---|---|
| Pure unit | TV-1 – TV-4; hash order-independence; `-0.0`; header grammar per kind; percent-encoding; parse back with `http-sfv` | `tests/unit/core/test_steering_state.py` |
| Schema | every §5.1 schema refusal; booleans; duplicates; conflicts; text-completion fields | `tests/unit/api/test_openai_schemas.py` (extend) |
| Service | inline apply and restore exact; others disabled not suppressed; unsteered; epoch skip and `changed`; `_has_steering_override` | `tests/unit/services/test_inline_steering.py` |
| **Report honesty** | the header is computed from live state: a test where the request asks for X while the hooks run Y must report Y | `tests/unit/services/test_steering_report.py` |
| Real model | tiny real Llama with a real `LoadedSAE`, the pattern of `tests/unit/services/test_scoring_completions.py:48-54`, `:416`: inline equals profile, unsteered equals detached, greedy | `tests/unit/services/test_inline_steering_real_model.py` |
| Path coverage | every generation entry point emits a report; entry points discovered from the service, not listed by hand | `tests/unit/services/test_steering_report_every_path.py` |
| Routes | header present and correct on chat, completions, streaming chunk; pre-load refusals do not call the loader | `tests/integration/api/test_steering_header_routes.py` |
| Hardware | BRD-04 acceptance 12 on LFM2.5-1.2B-Instruct with an attached SAE | FTASKS 7.x |

Fixtures must not agree by construction. Inline and profile tests use a profile and an inline set
built separately, from different literals. The honesty test injects a mismatch on purpose.

**Mutation controls** (each must turn the suite red; recorded in the review notes): skip the clamp
on the inline path; skip the restore; restore without the epoch guard; report from the request
record instead of the snapshot; hard-code the kind to `inline`; drop `steering` from
`_has_steering_override`; remove the header line in each route; remove the stream chunk.

## 11. Deployment & DevOps

- No migration, no environment variable, no Kubernetes change, no feature flag. Old clients send no
  `steering` and ignore unknown headers.
- Logs: `request_inline_steering_applied` (sae, layer, feature count, clamped count),
  `inline_steering_values_clamped`, `request_unsteered_applied`, `steering_report_unknown` (reason),
  `profile_sae_mismatch`.
- Rollback is a plain revert: no persisted state depends on the feature.
- The API reference and the manual's OpenAI-API page gain the request field, the header grammar and
  the test vectors.

## 12. Risk Assessment

| Risk | Effect | Mitigation |
|---|---|---|
| A path restores without capturing | That path reports nothing or something stale | `_finish_request_steering` replaces the bare restore; discovery test over entry points |
| Labelling by value equality mislabels a profile as `manual` | Consumer sees `manual` for a profile-steered answer | Same float formula in both apply paths (§5.3); test with λ ≠ 1 |
| Header grammar slip breaks the consumer's parser | Every steered pair recorded unreported | Parse miLLM's output with the consumer's library in a test |
| Hash spelling drifts between repos | Every inline pair mismatches | Published vectors pinned on both sides (X-07) |
| Disabled-not-suppressed differs from scoring's suppression | Unsteered generation still sensed (intended, T-80) | Documented; scoring keeps `_unsteered` |
| Streaming pre-check passes, in-slot check fails | A 200 stream that ends in an error chunk | Existing pattern; rare (needs a detach between check and admission) |
| First-SAE profile defect | A profile steers the wrong SAE | Out of scope; the header names the SAE actually steered; FPRD Open Question 1 |

**Alternatives considered:**
- *Echo the request:* rejected by PADR v1.5 §10; it cannot see an operator's concurrent change.
- *Provenance tag written by every authoritative writer:* exact, but 13 writer sites to keep in step,
  and a missed one mislabels silently. Value matching needs no writer changes.
- *JSON canonical form with decimal floats:* cross-language float text differs.
- *`suppressed()` for "unsteered":* also silences sensing and monitoring
  (`sae_wrapper.py:576`, `:681`), against T-80.

**Complexity:** medium.

## 13. Development Phases

| Phase | Content | Depends on | Milestone |
|---|---|---|---|
| 1 | Pure module: hash, canonical form, header serialiser, test vectors | — | Vectors pinned; consumer can start |
| 2 | Schema fields and validation on both request types | Feature 25's field list (soft) | Schema tests green |
| 3 | Inline and unsteered apply; restore branch; `_has_steering_override`; text-completion block | 2 | Real-model equality tests green |
| 4 | Snapshot, reader, contextvars, `_finish_request_steering` at every site | 1, 3 | Honesty and every-path tests green |
| 5 | Routes and stream chunk; pre-load refusals; reset in completions | 4 | Route tests green |
| 6 | Docs: API reference, manual page, Feature 25 outcome table note | 5 | Published |
| 7 | Acceptance: mutation controls, full suite, hardware | 6 | BRD-04 acceptance 12 passed |

Estimate: 3–5 days. Phase 1 can ship first so miDataworks 007 can pin the vectors.

## 14. Decisions from Clarifying Questions

Clarifying rounds were waived. Each question is answered from a cited source.

| # | Question | Answer | Source |
|---|---|---|---|
| TD1 | Reuse the restore or write a second one? | Reuse; branch on `layers` | PADR v1.5 §10; `inference_service.py:2206-2231` |
| TD2 | How are other SAEs silenced under inline steering? | `enable_steering(False)` | T-79, T-80 |
| TD3 | Where does the header's truth come from? | A snapshot of hook-visible state, before restore | PADR v1.5 §10 rationale; operator rule "computed from what the hooks applied" |
| TD4 | How is a source named? | Value equality against record, circuit plan, active profile; else `manual` | FR-28.3.2; T-83 |
| TD5 | Hash serialisation? | Line form, binary64 hex, LF, UTF-8, SHA-256 | X-07; consumer request (007 FTDD §6.2, TQ6) |
| TD6 | Header type system? | RFC 8941 List of Tokens with Parameters | FR-28.3.5; 007 FTDD TQ7 |
| TD7 | `intensity` type? | String, shortest round-trip | RFC 8941 §3.3.2 (Decimal has 3 fractional digits) |
| TD8 | Non-ASCII profile names? | Percent-encoded String | RFC 8941 String is ASCII; consumer is RFC 8941 only |
| TD9 | Mid-request change? | `changed` on every member; items describe the end | T-81 |
| TD10 | Refuse before auto-load? | Yes, for a non-empty set on a non-resident model | T-82; `sae_repository.py:294` has no caller |
| TD11 | Composition unreadable? | Header `unknown` (fail closed) | FR-28.3.7; honesty over the rung echo's fail-open |
| TD12 | Fix the first-SAE profile defect? | No; observable via `sae`, `layer`, and a log | FPRD D27 and Open Question 1 |
| TD13 | RFC 8941 library in miLLM? | No; a small serialiser checked by the consumer's parser in a test | PADR prefers libraries for *parsing*; miLLM only serialises |

**Open items** (none blocks implementation):
1. FPRD Open Question 1 (first-SAE profile targeting) is for the operator.
2. Whether `http-sfv` as a dev-only test dependency needs a PADR §5 entry. It adds nothing at runtime;
   flag it to the PADR owner in review. **Resolved (Stage 3, 2026-10-06, requested by 028):** PADR §5
   now lists `http-sfv==0.9.9` as a dev-only test dependency. Item 1 is recorded as tracked debt in
   PADR §10.
