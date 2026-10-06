# Feature PRD: Inline Steering and Steering-State Header

**Document ID:** 028_FPRD|Inline_Steering_And_Steering_Header
**Version:** 1.0 (planned)
**Status:** Planned. Feature PRD written 2026-10-06; FTDD, FTID and FTASKS follow.
**Source:** BRD-04 (miLLM — Dataworks Support) §5.8: R-04.31–R-04.34.
**PPRD:** Feature 28 (FR-28.1 – FR-28.4), PPRD v1.5 · **PADR:** v1.5 §10 "Dataworks Support
(Features 25–30)" trade-offs "Inline per-request steering vs saved profiles" and "Steering state
after generation and in a final stream chunk vs a request echo"
**Binding decisions:** checkpoint technical defaults of 2026-10-06
(`~/app/miDataworks/0xcc/docs/brd-decisions-2026-10-05.md`, "Checkpoint decisions")
**Depends on:** Feature 25 (the output-changing field list names `steering`, FR-25.3); F10/F14
per-request dial; F16 steering epoch. Reuses `_apply_request_steering`, `_restore_request_profile`,
`_unsteered` and `clamp_steering`.
**Co-release:** miStudio BRD-MIS-DATAWORKS-001 tool `millm_generate` (miStudio file `034_*`, FR-21),
which returns `X-miLLM-Steering` verbatim. Consumers: miDataworks BRD-03 R-03.35 (steered pairs,
each response checked against the reported steering state) and R-03.25 (each label records its
steering state). Feature 26 (Batch API) writes this feature's steering value into each output line
(FR-26.10.1).

Code references are to miLLM at `7aa659c` (HEAD, 2026-10-06).

---

## 1. Feature Overview

**Name:** Inline Steering and Steering-State Header.

**What it is:** three changes to the OpenAI-compatible `/v1` generation surface.

1. **Inline steering.** A chat or text completion can carry its own feature set:
   `steering: {sae_id?, features: [{index, strength}]}`. It steers that request only. No saved
   profile is created. `steering: {"features": []}` asks for an explicitly unsteered answer.
2. **Steering-state report.** Every generation response says how it was steered, in
   `X-miLLM-Steering`. The value states none, a profile with its intensity, inline steering with a
   feature count and hash, or a circuit. It describes what produced the answer, not what was asked.
3. **Steering on text completions.** `/v1/completions` gains `profile`, `steering_intensity` and
   `steering`. A text completion can then pick a profile, or refuse the active one.

**Problem:**
- **A steered experiment needs a saved profile today.** The only per-request steering fields are
  `profile` and `steering_intensity` (`millm/api/schemas/openai.py:120-128`). A job that compares
  two feature settings must create, store and name a profile per setting.
- **An answer does not say how it was steered.** The only steering echo is
  `X-miLLM-Steering-Intensity`. It is set only when the request sent a dial
  (`millm/api/routes/openai/chat.py:214-222`), and it is resolved *before* generation. A request
  steered by a globally active profile carries nothing.
- **Text completions cannot opt out of steering.** `TextCompletionRequest` has no steering fields
  (`openai.py:212-247`). `create_text_completion` never calls `_apply_request_steering`
  (`millm/services/inference_service.py:4780-4925`). So every text completion runs under whatever
  steering is live, and the client cannot see or change that.
- **The CBM routing check assumes text completions carry no steering.** `_has_steering_override`
  reads only `profile` and `steering_intensity`, and its docstring says text completions have
  neither (`inference_service.py:945-956`). (CBM is the continuous batching manager.)

**Goals:**
- steer one request with any feature set, with no stored artefact and no global change
- make every generation answer state its steering, so a caller can check rather than assume
- give text completions the same steering control chat already has
- never serve an answer under steering other than the one the response reports

**Connection to the project:** steering is miLLM's core product (Feature 4). Profiles (Feature 6),
clusters (Feature 8) and circuits (Features 12–19) all steer through the global state on the
attached sparse autoencoders (SAEs). This feature adds a fourth, request-scoped source. It also
adds the first report that covers every source. miDataworks builds steered preference pairs on it
(BRD-03 R-03.35).

## 2. User Stories & Scenarios

**US-1: a steered pair without profiles.** miDataworks generates a chosen and a rejected answer to
one prompt. The two differ on one feature axis: `{index: 1234, strength: 8}` and
`{index: 1234, strength: -8}`.
*Acceptance:* both requests succeed with no profile created. Each response carries
`X-miLLM-Steering` naming inline steering, one feature, and a hash. Recomputing the hash from the
request's own feature set matches.

**US-2: inline equals profile.** An operator saves a profile with the same features as an inline
request and sends both, greedy, on the same prompt.
*Acceptance:* the two outputs are identical (BRD-04 acceptance 12). The two headers report the same
feature count and hash, with different kinds.

**US-3: an explicitly unsteered answer.** A profile is globally active. A client sends
`steering: {"features": []}`.
*Acceptance:* the answer equals the answer with no SAE attached. The header says `none`. The next
request without `steering` runs under the active profile again.

**US-4: honest under an active profile.** A profile is active and a request sends no steering field.
*Acceptance:* the header names the profile, its source (`active`) and its effective intensity. Today
this response says nothing.

**US-5: text completion opt-out.** A labelling job sends `/v1/completions` with
`steering: {"features": []}` while a profile is active.
*Acceptance:* the completion is unsteered, and the header says `none`.

**US-6: streaming.** A streaming chat request is steered inline.
*Acceptance:* the stream ends with an extension chunk carrying the steering state, before `[DONE]`.

**US-7: an agent does it.** miStudio's `millm_generate` sends inline steering.
*Acceptance:* the tool returns the response body and `X-miLLM-Steering` verbatim (miStudio 034
FR-21).

**Edge and error scenarios:**

| Scenario | Outcome |
|---|---|
| `steering` and `profile` both sent | `400`, naming both fields (FR-28.2.1) |
| `steering` and `steering_intensity` both sent | `400` by default (Open Question 1) |
| `sae_id` names an SAE that is not attached | `400 SAE_NOT_ATTACHED`, naming the SAE (FR-28.1.4) |
| `sae_id` omitted while more than one SAE is attached | `400`, naming the attached SAEs (FR-28.1.3) |
| `sae_id` omitted and no SAE attached, `features` not empty | `400 SAE_NOT_ATTACHED` |
| `features: []` with no SAE attached | Accepted; nothing to suppress; header `none` |
| A feature index outside `[0, d_sae)` | `400 INVALID_FEATURE_INDEX`, naming index and `d_sae` |
| The same index twice | `400`, naming the index (FR-28.1.6) |
| A non-finite strength (NaN, infinity) | `400` (FR-28.1.6) |
| A strength beyond ±200 | Clamped; the header reports the clamp (FR-28.1.7) |
| `steering` on a GGUF model (llama.cpp's file format) | `400`, before any auto-load (FR-28.1.9) |
| `steering` on a scoring request | `400` (Feature 25, FR-25.7.2, unchanged) |
| `steering` on `/v1/embeddings` | `400` (FR-25.3 table, unchanged) |
| An operator changes global steering mid-request | Header reports the change (FR-28.3.6) |
| Steering state cannot be read (database error) | Header says `unknown`; never omitted, never guessed (FR-28.3.7) |

**User journey (US-1):** miDataworks builds two request bodies differing only in `strength` → sends
each with `X-miLLM-Strict: true` → miLLM admits each, applies the inline set inside the slot,
generates, restores, and reads the state the forward ran under → each response carries
`X-miLLM-Steering` → miDataworks compares each header with the setting it sent and stores the
pair only when both match.

## 3. Functional Requirements

Requirement IDs refine PPRD v1.5 FR-28.1–FR-28.4. Sub-requirements carry the parent's number.

### FR-28.1 Inline steering (R-04.31)

Chat and text completions SHALL accept `steering: {sae_id?, features: [{index, strength}]}`. It is
applied to this request only, inside the admission slot, and restored afterwards on the existing
per-request apply/restore lifecycle. No saved profile is created. An unattached SAE is refused.

- **FR-28.1.1 Schema.** `ChatCompletionRequest` and `TextCompletionRequest` gain an optional
  `steering` object. `sae_id` is an optional string. `features` is a required list of
  `{index: int ≥ 0, strength: float}`. A boolean `strength` is refused, as the dial already refuses
  booleans (`openai.py:182-188`). `features` may be empty only in the unsteered form (FR-28.2.2).
- **FR-28.1.2 Lifecycle.** The inline set is applied inside `_admit()` and restored in the same
  `finally` that restores a profile today. The three serial chat paths already do this for
  `profile`: batched chat (`inference_service.py:3460-3465`), chat (`inference_service.py:3629-3635`)
  and streaming chat (`inference_service.py:4349-4355`). Text completion gains the same block. The
  restore SHALL keep the F16 epoch guard: a restore superseded by an authoritative writer is skipped
  (`inference_service.py:2183-2186`).
- **FR-28.1.3 SAE selection.** `sae_id` selects an attached SAE. When omitted, miLLM uses the only
  attached SAE. When more than one SAE is attached and `sae_id` is omitted, the request is refused,
  naming each attached `(sae_id, layer)`. This follows `AttachedSAEState.by_layer`, which returns
  nothing rather than pick a basis when the choice is ambiguous (`millm/services/sae_service.py:553-562`).
  The same rule refuses an `sae_id` attached at more than one layer.
- **FR-28.1.4 Unattached SAE.** An `sae_id` with no attached entry is refused with
  `SAE_NOT_ATTACHED` (`400`, `millm/core/errors.py:327-331`), naming the SAE.
- **FR-28.1.5 Exactly this set.** The request runs under exactly the inline set: the selected SAE
  carries only these features, and every other attached SAE adds no steering for this request. The
  apply clears before it sets, as the profile path does, because `set_steering_batch` merges
  (`inference_service.py:2137-2138`; `millm/ml/sae_wrapper.py:413-429`). (Default; Open Question 2.)
- **FR-28.1.6 Validation.** Each index must lie in `[0, d_sae)` of the selected SAE, or the request
  is refused with `INVALID_FEATURE_INDEX` (`errors.py:355-359`). An index listed twice is refused,
  because a dictionary merge would keep one strength silently. A non-finite strength is refused.
  All validation happens before any state changes, as the profile path does
  (`inference_service.py:2104-2114`).
- **FR-28.1.7 Clamp.** Each strength passes through `clamp_steering`, the single ±200 clamp every
  apply path uses (`millm/core/steering_range.py:11-16`). A clamp that changes a value is never
  silent: the steering report counts the clamped features (FR-28.3.3) and a warning is logged.
- **FR-28.1.8 Serial routing.** A request carrying `steering` is served on the serial path, never by
  the CBM. `_has_steering_override` SHALL read `steering`, on both request types. Its docstring's
  claim that text completions carry no steering fields is corrected.
- **FR-28.1.9 Engine limits.** On the llama.cpp engine, `steering` is refused with
  `EngineUnsupportedError`, beside the existing `profile` refusal (`inference_service.py:3885-3890`).
  When the model row is GGUF, the refusal comes before any auto-load (FR-25.3.8).
- **FR-28.1.10 No stored artefact.** Inline steering writes no profile row, no active-profile change,
  and no steering epoch bump. It is not an authoritative writer.

### FR-28.2 Mutual exclusion and explicit unsteered (R-04.32)

`steering` and `profile` SHALL be mutually exclusive. `steering: {"features": []}` SHALL mean
explicitly unsteered.

- **FR-28.2.1** A request carrying both `steering` and `profile` is refused with `400`, naming both.
  The check runs at schema validation, before admission and before any auto-load.
- **FR-28.2.2** `steering: {"features": []}` disables steering on every attached SAE for this request,
  whatever profile, circuit or manual steering is live. An `sae_id` beside an empty list is refused,
  because it suggests a scope the empty form does not have.
- **FR-28.2.3** Sensing, monitoring and probes treat an explicitly unsteered generation as any other
  generation. The request is still sensed and still scored. (Default; Open Question 3.) Probes read
  the pre-steering residual from their own hook (PADR v1.4, Probe Monitor Runtime), so they are
  unaffected either way.
- **FR-28.2.4** `steering` with `steering_intensity` is refused with `400` by default. The dial
  scales a profile or circuit, and an inline set has no stored intensity to scale. (Open Question 1.)

### FR-28.3 Steering-state report (R-04.33)

Every generation response SHALL state its steering state in `X-miLLM-Steering`: none; profile with
intensity; inline with feature count and hash; or circuit. A non-streaming response sets it after
generation. A streaming response carries it in a final Server-Sent Events (SSE) chunk. Batch output
lines carry it in the body.

- **FR-28.3.1 Coverage.** The report is present on every response from `/v1/chat/completions` and
  `/v1/completions`, on every engine and path: serial, batched (`extra_messages`), streaming, CBM and
  llama.cpp. A scoring response carries it too, stating `none`, because scoring runs under
  `_unsteered` (`inference_service.py:1176-1199`); miStudio 034 FR-21 expects scoring tools to
  return it. `/v1/embeddings` is not a generation endpoint and does not carry it.
- **FR-28.3.2 Derived from what ran.** The value is derived from the steering the forward hooks
  applied: each attached SAE's live, unsuppressed values at generation time
  (`sae_wrapper.py:313`). It is not an echo of the request. It names a profile or circuit only when
  that source produced the applied values. Live values set directly through the steering routes,
  with no profile, are reported as `manual`.
- **FR-28.3.3 Content.** The value is a list with one item per steering source. No source means
  `none`. Each item carries:

  | Kind | Fields |
  |---|---|
  | `none` | — |
  | `profile` | profile name; `source` (`request` or `active`); effective intensity; SAE ID; feature count; hash |
  | `inline` | SAE ID; feature count; hash |
  | `manual` | SAE ID; feature count; hash |
  | `circuit` | circuit ID; effective intensity; `composed` when more than one circuit serves a layer |

  Any item whose applied values were clamped also carries the clamped count.
- **FR-28.3.4 Hash.** The hash is a SHA-256 digest of a canonical form of the applied set: the SAE ID
  and the `(index, strength)` pairs sorted by index. The canonical form is published in the API
  reference, so a client can recompute it from the set it sent. The same set always gives the same
  hash, whichever kind applied it. Float formatting in the canonical form is decided in the FTDD.
- **FR-28.3.5 Format.** The header is an RFC 8941 structured-field list, as `X-miLLM-Probe-Verdicts`
  and `X-miLLM-Circuit-Rung` are (`chat.py:46-47`, `chat.py:203-209`). Illustrative only; the FTDD
  fixes the grammar:

  ```
  X-miLLM-Steering: none
  X-miLLM-Steering: inline;sae="sae_4f2a";features=1;hash="sha256:9c1e…"
  X-miLLM-Steering: profile;name="humor";source=active;intensity=1.0;sae="sae_4f2a";features=12;hash="sha256:…"
  X-miLLM-Steering: circuit;id="crc_124fd83d1f2a";intensity=0.5
  ```
- **FR-28.3.6 Supersession.** If the steering epoch changes between admission and the end of
  generation, the report says so (a `changed` flag). An operator write landing mid-request means
  part of the answer ran under other steering, and the header must not describe one state as the
  whole (`sae_service.py:475-500`; restore skip at `inference_service.py:2183-2186`). (Default;
  Open Question 4.)
- **FR-28.3.7 Unknown, not guessed.** If the state cannot be determined, the header says `unknown`.
  It is never omitted on a successful response and never guessed. Reading the state never fails the
  request, as the rung echo already guarantees (`chat.py:203-212`).
- **FR-28.3.8 Non-streaming timing.** The header is set after generation, as
  `X-miLLM-Circuit-Rung` is (`chat.py:280-288`). The same applies to `/v1/completions`, whose route
  sets only `X-miLLM-Backend` today (`millm/api/routes/openai/completions.py:147`).
- **FR-28.3.9 Streaming.** A streaming chat response ends with one chunk shaped like the probe
  verdict chunk: `choices: []` plus an extension field `millm_steering`, emitted before `[DONE]`
  (`inference_service.py:2600-2611`). It is always emitted, even for `none`, so an absent chunk
  means an older server. No `X-miLLM-Steering` header is sent before the body: it would be a
  statement of intent, which is the PADR v1.5 trade-off this design rejects. Streaming on
  `/v1/completions` stays refused (`completions.py:79-84`).
- **FR-28.3.10 Batch lines.** This feature exposes one function that returns the steering value for
  a finished request. Feature 26 calls it for each output line (FR-26.10.1). The header, the stream
  chunk and the batch line SHALL use the same function, so the three cannot disagree.
- **FR-28.3.11 Existing headers unchanged.** `X-miLLM-Steering-Intensity` and
  `X-miLLM-Circuit-Rung` keep their current meaning and timing. `X-miLLM-Steering` is the
  authoritative statement; the API reference says so.

### FR-28.4 Steering on text completions (R-04.34)

`/v1/completions` SHALL gain `profile`, `steering_intensity` and `steering`, so a text completion can
choose or refuse the active profile.

- **FR-28.4.1** `TextCompletionRequest` gains the three fields with the same types and validators as
  `ChatCompletionRequest` (`openai.py:120-128`, `openai.py:182-195`).
- **FR-28.4.2** `create_text_completion` applies and restores them inside its existing `_admit()`
  block (`inference_service.py:4828`), around every prompt of a multi-prompt request. One request
  has one steering state and one header.
- **FR-28.4.3** A named profile that does not exist is refused with `404`, as on chat. Profile
  semantics are those of `_apply_request_steering` unchanged (`inference_service.py:1932-2147`).
- **FR-28.4.4** A text completion carrying none of the three fields runs under the live global
  steering, as today, and now reports it (FR-28.3).
- **FR-28.4.5** Feature 25's outcome table moves `steering` from "refused until Feature 28" to
  "honoured" for transformers chat and completions (FR-25.3.7). llama.cpp, scoring and embeddings
  stay "refused".

## 4. User Experience Requirements

No Admin UI change. The feature is an API surface. The API reference SHALL document:
- the `steering` object, its validation, and the clamp report;
- every `X-miLLM-Steering` kind, with examples;
- the hash's canonical form, with a worked example a client can reproduce;
- the final stream chunk;
- that `X-miLLM-Steering-Intensity` is a pre-generation echo and `X-miLLM-Steering` is authoritative.

Open WebUI (OWUI) is unaffected: it sends no `steering` field, ignores unknown headers, and already
tolerates the probe verdict chunk this design copies (`inference_service.py:2586-2592`).

## 5. Data Requirements

- No database table, column or migration.
- No persisted state. The inline set lives in the request and in the SAE's live values for the
  request's duration only.
- The steering value is computed per request from in-memory state plus the active-profile and
  active-circuit reads the echo paths already make.
- The hash is not stored by miLLM. miDataworks stores the header verbatim with each label and each
  generated row (BRD-03 R-03.25, R-03.35).

## 6. Technical Constraints

- `MAX_CONCURRENT_REQUESTS` stays 1 (`millm/core/config.py:255`). Inline steering mutates global SAE
  state and is correct only because admission is serial (PADR v1.1 per-request dial rationale).
- The apply/restore lifecycle is reused, not copied. A second lifecycle would drift from the epoch
  guard (PADR v1.5 §10; F16).
- Every effective strength goes through `clamp_steering` (PADR Feature 8 decision: one clamp).
- The sign rule holds: a negative strength is already directional and is applied as given, never
  re-signed (PADR, the canonical sign rule recorded for clusters and circuits).
- SAE attachment and steering are transformers-only (PADR v1.3, GGUF Serving).
- CBM is off in Kubernetes (`ENABLE_CONTINUOUS_BATCHING: "false"`), but its routing must still be
  correct (FR-28.1.8).

## 7. API/Integration Specifications

**Request (chat or text completion):**

```json
{
  "model": "LFM2.5-1.2B-Instruct",
  "messages": [{"role": "user", "content": "Tell me about the sea."}],
  "steering": {"sae_id": "sae_4f2a", "features": [{"index": 1234, "strength": 8.0}]}
}
```

**Response:** the existing OpenAI body, plus `X-miLLM-Steering` (non-streaming) or a final
`millm_steering` chunk (streaming).

**Errors:** the existing OpenAI error envelope. New refusals use existing codes where one fits:
`SAE_NOT_ATTACHED`, `INVALID_FEATURE_INDEX`, `invalid_request_error` for schema conflicts, and
`EngineUnsupportedError` for llama.cpp.

**Integration:**
- **Feature 25:** `steering` is on the output-changing list. This feature flips its outcome to
  honoured on transformers chat and completions.
- **Feature 26:** reads the steering value for each output line through FR-28.3.10.
- **miStudio `millm_generate`** (034 FR-21): passes `steering` through and returns the header.
- **miDataworks** (BRD-03 R-03.35, R-03.25; 005 FR-005.25): records the header; records "not
  reported" when it is absent.

No authentication change. miLLM runs unauthenticated on the local network (BRD-04 §3).

## 8. Non-Functional Requirements

- **Overhead.** Building the report adds no forward pass and no more than one database read beyond
  what the existing echo paths already make. The FTDD measures the added latency on the serial path.
- **Isolation.** A request's inline set is never visible to another request. After restore, global
  steering equals its pre-request value unless an authoritative writer superseded it.
- **Reliability.** Reading the state never fails a request (FR-28.3.7).
- **Honesty.** No response describes steering other than what ran. This is the feature's reason to
  exist (PADR v1.5 §10).

## 9. Feature Boundaries (Non-Goals)

- Inline steering across more than one SAE or layer in one request. One `sae_id` per request.
- Inline circuits or inline clusters. Use a saved circuit or profile.
- Storing or listing inline sets. The header and the caller's records are the audit trail.
- Steering on llama.cpp, on scoring requests, or on embeddings.
- Changing `X-miLLM-Steering-Intensity` or `X-miLLM-Circuit-Rung`.
- Streaming on `/v1/completions`.
- A steering report on `/v1/embeddings`.
- Admin UI changes.

## 10. Dependencies

- **Feature 25:** the output-changing field list and its per-endpoint outcome table (FR-25.3).
- **F10/F14** per-request dial and `_plan_effective_intensity`, for the profile and circuit
  intensity in the report.
- **F16** steering epoch, for restore safety and FR-28.3.6.
- **F18/F19** circuit serving and composition detection (`active_circuit_rung`,
  `inference_service.py:1629-1676`), for the circuit item and its `composed` flag.
- **Feature 24** probe verdict chunk, the shape FR-28.3.9 copies.
- **Feature 26** depends on this feature for FR-26.10.1.

## 11. Success Criteria

1. **BRD-04 acceptance 12.** Inline steering and a saved profile with the same features give
   identical greedy output. `X-miLLM-Steering` is correct for none, profile, inline and circuit,
   streaming and not.
2. A steered pair (US-1) produces two headers whose hashes match hashes recomputed from the request.
3. `features: []` under an active profile gives the same output as with no SAE attached.
4. A text completion under an active profile with `features: []` is unsteered, and the next one
   without the field is steered again.
5. Every wiring item has a test that fails when its call line is removed. The test asserts the
   payload and the call count, not only that a call happened (BRD-04 §6).

## 12. Testing Requirements

**Unit:**
- schema: every edge-scenario row in §2 returns its stated outcome;
- clamp: a strength of 500 applies 200 and the report counts one clamped feature;
- hash: stable across feature order; equal for inline, profile and manual with the same applied set;
- `_has_steering_override` is true for `steering` on both request types;
- restore after inline steering returns global state exactly, including `enabled`;
- an epoch bump mid-request sets `changed` and skips the restore.

**Integration (tiny real model on CPU, with a real attached SAE):**
- inline versus profile identical output, greedy;
- `features: []` versus no SAE attached identical output;
- every generation path emits the report: serial chat, batched chat, streaming chat, text completion,
  CBM chat and text, llama.cpp chat and text. The test discovers entry points from the service, as
  FR-27.9 does for probes; a hand-kept list does not count;
- streaming emits exactly one `millm_steering` chunk, before `[DONE]`.

**Mutation controls (required, recorded in the review notes):**
- remove the header-setting line in each route → red;
- remove `steering` from `_has_steering_override` → red;
- drop the clear before `set_steering_batch` on the inline path → red (output differs from profile);
- report the request's fields instead of the live state → red (US-4 case);
- skip `clamp_steering` on the inline path → red.

**User acceptance (hardware):** on LFM2.5-1.2B-Instruct with an attached SAE, a steered pair and an
inline-versus-profile pair, each header checked by hand.

## 13. Implementation Considerations

- **Complexity:** medium. The lifecycle exists. The risk is in the report: deriving the state from
  the live values rather than the request, on every path.
- **Approach:** add a small apply function for the inline set beside `_apply_request_steering`,
  returning the same saved-state shape so `_restore_request_profile` restores it unchanged. Add one
  `steering_state()` function that reads the applied state, and call it from the routes, the stream
  generator and Feature 26.
- **Request-scoped record.** The routes read the state after generation, but restore runs first.
  The state must be captured inside the admission slot, before restore, and handed to the route.
  The rung header already uses a request-scoped record (`circuit_apply_failed`, `chat.py:286`).
- **Unsteered mechanism.** `LoadedSAE.suppressed()` also stops monitoring and sensing capture
  (`sae_wrapper.py:576`, `sae_wrapper.py:681`). If FR-28.2.3 holds, the unsteered form must disable
  the steering delta only, as the λ = 0 path does with `enable_steering(False)`
  (`inference_service.py:2081`), on every attached SAE.
- **Observation, not in scope:** the profile path steers only the first attached SAE
  (`attached_sae`, `sae_service.py:509-512`). With several SAEs attached and no circuit, a named
  profile reaches only layer 0's entry. Recorded for the FTDD to confirm; inline steering avoids it
  through `sae_id`.
- **Estimate:** about 3–5 days including tests and the API reference.

## 14. Open Questions

Batched for the operator. Each carries the default this PRD assumes until answered.

1. **`steering` with `steering_intensity`.** Refuse with `400` (default), or scale the inline
   strengths by the dial? Scaling would let one feature set serve a dose sweep, but an inline set has
   no stored intensity, so the dial's absolute semantics have nothing to override.
2. **Other attached SAEs under inline steering.** Inline steering runs under exactly its own set,
   with every other attached SAE unsteered (default). The alternative leaves a live circuit or
   manual steering on other layers in place and reports both. R-04.32's empty-list rule suggests the
   default; BRD-04 does not say.
3. **Sensing and monitoring on an explicitly unsteered generation.** Keep recording (default), or
   suppress them as chat scoring does (R-04.8)? R-04.32 says "suppressed", which in code also stops
   capture (`sae_wrapper.py:576`).
4. **Mid-request steering changes.** Report a `changed` flag and describe the state at the end
   (default), or describe the state at admission? Neither describes an answer produced under two
   states; the flag at least says so.
5. **A request that would auto-load another model.** Can inline steering with a non-empty set be
   refused before the load? That is right only if a freshly loaded model never has an SAE attached.
   Not verified in code (technical; resolved in the FTDD).
6. **Header vocabulary.** BRD-04 names four states. This PRD adds `manual` and `unknown`, because
   live values with no profile, and an unreadable state, both occur. Confirm the added kinds.

## 15. Decisions from Clarifying Questions

Clarifying rounds were waived. Each question is pre-answered from a cited source. Questions with no
source are Open Questions above.

| # | Question | Answer | Source |
|---|---|---|---|
| D1 | Priority? | Planned, v1 of the Dataworks increment | PPRD v1.5 Feature 28; checkpoint decision 3 (everything in the spec) |
| D2 | Header name? | `X-miLLM-Steering` | R-04.33; miStudio 034 FR-21; miDataworks 005 FR-005.25 |
| D3 | Header timing? | After generation; final SSE chunk when streaming; body field in batch lines | R-04.33; PADR v1.5 §10 "Steering state after generation" |
| D4 | Header from request or from live state? | Live state | PADR v1.5 §10 rationale: "describe what produced the answer, not what was asked for" |
| D5 | Reuse the apply/restore lifecycle or add one? | Reuse | R-04.31; PADR v1.5 §10 "Inline per-request steering" |
| D6 | Clamp or refuse out-of-range strengths? | Clamp through `clamp_steering`, and report it | Task brief ("same apply/restore and clamp"); PADR Feature 8 clamp-at-apply |
| D7 | `steering` with `profile`? | Refused | R-04.32 |
| D8 | Meaning of `features: []`? | Explicitly unsteered, every attached SAE | R-04.32 |
| D9 | Unattached SAE? | Refused, `SAE_NOT_ATTACHED` | R-04.31; `errors.py:327-331` |
| D10 | `sae_id` omitted with several SAEs attached? | Refused, naming them | `by_layer` ambiguity rule, `sae_service.py:553-562` |
| D11 | Steering on llama.cpp? | Refused, before auto-load for a GGUF row | `inference_service.py:3885-3890`; FR-25.3 table; FR-25.3.8 |
| D12 | Steering on scoring? | Refused (unchanged) | R-04.8; FR-25.7.2 |
| D13 | Does a scoring response carry the header? | Yes, `none` | miStudio 034 FR-21 ("Scoring tools return it too") |
| D14 | Does inline steering need approval in miStudio? | No (miStudio's concern; recorded for context) | miStudio 034 D17 |
| D15 | CBM routing for inline steering? | Serial | `inference_service.py:1002-1007`; FR-25.3.6 |
| D16 | New database state? | None | R-04.31 ("No saved profile is created") |
| D17 | Streaming text completion? | Stays refused | `completions.py:79-84` |
