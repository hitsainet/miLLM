# Technical Design: Probe Scoring, Per-Request Activations and Probe-Path Fixes
## miLLM Feature 27

**Version:** 1.0 · **Created:** 2026-10-06 · **Status:** Planned
**References:** 027_FPRD v1.1 · BRD-04 §5.6, §5.7, §5.13 · PPRD v1.5 Feature 27 · PADR v1.5 §10
("Stateless probe scoring", "Packed scoring", "Probe-path coverage by discovery") · Feature 24 chain
(`0xcc/{prds,tdds,tids,tasks}/024_*`)
**Binding decisions:** checkpoint technical defaults; Feature-PRD decisions P-03, P-20, X-03, X-09;
technical defaults T-49, T-72 – T-77 (`~/app/miDataworks/0xcc/docs/`)
**Consumers:** miDataworks 009 FTDD §6.3 (probe-verdict loop, feature tagger); miStudio 034 FTDD §9
(`millm_score_probes`, ≤ 64 inputs per call)

Code is cited at miLLM `f5c71b6`, verified 2026-10-06. Clarifying rounds were waived; §14 records
each design question with its source.

---

## 1. Executive Summary

The feature adds two read paths into the loaded model and closes three gaps in the existing probe
paths. Every decision below reuses an existing mechanism rather than building a parallel one.

| Area | Decision | Rationale |
|---|---|---|
| Stateless scoring | New `ProbeScoringService` behind `POST /api/probes/score`. It builds probes with `armed_probe_from_row`, runs the parity forward, and decides with `_verdict_for` | Offline must equal live (RSK-07); sharing code, not copying it, is what makes them equal |
| Admission | New `InferenceService.run_model_work(fn)`: one `_admit()` slot, a worker thread, every SAE suppressed in that thread | One seam for score, parity and arm; suppression must be entered in the forward's thread |
| Granularity | One input per slot, one input per forward, never packed | Checkpoint default; bfloat16 is not batch-invariant |
| Parity and arm | Both routes run parity through `run_model_work` | Neither takes a slot today, and neither suppresses steering (T-73) |
| Activations | A request-scoped capture object registered on the attached `LoadedSAE`, read inside the existing SAE hook before and after `apply_steering` | No extra hook per request, and it works under suppression, where monitoring capture does not |
| Read point | `post_steering` default, `pre_steering` option, `unsteered` reported in scoring mode | Checkpoint default; T-77 rules out a counterfactual |
| Probe-path fixes | Batched chat, both CBM non-streaming paths and the three llama.cpp paths open a context and mark a reason | No path may reach generation silently (BR-006) |
| Guard | AST discovery of every generation site, plus a behavioural test that drives each site with a probe armed | A hand-kept list missed three paths; discovery cannot miss a new one |
| Boundary | `>=` in `_verdict_for` stays the only comparison; a guard keeps it so; docs state it | P-03; already true since commit `0c3f3fe` |

## 2. System Architecture

```
POST /api/probes/score ─▶ ProbeScoringService.score(req)
   loaded_identity(session)                          probe_arm_bridge.py:38
   resolve probes (ids given → refuse on mismatch; omitted → skip with reason)
   armed_probe_from_row(row, encoder, windows)       probe_arming.py:245
   for each input:
      prepare ids, prompt_tokens, last_user span     (pure; ProbeInputPreparer)
      InferenceService.run_model_work(score_one)     NEW — _admit() + to_thread + _unsteered
         pin check (model_id, loaded_at)
         ctx = ProbeRequestContext("score:i", probes) probe_runtime.py:214
         build_probe_forward(model, layers)(ids, ctx) probe_arm_bridge.py:119 (generalised)
         ctx.finish() → Verdicts                     _verdict_for, probe_runtime.py:537
   → results, parity status per probe, skipped probes     (nothing persisted)

POST /api/probes/{id}/parity ─▶ run_model_work(lambda: ProbeParityEngine(fwd).run(...))
POST /api/probes/{id}/arm    ─▶ ProbeArmingService.arm(..., executor=run_model_work)

/v1/chat|completions + return_sae_activations
   route validates shape (refuse n>1, extra_messages, several prompts)
   serial path: _activations_begin(spec) → LoadedSAE.begin_request_capture(capture)
      SAE hook (sae_hooker.py:186-211): capture.observe_pre(hidden) … apply_steering … capture.observe_post(modified)
   _activations_finish() → MillmExtension.sae_activations → response body / final stream chunk

Generation paths (FR-27.8): each opens _probe_begin, marks a reason where it cannot score,
   finishes before the response, records in finally.
```

**Integration points.** `probes.py` routes (`millm/api/routes/management/probes.py:340-418`),
`ProbeArmingService.arm` (`millm/services/probe_arming.py:285`), `InferenceService` generation paths,
the SAE forward hook (`millm/ml/sae_hooker.py:186-211`), and the OpenAI response schemas
(`millm/api/schemas/openai.py:334`, `:421`).

## 3. Technical Stack

- Python 3.11+, FastAPI, pydantic v2, SQLAlchemy async, torch ≥ 2.10 — all existing (PADR §2).
- **No new dependency.** The discovery guard uses the standard-library `ast` module, as
  `tests/unit/services/test_unload_admission.py:451-481` already does.
- `asyncio.to_thread` for the forward, matching scoring mode (`inference_service.py:4973`).

## 4. Data Design

- **No migration.** Stateless scoring and per-request activations persist nothing (FPRD §5).
- `probe_events` rows from FR-27.8 use existing columns: `scored`, `not_scored_reason`, `window`,
  `provisional` (`millm/db/models/probe.py:197-200`). P-20 is already satisfied by that column.
- **Parity report gains two keys, additively.** `ParityReport.as_details()`
  (`millm/services/probe_parity.py:246`) adds `model: {hf_id, revision, dtype, quantization}` and
  `checked_at`. Stored in the existing `probes.parity` JSON column (`db/models/probe.py:131`). Old
  reports lack the keys; the scoring response then says `checked_against: unknown`.
- **Validation.** All request models are `extra="forbid"` so a typo is a `422`, not a silent drop.
  Token ids must be non-negative and below the tokenizer's vocabulary size.

## 5. API Design

### 5.1 `POST /api/probes/score`

Request (`ProbeScoreRequest`, new in `millm/api/schemas/probe_scoring.py`):

```json
{
  "probe_ids": ["prb_…"],                    // optional (T-75)
  "inputs": [
    {"token_ids": [1, 2, 3], "prompt_tokens": 3},
    {"messages": [{"role": "user", "content": "…"}]},
    {"text": "…"}
  ],
  "windows": ["all", "prompt"],              // optional; resolve_windows default
  "return_token_ids": false
}
```

Response (`ApiResponse.ok(data)`):

```json
{
  "model": {"hf_id": "…", "revision": "…", "dtype": "bfloat16"},
  "probes": [{"probe_id": "prb_…", "name": "…", "layer": 11,
              "parity": {"status": "passed|failed|never_run",
                         "checked_against": {"hf_id": "…", "revision": "…", "dtype": "…"} }}],
  "skipped": [{"probe_id": "prb_…", "code": "PROBE_MODEL_MISMATCH", "reason": "…"}],
  "results": [{
    "index": 0, "input_kind": "token_ids", "n_tokens": 3, "prompt_tokens": 3,
    "token_ids": null,
    "verdicts": [{"probe_id": "…", "name": "…", "window": "all", "score": 4.2,
                  "threshold": 3.9, "verdict": true, "rung": 2, "rung_language": "…",
                  "provisional": false, "threshold_revision": 1, "n_scored_tokens": 3,
                  "not_scored_reason": null}],
    "error": null
  }]
}
```

- `verdict` is `Verdict.fires` renamed for the wire: `true`, `false` or `null`
  (`probe_runtime.py:190-192`). miDataworks maps `null` to skipped (009 FTDD §6.3).
- A per-input `error` (`{code, message}`) covers `MODEL_CHANGED` and a tokenisation failure. One bad
  input does not discard the others' results.

**Refusals (whole request, before any forward):**

| Condition | Error | Status |
|---|---|---|
| No model loaded | `ProbeNoModelLoadedError` (existing) | existing |
| GGUF / no hooks | `ProbeHookUnsupportedError` (existing) | existing |
| Unknown probe id | `ProbeNotFoundError` | 404 |
| Given probe fails identity | `ProbeModelMismatchError` / `ProbeDtypeMismatchError` | 409 |
| Unscorable scope | `ProbeScopeUnverifiableError` (arming's refusal) | 409 |
| SAE missing or mismatched | `ProbeSaeMissingError` / `ProbeSaeMismatchError` | 409 |
| Shape: zero or too many inputs, too many probes, two input kinds in one input, `prompt_tokens` > length | `ProbeScoreRequestError` (new, `INVALID_PROBE_SCORE_REQUEST`) | 400 |
| Every probe skipped (ids omitted) | `ProbeScoreRequestError` naming the skips | 400 |
| Input over the context window | existing `_check_context_length` error | existing |

### 5.2 Parity and arm routes

Shapes unchanged. The parity route stores the report with its new `model` and `checked_at` keys.

### 5.3 `return_sae_activations` on `/v1`

Request field on `ChatCompletionRequest` (`openai.py:47`) and `TextCompletionRequest`
(`openai.py:212`):

```json
"return_sae_activations": {"sae_id": "sae_…", "features": [12, 99], "top_k": 8,
                           "positions": "completion", "read_point": "post_steering"}
```

`positions` is `"last" | "prompt" | "completion" | "all" | {"start": int, "end": int}` (half-open,
absolute). `read_point` is `"post_steering"` (default) or `"pre_steering"`.

**Steering header on these responses (X-09; Stage 3, 2026-10-06, requested by 028).** Scoring is always
unsteered. A `/v1` request in scoring mode that carries `return_sae_activations` reports
`read_point: "unsteered"` in the body and `X-miLLM-Steering: none` in the header, as every scoring
response does (028 FR-28.3.1, FR-28.4.6). `POST /api/probes/score` is not a generation endpoint: it
carries no `X-miLLM-Steering` header, and its forward always runs with every SAE suppressed (T-73).

Response: an optional `millm` object on `ChatCompletionResponse` and `TextCompletionResponse`,
**omitted entirely when absent** so the OpenAI shape is unchanged:

```json
"millm": {"sae_activations": {
  "sae_id": "sae_…", "layer": 11, "read_point": "post_steering",
  "positions": [{"position": 17, "token_id": 345, "features": [{"index": 12, "value": 3.1}]}],
  "note": "post_steering is what the model computed at this layer; it is not an unsteered counterfactual"
}}
```

Streaming: one chunk `{"id", "object": "chat.completion.chunk", "choices": [], "millm": {…}}` after
the probe-verdict chunk and before `[DONE]` (T-76).

Refusals, all `400` with `SAE_ACTIVATIONS_REFUSED` (new) unless stated:
- no matching SAE attached → existing `SAENotAttachedError`;
- `sae_id` omitted with several attached → names the candidates;
- `n > 1`, `extra_messages`, several prompts (T-76);
- worst-case entries over `SAE_ACTIVATIONS_MAX_ENTRIES` (FR-27.2f), counted before generation;
- `top_k` over `SAE_ACTIVATIONS_MAX_TOP_K`; a feature index ≥ the SAE's width;
- a llama.cpp model → existing `EngineUnsupportedError`.

`return_sae_activations` is registered with Feature 25's known-field set so it never appears in
`X-miLLM-Ignored-Fields`.

### 5.4 Error strategy and security principles

Every new error is a `MiLLMError` subclass with class-level `code` and `status_code`; `/v1` refusals
get `ERROR_STATUS_MAP` rows. No prompt text enters logs at info level or any socket event.

## 6. Component Architecture

**New `millm/services/probe_scoring.py`:**
- `ProbeInputPreparer` (pure, no torch model): turns an input into `PreparedInput(ids, prompt_tokens,
  last_user_span, last_user_reason, input_kind)`.
  - `token_ids`: used as given; `prompt_tokens` optional; `last_user` reason `token_ids_have_no_turns`.
  - `messages`: rendered with live serving's renderer, `_format_chat_messages`
    (`inference_service.py:5580`), when the last turn is not an assistant turn. An assistant-ended
    conversation is rendered with `add_generation_prompt=False`, and `prompt_tokens` is the length
    of `messages[:-1]` rendered with the generation prompt, accepted only if it is a prefix of the
    full render (T-72). A user-ended conversation has no response: `prompt_tokens` equals its
    length. `last_user` uses `last_user_token_span` (`probe_turns.py:153`).
  - `text`: `messages = [{"role": "user", "content": text}]`, then as above (T-49).
- `ProbeScoringService(repository, inference)`: `score(request, session) -> dict`. Resolves probes,
  builds `ArmedProbe`s, groups by layer, scores input by input through `run_model_work`.

**Changed `millm/services/probe_arm_bridge.py`:** `build_probe_forward(model, layers)` installs one
`ProbeHooker` hook per distinct layer for one call. `build_parity_forward(model, layer)` becomes
`build_probe_forward(model, [layer])`, so parity's behaviour does not change.

**Changed `millm/services/inference_service.py`:**
- `run_model_work(fn)`: `async with self._admit(): return await asyncio.to_thread(self._unsteered_call, fn)`.
  `_unsteered_call` enters `self._unsteered()` inside the worker thread.
- `_activations_begin(spec, n_prompt)` / `_activations_finish()`: thin, never-raising seams beside
  `_probe_begin`, mirroring its shape (`inference_service.py:2498-2584`). Validation errors are raised
  earlier, by the route, before the slot.
- `_probe_begin_detached(request_id, reason)`: builds a `ProbeRequestContext` over the armed probes,
  marks the reason, and does **not** register it with `ProbeRuntimeState`. `_probe_record(ctx, …,
  detached=True)` skips `end_request()`. Used by every path that never scores, because concurrent CBM
  requests cannot share the runtime's single context (`probe_runtime.py:823-828`) and
  `_probe_record` would otherwise close another request's context (`inference_service.py:2626`).
  `_cbm_stream_chat_completion` (5296) migrates to it.
- FR-27.8 wiring in `_create_batched_chat_completion` (3393), `_cbm_chat_completion` (5230),
  `_cbm_text_completion` (5418), `_llamacpp_chat_completion` (3960),
  `_llamacpp_stream_chat_completion` (4012) and `_llamacpp_text_completion` (4221).
- **`millm/api/routes/openai/completions.py`:** the non-streaming branch sets `X-miLLM-Probe-Verdicts`
  from `get_probe_verdicts()` after `create_text_completion` (`completions.py:147-148`), as the chat
  route does (`chat.py:295-297`). Today the header is never sent on completions (FPRD FR-27.8g).

**New `millm/services/request_activations.py`:** `ActivationSpec` (validated request),
`RequestActivationCapture` (per-pass observe, position filter, chunked encode, top-k on device, one
host copy per pass), `MillmExtension` builder.

**Changed `millm/ml/sae_wrapper.py` / `sae_hooker.py`:** `LoadedSAE.begin_request_capture(capture)` /
`end_request_capture()`, refusing a second open capture as `ProbeRuntimeState.begin_request` does
(`probe_runtime.py:819-830`). The hook calls `capture.observe(hidden, phase="pre")` before steering
and `capture.observe(modified, phase="post")` after; the capture keeps only its read point. Both
calls run whether or not the SAE is suppressed.

**Changed `ProbeArmingService.arm`:** a required keyword `executor`. The parity call at
`probe_arming.py:400` becomes `await executor(lambda: ProbeParityEngine(forward).run(...))`. Required,
not optional, so no caller can forget it and silently run outside a slot.

## 7. State Management

- **Probe runtime state is untouched by scoring.** Scoring builds its own `ProbeRequestContext`
  objects and never calls `ProbeRuntimeState().begin_request` (`probe_runtime.py:819`). An armed hook
  on the same layer sees `_request is None` and records nothing (`probe_runtime.py:839-842`).
- **Activation capture is one-at-a-time state on the `LoadedSAE`**, opened inside the slot and closed
  in `finally`. A hung generation thread is the one way a late pass could reach the next capture; the
  existing hung-thread guard clears it beside disarming probes.
- **Pinning.** At the first input the service records `(model_id, loaded_at)` from
  `LoadedModelState().current` (`millm/ml/model_loader.py:154-161`). Each later slot compares; a change
  fails the remaining inputs with `MODEL_CHANGED`.
- **Caching.** None. A probe's `ArmedProbe` and encoder are built once per request and discarded.

## 8. Security Considerations

- No authentication, as for every miLLM route (BRD-04 §4).
- Input caps bound work per request: `PROBE_SCORE_MAX_INPUTS`, `PROBE_SCORE_MAX_PROBES`, and the model's
  context window per input.
- Token ids are range-checked before they reach the embedding table; an out-of-range id is a `400`,
  not a device-side assert.
- Privacy: no input text, rendered prompt or token list is logged above debug level; scoring emits
  no socket event at all. A test asserts the log records of a scoring call carry no input text.

## 9. Performance & Scalability

- **One forward per input** covering every requested probe, across layers. Cost ≈ one prefill per
  input. miStudio's 64-input cap at LFM2.5-1.2B is a few seconds.
- **Slot per input** keeps an interactive request's wait to one input's forward (BRD-04 RSK-03).
- **Hook churn.** `ProbeHooker.install` and `remove` reset dynamo each time
  (`millm/ml/probe_hooker.py:100`, `:111`). That is two resets per input, the same cost parity pays per
  vector today. Measured at acceptance; if it dominates, install once per request around the loop.
- **Activation memory.** Encoding all positions of a 4k prefill against a wide SAE is
  `4096 × d_sae` values. The capture slices the requested positions first, then encodes in chunks of
  `SAE_ACTIVATIONS_ENCODE_CHUNK` positions, keeps top-k on device, and copies only `(n, k)` to the
  host.
- **Defaults (config, measured at acceptance):** `PROBE_SCORE_MAX_INPUTS = 64` (miStudio's cap),
  `PROBE_SCORE_MAX_PROBES = 8` (matches `PROBE_MAX_ARMED`, `config.py:182`),
  `SAE_ACTIVATIONS_MAX_TOP_K = 64`, `SAE_ACTIVATIONS_MAX_ENTRIES = 65536`,
  `SAE_ACTIVATIONS_ENCODE_CHUNK = 512`.

## 10. Testing Strategy

**Philosophy:** every wiring line has a test that goes red when the line is removed, asserting payload
and call count. Offline-equals-live is tested through the real forward and the real decision code,
never a stub that agrees by construction.

**10.1 The discovery-based probe-wiring guard (replaces the hand-kept list).** The list at
`tests/unit/services/test_probe_wiring.py:49-53` names four methods. It missed batched chat and both
CBM non-streaming paths. New `tests/unit/services/test_probe_paths_discovered.py`:

1. **Discover generation sites by AST.** Parse `InferenceService`. A *generation site* is a method
   whose body calls, or passes as a callable, any primitive:
   `self._generate_sync`, `self._generate_in_thread`, `self._llamacpp_sync`,
   `self._cbm_backend.generate`, `self._cbm_backend.generate_stream`, `self._model.generate`,
   `self._model.create_completion`, `self._model.create_chat_completion`. Passing as a callable covers
   `asyncio.to_thread(self._generate_sync, …)` and `Thread(target=self._generate_in_thread)`.
2. **Discover entry points.** Build the `self.<method>` call graph; entry points are the public
   coroutines from which a generation site is reachable. Assert the discovered set is non-empty and
   contains the three known entries, so a broken parser cannot pass by finding nothing.
3. **Drive every site, with a probe armed.** Primitives are replaced by recording fakes. Spies on
   `_probe_begin` and `_probe_begin_detached`, wrappers on every discovered site, and the fakes all
   append to one lock-guarded event log (`enter`, `begin`, `gen`, `record`). A log, not a context
   lookup at generation time, because `_generate_in_thread` runs in a plain `Thread` that sees neither
   the caller's ContextVars nor a reliable frame stack. A scenario table maps
   each generation site to the request and service flags that route to it (CBM on with
   `PROBE_FORCE_SERIAL=False`, llama.cpp engine, `extra_messages`, `n=2`, stream). **The table is
   not the authority:** the test asserts `set(table) == discovered sites`, so a new site fails red
   until someone adds a way to reach it.
4. **Assert, per scenario:** its target site was entered; every `gen` is preceded by a `begin` since
   the scenario started; exactly one `record` follows, carrying a context with a verdict or a
   `not_scored_reason` (payload asserted); and afterwards `current_request() is None`.
5. **Exemptions** (scoring mode, embeddings) live in one dict with reasons. A test asserts no exempt
   method is a generation site, so an exemption cannot hide generation.
6. The old `TestEveryServingPathIsWired` parametrises over the discovered entry points instead of its
   literal list.

**Negative controls (recorded in the FTASKS):** delete each path's `_probe_begin` line, one at a time
(ten paths) → red each time; add a fake method that calls `self._generate_sync` → red ("site not in
the scenario table").

**10.2 Other tests.**
- Unit: `ProbeInputPreparer` (each kind, each boundary rule, prefix check, `text` as one user turn);
  probe resolution (given vs omitted, skips); pinning; per-input errors; response shape.
- **No event write:** a real SQLite session; count `probe_events` before and after a scoring call;
  assert `begin_request` is never called (spy) and `has_armed()` is unchanged.
- **Boundary (P-03):** a fixture whose score is exactly representable (as
  `test_probe_runtime.py:157-173`) scored through the route; asserts `verdict: true`. A source-wide AST
  guard asserts `_verdict_for` holds the only `Compare` between a probe score and a threshold.
- **Admission:** spies on `InferenceService._admit` show one entry per input and one per parity/arm.
  The existing `test_every_request_queue_slot_is_taken_through_admission` stays green.
- **Suppression:** a fake SAE that counts `suppressed()` entries **in the worker thread** proves score,
  parity and arm forwards run unsteered.
- **Activations:** isolation (two interleaved requests), read point (a steering SAE at the same layer
  makes `post` differ from `pre`), scoring mode reports `unsteered`, the unfed final token, cap before
  generation, shape refusals, stream chunk order, `millm` absent when not asked.
- Integration: tiny real transformer — import → score unarmed → arm → live chat with `max_tokens: 1`
  → compare prompt-window scores.
- Reachability: `/api/probes/score` present in `app.openapi()["paths"]`; removing the router line or
  the route goes red.
- Hardware: BRD-04 acceptance 10, 11 and 17 on mcs-lnxhost02 (§13).

**Fixtures.** Reuse `tests/unit/probe_fixtures.py`. Mocks stop at the primitive; the scoring service,
the context and `_verdict_for` always run for real.

## 11. Deployment & DevOps

- No migration, no k8s manifest change. New settings get defaults in `millm/core/config.py` and lines
  in `.env.example`.
- `docs/mcp-contract.md`: next additive minor version, adding `POST /api/probes/score` to the
  `millm_probes` inventory (§4) and the `>=` rule to §4d. miStudio's `millm_score_probes` waits on this.
- Manual: `manual/docs/features/probe-monitors.md` gains "Scoring stored text", the boundary rule, and
  the new not-scored reasons; the OpenAI API reference gains `return_sae_activations`.
- Logging: `probe_score` info line per request (probe count, input count, elapsed, skipped count —
  no content); `sae_activations` debug line per request.
- Rollback: the feature is additive. Reverting removes the route and the field; FR-27.8 wiring is
  independent and can stay.
- ⚠ A rollout kills in-flight work; hardware acceptance waits for the deploy to settle.

## 12. Risk Assessment

| Risk | Mitigation |
|---|---|
| Offline and live disagree on `messages` because the renders differ | Both use `_format_chat_messages`; acceptance 11 compares prompt-window scores with `max_tokens: 1` |
| The `text` render does not match miStudio's corpus | T-49 spike: reproduce one reported AUROC before shipping `text` (FTASKS 0.1) |
| A discovery test satisfied by comments or by finding nothing | AST calls only; assert known entries are found; negative controls per path |
| Fixtures agreeing by construction (offline == live because both are stubbed) | Real forward and real `_verdict_for` in every agreement test |
| Per-request capture on `LoadedSAE` leaks into the next request | Refuse a second open capture; close in `finally`; clear in the hung-thread guard |
| Dynamo resets per input slow scoring | Measured; fallback is one install per request |
| `executor` forgotten in a new `arm` caller | Required keyword; route test asserts it is `run_model_work` |
| Several features (25, 26, 28) also add a `millm` response object | One `MillmExtension` model, owned here; others add fields to it |

**Alternatives considered.** A separate prepended hook per activation request (simpler isolation, but
a dynamo reset per request and a second encode path); keeping `executor` optional (rejected: a default
that silently skips the slot is the defect being fixed); a static-only path guard (rejected: path
sensitivity — `create_chat_completion` calls `_probe_begin` *after* dispatching to batched chat, so a
"method contains the call" check passes wrongly).

**Complexity:** medium-high, dominated by `inference_service.py` (5,678 lines, ten generation paths).

## 13. Development Phases

| Phase | Content | Depends on |
|---|---|---|
| 0 | Spike T-49 (render vs miStudio corpus; AUROC reproduction) | — |
| 1 | Discovery guard (expected red on today's code) | — |
| 2 | FR-27.8 wiring (incl. the completions verdict header); guard goes green | 1 |
| 3 | `run_model_work`; parity and arm under it, unsteered; parity report keys | — |
| 4 | `ProbeScoringService`, route, schemas, errors, config | 3 (and 0 for `text`) |
| 5 | FR-27.10 boundary guard and stateless boundary test | 4 |
| 6 | Per-request activations | — |
| 7 | Contract, manual, reachability | 4, 6 |
| 8 | Feature acceptance incl. hardware (BRD-04 acceptance 10, 11, 17) | all |

Milestones: M1 = phases 1–3 (the defects closed); M2 = phases 4–5 (miDataworks probe labeler and
miStudio tool unblocked); M3 = phase 6 (feature tagger unblocked).

## 14. Decisions from Clarifying Questions

Rounds waived. Each answer is sourced; none is invented here.

| # | Question | Answer | Source |
|---|---|---|---|
| TD1 | One input at a time or packed? | One at a time, one slot each | Checkpoint default; FPRD FR-27.6b |
| TD2 | Steering during scoring, parity, arm? | Suppressed in all three | T-73; X-09 |
| TD3 | Parity required to score? | No; status reported | T-74 |
| TD4 | `probe_ids` omitted? | All matching; mismatches skipped with reasons | T-75 |
| TD5 | Window boundaries on stored input? | `prompt_tokens`; derived for assistant-ended `messages`; else `not_scored` | T-72 |
| TD6 | User-ended `messages` boundary? | Whole input is prompt (no response exists, so nothing is guessed) | Refinement of T-72, stated here |
| TD7 | `text` render? | One user turn; verified by AUROC reproduction | T-49 |
| TD8 | Activation read point? | `post_steering` default, `pre_steering` option, `unsteered` in scoring | Checkpoint default; T-77 |
| TD9 | Activation shapes? | Final `choices: []` chunk when streaming; refuse `n>1`, `extra_messages`, several prompts | T-76 |
| TD10 | Activation capture mechanism? | Request capture on `LoadedSAE`, read in the existing hook | Design choice; §12 alternatives |
| TD11 | llama.cpp paths in the guard? | Open a context and mark `engine_unsupported`; no exemption | FPRD FR-27.8f |
| TD12 | Verdict boundary? | `>=`, already in code; guard and docs added | P-03; X-03 |
| TD13 | Provisional verdicts? | Returned and stored with the flag | P-20 |
| TD14 | Config defaults? | §9 values, revisited at acceptance | miStudio 034 FTDD §9; `config.py:182` |
| TD15 | Context for paths that never score? | Detached context, not registered with the runtime | FPRD FR-27.8h (latent CBM collision) |
| TD16 | How does the guard observe ordering across threads? | One lock-guarded event log from spies and fakes | §10.1 step 3 |

**Open items:** none. FTASKS 0.1 (T-49 verification) must pass before `text` input is enabled.
