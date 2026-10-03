# Feature PRD: Probe Monitor Runtime

**Specified in:** `~/app/enhance/specs/ENH-001-probe-monitors` (handoff 2026-09-25)

**Document ID:** 024_FPRD|Probe_Monitor_Runtime
**Version:** 1.0 (planned)
**Status:** Planned. Specified in `~/app/enhance/specs/ENH-001-probe-monitors`, 2026-09-25.
**Source:** BRD-MILLM-PROBES-001 (BR-001 – BR-009)
**PPRD:** Feature 24 (FR-24.x), PPRD v1.4 · **PADR:** v1.4 "Probe Monitor Runtime" trade-offs
**Depends on:** the `mistudio.probe-definition/v1` contract (miStudio Feature 033), vendored here
**Co-release:** the MCP `millm_probes` category ships with miStudio 033 phase 7

---

## 1. Feature Overview

**Name:** Probe Monitor Runtime.

**What it is:** miLLM imports probe monitors trained in miStudio and runs them on live traffic. It
first checks each probe was built for exactly the loaded model, and proves on a set of test inputs
that it scores exactly as miStudio did. It then scores each request from a read-only hook that
doesn't depend on SAE attachment, and records the verdict. Both kinds of probe are supported: dense
probes over the residual, and k-sparse SAE probes, which use a private copy of their SAE's encoder
columns. The verdict is shown live and attached to the response.

**Problem:** probes are the cheap first stage of a monitoring cascade in every published production
system (Anthropic's Constitutional Classifiers++; DeepMind). miLLM can read activations on live
traffic (F11 and F15 sensing), but only through an attached SAE, and nothing can run a probe.

**Goals:**
- run probes at negligible overhead
- never score silently wrong: identity check plus parity gate
- never go silently quiet: every path either scores or says why
- never claim more than the probe's evidence rung

**Connection to the project:** F11/F15 sense concepts through SAEs. F24 senses concepts through probes
trained for detection, the first miLLM feature that consumes a miStudio *detector*.

## 2. User Stories & Scenarios

**US-1: import and arm.** An operator imports the LFM2 probe exported by miStudio (file, HF or MCP)
and arms it.
*Acceptance:* the identity check passes, parity passes (maximum deviation reported), and the probe
shows as armed with its rung.

**US-2: watch it work.** Requests to `/v1/chat/completions` are scored.
*Acceptance:* each scored request records an event (score, threshold, verdict, top firing positions),
the Probe Monitors page updates live, the non-streaming response carries an `X-miLLM-Probe-Verdicts`
header, and a streaming response ends with a verdict chunk before `[DONE]`.

**US-3: wrong model.** The operator tries to arm the probe while Qwen2.5-7B is loaded.
*Acceptance:* refused with `PROBE_MODEL_MISMATCH`, naming the mismatched field (hf_id / d_model /
n_layers / template / revision).

**US-4: low evidence.** A rung-1 probe needs an operator acknowledgement to arm (mirroring circuit
activation). The rung and its language are shown everywhere.

**US-5: an agent does it.** Through MCP (`millm_import_probe`, `millm_arm_probe`,
`millm_probe_events`), an agent imports, arms and reads events.

**Edge cases:**
- **GGUF model** → arming refused (`ENGINE_UNSUPPORTED`). A switch to a GGUF model while armed →
  probes auto-disarmed with a recorded reason.
- **Model unloaded or changed** → probes disarmed, reason `model_changed`.
- **Speculative decoding active** → probes paused, status reason `speculative_decoding`, and the
  request's verdict is `not_scored` with that reason.
- **`n > 1` or a batched request** → `not_scored`, reason `batched_request`.
- **Continuous batching** → the request is forced onto the serial path while any probe is armed.
- **Parity fails** → the probe can't be armed; the report names the vector and the largest deviation.
- **Too many probes** → above `PROBE_MAX_ARMED` (8), refused.
- **Hung stream thread** → probes disarmed as sensing is (the existing 5 s join guard).
- **Oversized or unknown-kind import** → refused (`PAYLOAD_TOO_LARGE`, `UNKNOWN_KIND`).

## 3. Functional Requirements

**FR-24.1 Import.** `POST /api/probes/import` accepts a definition JSON body (≤ 2 MB), validates
the kind, then validates against the pydantic mirror of the vendored v1 schema, and stores it with
provenance `origin=file|hub|mcp`. **`on_conflict=rename|fail`** follows the circuit convention.
(BR-001)

> **⚠ CORRECTED 2026-09-27.** This read `rename|replace|refuse` and claimed to follow the circuit
> convention; it did not. The real convention is `rename|fail` — `api/routes/management/circuits.py`
> pins the regex `^(rename|fail)$` and `api/schemas/cluster.py` the `Literal["rename","fail"]`.
> There is no `replace` and no `refuse` anywhere in miLLM, so building this as written would have
> given probes a third vocabulary for an operation two other importers already name.
>
> `refuse` is simply `fail` renamed. **`replace` is rejected on merit, not only for consistency:**
> overwriting a definition in place while that probe is ARMED would silently change the detector
> underneath a running monitor, and every event recorded before and after would carry the same
> `probe_id` — so the event history would describe two different detectors as one. Re-importing a
> rebuilt definition goes **disarm → delete → import**, which is explicit and leaves the old probe's
> events attached to the probe that actually produced them.
>
> MOVING A BAR IS NOT REPLACING A DETECTOR.
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

**FR-24.2 Hub.** `GET /api/probes/hub/search?q=&base_model=` lists HF repos tagged
`PROBE_HUB_TAG="mistudio-probe-definition"`, optionally filtered by `base_model:<id>`.
`GET /api/probes/hub/{repo_id}/definitions` reads `manifest.jsonl`, falling back to `*.probe.json`.
`POST /api/probes/hub/import` fetches and imports. It reuses the cluster hub machinery (TTL cache,
circuit breaker, `HUB_UNAVAILABLE`). (BR-001)

**FR-24.3 Identity check.** Arming compares the definition's `model` against the loaded model and
refuses with `PROBE_MODEL_MISMATCH`, naming each mismatch:
- `hf_id` vs the model row's `repo_id`
- `d_model` vs `config.hidden_size`
- `n_layers` vs `SAEHooker.get_layer_count`
- `chat_template_sha256` vs sha256 of `tokenizer.chat_template`
- `revision` vs the loaded snapshot SHA

When miLLM can only determine the *requested* revision (not a SHA), the revision check becomes a
recorded **warning**, `REVISION_UNVERIFIED`, and does not refuse. The engine must support hooks.
(BR-001, BR-006)

**FR-24.4 Parity gate.** Before arming, and on demand (`POST /api/probes/{id}/parity`), miLLM runs
the definition's test vectors through the live model and the probe head, using the vector's
`token_ids` directly. It compares per-token scores and combined scores with the vectors'
expectations, within `max(definition.test_vectors.tolerance, PROBE_PARITY_TOLERANCE)`. It also
re-renders each vector's `messages` with miLLM's own chat-template path and reports whether the
token ids match (**tokenization drift**, reported separately). Any score beyond tolerance →
`PROBE_PARITY_FAILED`, and the probe can't be armed. The result (maximum deviation, per-vector,
tokenization drift) is stored. Parity runs inside the request queue, serialised with generation.
(BR-005)

**FR-24.5 Attachment-independent read hook.** An armed probe's layer gets one read-only forward hook, **prepended**
(`register_forward_hook(fn, prepend=True)`) so it reads the **pre-steering** residual, like sensing.
All armed probes on a layer share one hook. The hook never modifies the output. Hook add and remove
call the dynamo reset used for SAE hook changes. (BR-002)

**FR-24.6 Per-request scoring.** For each request on a scored path, it tracks:
- prefill and decode phases
- absolute positions
- scope: `all`, `prompt` (system and user roles in the rendered prompt, found by the same
  prefix-rendering method miStudio uses), or `response` (decode tokens)

It updates each probe's running aggregate. Streamable rules update online, and `last` is taken at the
end. At the end of the request it computes the verdict (`score >= threshold` — at-or-above, matching how the producer cuts the bar; `>` until 2026-10-03) and the top-5 firing
token positions. There is exactly one device-to-host copy per forward pass. (BR-003)

**FR-24.7 Verdict delivery.** (BR-004)
- **Non-streaming chat and completions:** the response header `X-miLLM-Probe-Verdicts` is an RFC 8941
  list, `"<name>";score=<float>;threshold=<float>;verdict=?1|?0;rung=<int>`, with `not_scored` entries
  carrying `reason`.
- **Streaming:** a final chunk with `choices: []` and an extension field `millm_probe_verdicts`, sent
  **before** `[DONE]` (the same empty-choices shape as a usage chunk).
- It is set only when at least one probe is armed. With none armed, there is no header and no chunk.

**FR-24.8 Events and live view.** Each scored (or not-scored) request records one `probe_events` row
per armed probe: score, threshold, verdict, rung, top positions, phase coverage, and a context window
(decoded tokens around the top position, `PROBE_EVENT_CONTEXT_TOKENS`). Rows are pruned by
`PROBE_MAX_EVENTS_PER_PROBE` and `PROBE_MAX_AGE_DAYS`. A socket event `probe:event` is emitted,
throttled like sensing, and it **never carries prompt or context text** (context keys stripped, as
sensing does). (BR-004)

**FR-24.9 Status.** `GET /api/probes/status` returns:
- the armed probes (layer, rule, rung)
- the paused reasons (`speculative_decoding`, `model_changed`, …)
- the last request overhead in ms and the warning threshold
- events recorded, and socket events dropped

A probe is never silently quiet. (BR-006, BR-009)

**FR-24.15 SAE probes with a private encoder copy (D14).** A definition with
`basis=sae_features` arms only if its SAE is **downloaded in miLLM** (existing SAE Management) and
matches the definition's `sae` block: HF repo and path, revision, weights SHA-256, architecture,
`d_model` and `n_features`.
- **Refusals:** a missing SAE → `PROBE_SAE_MISSING` (naming the repo and path to download); a
  mismatch → `PROBE_SAE_MISMATCH` (naming the fields).
- **Arming** loads **only the encoder columns of the probe's k features** (`W_enc[:, idx]`,
  `b_enc[idx]`, and per-feature thresholds for JumpReLU) onto the layer's card.
- **Scoring:** the probe hook encodes the pre-steering residual with the SAE's activation function
  and the definition's `normalization`, then applies the head to the k features.
- The SAE is **never attached and never steers**. Attaching, detaching or steering with the same or
  any other SAE doesn't affect the probe.
- Parity (FR-24.4) runs through this same path, so an encoder difference is caught before arming.
- **Memory:** k × `d_model` per probe (k=128, d=2048: about 0.5 MB in fp16). (BR-010)

**FR-24.10 Serving-path rules.** (BR-006)
- CBM is forced serial while any probe is armed (`PROBE_FORCE_SERIAL`, default true).
- Speculative decoding pauses scoring, with the reason recorded.
- `n>1` and batched requests are `not_scored`.
- llama.cpp and GGUF refuse arming.
- Model unload or swap auto-disarms.

**FR-24.11 Evidence.** `millm/core/probe_evidence.py` mirrors miStudio's `PROBE_RUNG_LANGUAGE` and
`PROBE_RUNG_NEXT_STEP` verbatim (a test compares them with miStudio's module when it's checked out).
Arming below rung 2 requires `acknowledge_below_rung2=true` (`UNVALIDATED_PROBE` otherwise, refusal in
the envelope as for circuits), and the operator acknowledgement is stored. The rung and language are
shown in lists, status, events and headers. (BR-007)

**FR-24.12 Admin UI.** (BR-004, D8)
- A new **Probe Monitors** page (`/probe-monitors`): import (file / hub), a list with rung chips, arm
  and disarm (with the acknowledgement dialog), a parity report, a live events feed, and status.
- The existing **"Probe" page is renamed "Feature Monitor"** in the sidebar, dashboard, quick actions,
  manual and e2e. Its route `/monitoring` is unchanged.

**FR-24.13 MCP.** `docs/mcp-contract.md` v1.6 adds category `millm_probes` (import, hub import,
list, arm, disarm, status, events) with a three-state status column, a §4 probe rung rule, and §5
error codes (`PROBE_MODEL_MISMATCH`, `PROBE_PARITY_FAILED`, `UNVALIDATED_PROBE`, `PROBE_LIMIT`,
`PROBE_NOT_FOUND`). Co-released with miStudio 033 phase 7. (BR-008)

**FR-24.14 Overhead.** Scoring overhead is measured and stays under
`PROBE_MAX_OVERHEAD_MS_PER_PASS` (default 0.25 ms, NFR-1.4) **per forward pass**, on any model and
any request shape. It warns above that. A per-request backstop, `PROBE_MAX_OVERHEAD_MS` (500 ms),
catches pathology. (BR-009)

> **⚠ RE-SPECIFIED 2026-09-30 — THE ORIGINAL NAMED THE WRONG VARIABLE.** It read *"stays under
> `PROBE_MAX_OVERHEAD_MS` (default 5 ms) at 4k-token contexts with 2 armed probes on LFM2"*.
> Measured on Llama-3.1-8B, one probe, layer 11, varying ONLY `max_tokens` on an identical prompt:
>
> | generated tokens | overhead |
> |---|---|
> | 4 | 0.837 ms |
> | 30 | 3.425 ms |
> | 120 | 11.223 ms |
>
> while **2820 prompt tokens with 8 generated cost 1.94 ms** — twenty-one times the tokens for a
> third of the overhead. Prefill scores the whole prompt in ONE call; decode scores one token per
> call. A prompt token is roughly **300x cheaper** than a generated one, and the total is the pass
> count times a near-constant (~0.09 ms).
>
> Two consequences. **The criterion was measurable on its cheapest case** — a 4k context with a
> short completion is nearly free, so "under 5 ms at 4k tokens" could be satisfied while normal use
> was far above it. And **the threshold warned on normal use on every model**, LFM2 included: it is
> crossed at ~50 generated tokens with one probe, ~25 with two. A warning that fires on healthy
> requests is noise, and an operator who learns to ignore it has lost the guard while it still
> looks like coverage.
>
> In absolute terms the cost was never the problem: 11.36 ms on a 2.855 s request is **0.40%**.

## 4. User Experience Requirements
- The new page follows existing admin-ui patterns (react-query, toasts, the Tailwind dark theme).
- The events feed shows score bars against the threshold, the verdict badge, the rung chip, and a
  context expander (fetched on demand from the event detail route, never from the socket).
- Refusals show the server's code and message, with mismatches listed field by field.

## 5. Data Requirements
Migration **016** (`down_revision="015"`) adds two tables.

**`probes`** (`prb_`):
- `name`, `definition` JSON (the raw payload, so round trips are lossless)
- `hf_id`, `revision`, `d_model`, `n_layers`, `template_sha256`, `layer`, `rule`, `streamable`, `scope`
- `threshold`, `target_fpr`, `rung`, `definition_acknowledgement` JSON
- `parity` JSON, `armed` bool, `arm_acknowledgement` JSON, `paused_reason`
- `provenance` JSON
- `created_at`, `updated_at`

**`probe_events`**:
- `probe_id` FK CASCADE, `request_id`
- `scored` bool, `not_scored_reason`
- `score`, `threshold`, `verdict`, `rung`, `top_positions` JSON, `n_scored_tokens`
- `context_text`, `context_token_ids` JSON, `summary`
- `created_at`
- indexes `(probe_id, created_at)` and `request_id`

A separate table is used because `sensing_events` is keyed to profiles.

## 6. Technical Constraints
- torch ≥ 2.10, so `prepend=True` is available.
- Hooks apply only to the transformers engine (`LoadedModel.supports_hooks`).
- The request-scoped ContextVar pattern (`reset_steering_memo`) is used for the header verdict.
- Every refusal is a `MiLLMError` subclass with class-level `code`/`status_code`. `/v1` refusals get
  `ERROR_STATUS_MAP` rows.

## 7. API/Integration Specifications
All management routes live under `/api/probes`:
- `POST /import`
- `GET /hub/search`, `GET /hub/{repo_id:path}/definitions`, `POST /hub/import`
- `GET /`, `GET /{id}`, `DELETE /{id}`
- `POST /{id}/arm {acknowledge_below_rung2?: bool, reason?: str}`, `POST /{id}/disarm`
- `POST /{id}/parity`
- `POST /{id}/recalibrate {decision, mistudio_probe_id, mistudio_run_id?, calibration_id?, reason?}` — move the DECISION BAR of an already-imported, possibly armed probe, without touching its detector. Permitted where `on_conflict=replace` is not, because `probe_events` already records the threshold each verdict was judged against: a re-cut leaves every old and new event true and comparable, since the score beneath each came from the same weights at the same layer under the same scope with the same rule. The request body is `extra="forbid"` and therefore structurally incapable of carrying a detector; a cut that cannot be matched to `provenance.probe_id` is refused. An armed probe's live runtime shape is refreshed in the same call, because a database write alone changes nothing that is served.
- `GET /status`
- `GET /events?probe_id=&limit=`, `GET /events/{event_id}`, `DELETE /events?probe_id=`

OpenAI routes: only the headers and the final chunk change. The request body is unchanged in v1
(no per-request probe selection).

## 8. Non-Functional Requirements
- **Overhead:** under 0.25 ms **per forward pass** (FR-24.14), on any model and any request
  shape. Re-specified 2026-09-30; the original per-request form named the wrong variable.
- **Correctness:** parity at 1e-3 on LFM2 fp16.
- **Privacy:** no prompt or context text on the socket, pinned by a test.
- **Durability:** events are pruned by count and age.

## 9. Feature Boundaries (Non-Goals)
- Acting on verdicts (stop, escalate, re-route): D2's later phase.
- Per-request probe selection in the request body.
- Probes on GGUF.
- Open WebUI outlet display.
- Batched scoring under CBM.

## 10. Dependencies
- **Contract:** miStudio 033 (vendored schema and co-release).
- **Existing miLLM:** `SAEHooker._get_layer`, `layer_device`, `get_layer_count`; the sensing
  lifecycle in `inference_service.py`; `_reset_dynamo_for_hook_change`; `RequestQueue`;
  `ClusterHubService` helpers; `ApiResponse` envelope; `progress_emitter`; `core/errors.py`;
  admin-ui `useSensing` pattern.
- **Hardware:** mcs-lnxhost02, LFM2.5-1.2B.

## 11. Success Criteria
- **SC-1b:** miStudio's k-sparse **SAE probe** for LFM2 arms with its SAE downloaded but **not
  attached**, passes parity, and keeps scoring while another SAE is attached and steering on the same
  layer. It is refused with `PROBE_SAE_MISSING` when its SAE isn't downloaded.
- **SC-1:** miStudio's Stage 2 LFM2 probe imports from a file and from HF, passes parity (maximum
  deviation ≤ 1e-3) and arms.
- **SC-2:** arming refuses on Qwen2.5-7B (mismatch named) and on a GGUF model.
- **SC-3:** live requests produce events, the page updates live, and the header / final chunk carry
  verdicts. Non-streaming and streaming agree on the verdict for the same prompt.
- **SC-4:** overhead is under 0.25 ms per forward pass, measured across at least two
  completion lengths **and** one long-prompt/short-completion request, so the figure cannot
  come from the cheap case alone (measured, recorded).
- **SC-5:** every wiring point (hook install, begin/finish in each generation path, record, emit,
  header, chunk, routes, sidebar, MCP registration) has a removal test that fails. The socket-privacy
  test fails if context text is emitted.

## 12. Testing Requirements
- **Unit:**
  - mirror / sync
  - import validation
  - identity check (each field)
  - the parity engine (a fake model with known outputs)
  - the hook (pre-steer, prepend order, output unchanged)
  - runtime aggregation (streaming equals batch)
  - scope masks
  - verdict header formatting
  - final-chunk shape
  - event recording, pruning and socket stripping
  - serving-path rules
  - rung mirror and ack
- **Integration:** a tiny transformer end to end: import → parity → arm → chat (stream and
  non-stream) → events → header / chunk.
- **Performance:** an overhead test at 4k tokens (the edge-sensing baseline style).
- **Frontend:** vitest for the page, the ack dialog, the live feed (socket mock) and the rename.
- **E2E:** the Playwright navigation spec updated.
- **Cross-repo:** schema byte identity and rung-language identity with miStudio; the MCP contract
  consistency tests with `MILLM_REQUIRE_CROSS_REPO_CHECKS=1`.

## 13. Implementation Considerations
- Complexity is high: `inference_service.py` has 5k lines and several generation paths.
- **Order:** contract mirror → persistence and import → identity and parity → hook and runtime →
  lifecycle wiring per path → verdict delivery → events and status → UI and rename → MCP → acceptance.
- **Main risks:**
  - the streaming verdict must be computed before `[DONE]`, which today is yielded before sensing's
    `finally`
  - the prompt-scope role mask must match miStudio's method

## 14. Open Questions
1. **Parity tolerance:** 1e-3 absolute on the combined score at fp16 is proposed. It is measured
   during miStudio 033 acceptance (vectors scored at two batch sizes); confirm there.
2. **Revision:** can miLLM resolve the loaded snapshot SHA from `cache_path` for every download
   path? If not, `REVISION_UNVERIFIED` is a warning (FR-24.3). Spike to confirm.
4. **Is SAE encoding identical in both apps?** Does miLLM's SAE encode path (normalization, JumpReLU
   thresholds) equal miStudio's `encode_with_training_normalization` for the same SAE and input?
   Spike before building FR-24.15. Parity will catch any difference, but it's better found first.
3. **Do real clients tolerate the final `choices: []` chunk?** The OpenAI SDK and Open WebUI need
   testing. If any breaks, make the chunk opt-in with the request header `X-miLLM-Probe-Stream: 1`.

## 15. Decisions from Clarifying Questions
BRD decisions D2, D7, D8, D11 and D13 carry forward. **Planner defaults** for review:

| # | Question | Default | Why |
|---|---|---|---|
| R1 | Separate events table or shared? (BRD OQ-2) | Separate `probe_events` | `sensing_events` is profile-keyed; probes aren't profiles |
| R2 | Does the probe read pre- or post-steering? | Pre-steering (prepend hook) | Matches sensing; steering can't blind a monitor at its own layer |
| R3 | How is the streaming verdict transported? | A final `choices: []` chunk with `millm_probe_verdicts` | The OpenAI usage-chunk shape; spike OQ-3 decides whether it needs to be opt-in |
| R4 | What does parity check? | token_ids-driven scores, plus a separate tokenization-drift report | Separates model drift from tokenizer drift |
| R5 | When does parity run? | On arm and on demand, inside the request queue | Uses the live model without racing generation |
| R6 | What is the scope semantics at runtime? | `prompt` by the same prefix rendering method; `response` = decode | Must equal miStudio's mask for parity |
| R7 | What is the maximum number of armed probes? | 8 | Bounds overhead; adjustable |
| — | SAE probes: private encoder copy of the k features; SAE must be downloaded, not attached | *User decision D14* | |
| R8 | Should the rename happen now? | Yes, with the route unchanged | Avoids the "Probe" name collision (D8) without breaking links |
