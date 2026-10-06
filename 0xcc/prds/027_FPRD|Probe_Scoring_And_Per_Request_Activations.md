# Feature PRD: Probe Scoring, Per-Request Activations and Probe-Path Fixes

**Specified in:** BRD-04 §5.6, §5.7 and §5.13 (Dataworks Support, 2026-10-06)

**Document ID:** 027_FPRD|Probe_Scoring_And_Per_Request_Activations
**Version:** 1.0 (planned)
**Status:** Planned. Written 2026-10-06 from BRD-04 and the 2026-10-06 checkpoint decisions.
**Source:** Business Requirements Document (BRD) BRD-04 (miLLM — Dataworks Support), R-04.24 – R-04.30, R-04.46, R-04.47
**Project PRD (PPRD):** Feature 27 (FR-27.1 – FR-27.9), PPRD v1.5 · **Project Architecture Decision Record (PADR):** v1.5 §10 "Dataworks Support" trade-offs
**Depends on:** Feature 24 (Probe Monitor Runtime), Feature 25 (request validation, scoring mode)
**Consumers:** miDataworks BRD-03 R-03.52 (probe-verdict labeler, feature tagging);
miStudio BRD-MIS-DATAWORKS-001 (`millm_score_probes`, its 034 FPRD FR-17)
**Binding decisions:** `~/app/miDataworks/0xcc/docs/brd-decisions-2026-10-05.md`, "Checkpoint
decisions — 2026-10-06", technical defaults

Code references are to miLLM at `7aa659c`, verified on 2026-10-06.

---

## 1. Feature Overview

**Name:** Probe Scoring, Per-Request Activations and Probe-Path Fixes.

**What it is:** three related changes to how miLLM reads a model's internals for one caller.

1. **Stateless probe scoring.** `POST /api/probes/score` asks one or more imported probes about
   stored text. It generates nothing, writes no probe event and arms nothing.
2. **Per-request sparse autoencoder (SAE) activations.** A chat or text completion can ask for the
   SAE feature activations of its own request, returned in its own response body.
3. **Probe-path fixes.** Batched chat and the continuous batching manager (CBM) non-streaming paths
   stop skipping armed probes silently. The parity route takes a request slot. A test discovers
   every generation entry point instead of trusting a hand-kept list.

**Problem:**
- A detector operator cannot ask "what would this probe say about these 10,000 rows". Parity scores
  only a definition's own test vectors. Scoring mode records no verdicts by design
  (`millm/services/inference_service.py:4934-4936`).
- Live scoring needs global arming. Any armed probe pushes every request off continuous batching
  (`inference_service.py:1041-1046`).
- The only per-request activation read is a shared history buffer. It holds 100–1000 entries
  (`millm/api/schemas/monitoring.py:22-27`, `millm/services/monitoring_service.py:122`). It keeps
  the last position only (`monitoring_service.py:288-289`). Under load, other requests evict a
  caller's entries.
- `_create_batched_chat_completion` never calls `_probe_begin`
  (`inference_service.py:3393-3521`). Armed probes are skipped with no trace.
- `_cbm_chat_completion` (`inference_service.py:5230`) and `_cbm_text_completion`
  (`inference_service.py:5418`) never call `_probe_begin` either. They are safe today only because
  `PROBE_FORCE_SERIAL` routes armed traffic away from CBM. The setting's own comment says not to rely
  on that (`millm/core/config.py:177-179`).
- The parity route runs a forward pass with no request slot
  (`millm/api/routes/management/probes.py:385-418`). It can overlap a generation.

**Goals:**
- Score stored text with any imported probe, with the same numbers live scoring would produce.
- Return a request's own activations, never another request's.
- Never let a probe go silently quiet on any generation path, including paths not yet written.
- Never run a forward pass outside the single admission path.

**Connection to the project:** Feature 24 made miLLM a live probe runtime. This feature makes it an
offline one too. miDataworks labels stored rows with probe verdicts and tags rows with SAE features.
miStudio's Model Context Protocol (MCP) server exposes the scoring endpoint to agents.

## 2. User Stories & Scenarios

**US-1: label stored rows with a probe.** A miDataworks operator runs the probe-verdict labeler over
10,000 stored rows. Nothing is armed.
*Acceptance:* every row gets a score, threshold, verdict, evidence rung and provisional flag per
window. No `probe_events` row is written. `GET /api/probes/status` shows the same armed set before and
after. Other traffic keeps its routing.

**US-2: offline equals live.** The operator scores the same inputs offline, then arms the probe and
sends them live.
*Acceptance:* the scores agree within the parity tolerance (BRD-04 acceptance 11).

**US-3: wrong model.** An operator scores with a probe fitted for another model.
*Acceptance:* refused, naming each mismatched identity field, as arming refuses
(`millm/services/probe_identity.py:154`).

**US-4: an agent asks.** An agent calls miStudio's `millm_score_probes` on 200 rows.
*Acceptance:* the result carries `rung_language` verbatim, so the tool composes no phrase from a rung
number (miStudio 034 FPRD FR-17).

**US-5: tag stored text with features.** miDataworks sends a scoring-mode completion with
`return_sae_activations`.
*Acceptance:* the response holds this request's activations only, keyed by position, under a `millm`
extension object. It states the read point.

**US-6: two interleaved callers.** Two clients send activation requests at the same time.
*Acceptance:* each response holds only its own request's activations (BRD-04 acceptance 10).

**US-7: batched chat with a probe armed.** A client sends a chat request with `extra_messages`.
*Acceptance:* the response's `X-miLLM-Probe-Verdicts` header and the probe event say `not_scored`
with reason `batched_request` (BRD-04 acceptance 17).

**Edge cases:**
- **A GGUF model loaded** (llama.cpp's model file format) → scoring and activation requests are
  refused before any work (`millm/services/probe_arm_bridge.py:53-60` for probes).
- **No model loaded** → refused with the existing no-model error (`probe_arm_bridge.py:49-52`).
- **SAE probe whose SAE is not downloaded** → refused with `PROBE_SAE_MISSING`, as arming refuses.
- **Probe whose scope the runtime cannot score** → refused with arming's scope refusal
  (`millm/services/probe_arming.py:357`).
- **A window whose boundary is unknown** (for example `response` on bare `token_ids`) → that verdict
  is `not_scored` with reason `prompt_boundary_unknown`. It is never guessed (see Open Question 1).
- **Model unloaded or swapped mid-request** → the remaining inputs fail with a stated reason. No input
  is scored on a different model from the first.
- **Activation request over the size cap** → `400` before generation.
- **No attached SAE matches `sae_id`** → refused, naming the SAE.
- **`sae_id` omitted with several SAEs attached** (a circuit) → refused, naming the candidates.
- **A probe is armed on the same layer during stateless scoring** → the armed hook records nothing,
  because no request context is open (`millm/services/probe_runtime.py:839-842`).

## 3. Functional Requirements

FR IDs are those of PPRD v1.5. Each is refined here; lettered items are refinements.

### 3.1 Per-request SAE activations (BRD-04 §5.6)

**FR-27.1 Request and response.** Chat and text completions accept
`return_sae_activations: {sae_id?, features?, top_k, positions}`. (R-04.24)
- a. `positions` is `last`, `prompt`, `completion`, `all`, or an index range.
- b. Positions are absolute over the processed sequence: prompt then generated tokens.
- c. `last` is the last position the model processed. In generation, the final sampled token is never
  fed back, so it has no activation. The response lists the positions actually read.
- d. `features` restricts the candidate features. `top_k` picks the largest among them. This follows
  the monitoring configuration's `features` and `top_k` (`millm/api/schemas/monitoring.py`).
- e. The response carries a `millm` extension object holding `sae_activations`: the SAE id, its
  layer, the read point, and one entry per position with token id and top features.
- f. Capture is private to the request. It never reads or writes the monitoring history buffer
  (`monitoring_service.py:122`). It works whether monitoring is enabled or not.
- g. A request carrying `return_sae_activations` is served on the serial path, as a steering
  override is (`inference_service.py:945-957`).

**FR-27.2 Refusals and the read point.** (R-04.25)
- a. The request is refused, naming the SAE, unless a matching SAE is attached.
- b. With `sae_id` omitted, exactly one SAE must be attached. Otherwise the refusal names the
  attached candidates.
- c. The default read point is **post-steering**: what the model computed, after this layer's
  steering delta. An option selects **pre-steering**: before this layer's delta, which is where
  monitoring reads today (`millm/ml/sae_hooker.py:186-192` capture, `:211` steering). (Checkpoint
  technical default.)
- d. The response states the read point used: `post_steering`, `pre_steering` or `unsteered`.
- e. Neither read point is an unsteered counterfactual. Steering at earlier layers, and tokens
  generated under steering, still shape the residual. The response and the manual say so.
- f. The worst-case entry count is checked **before** generation: positions × `top_k`, counting
  `prompt tokens + max_tokens` for `completion` and `all`. Over the configured cap returns `400`. A
  request is never refused after it has generated.
- g. A GGUF model is refused before any auto-load. llama.cpp exposes no hook points.

**FR-27.3 Scoring mode.** `return_sae_activations` works in scoring mode, so stored text is tagged
without generating. (R-04.26)
- a. Scoring mode runs with every attached SAE suppressed (`inference_service.py:1176-1199`). The read
  point is reported as `unsteered`.
- b. Suppression also disables the SAE hook's monitoring capture (`millm/ml/sae_wrapper.py:180-184`).
  So per-request capture cannot rely on that capture path. It needs its own read.
- c. `last` in scoring mode is the last prompt position: the one whose next-token distribution is
  scored.

### 3.2 Stateless probe scoring (BRD-04 §5.7)

**FR-27.4 The endpoint.** `POST /api/probes/score` takes `{probe_ids?, inputs, windows?}`. (R-04.27)
- a. Each input is exactly one of `token_ids`, `messages` or `text`. `token_ids` is authoritative
  when given.
- b. The result holds, per input, probe and window: `score`, `threshold`, `verdict`, `rung`,
  `rung_language`, `provisional`, `threshold_revision`, `n_scored_tokens` and, when not scored,
  `not_scored_reason`. These are the fields of `Verdict` (`probe_runtime.py:180-211`).
- c. `verdict` is three-valued. `null` means the probe said nothing, as `Verdict.fires` defines
  (`probe_runtime.py:190-192`). It is never coerced to false.
- d. `rung_language` is returned verbatim, so no consumer maps a number to a phrase (miStudio
  BRD-MIS-DATAWORKS-001 BR-019).
- e. Per input, the result states the input kind used and the number of token ids scored. On request
  it echoes the token ids. A `messages` render can differ from a definition's recorded ids
  (`millm/services/probe_parity.py:503-509`), so the caller must be able to see what was scored.
- f. It generates nothing. It writes no `probe_events` row. It never calls the runtime's
  `begin_request` (`probe_runtime.py:819`). It changes no armed state and no stored parity report.
- g. The route is registered in the live application and returns the standard `ApiResponse`
  envelope.

**FR-27.5 Same construction and decision as live.** (R-04.28)
- a. It works on any imported probe, armed or not, at any evidence rung. No acknowledgement is
  needed, because nothing is armed and every result carries its rung language.
- b. It builds the runtime probe with `armed_probe_from_row` (`probe_arming.py:245`), as the parity
  route does (`probes.py:407`). The encoder comes from `build_probe_encoder`
  (`probe_arm_bridge.py:143`).
- c. Windows resolve through `resolve_windows` (`probe_arming.py:86`). Omitted `windows` gives the
  arm-time default (`probe_arming.py:83`, `:99-100`).
- d. The decision uses the same window bars and length bands as live scoring: `_verdict_for`
  (`probe_runtime.py:537`) and `threshold_for_length` (`probe_runtime.py:156`). No second copy of
  either is written.
- e. The forward pass is the parity forward (`build_parity_forward`, `probe_arm_bridge.py:119`):
  the prepended read-only probe hook, removed after each call.
- f. Identity is checked with `check_identity` (`probe_identity.py:154`) against
  `loaded_identity` (`probe_arm_bridge.py:38`). A mismatch is refused, naming each field.
- g. Arming's other refusals that do not depend on arming also apply: an unscorable scope
  (`probe_arming.py:357`), a missing or mismatched SAE, a GGUF engine.

**FR-27.6 Admission and shared passes.** (R-04.29)
- a. Each input runs inside a slot taken through `_admit()` (`inference_service.py:649`). The slot is
  released between inputs, so interactive traffic waits at most one input's forward pass. This
  follows the batch API's chunk rule (PADR v1.5 "Batch runner inside the admission path").
- b. Inputs run **one at a time**, never packed. bfloat16 is not batch-invariant, and miStudio scores
  test vectors one input at a time (checkpoint technical default). A Feature 26 batch row for this
  endpoint runs single-row whatever its `pack` setting.
- c. All requested probes on one layer share one forward pass per input. Probes on different layers
  may share one pass too; the technical design decides.
- d. The model identity is pinned at the first input. A later slot that finds a different loaded
  model fails the remaining inputs with a stated reason.
- e. **The parity route is brought under `_admit()`** (`probes.py:385-418`).
- f. **Refinement beyond BRD-04's text: the arm route too.** `arm_probe` runs the same parity forward
  through `ProbeArmingService.arm` (`probes.py:363-372`, `probe_arming.py:400`). It has no slot either.
  Feature 24's FR-24.4 promised parity "inside the request queue". Both routes are fixed.

**FR-27.7 No global arming, no routing change.** (R-04.30)
- a. Offline scoring needs no armed probe.
- b. It installs only temporary hooks, so `ProbeRuntimeState().has_armed()` stays false. CBM routing
  for other requests is unchanged (`inference_service.py:1041-1046`).
- c. This closes BRD-03 §7.1 audit item 6.

### 3.3 Probes on every generation path (BRD-04 §5.13)

**FR-27.8 Batched chat opens a probe context.** (R-04.46)
- a. `_create_batched_chat_completion` calls `_probe_begin` inside its slot. Until per-row scoring
  exists, it marks the request `not_scored` with reason `batched_request`, as the `n > 1` path does
  (`inference_service.py:3641-3648`).
- b. It calls `_probe_finish` before the response is returned, so the chat route's header read
  (`millm/api/routes/openai/chat.py:295-297`) finds the verdicts.
- c. It calls `_probe_record` in a `finally`, so an event is written even when generation fails.
  `_probe_record` also closes the runtime's request context (`inference_service.py:2626`).
- d. **The same fix applies to `_cbm_chat_completion` and `_cbm_text_completion`.** Each marks
  reason `continuous_batching`, as `_cbm_stream_chat_completion` already does
  (`inference_service.py:5296-5298`).
- e. `_serial_chat_fallback` needs no change: it calls `create_chat_completion` per conversation
  (`inference_service.py:3540-3544`), which is already wired.

**FR-27.9 A discovery-based guard.** (R-04.47)
- a. A test discovers every generation entry point from `InferenceService` itself. An entry point is
  any method whose call graph reaches a generation primitive: `_generate_sync`
  (`inference_service.py:5492`), the streaming generate thread, the CBM backend's `generate` or
  `generate_stream`, or a llama.cpp completion.
- b. For each, with a probe armed, the test fails if generation is reached with neither a probe
  context nor a recorded not-scored reason.
- c. It covers serial, streaming and batched chat; text completion; the three CBM paths; and the
  llama.cpp paths.
- d. A hand-kept list does not satisfy this. The existing list in
  `tests/unit/services/test_probe_wiring.py:49-53` names four methods and missed three. It stays as a
  secondary check, not the guard.
- e. Scoring mode and embeddings run forward passes but no generation. If discovery finds them, any
  exemption names the method and its reason in one place. A test asserts each exempt method reaches
  no generation primitive.
- f. Removing the FR-27.8 call turns the test red (BRD-04 acceptance 17).

## 4. User Experience Requirements

- No new Admin UI page. The Probe Monitors page and `GET /api/probes/status` already show not-scored
  reasons; `batched_request` and `continuous_batching` appear there as other reasons do.
- Refusals use the envelope's error code and name the field, SAE, probe or window at fault.
- Error messages say what to do next, as Feature 24's refusals do (for example, "download the SAE").

## 5. Data Requirements

- **No migration.** Stateless scoring persists nothing. Per-request activations persist nothing.
- `probe_events` gains rows only from FR-27.8's not-scored events, using existing columns
  (`not_scored_reason`).
- New configuration settings, with defaults fixed in the technical design:
  - the activation entry cap (positions × `top_k`);
  - inputs per scoring request, and probes per scoring request.
- No change to the `mistudio.probe-definition/v1` contract.

## 6. Technical Constraints

- PADR v1.5 "Stateless probe scoring vs arming a probe": share construction and decision code; never
  copy them.
- PADR v1.5 "Probe-path coverage by discovery vs a list of entry points".
- `MAX_CONCURRENT_REQUESTS` is 1 (`millm/core/config.py:255`). `_admit()` is the only way to take a
  slot (`inference_service.py:654-655`).
- SAE suppression is per thread (`sae_wrapper.py:186-190`). It must be entered in the thread that runs
  the forward pass (`inference_service.py:1184-1187`).
- The probe hook is prepended, so it reads before any SAE steering hook on its layer
  (`millm/ml/probe_hooker.py:5-9`).
- Hooks exist only on the transformers engine (`LoadedModel.supports_hooks`).
- Every refusal is a `MiLLMError` subclass with class-level `code` and `status_code`. A `/v1`
  refusal gets an `ERROR_STATUS_MAP` row.
- Python 3.11+, FastAPI, pydantic v2; Black (line length 100), Ruff.

## 7. API/Integration Specifications

**New management route:**
- `POST /api/probes/score` — body `{probe_ids?: [str], inputs: [Input], windows?: [str],
  return_token_ids?: bool}`.
  `Input` is `{token_ids: [int]} | {messages: [...]} | {text: str}`. Response: `ApiResponse` with one
  result per input, each holding per-probe, per-window verdict fields (FR-27.4b).

**Changed OpenAI routes:**
- `/v1/chat/completions` and `/v1/completions` accept `return_sae_activations`. The response body
  gains a `millm` extension object. No other field changes.
- `return_sae_activations` must be a known field to Feature 25's request validation, so it is never
  reported in `X-miLLM-Ignored-Fields` where it is implemented.

**Changed management routes:**
- `POST /api/probes/{id}/parity` and `POST /api/probes/{id}/arm` take a slot through `_admit()`. Their
  request and response shapes are unchanged.

**Contracts and consumers:**
- `docs/mcp-contract.md` gains `POST /api/probes/score` in the `millm_probes` endpoint inventory, as
  an additive minor version. miStudio's BRD-MIS-DATAWORKS-001 needs that section before it builds
  `millm_score_probes`.
- Feature 26 lists `/api/probes/score` as a batch endpoint (R-04.16). Its rows run single-row
  (FR-27.6b).
- No authentication, as for every miLLM route (BRD-04 §4).

## 8. Non-Functional Requirements

- **Correctness:** stateless scores reproduce a definition's test vectors within the parity
  tolerance, `max(definition tolerance, PROBE_PARITY_TOLERANCE)` (`millm/core/config.py:186`).
- **Agreement:** offline and live scores for the same inputs agree within the same tolerance.
- **Isolation:** no request ever receives another request's activations.
- **Interactivity:** an interactive request waits at most one scoring input's forward pass.
- **Privacy:** no prompt, message or text content goes to a socket event or a log line at info level.
- **Memory:** the activation cap bounds response size and host memory before generation starts.

## 9. Feature Boundaries (Non-Goals)

- Probe scoring of each row inside a batched chat request. FR-27.8 records the gap honestly (BRD-04
  §7).
- Packed (multi-input) probe scoring.
- Probes or activations on GGUF models.
- Re-enabling continuous batching in production, or raising `MAX_CONCURRENT_REQUESTS`.
- The MCP tool itself. miStudio owns `millm_score_probes`.
- Storing scoring results. A caller stores what it needs.
- Changing the probe definition contract or the parity tolerance.

## 10. Dependencies

- **Feature 24:** probe import, identity check, parity forward, windows, length bands, the not-scored
  vocabulary and the verdict header.
- **Feature 25:** scoring mode and request validation. `return_sae_activations` is registered there.
- **Feature 26:** consumes `/api/probes/score` as a batch endpoint. Not a prerequisite of this
  feature.
- **F11/F17 sensing lifecycle:** the per-request context pattern the probe calls mirror.
- **Hardware:** mcs-lnxhost02 and LFM2.5-1.2B for BRD-04 acceptance 11.
- **Consumers waiting on this:** miDataworks R-03.52; miStudio 034 FR-17.

## 11. Success Criteria

Owned BRD-04 acceptance items: 10, 11 and 17.

- **SC-1 (acceptance 10):** two interleaved requests each get only their own activations. A request
  over the cap returns `400`.
- **SC-2 (acceptance 11, hardware):** on LFM2.5-1.2B, `/api/probes/score` reproduces a definition's
  test vectors within the parity tolerance, unarmed. Arming the probe and scoring the same inputs live
  gives the same scores. No `probe_events` row is written by the stateless call.
- **SC-3 (acceptance 17):** with a probe armed, a batched chat request records reason
  `batched_request`. Removing the FR-27.8 call turns the FR-27.9 test red.
- **SC-4:** the CBM non-streaming chat and text paths record `continuous_batching` with
  `PROBE_FORCE_SERIAL` false. Removing either call turns a test red.
- **SC-5:** the parity, arm and score routes each take a slot. Removing the `_admit()` call from any
  of them turns a test red.
- **SC-6:** every wiring item has a removal test that asserts the payload and the call count, not only
  that a call happened (PPRD FR-20.3).

## 12. Testing Requirements

- **Unit:**
  - request schema validation for both new request shapes;
  - position resolution, including the unfed final token and scoring mode's `last`;
  - the pre-generation cap check with a worst-case count;
  - the read point: post-steering differs from pre-steering when this layer steers, and scoring mode
    reports `unsteered`;
  - stateless scoring calls `armed_probe_from_row`, `_verdict_for` and `threshold_for_length`, asserted
    by call, not by source text;
  - no `begin_request`, no event row, armed set unchanged;
  - identity, scope, SAE and GGUF refusals;
  - one input per slot, released between inputs; model swap mid-request;
  - the batched chat and CBM wiring (begin, mark, finish, record).
- **Discovery guard:** FR-27.9, with negative controls that delete each new `_probe_begin` call.
- **Integration:** a tiny transformer: import → score unarmed → arm → live chat → compare scores; two
  interleaved activation requests.
- **Reachability:** `/api/probes/score` present in the live application's OpenAPI paths, not merely
  importable. Removing its registration turns a test red.
- **Hardware:** SC-2 on mcs-lnxhost02.
- **Cross-repo:** the MCP contract consistency tests with `MILLM_REQUIRE_CROSS_REPO_CHECKS=1`.
- **Mutation controls:** break each load-bearing line, run the suite, restore, and verify the restore.
  Prioritise the event-write guard, the activation isolation, and the `_admit()` calls.

## 13. Implementation Considerations

- **Complexity: medium-high.** `inference_service.py` is 5,678 lines with many generation paths.
- **Suggested order:** FR-27.9 guard first (it should go red on today's code) → FR-27.8 fixes →
  admission for parity and arm (FR-27.6e–f) → stateless scoring → per-request activations.
- **Parity runs synchronously in an async route** (`probes.py:412`, `probe_arming.py:400`). It
  blocks the event loop for every test vector. Moving the forward to a worker thread is a design
  choice; if made, suppression must be entered in that thread (§6).
- **The parity forward runs with steering live.** `build_parity_forward` suppresses no SAE
  (`probe_arm_bridge.py:119-140`). The probe hook reads before its own layer's steering, but a profile
  steering an earlier layer still moves the residual. See Open Question 3.
- **Per-request capture needs its own read** in scoring mode, because suppression disables the SAE
  hook's capture (FR-27.3b). A prepended read-only hook, like the probe hook, is one option.
- **Risks:** a discovery test that matches comments instead of calls (use the call graph, not
  source text); fixtures where offline and live agree by construction (score through the real
  forward and real decision code).

## 14. Open Questions

1. **Window boundaries on stored input.** How does a caller state where the prompt ends for
   `prompt`, `response` and `last_user` on stored input? Proposed: optional `prompt_tokens` beside
   `token_ids`; derive it for `messages` ending in an assistant turn; otherwise `not_scored` with
   `prompt_boundary_unknown`.
2. **Rendering `text`.** Is `text` tokenized raw, or rendered as one user turn, as miStudio built its
   training corpus? The two give different scores.
3. **Steering during stateless scoring and parity.** Should both run with every attached SAE
   suppressed, as chat and completion scoring do (R-04.8)? Today the parity and arm forwards suppress
   nothing, so an earlier-layer profile can move parity.
4. **Parity status.** Should stateless scoring refuse a probe with no passing parity report against
   the loaded model, or score it and report its parity status?
5. **`probe_ids` omitted.** Does that mean every imported probe matching the loaded model, with
   mismatches listed as skipped, or should `probe_ids` be required?
6. **Request shapes for activations.** With streaming, should activations come in a final
   `choices: []` chunk, as probe verdicts do, or be refused? With `n > 1`, `extra_messages` or several
   prompts, refuse (as sensing skips them) or return per choice?
7. **An unsteered counterfactual.** Is a third read point needed: a second forward with every SAE
   suppressed, so completion tokens are read as an unsteered model would read them?

## 15. Decisions from Clarifying Questions

Clarifying rounds were waived for this chain. Each answer below comes from a cited source. Questions
the sources do not settle are in §14.

| # | Question | Answer | Source |
|---|---|---|---|
| D1 | Priority? | Important; R-04.46 closes a latent defect in shipped Feature 24 | PPRD v1.5 Feature 27 |
| D2 | Primary users? | Integration feature: miDataworks and miStudio's MCP tool | BRD-04 header; BRD-03 R-03.52 |
| D3 | Batched or one input at a time for stored-text scoring? | One input at a time; never packed | Checkpoint technical default; PADR v1.5 packing trade-off |
| D4 | Activation read point default? | Post-steering (what the model computed); pre-steering by option | Checkpoint technical default; resolves BRD-04 OQ-5 |
| D5 | Reuse or copy the probe decision code? | Reuse: `armed_probe_from_row`, the parity forward, `_verdict_for`, `threshold_for_length` | R-04.28; PADR v1.5 |
| D6 | Does scoring need arming or an acknowledgement below rung 2? | No; it works on any imported probe and returns rung language | R-04.28, R-04.30; miStudio BR-019 |
| D7 | Is the parity route fixed or copied? | Fixed: brought under `_admit()`; the arm route too | R-04.29; FR-24.4's own promise |
| D8 | How are generation paths enumerated? | Discovered from the service; a hand-kept list does not count | R-04.47; PADR v1.5 |
| D9 | What does batched chat report? | `not_scored`, reason `batched_request`; per-row scoring is a non-goal | R-04.46; BRD-04 §7 |
| D10 | Data persistence? | None; no migration | R-04.27 ("writes no probe event") |
| D11 | User interface? | None new; existing status shows the new reasons | PPRD v1.5 Feature 27 "UI Tab" |
| D12 | Security? | No authentication, as for all miLLM routes | BRD-04 §4, decision 7 |
| D13 | MCP tool ownership? | miStudio; miLLM documents the route in its MCP contract | BRD-04 §4; BRD-MIS-DATAWORKS-001 |

## Appendix: BRD-04 Coverage

Every BRD-04 requirement owned by Feature 27 maps to exactly one FR.

| BRD-04 requirement | Section | FR | Acceptance item |
|---|---|---|---|
| R-04.24 | 5.6 Per-request SAE activations | FR-27.1 | 10 |
| R-04.25 | 5.6 | FR-27.2 | 10 |
| R-04.26 | 5.6 | FR-27.3 | 10 |
| R-04.27 | 5.7 Stateless probe scoring | FR-27.4 | 11 |
| R-04.28 | 5.7 | FR-27.5 | 11 |
| R-04.29 | 5.7 | FR-27.6 | 11 |
| R-04.30 | 5.7 | FR-27.7 | 11 |
| R-04.46 | 5.13 Probes on every generation path | FR-27.8 | 17 |
| R-04.47 | 5.13 | FR-27.9 | 17 |

**9 of 9** owned requirements covered, matching PPRD v1.5's count for Feature 27.
