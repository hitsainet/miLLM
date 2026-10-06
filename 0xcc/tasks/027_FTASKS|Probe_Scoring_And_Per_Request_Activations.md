# Feature 27: Probe Scoring, Per-Request Activations and Probe-Path Fixes — Task List

**Specified in:** BRD-04 §5.6, §5.7, §5.13 (Dataworks Support, 2026-10-06)

**Status:** Planned (2026-10-06). Clarifying rounds and the "Go" pause after the parent tasks were
**waived** by the coordinator for this chain; parent and sub-tasks are generated together.
**Inputs:** 027_FPRD v1.1, 027_FTDD v1.0, 027_FTID v1.0 · BRD-04 · PPRD v1.5 Feature 27 · PADR v1.5 §10
**Binding decisions:** checkpoint technical defaults; P-03, P-20, X-03, X-09; T-49, T-72 – T-77.
**Consumers waiting:** miDataworks 009 (probe-verdict labeler, feature tagger); miStudio 034
(`millm_score_probes`).

## Relevant Files
- `millm/services/probe_scoring.py`: preparer and scoring service · `tests/unit/services/test_probe_scoring.py`, `test_probe_input_preparer.py`, `test_probe_score_writes_nothing.py`
- `millm/services/request_activations.py`: capture and extension builder · `tests/unit/services/test_request_activations.py`
- `millm/services/inference_service.py`: `run_model_work`, detached probe contexts, FR-27.8 wiring, activation seams · `tests/unit/services/test_run_model_work.py`, `test_probe_paths_discovered.py`, `test_probe_wiring.py` (reworked)
- `millm/services/probe_arm_bridge.py`: `build_probe_forward`
- `millm/services/probe_arming.py`: `arm(..., executor)` · `tests/unit/services/test_probe_arming.py`
- `millm/services/probe_parity.py`: report keys · `tests/unit/services/test_probe_parity.py`
- `millm/services/probe_runtime.py`: `_verdict_for` (unchanged; guarded) · `tests/unit/services/test_verdict_boundary_is_one_place.py`, `test_probe_runtime.py`
- `millm/api/routes/management/probes.py`: score route; parity and arm under admission · `tests/unit/api/test_probe_score_route.py`, `tests/unit/api/test_probe_routes.py`
- `millm/api/routes/openai/chat.py`, `completions.py`: activation refusals, completions verdict header · `tests/unit/api/test_return_sae_activations.py`
- `millm/api/schemas/probe_scoring.py`, `millm/api/schemas/millm_extension.py`, `millm/api/schemas/openai.py`
- `millm/ml/sae_wrapper.py`, `millm/ml/sae_hooker.py`: request capture
- `millm/core/errors.py`, `millm/api/routes/openai/errors.py`, `millm/core/config.py`, `.env.example`
- `tests/integration/test_probe_score_matches_live.py`
- `docs/mcp-contract.md`, `manual/docs/features/probe-monitors.md`, the OpenAI API manual page
- `tests/unit/test_mcp_contract_consistency.py`, `tests/unit/test_mcp_tool_paths_are_real.py` (must stay green)

### Notes
- Tests: `pytest tests/unit` (backend), `ruff`, `mypy millm/`. Cross-repo checks with
  `MILLM_REQUIRE_CROSS_REPO_CHECKS=1`.
- **Reachability is a shipping gate:** every wiring line has a test that fails when the line is
  removed, asserting the payload and call count.
- **Mutation controls** (FTID §8, M1–M21): back up, mutate one line, run, restore. **Verify the
  restore** with `git diff` and a re-grep of the mutated line. Never leave a mutation in the tree. An
  unlanded mutation looks exactly like a surviving one, so confirm each edit landed first.
- **No hand-kept lists.** The guard's scenario table is checked for equality against discovery.
- Re-verify every line number from the FTID before editing.

### Category Checklist Results
- **Data layer:** N/A for schema — no migration (FPRD §5). Stored-data change is the two additive parity-report keys, task 3.4.
- **Backend/API:** 4.x (score route), 3.x (parity, arm), 6.x (`/v1` field and response), 2.7 (completions header)
- **Frontend/UI:** N/A — no new UI (FPRD §4); task 2.9 checks the existing page renders the new reasons.
- **Business logic:** 4.x (preparer, scoring), 6.x (capture), 5.x (boundary)
- **Integration wiring:** 2.x (generation paths), 3.x (arm, parity), 6.6 (Feature 25 field), 7.3 (Feature 26 hand-off)
- **Error handling and logging:** 4.6, 4.7, 6.4, 2.8 (hung-thread capture clear), 4.10 (no content in logs)
- **Testing:** throughout; discovery guard 1.x; integration 8.4; hardware 8.5–8.7
- **Performance and security:** 4.9 (caps, id range), 6.5 (chunked encode), 8.8 (dynamo-reset cost)
- **Configuration/deployment:** 7.1 (config, `.env.example`); no k8s change
- **Documentation:** 7.2 (MCP contract), 7.4 (manual)

## Tasks

- [?] 0.0 Verification before `text` ships (covers FR-27.4a2; T-49)
  - [?] 0.1 On one miStudio probe definition, compare each test vector's `token_ids` with (a) live
        serving's render of its `messages` (`_format_chat_messages`, generation prompt on) and (b) the
        drift-check render (generation prompt off, `probe_parity.py:517-519`). Record which reproduces
        the recorded ids. Record: `0xcc/reviews/probe_scoring_phase0_2026-10.md`. **[?] needs hardware — operator session.**
  - [?] 0.2 Score one miStudio-evaluated set as `text` (one user turn) through the score service on
        the probe's model; compute AUROC; it must lie inside miStudio's reported interval. If it does
        not, `text` stays refused and the finding goes to the operator — do not change the render rule
        silently. **[?] needs hardware — operator session.**
  - [x] 0.3 Until 0.2 passes, `text` inputs are refused with `INVALID_PROBE_SCORE_REQUEST` naming the
        pending verification (FTID §9); test that refusal; remove the gate in the same commit that
        records 0.2's pass.

- [x] 1.0 The discovery-based probe-path guard, written first (covers FR-27.9)
  - [x] 1.1 `tests/unit/services/test_probe_paths_discovered.py`: AST discovery of generation sites in
        `InferenceService` from the primitive sets in FTID §8, counting calls **and** callables passed
        as arguments, walking nested closures (`inference_service.py:4246`).
  - [x] 1.2 Entry-point discovery over the `self.<method>` call graph; assert the set is non-empty and
        contains `create_chat_completion`, `stream_chat_completion`, `create_text_completion`, so a
        broken parser cannot pass by finding nothing.
  - [x] 1.3 Event-log harness: lock-guarded log; spies on `_probe_begin` and `_probe_begin_detached`;
        wrappers on every discovered site; fake primitives. Scenario table keyed by site, with
        `assert set(table) == discovered_sites`.
  - [x] 1.4 Per-scenario assertions (FTDD §10.1 step 4): site entered; every `gen` preceded by a
        `begin`; exactly one `record` with a verdict or `not_scored_reason` (payload asserted);
        `current_request() is None` afterwards.
  - [x] 1.5 Exemptions dict (scoring mode, embeddings) with reasons, and a test that no exempt method is
        a generation site.
  - [x] 1.6 Run on today's code and **record the expected red**: batched chat, `_cbm_chat_completion`,
        `_cbm_text_completion` and the three llama.cpp paths fail. A guard that is green before the
        fix is not guarding the fix.
  - [x] 1.7 Rework `tests/unit/services/test_probe_wiring.py:49-53`: parametrise
        `TestEveryServingPathIsWired` over the discovered entry points; delete the literal list.

- [x] 2.0 Probe-path fixes (covers FR-27.8a–h)
  - [x] 2.1 `_probe_begin_detached(request_id, reason)` and `_probe_record(..., detached=True)` (skips
        `end_request()`), beside the existing seams (`inference_service.py:2498-2640`). Unit-test: a
        detached context is never registered with `ProbeRuntimeState`, and recording it leaves another
        open context untouched (FR-27.8h).
  - [x] 2.2 `_create_batched_chat_completion` (`inference_service.py:3393`): detached context with
        `batched_request` inside the slot (3459); `_probe_finish` before the return (3511);
        `_probe_record` in the `finally` (3505). Test: the verdict header on a batched response says
        `not_scored; reason="batched_request"` (BRD-04 acceptance 17; FPRD US-7).
  - [x] 2.3 `_cbm_chat_completion` (5230) and `_cbm_text_completion` (5418): detached context with
        `continuous_batching`; finish before return; record in `finally`. Test with
        `PROBE_FORCE_SERIAL=False`.
  - [x] 2.4 Migrate `_cbm_stream_chat_completion` (5296) to the detached context. Test: two concurrent
        CBM requests with a probe armed **both** record `continuous_batching` (the latent collision,
        FR-27.8h).
  - [x] 2.5 The three llama.cpp paths (3960, 4012, 4221): detached context with `engine_unsupported`.
        Test each.
  - [x] 2.6 Error paths: a generation failure on each fixed path still records an event (record in
        `finally`). Test by making the fake primitive raise.
  - [x] 2.7 `/v1/completions` sets `X-miLLM-Probe-Verdicts` after `create_text_completion`
        (`completions.py:147-148`), as `chat.py:295-297` (FR-27.8g). Test the header is present with a
        probe armed and absent with none.
  - [x] 2.8 The hung-thread guard (`inference_service.py:4750-4755`) also closes any open activation
        capture. Test with a capture open.
  - [x] 2.9 Admin UI: confirm the Probe Monitors page shows `batched_request`, `continuous_batching`
        and `engine_unsupported` as text (no code change expected; add a vitest case if it renders
        unknown reasons blank).
  - [x] 2.10 Discovery guard (1.x) now green. **Negative controls M1–M10:** delete each of the ten
        paths' begin call, one at a time — create_chat (3645), stream_chat (4380), text (4838), CBM
        stream, batched, CBM chat, CBM text, three llama.cpp — each must turn the guard red. **M11:** add
        a method calling `self._generate_sync` with no scenario → red. Record all eleven.

- [x] 3.0 Admission and suppression for model work (covers FR-27.6e–g, FR-27.7, FR-27.5a1)
  - [x] 3.1 `InferenceService.run_model_work(fn)` and `_unsteered_call` (FTID §3). Tests: one `_admit`
        entry per call; `_unsteered` entered **in the worker thread** (a fake SAE counting
        `suppressed()` per thread); an unloading model refuses before `fn` runs.
        `test_every_request_queue_slot_is_taken_through_admission` stays green.
  - [x] 3.2 `build_probe_forward(model, layers)`; `build_parity_forward` delegates. Existing parity
        tests unchanged and green.
  - [x] 3.3 Parity route (`probes.py:385-418`) runs the engine through `run_model_work`. Arm route
        (`probes.py:340-383`) passes `executor=inference.run_model_work`; `ProbeArmingService.arm`
        takes `executor` as a required keyword and awaits it at `probe_arming.py:400`. Tests: payload
        (the executor is the service's `run_model_work`) and call count (once per arm, once per parity).
        Update `test_probe_arming.py` callers with an inline executor.
  - [x] 3.4 `ParityReport.as_details()` adds `model` and `checked_at` (`probe_parity.py:246`); readers
        treat both as optional. Test old reports without the keys still read.
  - [x] 3.5 Test the steering fix itself: with a steering profile active on an **earlier** layer,
        parity scores equal the unsteered scores (they did not before; record the pre-fix difference).
  - [x] 3.6 Controls: **M15** (drop `_admit` from `run_model_work`), **M16** (drop `executor=` from the
        arm route), **M17** (enter `_unsteered` outside the worker) — each red. Record.

- [x] 4.0 Stateless probe scoring (covers FR-27.4, FR-27.5, FR-27.6a–d, FR-27.7)
  - [x] 4.1 Schemas in `millm/api/schemas/probe_scoring.py`, all `extra="forbid"`: one input kind per
        input; `prompt_tokens` only with `token_ids`; caps. Tests for each refusal.
  - [x] 4.2 `ProbeInputPreparer` (FTID §3): `token_ids` as given; assistant-ended `messages` with the
        prefix check; user-ended `messages` (whole input is prompt); `text` as one user turn;
        `last_user` span via `probe_turns.py:153` (T-72, T-49). Tests per rule, including a failed prefix
        check → `prompt_boundary_unknown`, and `last_user` on `token_ids` → `token_ids_have_no_turns`.
  - [x] 4.3 Probe resolution: given ids refuse on mismatch with the existing errors (identity, scope,
        SAE); omitted ids → all matching probes, mismatches in `skipped` with code and reason (T-75);
        all skipped → `400`. Tests for each (FPRD US-3).
  - [x] 4.4 Scoring loop: build with `armed_probe_from_row`; one `run_model_work` per input; one forward
        per input across all layers; `context.finish()`. Tests assert by **call** (spies with
        `wraps=`) that `armed_probe_from_row`, `_verdict_for` and `threshold_for_length` run, and that
        `_admit` is entered once per input (FR-27.6a).
  - [x] 4.5 Response mapping via `verdict_payload` (FTID §11): `verdict` keeps `null`; `rung_language`
        verbatim; `provisional` and `threshold_revision` carried (P-20); `input_kind`, `n_tokens`,
        `prompt_tokens`; `token_ids` only when asked. Test each field; test a provisional window keeps
        its flag.
  - [x] 4.6 Pinning: `(model_id, loaded_at)` at the first input; a change fails the remaining inputs
        with `MODEL_CHANGED` and keeps the earlier results. Test with a simulated reload between inputs.
  - [x] 4.7 Whole-request refusals: no model, GGUF (existing errors), shape errors
        (`ProbeScoreRequestError`), over-context input. Tests for each edge case in FPRD §2.
  - [x] 4.8 Route `POST /api/probes/score` (`probes.py`, prefix line 43). Reachability: the path is in
        `app.openapi()["paths"]`; it reaches the score handler, not a `{probe_id}` handler. **M20:** remove
        the route → red.
  - [x] 4.9 Caps and id range: `PROBE_SCORE_MAX_INPUTS`, `PROBE_SCORE_MAX_PROBES`, token ids within the
        vocabulary. Tests for each.
  - [x] 4.10 Privacy: a scoring call's log records contain no input text or token list above debug.
        Test with `caplog`.
  - [x] 4.11 **No event write, no arming change** (`test_probe_score_writes_nothing.py`): real SQLite
        session; `probe_events` count unchanged; `begin_request` call count 0; `has_armed()` unchanged;
        armed probe on the same layer records nothing; stored parity report unchanged. **M13** (call
        `begin_request` in `_score_one`) and **M14** (write an event row from the route) → red. Record.
  - [x] 4.12 Parity status per probe (T-74): `never_run`, `passed`, `failed`, `checked_against`
        (`unknown` for old reports). Test each.
  - [x] 4.13 No routing change (FR-27.7): after a scoring call, `_use_cbm_for_request` answers as before.

- [x] 5.0 The verdict boundary (covers FR-27.10; P-03, X-03, P-20)
  - [x] 5.1 Stateless boundary test: an exactly representable fixture (as
        `test_probe_runtime.py:157-173`) scored through the score route returns `verdict: true` with
        `score == threshold`.
  - [x] 5.2 `test_verdict_boundary_is_one_place.py`: AST guard; the set of probe-score/threshold
        comparisons in `millm/` is exactly `{_verdict_for}` (FTID §8).
  - [x] 5.3 **M12:** `>=` → `>` at `probe_runtime.py:665` → the live test (`test_probe_runtime.py:157`)
        **and** 5.1 go red. Add a second comparison anywhere in `millm/` → 5.2 red. Record both.
  - [x] 5.4 Confirm P-20 on the live path: `probe_events.provisional` is written for a provisional
        window (`db/models/probe.py:200`); existing test or add one.

- [x] 6.0 Per-request SAE activations (covers FR-27.1, FR-27.2, FR-27.3)
  - [x] 6.1 Schemas: `ReturnSaeActivations` (`extra="forbid"`, positions union, `read_point`), and
        `MillmExtension` in its own module; `millm` on both responses, omitted when `None` by a wrap
        serializer. Test a response without the field serialises byte-identical to today's.
  - [x] 6.2 `LoadedSAE.begin_request_capture` / `end_request_capture`, refusing a second open capture;
        hook calls before and after `apply_steering` (`sae_hooker.py:186-211`), running under
        suppression too. Tests: pre vs post differ when this layer steers; under suppression the capture
        still records and reports `unsteered`.
  - [x] 6.3 `RequestActivationCapture`: position resolution (`last`, `prompt`, `completion`, `all`,
        range); offset advances by the full pass width; the final sampled token is not reported;
        `features` then `top_k`; scoring mode's `last` is the last prompt position (FR-27.3c). Tests for
        each.
  - [x] 6.4 Route-level refusals before the slot: no matching SAE (`SAENotAttachedError`), ambiguous SAE
        (names candidates), `n > 1`, `extra_messages`, several prompts (T-76), llama.cpp, feature index
        out of range, `top_k` over cap, worst-case entries over cap (FR-27.2f). Test each, and that no
        generation ran.
  - [x] 6.5 Chunked encode and one host copy per pass. Test with a wide fake SAE that encode is called
        in chunks of `SAE_ACTIVATIONS_ENCODE_CHUNK`.
  - [x] 6.6 Wire the seams into the serial chat, streaming chat and text paths, and scoring mode. Serial
        routing for activation requests (as `_has_steering_override`, `inference_service.py:945-957`).
        Register the field with Feature 25's known-field set. Removal tests per seam (payload + count).
  - [x] 6.7 Streaming: one `choices: []` chunk with `millm` after the probe chunk and before `[DONE]`
        (T-76). Test the order.
  - [x] 6.8 Isolation (BRD-04 acceptance 10): two interleaved requests each get only their own
        positions. **M18** (capture never cleared) and **M19** (read point ignored) → red. Record.
  - [x] 6.9 The response `note` states the read point is not a counterfactual (FR-27.2e, T-77). Test the
        text is present.

- [x] 7.0 Configuration, contracts and documentation (covers FR-27.4g, FR-27.10e)
  - [x] 7.1 Config block and `.env.example` lines (FTID §9). Test defaults load.
  - [x] 7.2 `docs/mcp-contract.md`: next additive minor version; `POST /api/probes/score` in the
        `millm_probes` inventory; §4d states `score >= threshold`. Contract consistency tests green.
  - [x] 7.3 Tell Feature 26's chain: `/api/probes/score` batch rows call the service with one input,
        never packed (FR-27.6b). Note in this file's follow-ups; do not edit 026's files.
  - [x] 7.4 Manual: "Scoring stored text" section, the boundary rule, the new not-scored reasons, and
        `return_sae_activations` in the OpenAI API page with the read-point note.

- [?] 8.0 Feature Acceptance
  - [x] 8.1 Verify each FPRD success criterion (SC-1 … SC-7) and user story (US-1 … US-7) one by one;
        record evidence.
  - [x] 8.2 Re-run every mutation control M1–M21; each red, each restore verified. Record in
        `0xcc/reviews/review_feature027_<date>.md`.
  - [x] 8.3 Full suite: `pytest tests/unit`, with `0xcc/` hidden as well (the public mirror's view), plus
        `ruff` and `mypy`. Cross-repo checks with miStudio present.
  - [x] 8.4 Integration (`tests/integration/test_probe_score_matches_live.py`): tiny real transformer —
        import → score unarmed → arm → live chat with `max_tokens: 1` → prompt-window scores equal.
  - [?] 8.5 **Hardware, BRD-04 acceptance 11** (mcs-lnxhost02, LFM2.5-1.2B): `/api/probes/score`
        reproduces a definition's test vectors within the parity tolerance, unarmed; armed live scores
        on the same inputs agree; `probe_events` count unchanged by the stateless call. **[?] needs hardware — operator session.**
  - [?] 8.6 **Hardware, BRD-04 acceptance 10:** two interleaved activation requests each get only their
        own activations; an over-cap request returns `400`. **[?] needs hardware — operator session.**
  - [?] 8.7 **Hardware, BRD-04 acceptance 17:** with a probe armed, a batched chat request records
        `batched_request` in the header and the event; removing the FR-27.8 call turns the guard red. **[?] needs hardware — operator session.**
  - [?] 8.8 Measure scoring throughput and the dynamo-reset share per input; hoist hook installs if it
        dominates (FTDD §9). Record the figures. **[?] needs hardware — operator session.**
  - [?] 8.9 Wait for any rollout to settle before 8.5–8.8 (a rollout kills in-flight GPU work). **[?] needs hardware — operator session.**
  - [x] 8.10 Update the miLLM project status and Document Inventory; record follow-ups (per-row
        batched probe scoring stays out of scope, BRD-04 §7).

## Coverage Audit
- **FRs:** 27.1 → 6.1, 6.3, 6.6, 6.7 · 27.2 → 6.2, 6.4, 6.9 · 27.3 → 6.2, 6.3 · 27.4 → 0.x, 4.1, 4.2,
  4.5, 4.8, 4.11 · 27.5 → 3.4, 4.3, 4.4, 4.12 · 27.6 → 3.1–3.5, 4.4, 4.6 · 27.7 → 4.11, 4.13 · 27.8 →
  2.1–2.9 · 27.9 → 1.x, 2.10 · 27.10 → 5.x, 7.2. Every FR is cited by a parent task.
- **Acceptance criteria (BRD-04):** 10 → 6.8 (impl/test), 8.6 · 11 → 4.x (impl), 4.11, 8.4, 8.5 (test) ·
  17 → 2.2 (impl/test), 2.10, 8.7.
- **Edge cases (FPRD §2), implementing → testing:** GGUF → 4.7/6.4 · no model → 4.7 · SAE probe missing
  SAE → 4.3 · unscorable scope → 4.3 · unknown window boundary → 4.2 · score on the bar → 5.1 · model
  swapped mid-request → 4.6 · over cap → 6.4 · shape refusals → 6.4 · no matching SAE / ambiguous SAE →
  6.4 · armed probe on the same layer during scoring → 4.11.
- **TDD sections:** data 3.4 · API 4.x, 6.x · components 2.x–6.x · state 2.1, 4.6, 6.2 · security 4.9,
  4.10 · performance 6.5, 8.8 · testing 1.x and throughout · deployment 7.x, 8.9.
- **TID sections:** file structure (Relevant Files) · components 3.x, 4.x, 6.x · database 3.4 · API 4.8,
  6.4, 2.7 · frontend 2.9 · business logic 4.x · testing 1.x, mutation controls · config 7.1 ·
  integration 6.6, 7.3 · utilities 4.5 · errors and logging 4.7, 4.10, 2.8 · performance 6.5, 8.8 ·
  quality 8.3.
- **Open questions:** none open in the FPRD (all seven decided by T-72 – T-77 and T-49). The one
  verification T-49 requires is task 0.x.
- **The final parent task is Feature Acceptance.** ✔

## Follow-ups (recorded during implementation, 2026-10-06)
- **For Feature 26's chain (task 7.3, FR-27.6b):** a batch row targeting `/api/probes/score` must
  call `ProbeScoringService.score` with ONE input per row and never pack rows, whatever the batch's
  `pack` setting — bfloat16 is not batch-invariant and each input takes its own admission slot.
  The one wire mapping to reuse for batch output lines is `probe_scoring.verdict_payload`. 026's
  files were not edited.
- **Out of scope, still open (BRD-04 §7):** per-row probe scoring inside batched chat — batched
  chat records `batched_request` until it exists.
- `probe_arming.scope_refusal` cannot fire today (`RUNTIME_SCORABLE_SCOPES` admits every scope);
  the arming docstring describing an `all`-only runtime is stale.
