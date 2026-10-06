# Feature 25 — Implementation Controls Record

**Date:** 2026-10-06 · **Branch:** `feat/025-chat-scoring` (from `main` @ `0efff20`)
**Scope:** 025_FTASKS parent tasks 0.0–8.0 and the non-hardware part of 9.0.
**Method (every row):** back up the file, apply ONE edit, run the affected tests and require a red,
restore from the backup, then verify the restore by sha256 against the pre-mutation file AND by
re-grepping the original line. Each mutation is checked to have LANDED (the file on disk equals
the mutated text) before its result is read. Automated by a scratchpad helper (`mutate.py`), one
control at a time, never two pytest processes at once.

A surviving mutation is a **test finding**: the regression test is written, and the same mutation
is re-run as a negative control (rows suffixed `-rerun`).

## Suite counts

| When | `venv/bin/pytest tests/unit` |
|---|---|
| Before (main @ `0efff20`) | 3486 passed / 3 skipped / 0 failed (102 s) |
| After 3.0 (errors) | 3624 passed / 3 skipped / 0 failed |
| After 2.0 (request policy) | 3872 passed / 3 skipped / 0 failed |
| After 4.0 (small refusals) | 3898 passed / 3 skipped / 0 failed |
| After 5.0 (chat scoring) | 3932 passed / 3 skipped / 0 failed |
| After 6.0 (seed + fingerprint) | 3964 passed / 3 skipped / 0 failed |
| After 7.0 (structured output) | 4050 passed / 3 skipped / 0 failed |

## Discrepancies between the documents and the code (the code won)

1. **`InvalidParameterError` did not exist.** FTID §3.1 says `parse_strict` raises it and that
   `INVALID_PARAMETER` "already has a map row". The row existed; the class did not. Added in
   `millm/core/errors.py`.
2. **The /v1 handler never set `param`.** `millm_error_handler` built every MiLLMError envelope with
   `param=None`, so no MiLLMError refusal could name its field as FPRD §3 requires. It now copies
   `details["param"]` (control `C3-param`).
3. **34 codes had no `ERROR_STATUS_MAP` row, not 2.** The FTDD names `INVALID_SCORING_REQUEST` and
   `NON_FINITE_LOGITS`; the registry walk (3.3) found 34 (`MODEL_LOCKED`, `SAE_NOT_FOUND`,
   `PROFILE_INCOMPATIBLE`, …), every 4xx among them reaching OpenAI clients typed `server_error`.
   All 34 now have rows; statuses are each class's own.
4. **Intermediate table cells.** Phase 2 lands before the fields it lists exist on the chat schema
   (`seed`, `logprobs`, `response_format`, …). Those cells were committed as *refused* ("not
   implemented on this endpoint yet") and flipped to *honoured* in the phase that implements them,
   so no intermediate commit honours a field it drops. A test asserts that every honoured cell's
   field is declared on that endpoint's schema, and `evaluate` fails closed if one arrives undeclared.
5. **`profile` and `steering_intensity` are list fields.** FTASKS 5.5 puts their scoring refusal in
   the table; the table also refuses them on `/v1/completions` and `/v1/embeddings` (previously
   silently dropped there — undeclared) and on llama.cpp before auto-load (previously an
   `EngineUnsupportedError` after the load).
6. **Stale comment in `embeddings.py`.** It described a GGUF refusal "before the auto-load" with no
   code under it; GGUF embeddings are served by design (`test_gguf_refused_before_load.py`). Comment
   replaced.
7. **The existing schema tests pinned `extra="ignore"`.** `TestChatMessage::test_extra_fields_ignored`
   and `TestSteeringIntensityField::test_extra_ignore_retained` asserted extras vanish — the exact
   behaviour this feature replaces. Rewritten to the new contract: extras still parse (rollout safety)
   and are kept in `model_extra`.

## Measurements

- **Policy cost (2.10, FPRD §8):** `evaluate()` on a 50-message chat request, CPU, mean of 5,000
  calls: **11.6 µs** with no unused field, **15.9 µs** with 51 unused locations. No forward pass, no
  database read.

## Latent defects found and fixed in touched code

- **The continuous batching manager (CBM) dropped `n` and the penalties.** `_cbm_chat_completion`
  builds exactly one choice, and its GenerationConfig is fixed at start-up, while the gate checked
  only temperature and top_p. A CBM-served `n: 3` returned one choice; a `frequency_penalty` was
  silently not applied. The gate now routes `n > 1`, a seed, a constraint and a non-zero penalty
  serial, read in one place (`_cbm_route_kwargs`). Penalties are not on the FR-25.3 list; fixed
  because it is the same silent drop in the same line (operator rule: fix latent defects in touched
  code). The CBM is off in Kubernetes.
- **`n` on `/v1/completions` and a streamed `n`/`extra_messages`** — the FPRD's own FR-25.3.5 and
  FR-25.4.1 latent defects — refused at the schema and (streaming) in the service.
- **`logprobs: null` on every chat choice avoided.** Adding `ChatCompletionChoice.logprobs` would
  have serialised `"logprobs": null` into every generation response; it is declared with
  `exclude_if=None` (pydantic 2.12) so a client sending no new field gets the body it got before
  (FPRD §8). Pinned by `test_a_generation_response_carries_no_logprobs_key`.
- **Seed devices.** FTID §7.4 forks the RNG of "the CUDA devices the model occupies". But
  `torch.manual_seed` reseeds EVERY CUDA device, so forking only the model's cards would leave the
  others permanently reseeded. `seeded_rng` forks every initialised device (`_rng_devices`).
- **Seed over the serial `n` loop.** FTID §7.4 seeds inside `_generate_sync`, i.e. per
  `generate()` call. On the serial path with `n > 1` that makes all n choices identical. The serial
  chat path enters `seeded_rng` once around the whole n-loop (inside the slot; RNG state is
  process-global, so the worker threads draw from it). Text completions keep per-prompt seeding
  (FTID I7), as do the batched and streaming paths. Pinned by
  `test_n_choices_are_reproducible_and_not_all_identical` and control `C6-serial-loop`.
- **`GenerationConfig` copies dropped new fields.** `with_max_tokens`, `with_stop_sequences` and the
  static-cache branch of `_build_generate_kwargs` rebuilt the config field by field, so a seed or a
  constraint would have vanished on those paths. All three now use `dataclasses.replace`.
- **`system_fingerprint` on streams.** FTDD §5.2 assigns it on non-streaming responses only and sends
  only headers on a stream; FR-25.13.8 reads "every chat and text completion response". Followed the
  FTDD; a streamed chunk carries no fingerprint. Recorded as a follow-up.
- **Interrupted control.** The session was killed while control `C6-best-effort` held its mutation
  in `inference_service.py`. On resume the tree still carried `return base`; the file was restored
  from the control's pre-mutation backup (sha256 `52d4643c…` matched), every earlier mutated line
  was re-grepped present, the diff was scanned for mutation artifacts, and the control was re-run.
- **`json_object` compiled with xgrammar's built-in JSON grammar accepted ANY JSON value.** Found
  by the end-to-end test, not by reading: a random model under the built-in grammar produced a
  complete JSON array, and `validate_output` correctly refused it with a 500. `json_object` now
  compiles `{"type": "object"}` (control `C7-json-object`). The second line of defence did exactly
  its job, and the first line was wrong.
- **The FTDD's unenforced-keyword list is partly stale on this probe.** Re-running the probe against
  the installed xgrammar 0.2.8 with a character-level vocabulary: `uniqueItems` and `not` compile and
  are NOT enforced (as the FTDD says), but `multipleOf` (`{"type":"integer","multipleOf":2}`,
  document `3`) WAS rejected. `multipleOf` stays refused per the FTDD; it is a candidate for the
  allowlist once a probe on a served tokenizer confirms it.
- **`max_whitespace_cnt` is not set.** Under xgrammar's default `any_whitespace=True` a random model
  emits long whitespace runs inside a valid document (observed on the tiny model). A trained model
  rarely does, but a budget can be spent on whitespace; worth measuring in 9.5.

## Mutation controls

| # | Control | File | Mutation | Landed | Result | Red | Restore (sha256 + re-grep) |
|---|---|---|---|---|---|---|---|
| 1 | M15 | `millm/api/routes/openai/errors.py` | delete the `INVALID_SCORING_REQUEST` row from ERROR_STATUS_MAP | yes | 1 failed, 20 passed in 2.55s | **red** | ok |
| 2 | C3-param | `millm/api/exception_handlers.py` | /v1 handler: `param=param ...` -> `param=None` | yes | 1 failed, 134 passed in 2.61s | **red** | ok |
| 3 | M1 | `millm/api/request_policy.py` | `evaluate()`: strict branch `if unused and strict:` -> `if False:` | yes | 1 failed, 49 passed in 2.28s | **red** | ok |
| 4 | M2 | `millm/api/request_policy.py` | `evaluate()`: refused-outcome check removed (refused fields pass) | yes | 1 failed, 2 passed in 2.94s | **red** | ok |
| 5 | M3 | `millm/api/request_policy.py` | `evaluate()`: `unused.extend(_message_locations(request))` removed | yes | 1 failed, 32 passed in 2.25s | **red** | ok |
| 6 | M4 | `millm/api/request_policy.py` | `apply_request_policy`: log event also carries `repr(request.model_extra)` (values) | yes | 1 failed, 57 passed in 2.26s | **red** | ok |
| 7 | M5-chat | `millm/api/routes/openai/chat.py` | chat route: `apply_request_policy(...)` call replaced by an empty result | yes | 1 failed, 60 passed in 6.22s | **red** | ok |
| 8 | M5-completions | `millm/api/routes/openai/completions.py` | completions route: policy call replaced by an empty result | yes | 1 failed, 65 passed in 11.74s | **red** | ok |
| 9 | M5-embeddings | `millm/api/routes/openai/embeddings.py` | embeddings route: policy call replaced by an empty result | yes | 1 failed, 66 passed in 11.81s | **red** | ok |
| 10 | H-chat-nonstream | `millm/api/routes/openai/chat.py` | chat route: non-streaming `X-miLLM-Ignored-Fields` assignment removed | yes | 1 failed in 6.14s | **red** | ok |
| 11 | H-chat-stream | `millm/api/routes/openai/chat.py` | chat route: streaming `X-miLLM-Ignored-Fields` assignment removed | yes | 1 failed, 4 passed in 11.53s | **red** | ok |
| 12 | H-completions | `millm/api/routes/openai/completions.py` | completions route: header assignment removed | yes | 1 failed, 5 passed in 11.64s | **red** | ok |
| 13 | H-embeddings | `millm/api/routes/openai/embeddings.py` | embeddings route: header assignment removed | yes | 1 failed, 6 passed in 11.79s | **red** | ok |
| 14 | P-engine-unused | `millm/api/request_policy.py` | `evaluate()`: ENGINE_UNUSED append removed (chat_template_kwargs on llama.cpp) | yes | 1 failed, 34 passed in 2.30s | **red** | ok |
| 15 | P-fail-closed | `millm/api/request_policy.py` | `evaluate()`: fail-closed branch for an undeclared field on an honoured cell disabled — SURVIVED: no test reached it | yes | 230 passed in 13.92s | **SURVIVED** | ok |
| 16 | P-fail-closed-rerun | `millm/api/request_policy.py` | same mutation, negative control after adding `test_an_undeclared_list_field_on_an_honoured_cell_fails_closed` | yes | 1 failed, 35 passed in 2.21s | **red** | ok |
| 17 | M16 | `millm/services/inference_service.py` | `stream_chat_completion`: the n>1 / extra_messages guard disabled | yes | 1 failed, 6 passed in 2.81s | **red** | ok |
| 18 | S-comp-n | `millm/api/schemas/openai.py` | `TextCompletionRequest`: n>1 validator disabled — SURVIVED: over HTTP the table refuses `n` too, so no test saw the schema rule direct callers rely on | yes | 85 passed in 6.98s | **SURVIVED** | ok |
| 19 | S-chat-stream-n | `millm/api/schemas/openai.py` | chat schema: streaming + n>1 validator disabled | yes | 1 failed, 3 passed in 2.75s | **red** | ok |
| 20 | S-chat-stream-extra | `millm/api/schemas/openai.py` | chat schema: streaming + extra_messages validator disabled | yes | 1 failed, 4 passed in 2.80s | **red** | ok |
| 21 | S-mct-fold | `millm/api/schemas/openai.py` | `max_completion_tokens` no longer folded into `max_tokens` | yes | 1 failed, 8 passed in 2.85s | **red** | ok |
| 22 | S-mct-disagree | `millm/api/schemas/openai.py` | disagreeing `max_tokens`/`max_completion_tokens` check disabled | yes | 1 failed, 9 passed in 5.32s | **red** | ok |
| 23 | CBM-n | `millm/services/inference_service.py` | CBM gate: `n > 1` no longer routes serial | yes | 1 failed, 18 passed in 6.89s | **red** | ok |
| 24 | CBM-penalty | `millm/services/inference_service.py` | CBM gate: penalties no longer route serial | yes | 1 failed, 19 passed in 6.85s | **red** | ok |
| 25 | S-user-removed | `millm/api/schemas/openai.py` | `user` re-declared on the chat schema (so it would no longer be reported) | yes | 1 failed, 14 passed in 6.65s | **red** | ok |
| 26 | S-comp-n-rerun | `millm/api/schemas/openai.py` | same mutation, negative control after `test_the_schema_refuses_it_for_direct_callers_too` | yes | 1 failed, 2 passed in 2.69s | **red** | ok |
| 27 | M6 | `millm/services/inference_service.py` | `_unsteered_next_token_logits`: `with self._unsteered()` -> `nullcontext()` | yes | 1 failed, 5 passed in 4.95s | **red** | ok |
| 28 | M7 | `millm/services/inference_service.py` | chat scoring branch moved after the llama.cpp and `extra_messages` branches | yes | 1 failed, 1 passed in 4.92s | **red** | ok |
| 29 | M8 | `millm/services/inference_service.py` | `_score_chat_completion`: `add_special_tokens=False` -> `True` (double BOS) | yes | 1 failed in 4.79s | **red** | ok |
| 30 | M9 | `millm/services/inference_service.py` | `_score_chat_completion` reimplements tokenise/forward/score instead of calling `_score_prompts` | yes | 1 failed, 1 passed in 4.76s | **red** | ok |
| 31 | C5-top-truncate | `millm/services/inference_service.py` | chat `top_logprobs` no longer truncated to the requested count | yes | 1 failed, 29 passed in 8.12s | **red** | ok |
| 32 | C5-logprobs-null | `millm/services/inference_service.py` | chat logprobs object built even when only `allowed_token_ids` was sent | yes | 1 failed, 30 passed in 7.99s | **red** | ok |
| 33 | C5-no-template | `millm/services/inference_service.py` | NO_CHAT_TEMPLATE refusal disabled (scoring would use the generic fallback) | yes | 1 failed, 22 passed in 5.73s | **red** | ok |
| 34 | C5-bytes | `millm/services/inference_service.py` | chat `bytes` built from the id instead of the decoded text | yes | 1 failed, 27 passed in 8.04s | **red** | ok |
| 35 | C5-index | `millm/services/inference_service.py` | `_score_prompts` no longer records the failing index | yes | 1 failed, 25 passed in 8.09s | **red** | ok |
| 36 | C5-schema-stream | `millm/api/schemas/openai.py` | chat scoring `stream=false` limit disabled | yes | 1 failed, 11 passed in 5.17s | **red** | ok |
| 37 | C5-top-needs-logprobs | `millm/api/schemas/openai.py` | `top_logprobs` without `logprobs: true` accepted | yes | 1 failed, 13 passed in 5.04s | **red** | ok |
| 38 | C5-table-profile | `millm/api/request_policy.py` | table: `profile` honoured on a scoring request (refuse_if removed) | yes | 1 failed, 17 passed in 5.29s | **red** | ok |
| 39 | M12 | `millm/services/inference_service.py` | `seeded_rng`: `torch.random.fork_rng(...)` removed (seed applied, global state not restored) | yes | 1 failed, 6 passed in 5.18s | **red** | ok |
| 40 | M13 | `millm/services/inference_service.py` | `seeded_rng`: `torch.manual_seed(seed)` removed | yes | 1 failed in 4.89s | **red** | ok |
| 41 | C6-generate-sync | `millm/services/inference_service.py` | `_generate_sync`: `seeded_rng(seed)` dropped (batched + text completion unseeded) | yes | 1 failed, 3 passed in 5.12s | **red** | ok |
| 42 | C6-stream-thread | `millm/services/inference_service.py` | streaming: seed kwargs no longer passed to the generation thread | yes | 1 failed, 5 passed in 5.14s | **red** | ok |
| 43 | C6-in-thread | `millm/services/inference_service.py` | `_generate_in_thread`: `seeded_rng(seed)` dropped | yes | 1 failed, 5 passed in 5.15s | **red** | ok |
| 44 | C6-serial-loop | `millm/services/inference_service.py` | serial chat: `with seeded_rng(gen_config.seed)` around the n-loop -> nullcontext | yes | 1 failed in 4.95s | **red** | ok |
| 45 | C6-batched | `millm/services/inference_service.py` | batched chunk: seed kwargs not passed to `_generate_sync` | yes | 1 failed, 4 passed in 5.07s | **red** | ok |
| 46 | C6-text | `millm/services/inference_service.py` | text completion: seed kwargs not passed to `_generate_sync` | yes | 1 failed, 3 passed in 5.04s | **red** | ok |
| 47 | C6-cbm-seeded | `millm/services/inference_service.py` | CBM gate: seeded requests no longer route serial | yes | 1 failed, 10 passed in 5.36s | **red** | ok |
| 48 | C6-scope-batch | `millm/services/inference_service.py` | batched path records scope `request` instead of `batch-shape` | yes | 1 failed, 9 passed in 5.25s | **red** | ok |
| 49 | C6-best-effort | `millm/services/inference_service.py` | `_seed_scope` never downgrades to `best-effort` while the CBM runs (first run interrupted mid-control; see note below; re-run here) | yes | 1 failed, 10 passed in 5.29s | **red** | ok |
| 50 | C6-route-chat-seed | `millm/api/routes/openai/chat.py` | chat route: non-streaming `X-miLLM-Seed` assignment removed | yes | 1 failed, 12 passed in 7.65s | **red** | ok |
| 51 | C6-route-stream-seed | `millm/api/routes/openai/chat.py` | chat route: streaming `X-miLLM-Seed` assignment removed | yes | 1 failed, 14 passed in 10.29s | **red** | ok |
| 52 | C6-route-comp-seed | `millm/api/routes/openai/completions.py` | completions route: `X-miLLM-Seed` assignment removed | yes | 1 failed, 15 passed in 10.00s | **red** | ok |
| 53 | C6-fp-chat | `millm/api/routes/openai/chat.py` | chat route: `system_fingerprint` assignment removed | yes | 1 failed, 24 passed in 15.04s | **red** | ok |
| 54 | C6-fp-comp | `millm/api/routes/openai/completions.py` | completions route: `system_fingerprint` assignment removed | yes | 1 failed, 24 passed in 15.11s | **red** | ok |
| 55 | C6-llamacpp-chat-seed | `millm/services/inference_service.py` | service defence in depth: llama.cpp chat seed refusal disabled | yes | 1 failed, 25 passed in 15.39s | **red** | ok |
| 56 | C6-table-seed | `millm/api/request_policy.py` | table: seed honoured on llama.cpp (T-61 refusal removed) | yes | 1 failed, 22 passed in 13.81s | **red** | ok |
| 57 | C6-bool-seed | `millm/api/schemas/openai.py` | `seed: true` accepted as seed 1 | yes | 1 failed, 19 passed in 13.47s | **red** | ok |
| 58 | M10 | `millm/services/inference_service.py` | `_build_generate_kwargs`: `processors.append(processor)` removed (unconstrained generation) | yes | 1 failed in 5.06s | **red** | ok |
| 59 | M11 | `millm/services/inference_service.py` | `_finish_constrained` reads matcher `is_terminated()` instead of the last generated token | yes | 1 failed in 5.01s | **red** | ok |
| 60 | M14 | `millm/api/json_schema_subset.py` | `uniqueItems` added to ALLOWED_KEYWORDS without enforcement | yes | 1 failed, 62 passed in 10.84s | **red** | ok |
| 61 | C7-route-subset | `millm/api/routes/openai/chat.py` | chat route: `json_schema_subset.check(...)` call removed | yes | 1 failed, 17 passed in 10.09s | **red** | ok |
| 62 | C7-route-cbm | `millm/api/routes/openai/chat.py` | chat route: CBM-enabled refusal disabled | yes | 1 failed, 25 passed in 10.66s | **red** | ok |
| 63 | C7-route-header | `millm/api/routes/openai/chat.py` | chat route: `X-miLLM-Constrained` assignment removed | yes | 1 failed, 12 passed in 6.79s | **red** | ok |
| 64 | C7-compile-serial | `millm/services/inference_service.py` | serial chat: constraint not compiled (`constraint = None`) | yes | 1 failed in 5.03s | **red** | ok |
| 65 | C7-compile-batched | `millm/services/inference_service.py` | batched chat: constraint not compiled | yes | 1 failed, 8 passed in 5.29s | **red** | ok |
| 66 | C7-validate | `millm/services/inference_service.py` | `_finish_constrained`: `validate_output` call removed | yes | 1 failed, 16 passed in 10.21s | **red** | ok |
| 67 | C7-spec-drop | `millm/services/inference_service.py` | `_build_generate_kwargs`: assistant_model kept for a constrained request | yes | 1 failed, 29 passed in 10.93s | **red** | ok |
| 68 | C7-unload-drop | `millm/services/inference_service.py` | `on_model_unloading` no longer drops the grammar cache | yes | 1 failed, 31 passed in 10.69s | **red** | ok |
| 69 | C7-stream-guard | `millm/services/inference_service.py` | `stream_chat_completion`: response_format guard disabled | yes | 1 failed, 27 passed in 10.71s | **red** | ok |
| 70 | C7-json-object | `millm/ml/constrained_decoding.py` | json_object compiled with xgrammar's builtin JSON grammar (any value) | yes | 1 failed, 5 passed in 5.18s | **red** | ok |
| 71 | C7-table-scoring | `millm/api/request_policy.py` | table: response_format honoured on a scoring request — SURVIVED: the schema validator refuses first over HTTP, so no test reached the cell a direct caller relies on | yes | 140 passed in 15.78s | **SURVIVED** | ok |
| 72 | C7-table-scoring-rerun | `millm/api/request_policy.py` | same mutation, negative control after `test_the_table_refuses_response_format_on_scoring_for_a_direct_caller` | yes | 1 failed, 63 passed in 2.29s | **red** | ok |

