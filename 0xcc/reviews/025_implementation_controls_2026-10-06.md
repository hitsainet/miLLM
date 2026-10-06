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

