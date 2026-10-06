# Feature 25: Chat Scoring, Structured Output, Seed and Request Validation — Task List

**Status:** Planned (2026-10-06). Clarifying rounds and the "Go" pause after parent tasks were
**waived** by the coordinator; the full list is generated in one pass.
**Inputs:** 025_FPRD v1.1, 025_FTDD v1.0, 025_FTID v1.0 · BRD-04 §5.1–§5.4, §6 · PADR v1.5 §10 ·
decisions of 2026-10-06 (checkpoint defaults; P-01–P-25, X-01–X-11; T-55–T-62)
**Code verified at** miLLM `7aa659c`. Re-check every line number before editing.
**Build order:** first in BRD-04's order (RSK-09). Phase 2.0 alone is enough for miDataworks and
miStudio 034 to send `X-miLLM-Strict: true` safely.

## Relevant Files
- `millm/api/request_policy.py` (new): table, neutral values, engine-unused map, strict parsing,
  header encoding · `tests/unit/api/test_request_policy.py`, `tests/unit/api/test_request_policy_coverage.py`
- `millm/api/json_schema_subset.py` (new): allowlist and check · `tests/unit/api/test_json_schema_subset.py`
- `millm/ml/constrained_decoding.py` (new): grammar cache, processor, output validation ·
  `tests/unit/ml/test_constrained_decoding.py`
- `millm/services/system_fingerprint.py` (new) · `tests/unit/services/test_seed_and_fingerprint.py`
- `millm/api/schemas/openai.py`: fields, `extra="allow"`, validators, response shapes ·
  `tests/unit/api/test_openai_schemas.py` (extended)
- `millm/ml/generation_config.py`: seed, constraint, alias · `tests/unit/api/test_generation_config.py` (extended)
- `millm/services/inference_service.py`: `_score_prompts`, chat scoring, constraint and seed wiring,
  CBM gate, streaming guard · `tests/unit/services/test_chat_scoring.py`,
  `tests/unit/services/test_structured_output.py`, `tests/unit/services/test_streaming_refusals.py`,
  `tests/unit/services/test_scoring_completions.py` (must stay green unchanged)
- `millm/api/routes/openai/chat.py`, `completions.py`, `embeddings.py`: policy, headers,
  fingerprint · `tests/unit/api/test_unused_fields_http.py`
- `millm/core/errors.py`, `millm/api/routes/openai/errors.py`: five errors, map rows ·
  `tests/unit/api/test_error_map_complete.py`
- `millm/core/config.py`, `.env.example`, `pyproject.toml`
- `tests/hardware/` (new scripts): chat-scoring parity, structured output, seed, constraint cost,
  llama.cpp seed
- `manual/` OpenAI API reference page
- `0xcc/reviews/review_feature025_<date>.md` (new at acceptance): mutation-control record

### Notes
- Backend tests: `pytest tests/unit` (and the new files individually). Lint/type: `ruff`, `mypy millm/`.
- **Reachability is a shipping gate:** every wiring line needs a test that FAILS when the line is
  removed, asserting payload and call count. Mutation controls are listed in FTID §8.3; each is
  re-run and recorded in 9.x. Verify a mutation landed (re-grep) before concluding it survived;
  confirm `git diff` is clean after each restore.
- **No hand-kept lists where a registry can be read:** `/v1` paths from `app.openapi()["paths"]`
  (FastAPI 0.141.1), error codes from `MiLLMError.__subclasses__()`, list fields from
  `OUTPUT_CHANGING` itself.
- Never a mocked model: tiny real transformers (FTID §8.1).
- **Not done by this task list's author:** editing `CLAUDE.md`, the PPRD or the PADR (forbidden for
  this pass). 9.9 records what the operator must update.

### Category Checklist Results
- **Data layer:** N/A — no table, column or migration (FTDD §4.1); schema (pydantic) changes are in 2.x/4.x.
- **Backend/API:** 2.x, 3.x, 4.x, 5.x, 6.x, 7.x
- **Frontend/UI:** N/A — no Admin UI change (FPRD §4); the contract is HTTP headers and body.
- **Business logic:** 2.x (policy), 5.x (scoring), 6.x (seed), 7.x (constraints)
- **Integration wiring:** 2.6–2.8 (routes), 5.3 (service branch), 6.3–6.5, 7.5–7.7, 8.x (Features 26/27/28/30 hand-offs)
- **Error handling & logging:** 3.x; 2.5 (log event); 7.8 (invalid output)
- **Testing:** throughout; HTTP tests in each phase; hardware 9.3–9.6; mutation controls 9.2
- **Performance & security:** 2.9 (header injection), 2.5/9.2 M4 (log privacy), 7.4 (schema size caps), 9.5 (GPU per-token cost), 2.10 (policy cost)
- **Configuration/deployment:** 1.x (dependencies, settings, `.env.example`); no k8s change
- **Documentation:** 8.5 (API reference), 9.9 (records for the operator)

## Tasks

- [ ] 0.0 Spikes and measurements that gate later work (covers FR-25.10.8, FR-25.14.3; FTDD §12)
  - [ ] 0.1 **gemma-4 tokenizer spike** (before 7.x ships): on the node, build
        `xgr.TokenizerInfo.from_huggingface` from the served gemma-4 tokenizer and compile the FTDD
        §3.1 test schema; require the target string accepted. Record compile time and mask timing.
        If it fails, add a refusal row for that model family and record why.
  - [ ] 0.2 **xgrammar on the node's CUDA stack:** in the deployed image, run one constrained
        `generate()` on LFM2.5-1.2B on the 3090 (the triton bitmask kernel path). A failure blocks
        7.x.
  - [ ] 0.3 **T-61 llama.cpp seed measurement** (before 6.6): forward `seed` to llama.cpp on the
        reference GGUF model, run the same sampled request twice, compare bytes. Record the
        result; 6.6 implements the matching outcome. Until it passes the outcome stays *refused*.

- [x] 1.0 Dependencies and configuration (covers FR-25.10.8; FTDD §11)
  - [x] 1.1 `pyproject.toml`: add `xgrammar>=0.2.8,<0.3` and `jsonschema>=4.23,<5` with a comment
        citing FTDD §3.1. Confirm the resolver keeps `transformers>=5.15.1,<6` and `torch>=2.10`.
  - [x] 1.2 `millm/core/config.py`: `STRUCTURED_OUTPUT_GRAMMAR_CACHE=64`,
        `IGNORED_FIELDS_HEADER_MAX_BYTES=1024`; `.env.example` entries.
  - [x] 1.3 Test: settings load with defaults and env overrides.

- [x] 2.0 Request policy: unused fields, strict mode, output-changing table (covers FR-25.1, FR-25.2, FR-25.3)
  - [x] 2.1 Schemas: `extra="allow"` at `openai.py:39,198,249,296`; update the docstring note
        (`:12`) and the stale comment (`:87-89`).
  - [x] 2.2 `request_policy.py`: `Endpoint`, `Engine`, `OUTPUT_CHANGING` with **every cell filled**
        per FPRD FR-25.3.3 (incl. 3.3a `max_completion_tokens`), `NEUTRAL` (FR-25.3.4),
        `ENGINE_UNUSED` (`chat_template_kwargs` on llama.cpp, FR-25.1.2b).
  - [x] 2.3 `evaluate()`: list fields first (declared via `model_fields_set`, extras via
        `model_extra`), refused + non-neutral → `FieldNotHonouredError`; then unused locations at
        top level, `messages[i]`, `extra_messages[j][i]` (FR-25.1.3, FR-25.1.4).
  - [x] 2.4 `parse_strict()`: `true`/`1` on, `false`/`0`/absent off, else `400 INVALID_PARAMETER`
        naming the header (FR-25.2.2). Strict + unused → `UnusedFieldsRefusedError` listing every
        location (FR-25.2.1).
  - [x] 2.5 Log event `request_fields_unused` (warning): endpoint, request id, locations — never
        values (FR-25.1.8).
  - [x] 2.6 Chat route: call `apply_request_policy` after the embedding-only check
        (`chat.py:124-125`), before auto-load (`chat.py:143`); set the header on non-streaming
        responses after generation and on `stream_headers` (`chat.py:249-258`) (FR-25.1.5, FR-25.1.7).
  - [x] 2.7 Completions route (`completions.py:71-72` → before `:110`) and embeddings route
        (before its auto-load): same call and header.
  - [x] 2.8 Engine from the row (`gguf_files`, `millm/db/models/model.py:126`) so refusals precede
        auto-load (FR-25.2.3, FR-25.3.8).
  - [x] 2.9 `encode_field_list()`: RFC 8941 sf-strings, escaping, percent-encoding outside
        printable ASCII, 1,024-byte bound with `"+N more"` (FR-25.1.6; FTDD §8).
  - [x] 2.10 Measure policy cost on a 50-message request; record in the review file (FPRD §8).
  - [x] 2.11 Tests — unit: table completeness (every field × endpoint × engine has a cell);
        neutral values; `model_fields_set` (default `n=1` not a presence); location strings;
        strict values incl. `yes` → 400; header bound; a key containing `\r\n` is escaped.
  - [x] 2.12 Tests — coverage (`test_request_policy_coverage.py`): enumerate `/v1` POST paths from
        `app.openapi()["paths"]`; an unmapped path fails; for every list field and path, POST it
        through the test client with a non-neutral value and assert `400` with `param` = field where
        refused, and not `400`-for-that-field where honoured, **with and without** strict
        (FR-25.3, SC-2). Spy `load_model_and_wait`: call count 0 for every refusal decidable from
        the row.
  - [x] 2.13 Tests — HTTP (`test_unused_fields_http.py`): `foo: 1` → `X-miLLM-Ignored-Fields:
        "foo"`; with strict → `400` naming `foo`; `messages[0].name` reported; absent header when
        nothing ignored; streaming carries the header; `chat_template_kwargs` on a GGUF row reported
        (and refused under strict, before load); the response body is unchanged by an unused field
        (FR-25.1.9; SC-1).
  - [x] 2.14 Test: a message extra never reaches `apply_chat_template` (spy the template call's
        message dicts) — guards the `extra="allow"` switch.
  - [x] 2.15 Test: the log event carries locations and not a sentinel value placed in the field
        (privacy; mutation M4).

- [x] 3.0 Errors and the error map (covers FR-25.3.2, FR-25.11; latent defect FTDD §5.5)
  - [x] 3.1 `millm/core/errors.py`: `FieldNotHonouredError`, `UnusedFieldsRefusedError`,
        `ResponseFormatUnsupportedError`, `NoChatTemplateError`, `ConstrainedOutputInvalidError`
        (class-level `code`/`status_code`).
  - [x] 3.2 `ERROR_STATUS_MAP` rows for the five, **plus** the missing `INVALID_SCORING_REQUEST`
        (400, invalid_request_error) and `NON_FINITE_LOGITS` (500, server_error) — today a scoring
        400 reaches clients typed `server_error` (`millm/api/exception_handlers.py:93-94`).
  - [x] 3.3 Test (`test_error_map_complete.py`): walk `MiLLMError.__subclasses__()` recursively;
        every `code` has a row (registry-derived, mutation M15).
  - [x] 3.4 Test: an out-of-vocabulary `allowed_token_ids` on `/v1/completions` returns type
        `invalid_request_error`.

- [x] 4.0 Small refusals and aliases (covers FR-25.3.3a, FR-25.3.3b, FR-25.3.5, FR-25.4)
  - [x] 4.1 `TextCompletionRequest`: `n > 1` → error naming `n` (T-56, FR-25.4.3).
  - [x] 4.2 Remove `user` from the three request classes (`openai.py:65,232,294`) so it is reported
        (T-57, FR-25.3.3b).
  - [x] 4.3 `max_completion_tokens` on chat and completions: equal-or-absent check, folded into
        `max_tokens`; on embeddings the table refuses it (T-58, FR-25.3.3a).
  - [x] 4.4 Chat schema: `stream` with `n > 1` or `extra_messages` → error; service guard at the top
        of `stream_chat_completion` (`inference_service.py:4288`) raising `FieldNotHonouredError`
        (FR-25.3.5).
  - [x] 4.5 CBM: confirm a listed field the CBM cannot honour never reaches `_cbm_chat_completion`
        (`:5230`) or `_cbm_text_completion` (`:5418`); route or refuse per the table (FR-25.3.6).
  - [x] 4.6 Tests: `n: 2` on completions → `400` on transformers and on a GGUF row, before load
        (SC-3); streaming `n: 2` → `400`; streaming `extra_messages` → `400`; direct service call
        with `n: 2` streaming raises (M16); `user` reported, strict refuses it;
        `max_completion_tokens: 5` limits generation to 5 tokens; conflicting values → `400` naming
        both; `max_completion_tokens: 1` satisfies scoring's limit.
  - [x] 4.7 Test: with CBM enabled in a test service, a request with `n > 1` is served serially
        (or refused) and never by the CBM path (spy call count 0).

- [ ] 5.0 Chat scoring (covers FR-25.5, FR-25.6, FR-25.7, FR-25.8, FR-25.9)
  - [ ] 5.1 Extract `_score_prompts` from `_score_text_completion` (`inference_service.py:4964-5022`);
        `_score_text_completion` calls it. `test_scoring_completions.py` stays green **unchanged**.
  - [ ] 5.2 Chat schema fields (`logprobs` bool, `top_logprobs` 0–20, `allowed_token_ids`,
        `return_tokens_as_token_ids`), `wants_scores()`, `validate_scoring_mode` with `stream` false,
        `top_logprobs` requires `logprobs: true` (FR-25.5.1–5.3, FR-25.6.1–6.2).
  - [ ] 5.3 Branch first in `create_chat_completion`, before `:3595` (FR-25.5.8); add
        `_score_chat_completion` per FTID §7.2 — render inside `_admit()`, `add_special_tokens=False`,
        no probe/sensing/steering calls (FR-25.5.4–5.6, FR-25.7.1, FR-25.7.3).
  - [ ] 5.4 No chat template → `NoChatTemplateError` (T-55, FR-25.5.9).
  - [ ] 5.5 Table rows: scoring with `profile`, `steering_intensity`, `steering`, `response_format`
        refused (FR-25.6.4, FR-25.7.2; X-09).
  - [ ] 5.6 GGUF row refused before auto-load; resident llama.cpp refused in the service
        (FR-25.6.3).
  - [ ] 5.7 Response shape: `ChatLogprobs`, `bytes` from decoded text, id form in `token`,
        `top_logprobs` length, `logprobs: null` with `allowed_token_ids` alone, `finish_reason:
        "length"`, usage sums (FR-25.8.1–8.9).
  - [ ] 5.8 `extra_messages`: one conversation at a time, index order, one slot, failure names the
        index, `X-miLLM-Batch` = choices (FR-25.9.1–9.5).
  - [ ] 5.9 Tests — parity (`test_chat_scoring.py`): chat scoring of `messages` equals completion
        scoring of `_format_chat_messages(messages)` with `add_special_tokens=False`: identical
        token ids and logprobs, exact on CPU (SC-4 unit form; M8).
  - [ ] 5.10 Test: `_score_prompts` is called once per request with the rendered texts and
        `add_special_tokens=False` (payload + count; M9).
  - [ ] 5.11 Test: an `extra_messages` scoring request returns one scored choice per conversation
        in order — proves the branch precedes the batched path (M7).
  - [ ] 5.12 Test: with an SAE attached and a profile active, chat scoring equals scoring with none
        attached; suppression entered in the worker thread (SC-5; M6, for both endpoints).
  - [ ] 5.13 Test: no probe, sensing or circuit-sensing context opens during chat scoring (spies,
        count 0) and `generate()` is never called (FR-25.7.3–7.4).
  - [ ] 5.14 Tests — refusals: `max_tokens: 2`, `n: 2`, `stream: true`, temperature 1e-4,
        `top_logprobs` without `logprobs`, `profile` with scoring, `response_format` with scoring,
        no template, GGUF row (no load), out-of-vocabulary id, empty render (index named).
  - [ ] 5.15 Tests — shape: `token_id:<id>` form with correct `bytes`; `top_logprobs: 0` → `[]`;
        `allowed_token_ids` alone → `logprobs: null` and a constrained `content`.

- [ ] 6.0 Seed and system fingerprint (covers FR-25.13, FR-25.14)
  - [ ] 6.1 Schemas: `seed` on chat and completions, range 0–2³²−1 (FR-25.13.1); table: refused on
        embeddings.
  - [ ] 6.2 `GenerationConfig.seed`; `_generate_sync` (`:5492`) under `fork_rng` + `manual_seed` in
        the worker thread (FR-25.13.2; FTID §7.4).
  - [ ] 6.3 Thread the seed through every transformers generation path: serial chat (incl. the `n`
        loop at `:3675`), streaming chat, batched chat, text completion.
  - [ ] 6.4 CBM gate (`:958-982`): seeded requests never use the CBM (FR-25.13.7).
  - [ ] 6.5 Scope reporting: `request` / `batch-shape` / `best-effort`; route sets `X-miLLM-Seed`
        only when a seed was sent (FR-25.13.6, FR-25.14.1–14.2, FR-25.14.4; T-60).
  - [ ] 6.6 llama.cpp: implement the outcome 0.3 measured — forward `seed` in `_llamacpp_params`
        (`:3933-3954`) and flip the table cell, or keep *refused* (FR-25.14.3; T-61).
  - [ ] 6.7 `system_fingerprint.py` and the route assignment on chat and text completion responses
        (FR-25.13.8–13.10).
  - [ ] 6.8 Tests: same sampled request with `seed: 7` twice → byte-identical text and
        `finish_reason` (SC-7; M13); seed 8 differs; an unseeded request after a seeded one is not
        reproducible from the seed (M12); greedy and scoring echo the seed; batched chat reports
        `batch-shape`; out-of-range seed → `400`; GGUF seed refused before load (until 6.6 flips).
  - [ ] 6.9 Tests: fingerprint contains model, revision, precision and engine; NULL revision →
        `unrecorded`; `"unknown"` dtype → `unrecorded`; changes when the loaded dtype changes; present
        on unseeded responses.

- [ ] 7.0 Structured output (covers FR-25.10, FR-25.11, FR-25.12)
  - [ ] 7.1 Schemas: `ResponseFormat` union (`text`, `json_object`, `json_schema` with `name`,
        `schema`, `description`, `strict`) (FR-25.10.1, FR-25.10.4); validators refuse it with
        `stop` (FR-25.11.5), `stream` (T-59) and scoring.
  - [ ] 7.2 `json_schema_subset.py`: allowlist per FTDD §4.3; `check()` with JSON pointers; size caps
        (FR-25.10.9, FR-25.11.2).
  - [ ] 7.3 Chat route: subset check after policy, before auto-load; table refuses GGUF
        (FR-25.11.1), completions and embeddings (FR-25.11.4), and the CBM when enabled
        (FR-25.11.3).
  - [ ] 7.4 `constrained_decoding.py`: `GrammarCache` (per load, LRU 64, stop ids from the generation
        config, logits-width vocab), `JsonConstraintProcessor` (no `assert`, device bitmask once),
        `validate_output` (FTID §3.3).
  - [ ] 7.5 Service: compile before `_admit()` via `asyncio.to_thread`; `_build_generate_kwargs`
        (`:2366`) appends a fresh processor per `generate()` and drops `assistant_model` with a log
        (FR-25.10.5).
  - [ ] 7.6 Batched rows: one matcher per row in `_create_batched_chat_completion`; serial `n`:
        fresh processor per loop iteration (FR-25.10.10).
  - [ ] 7.7 `finish_reason`: complete iff last generated id ∈ stop ids, else `"length"`
        (FR-25.12.1–12.2); route sets `X-miLLM-Constrained` when a constraint ran (FR-25.12.3).
  - [ ] 7.8 Complete output validated; failure → `ConstrainedOutputInvalidError` (500), logged
        without content (FR-25.10.6).
  - [ ] 7.9 Defence in depth: `_llamacpp_chat_completion` raises `ResponseFormatUnsupportedError` if
        a constraint arrives.
  - [ ] 7.10 Tests — subset honesty: for each allowlisted keyword, a violating string is rejected by
        the installed xgrammar matcher; a missing example fails; `uniqueItems` added to the allowlist
        turns it red (M14).
  - [ ] 7.11 Tests — end to end on the tiny char-tokenizer model: `json_object` output parses;
        `json_schema` output validates; `strict: false` still enforced; `max_tokens: 5` →
        `finish_reason: "length"` and `X-miLLM-Constrained` present (M11); processor removed →
        validation test red (M10); two rows of `extra_messages` both validate; `n: 2` both validate.
  - [ ] 7.12 Tests — refusals before load: GGUF row; `multipleOf` (pointer named); schema over
        64 KB; with `stop`; with `stream`; on `/v1/completions`; CBM enabled.
  - [ ] 7.13 Tests: speculative draft dropped for a constrained request (spy, log); grammar cache
        hit on the second identical schema; cache dropped on unload.

- [ ] 8.0 Integration hand-offs and documentation (covers FR-25.1.1, FR-25.3.7; FPRD §7, §10)
  - [ ] 8.1 Feature 26 note in code and FTID: batch lines call `evaluate(strict=True)` and reuse
        `encode_field_list`, `build_system_fingerprint`, `validate_output` (FR-25.1.1; 026 FR-26.1.6,
        FR-26.10.1). No Feature 26 code here.
  - [ ] 8.2 Feature 27/28/30 hooks: table cells for `steering` and `dimensions` read *refused* with
        a reason naming the owning feature (FR-25.3.7); `_score_prompts` docstring names the Feature 27
        capture seam (R-04.26).
  - [ ] 8.3 Confirm with miStudio 034's FTDD that `X-miLLM-Strict: true` (TD5) and
        `X-miLLM-Ignored-Fields` are the names it reads; record agreement in the review file.
  - [ ] 8.4 Delete the route-level GGUF scoring check in `completions.py:86-94` **only** after a test
        proves the table produces the identical response (status, type, `param`) (FTID I9).
  - [ ] 8.5 Manual API reference: headers (`X-miLLM-Ignored-Fields`, `X-miLLM-Strict`,
        `X-miLLM-Constrained`, `X-miLLM-Seed`), the outcome table, the JSON Schema subset, seed
        scopes, chat logprobs shape, `system_fingerprint` format.

- [ ] 9.0 Feature Acceptance
  - [ ] 9.1 Verify each FPRD success criterion SC-1…SC-8 and each user story US-1…US-7 against its
        test; tick or file the gap.
  - [ ] 9.2 Run every FTID §8.3 mutation control (M1–M16): back up, mutate one line, confirm it
        landed, run, require red, restore, confirm `git diff` clean. Record results in
        `0xcc/reviews/review_feature025_<date>.md`. Any survivor gets a regression test and is
        re-run as a negative control (SC-8).
  - [ ] 9.3 **Hardware — BRD-04 acceptance 3 (chat scoring parity):** JEV-9B-decision, bfloat16, on
        the 3090: 200 prompts, chat scoring of `messages` vs completion scoring of the same rendered
        prompt — identical token ids, every logprob within 1e-5 absolute.
  - [ ] 9.4 **Hardware — BRD-04 acceptance 4:** with a profile active, chat scoring equals scoring
        with no SAE attached.
  - [ ] 9.5 **Hardware — BRD-04 acceptance 5:** 100 `json_schema` requests on transformers all parse
        and validate; record the GPU per-token constraint cost against unconstrained; the same request
        on a GGUF model returns `400` with no model load (verify by the resident model's `loaded_at`
        unchanged).
  - [ ] 9.6 **Hardware — BRD-04 acceptance 6:** the same sampled request with `seed: 7` twice on the
        serial path gives byte-identical text and echoes the seed; include a run with the speculative
        draft configured if the node has one.
  - [ ] 9.7 **BRD-04 acceptance 1 and 2** over HTTP against the deployed pod: `foo: 1` reported;
        strict refuses; every refused list field returns `400` with and without strict; `n: 2` on
        completions returns `400`.
  - [ ] 9.8 Full suite: `pytest tests/unit` green; `ruff`, `mypy millm/` clean; the existing
        `test_scoring_completions.py` unchanged and green; also run with `0xcc/` hidden (the public
        mirror's view) so no test depends on these documents.
  - [ ] 9.9 Records for the operator (this pass may not edit them): CLAUDE.md Document Inventory and
        status, PPRD Feature 25 status, and the PADR amendments listed in FTDD §14 (stack: xgrammar,
        jsonschema; GGUF structured output closed as refused; `_score_prompts` as the shared scorer).
        File any follow-up found during acceptance as new tasks.

## Coverage Audit
- **FRs → tasks:** FR-25.1 → 2.1, 2.3, 2.5–2.9, 2.13–2.15, 8.1 · FR-25.2 → 2.4, 2.8, 2.11, 2.13 ·
  FR-25.3 → 2.2, 2.3, 2.12, 3.1–3.2, 4.2–4.5, 5.5, 8.2 · FR-25.4 → 4.1, 4.6 · FR-25.5 → 5.1–5.4,
  5.9–5.11 · FR-25.6 → 5.2, 5.6, 5.14 · FR-25.7 → 5.3, 5.5, 5.12, 5.13 · FR-25.8 → 5.7, 5.15 ·
  FR-25.9 → 5.8, 5.11 · FR-25.10 → 0.1, 0.2, 1.1, 7.1, 7.2, 7.4–7.6, 7.8, 7.10, 7.11 · FR-25.11 →
  3.1, 7.1, 7.3, 7.9, 7.12 · FR-25.12 → 7.7, 7.11 · FR-25.13 → 6.1–6.5, 6.7–6.9 · FR-25.14 → 0.3,
  6.5, 6.6, 6.8. **14 of 14.**
- **Acceptance criteria (implement + test):** SC-1 → 2.6/2.13 · SC-2 → 2.2/2.12 · SC-3 → 4.1, 4.4/4.6
  · SC-4 → 5.3/5.9, 9.3 · SC-5 → 5.3/5.12, 9.4 · SC-6 → 7.3–7.7/7.11–7.12, 9.5 · SC-7 → 6.2/6.8, 9.6
  · SC-8 → every phase's removal tests/9.2.
- **Edge cases (implement + test):** GGUF scoring/structured (5.6, 7.3 / 5.14, 7.12) · scoring
  limits (5.2 / 5.14) · `top_logprobs` without `logprobs` (5.2 / 5.14) · steering fields on scoring
  (5.5 / 5.14) · schema keyword outside subset (7.2 / 7.12) · `response_format` with scoring (7.1 /
  5.14) · streaming `n`/`extra_messages` (4.4 / 4.6) · seed on batched rows (6.5 / 6.8) · strict +
  message field (2.3–2.4 / 2.13) · no chat template (5.4 / 5.14).
- **TDD/TID sections:** data design N/A (no persistence); API design 2.x–7.x; components 2.2, 5.1,
  6.7, 7.2, 7.4; state 6.2, 7.4 (cache lifecycle); security 2.9, 2.15, 7.2; performance 2.10, 9.5;
  testing throughout; deployment 1.x; frontend N/A.
- **Open questions:** none open in the FPRD (all resolved, T-55–T-62). Outstanding **measurements**
  are tasks: T-61 → 0.3; gemma-4 tokenizer → 0.1; CUDA kernel → 0.2; GPU per-token cost → 9.5.
- **The final parent task is Feature Acceptance.** ✔
