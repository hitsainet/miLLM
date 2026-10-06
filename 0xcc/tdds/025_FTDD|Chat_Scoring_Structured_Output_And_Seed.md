# Technical Design: Chat Scoring, Structured Output, Seed and Request Validation
## miLLM Feature 25

**Version:** 1.0 · **Created:** 2026-10-06 · **Status:** Planned
**References:** 025_FPRD v1.1 · BRD-04 §5.1–§5.4 · PPRD v1.5 Feature 25 · PADR v1.5 §1 rows "Request
validation (v1.5)" and "Chat scoring (v1.5)", §10 "Dataworks Support" trade-offs · checkpoint and
Feature-PRD decisions 2026-10-06 (`~/app/miDataworks/0xcc/docs/brd-decisions-2026-10-05.md`) ·
technical register T-55–T-62 (`~/app/miDataworks/0xcc/docs/fprd-open-questions-2026-10-06.md`)
**Code verified at** miLLM `7aa659c` (HEAD, 2026-10-06). Re-check line numbers before editing.

---

## 1. Executive Summary

Feature 25 makes every `/v1` field either honoured or refused, adds chat scoring, structured output
and seeds. It changes no data model and adds no route.

| Area | Decision | Rationale |
|---|---|---|
| Unknown fields | Request schemas switch from `extra="ignore"` to `extra="allow"`; one policy module reads `model_extra` (top level, each message) | Pydantic already parses the body; reading its extras needs no second parse and cannot drift from validation |
| Output-changing list | One table in `millm/api/request_policy.py`: field × endpoint × engine → *honoured* or *refused* | FR-25.3.1: one place; the test reads the table and the live OpenAPI paths |
| Strict mode | `X-miLLM-Strict` parsed by the same module; unknown value refused | FR-25.2; aligned with miStudio 034 TD5 |
| Chat scoring | Extract the per-prompt loop of `_score_text_completion` into `_score_prompts`; both endpoints call it | One arithmetic path (PADR §10 "One scoring path"); chat needs the decoded token for `bytes`, which the completion response shape drops |
| Constrained decoding | **xgrammar 0.2.8**, driven by miLLM's own thin logits processor | Measured on transformers 5.15.1 and five served tokenizers: fastest masks, every target accepted, a transformers integration path already used by the library (§3) |
| Schema subset | miLLM declares an allowlist of JSON Schema keywords, checked before load, and validates every output with `jsonschema` | xgrammar compiles some keywords **without enforcing them** (measured: `multipleOf`, `not`, `uniqueItems`) |
| Seed | Applied inside the admission slot under `torch.random.fork_rng`; range 0–2³²−1 | Scoped to the request; unseeded traffic is unaffected; the range fits llama.cpp too |
| Fingerprint | `system_fingerprint` built from the model row and `LoadedModel` | R-04.14; unknown parts written `unrecorded` |
| Latent defect | Add the missing `ERROR_STATUS_MAP` rows for scoring errors | A 400 currently reaches OpenAI clients typed `server_error` |

## 2. System Architecture

```
POST /v1/{chat/completions | completions | embeddings}
  pydantic parse (extra="allow")  ── schema validators: scoring limits, n, alias, response_format shape, seed range
  route:
    row = find_model_by_name()           (exists today)
    engine = engine_of(row)              transformers | llamacpp   (row.gguf_files)
    policy = request_policy.evaluate(request, endpoint, engine, strict=parse_strict(headers))
        ├─ refused list field present  → 400 FIELD_NOT_HONOURED       (always)
        ├─ unused fields + strict      → 400 UNUSED_FIELDS_REFUSED
        └─ unused fields               → header X-miLLM-Ignored-Fields + log event
    response_format subset check (pure)  → 400 RESPONSE_FORMAT_UNSUPPORTED
    auto-load (exists today)
    InferenceService:
      chat scoring   → render template (in slot) → _score_prompts(add_special_tokens=False) → chat logprobs
      text scoring   → _score_prompts(add_special_tokens=request) → legacy logprobs
      generation     → GenerationConfig(+seed, +constraint) → _build_generate_kwargs(+logits_processor)
                       → _generate_sync under fork_rng(seed) → finish_reason → JSON validation
    route sets X-miLLM-Constrained, X-miLLM-Seed, body.system_fingerprint
```

**Integration points (existing code):**
- Schemas: `millm/api/schemas/openai.py:25-39` (`ChatMessage`), `:47-205` (`ChatCompletionRequest`),
  `:212-284` (`TextCompletionRequest`), `:287-296` (`EmbeddingRequest`), `:326-351` (chat response).
- Routes: `millm/api/routes/openai/chat.py:101-298`, `completions.py:51-148`,
  `embeddings.py:47-113`; aggregated in `millm/api/routes/openai/__init__.py:20-26`.
- Service: `create_chat_completion` (`millm/services/inference_service.py:3574`),
  `create_text_completion` (`:4780`), `_score_text_completion` (`:4930-5035`),
  `_build_generate_kwargs` (`:2366-2486`), `stream_chat_completion` (`:4288`),
  `_create_batched_chat_completion` (`:3393`), `_llamacpp_params` (`:3933-3954`).
- Errors: `millm/core/errors.py:42-73`; `ERROR_STATUS_MAP` at `millm/api/routes/openai/errors.py:61`;
  the live handler `millm/api/exception_handlers.py:77-107`.

## 3. Technical Stack

### 3.1 Constrained-decoding library: xgrammar (T-62)

**What was checked** (2026-10-06, in a private scratch target, never in the repo's venv):
miLLM's own venv (`python 3.12.3`, `transformers 5.15.1`, `torch 2.10.0+cu128`, `triton 3.6.0`,
`numpy 2.4.2`); PyPI metadata for five candidates; tokenizers downloaded for the served families; one
end-to-end generation on the reference model.

**Candidates (PyPI, 2026-10-06):**

| Library | Latest | Released | Licence | Dependencies of note | transformers integration |
|---|---|---|---|---|---|
| xgrammar | 0.2.8 | 2026-09-24 | Apache-2.0 | `apache-tvm-ffi`, `torch`, `transformers>=4.38`, `triton` (Linux x86), `numpy`, `pydantic` | ships `xgrammar/contrib/hf.py` (a `transformers.LogitsProcessor`) |
| llguidance | 1.9.1 | 2026-09-30 | MIT | none | tokenizer adapter only (`llguidance.hf.from_tokenizer`); processor must be written |
| outlines | 1.3.3 | 2026-08-06 | Apache-2.0 | `outlines_core`, `jinja2`, `diskcache`, `pillow`, `genson`, `cloudpickle`… | a framework wrapping xgrammar/llguidance/outlines-core backends |
| outlines-core | 0.2.14 | 2026-01-09 | Apache-2.0 | none | index only |
| lm-format-enforcer | 0.11.3 | 2025-08-24 | MIT | `interegular` | pure Python; no release in 13 months |

Every hard dependency of xgrammar except `apache-tvm-ffi` is already in miLLM's environment.
outlines would add five unrelated packages to reach the same backends. lm-format-enforcer is pure
Python and stale. The measured comparison was therefore xgrammar against llguidance.

**Measured on the served tokenizers** (one 3-field schema with `enum`, `minimum`/`maximum`,
`maxItems`; target `{"label":"humor","confidence":0.75,"reasons":["pun"]}`; CPU):

| Tokenizer (vocab) | xgrammar compile (s) | xgrammar mask median / max (ms) | llguidance compile (s) | llguidance mask median / max (ms) | Both accept target |
|---|---|---|---|---|---|
| LiquidAI/LFM2.5-1.2B-Instruct (64,402) | 0.126 | 0.001 / 0.037 | 0.313 | 0.014 / 0.991 | yes |
| autotrust/JEV-9B (248,077) | 0.610 | 0.004 / 0.059 | 1.543 | 0.016 / 0.578 | yes |
| ibm-granite/granite-4.1-8b (100,352) | 0.254 | 0.001 / 0.041 | 0.648 | 0.020 / 5.707 | yes |
| unsloth/Llama-3.1-8B-Instruct (128,256) | 0.486 | 0.002 / 0.036 | 1.219 | 0.021 / 3.904 | yes |
| unsloth/gemma-3-12b-it (262,145) | 0.728 | 0.002 / 0.038 | 2.398 | 0.017 / 3.230 | yes |

(`google/gemma-3-12b-it` is gated without a token; the unsloth mirror carries the same tokenizer.
gemma-4's tokenizer was not checked — see §12.)

**End to end on the reference model** (LFM2.5-1.2B-Instruct, float32, CPU, transformers 5.15.1
`generate()` with an xgrammar processor):
- Unconstrained, the model answered a "reply in JSON" prompt with a schema of its own
  (`{"analysis": …}`). Constrained, it produced `{"label": "humor", "confidence": 0.95}`, which
  validates.
- `max_new_tokens=5` gave `{"label":` — does not parse. This is the case FR-25.12 reports as
  `"length"`.
- Sampling with `torch.manual_seed(7)` twice gave identical text; seed 8 differed.
- A batch of two rows (one matcher per row) validated on both rows.

**The JSON Schema subset is not what the library compiles.** Probing xgrammar 0.2.8 with violating
inputs: `minimum`, `maximum`, `enum`, `pattern`, `minLength`, `maxItems`, `required`,
`additionalProperties: false` and `format: date` were enforced. **`multipleOf`, `not` and
`uniqueItems` compiled and were NOT enforced** — a violating string was accepted. A library upgrade
can change this list in either direction. Hence §4.3: miLLM's own allowlist, pinned by a test that
runs the same probe against the installed library, and a `jsonschema` validation of every output.

**Decision:** `xgrammar>=0.2.8,<0.3` (capped per the pyproject rule, `pyproject.toml:40-47`) and
`jsonschema>=4.23,<5` as a direct dependency (today transitive at 4.26.0). Rejected: llguidance —
5–20× slower median masks, millisecond worst cases, and a processor to write anyway; outlines — a
framework over the same backends; lm-format-enforcer — pure Python, stale.

**Why miLLM writes its own processor rather than using `xgrammar/contrib/hf.py`** (xgrammar 0.2.8):
- it checks acceptance with `assert` (line ~95), which `python -O` strips, so a rejected token would
  pass silently;
- it copies the bitmask host-to-device on every step (`self.token_bitmask.to(scores.device)`);
- it is never called after the final token (its own comment), so `is_terminated()` after
  `generate()` reads `False` on a complete document — measured on a tiny model. miLLM therefore
  derives completeness from the last generated token and validates the output, never from matcher
  state alone.

### 3.2 Other stack points
- No new service, queue or table. `torch.random.fork_rng` (torch ≥ 2.10, already pinned) scopes the
  seed.
- `jsonschema.Draft202012Validator` validates structured output after generation.

## 4. Data Design

### 4.1 No persistence
No migration, no table, no column. Nothing in this feature is stored.

### 4.2 Request schemas
- `ChatMessage`, `ChatCompletionRequest`, `TextCompletionRequest`, `EmbeddingRequest`:
  `model_config = {"extra": "allow"}`. Extras land in `model_extra`. Nothing in `millm/` dumps a
  request object (`grep model_dump` over request objects finds none at HEAD), so extras cannot leak
  into generation. A test pins that a message extra never reaches `apply_chat_template`.
- `ChatCompletionRequest` gains: `logprobs: Optional[bool]`, `top_logprobs: Optional[int]` (0–20),
  `allowed_token_ids` (as on completions, `openai.py:241`), `return_tokens_as_token_ids: bool`,
  `response_format: Optional[ResponseFormat]`, `seed: Optional[int]`,
  `max_completion_tokens: Optional[int]`.
- `TextCompletionRequest` gains `seed`, `max_completion_tokens`.
- `user` is **removed** from all three request schemas, so it lands in `model_extra` and is reported
  (T-57). Nothing reads it.
- `ResponseFormat` is a discriminated union on `type`: `text`, `json_object`, `json_schema`
  (`name` 1–64 chars `[A-Za-z0-9_-]`, `schema` object, `description`, `strict`).

### 4.3 Validation strategy
Three layers, each with one job:
1. **Schema validators (pydantic):** shapes and intra-request rules — scoring limits (extended to
   chat, with `stream` false), `top_logprobs` requires `logprobs: true`, `n > 1` on completions,
   `max_tokens`/`max_completion_tokens` agreement, seed range, `response_format` with scoring, with
   `stop`, with `stream`. These need no model and run before the row lookup.
2. **Request policy (`request_policy.evaluate`):** the output-changing table and unused fields, per
   endpoint and engine. Needs the model row, not the loaded model.
3. **Schema subset (`json_schema_subset.check`):** walks the user schema; any keyword outside the
   allowlist is refused with its JSON path. Pure; before auto-load.

**Declared subset (v1):** `type` (all seven), `properties`, `required`, `additionalProperties`
(boolean or schema), `items`, `minItems`, `maxItems`, `enum`, `const`, `minimum`, `maximum`,
`minLength`, `maxLength`, `pattern`, `anyOf`, local `$ref` with `$defs`, and the annotations
`title`, `description`, `default`, `examples`. Everything else is refused, including `multipleOf`,
`not`, `uniqueItems`, `allOf`, `oneOf`, `if`/`then`/`else`, `format`, `patternProperties`,
`dependentRequired` and remote `$ref`. A keyword joins the allowlist only with a test proving the
installed library enforces it (§10).

## 5. API Design

### 5.1 Request policy table (FR-25.3.3)
`OUTPUT_CHANGING: dict[str, dict[tuple[Endpoint, Engine], Outcome]]` in `millm/api/request_policy.py`.
`Endpoint` is `chat`, `completions`, `embeddings`; `Engine` is `transformers`, `llamacpp`. `Outcome`
is `HONOURED` or `Refused(reason)`. Neutral values (FR-25.3.4) live beside the table as
`NEUTRAL: dict[str, Callable[[Any], bool]]`. Features 26–30 add rows; they never add a second table.

Engine-specific unused declared fields (FR-25.1.2 case b) live in the same module:
`ENGINE_UNUSED = {("chat", "llamacpp"): {"chat_template_kwargs"}}` — today logged only at `info`
(`inference_service.py:3911-3920`).

### 5.2 Evaluation order (chat route; the others are subsets)
1. pydantic validation → `400` (existing handler, `exception_handlers.py:35`).
2. `X-miLLM-Strict` parse → `400 INVALID_PARAMETER` on an unknown value.
3. Row lookup and embedding-only check (existing, `chat.py:115-125`).
4. `request_policy.evaluate` → `400 FIELD_NOT_HONOURED` or `400 UNUSED_FIELDS_REFUSED`, else a list
   for the header.
5. Engine refusals from the row: scoring on GGUF (completions already does this,
   `completions.py:86-94`), `response_format` on GGUF, `seed` on GGUF until T-61 passes. All of these
   are rows in the table, so step 4 performs them.
6. Schema subset check.
7. Auto-load (existing, `chat.py:143-170`).
8. Service call; then headers set from the returned outcome.

### 5.3 Headers
- `X-miLLM-Ignored-Fields`: an RFC 8941 list of sf-strings, e.g. `"foo", "messages[2].name"`.
  Field names are user-controlled, so each is escaped as an sf-string; characters outside printable
  ASCII are percent-encoded. Bounded at 1,024 bytes; when entries are dropped, the last member is
  `"+N more"` (FR-25.1.6).
- `X-miLLM-Constrained`: `json_object` or `json_schema;name="judge_v1"` (the `X-miLLM-Circuit-Rung`
  style, `chat.py:209`).
- `X-miLLM-Seed`: `7;scope="request"` | `scope="batch-shape"` | `scope="best-effort"` (the last when
  the continuous batching manager (CBM) is running in this process, because its thread shares the
  global random generator — §7).
- Request header `X-miLLM-Strict`: `true`/`1` on, `false`/`0`/absent off, else `400`.

### 5.4 Chat scoring response
`ChatCompletionChoice` gains `logprobs: Optional[ChatLogprobs]`:
`{"content": [{"token", "logprob", "bytes": [int], "top_logprobs": [{"token","logprob","bytes"}]}]}`.
`finish_reason` stays `"length"` (as `inference_service.py:5011`). `system_fingerprint: Optional[str]`
is added to `ChatCompletionResponse` and `TextCompletionResponse`.

### 5.5 Errors
New `MiLLMError` subclasses in `millm/core/errors.py`, each with a row in `ERROR_STATUS_MAP`:

| Class | code | status / type | When |
|---|---|---|---|
| `FieldNotHonouredError` | `FIELD_NOT_HONOURED` | 400 / invalid_request_error | a list field whose outcome is *refused* |
| `UnusedFieldsRefusedError` | `UNUSED_FIELDS_REFUSED` | 400 / invalid_request_error | strict mode with unused fields; `details.fields` |
| `ResponseFormatUnsupportedError` | `RESPONSE_FORMAT_UNSUPPORTED` | 400 / invalid_request_error | keyword outside the subset; engine; CBM; combination |
| `NoChatTemplateError` | `NO_CHAT_TEMPLATE` | 400 / invalid_request_error | chat scoring on a model without a template (T-55) |
| `ConstrainedOutputInvalidError` | `CONSTRAINED_OUTPUT_INVALID` | 500 / server_error | a complete constrained output fails validation |

**Latent defect fixed here:** `INVALID_SCORING_REQUEST` (400) and `NON_FINITE_LOGITS` (500) have no
`ERROR_STATUS_MAP` row, so the live handler falls back to `(exc.status_code, "server_error")`
(`exception_handlers.py:93-94`). A scoring client sending a bad token id gets a 400 typed as a server
fault, which OpenAI SDKs retry. Rows are added, and a test walks every `MiLLMError` subclass (from
`__subclasses__`, not a list) and requires a row for each code.

## 6. Component Architecture

| Module | Status | Responsibility |
|---|---|---|
| `millm/api/request_policy.py` | new | the table, neutral values, engine-unused map, `parse_strict`, `evaluate`, header encoding |
| `millm/api/json_schema_subset.py` | new | the allowlist and `check(schema) -> None | Refusal` |
| `millm/ml/constrained_decoding.py` | new | `GrammarCache` (per loaded tokenizer, LRU of compiled grammars), `JsonConstraintProcessor` (`transformers.LogitsProcessor`), `validate_output` |
| `millm/services/system_fingerprint.py` | new | `build_system_fingerprint(row, loaded) -> str` |
| `millm/api/schemas/openai.py` | modified | fields, `extra="allow"`, validators, response shapes |
| `millm/ml/generation_config.py` | modified | `seed`, `constraint`, `max_completion_tokens` alias |
| `millm/services/inference_service.py` | modified | `_score_prompts`, chat scoring branch, processor and seed wiring, finish reason, streaming refusals |
| routes `chat.py`, `completions.py`, `embeddings.py` | modified | policy call, headers, fingerprint |
| `millm/core/errors.py`, `openai/errors.py` | modified | five errors, map rows, the two missing rows |

**Separation of concerns:**
- `request_policy` knows fields and engines, never tensors.
- `constrained_decoding` knows grammars and logits, never HTTP.
- The service owns ordering inside the admission slot.

### 6.1 Chat scoring flow
`create_chat_completion` branches on `request.wants_scores()` **first**, before llama.cpp, batched
and CBM (`inference_service.py:3595-3610`), mirroring `create_text_completion` (`:4795-4796`).
`_score_chat_completion`:
1. refuses a llama.cpp engine (as `:4945-4949`);
2. inside `_admit()`, checks `tokenizer.chat_template` (refuse `NO_CHAT_TEMPLATE`, T-55), renders
   each conversation with `_format_chat_messages(..., chat_template_kwargs)`
   (`inference_service.py:5580`) — inside the slot because an unload deletes the tokenizer
   (`:3065-3089`);
3. calls `_score_prompts(texts, add_special_tokens=False, allowed=…, temperature=…, top_k=…)`;
4. maps each result to a chat choice.

`_score_prompts` is the body of today's `_score_text_completion` loop (`:4964-5022`), returning
`(NextTokenScores, prompt_tokens)` per prompt and raising the same errors. `_score_text_completion`
calls it too, so the arithmetic exists once. It never calls `generate()` and never opens a probe or
sensing context, so Feature 27's discovery test (FR-27.9) can classify it (FR-25.7.4). Feature 27
(R-04.26) later adds activation capture inside `_score_prompts`, which is why it stays one function.

### 6.2 Constrained generation flow
1. The route has already checked the subset.
2. `GrammarCache.get(tokenizer_key, schema_json)` compiles off the event loop
   (`asyncio.to_thread`) **before** `_admit()`, so compilation (0.1–0.7 s cold, measured) never holds
   the GPU slot. `TokenizerInfo` is built once per load with `stop_token_ids` taken from the model's
   generation config, so the chat end-of-turn token counts as a stop.
3. `GenerationConfig.constraint` carries the compiled grammar. `_build_generate_kwargs` adds a fresh
   `JsonConstraintProcessor` per `generate()` call (matchers are stateful; the serial `n` loop at
   `inference_service.py:3675` calls `generate()` n times) and drops `assistant_model` with a log,
   as the batched path already does (`:3206-3210`).
4. After generation: complete when the last generated token is a stop token; otherwise
   `finish_reason: "length"`. A complete output is parsed and validated; failure raises
   `CONSTRAINED_OUTPUT_INVALID`.
5. Batched rows (`extra_messages`) use one matcher per row, as measured.

### 6.3 Seed flow
`GenerationConfig.seed` → `_generate_sync` (`inference_service.py:5492`) wraps `generate()` in
`torch.random.fork_rng(devices=[device])` + `torch.manual_seed(seed)` **in the worker thread**.
Scope is reported by the service: `request` for the serial path, `batch-shape` for
`_create_batched_chat_completion`, `best-effort` when the CBM is running. A seeded sampled request
is routed away from the CBM by adding `seed is not None` to the CBM gate (`:958-982`). On llama.cpp
the table says *refused* until T-61's measurement passes; then `_llamacpp_params` forwards `seed`.

## 7. State Management
- **No new global state** except `GrammarCache`, owned by the loaded model: built on load, dropped
  on unload (it holds a reference to the tokenizer's vocabulary). Keyed by `(model_id, loaded_at)`
  so a reload never reuses a stale `TokenizerInfo`.
- **Random state:** `fork_rng` saves and restores the CPU and the device generator, so an unseeded
  request after a seeded one is not made deterministic. The CBM's own thread shares the global
  generator, which is why its presence downgrades the seed scope to `best-effort` rather than
  claiming more (FR-25.14.4).
- **Per-request outcomes** (ignored fields, constrained format, seed scope) travel back to the
  route on the response object or a request-scoped ContextVar, following `get_probe_verdicts()`
  (`chat.py:295`). The route sets headers after generation for non-streaming responses.

## 8. Security Considerations
- **Header injection:** field names are attacker-chosen JSON keys. They are emitted only as escaped
  sf-strings; CR, LF and non-ASCII are percent-encoded. A test sends a key containing `\r\n`.
- **Privacy:** the `request_fields_unused` log event carries locations, never values (FR-25.1.8),
  with a mutation-controlled test.
- **Denial of service:** schema size capped (64 KB serialised, depth 32, 256 properties); grammar
  compile time bounded by the cache and run off the slot; the LRU holds 64 compiled grammars.
- **No authentication** (BRD-04 §3, §7): unchanged.

## 9. Performance & Scalability
- Unknown-field evaluation: a walk over `model_extra`, no forward pass, no database read. Measured on
  a 50-message request and recorded.
- Constrained decoding: CPU mask median 1–4 µs and worst case ≤ 0.06 ms across the served tokenizers
  (§3.1); the bitmask is allocated once per request on the logits' device and applied in place with
  xgrammar's kernel. The GPU per-token figure is a hardware-acceptance measurement (FTASKS).
- Grammar compile: cached; the first request with a new schema pays 0.1–0.7 s outside the slot.
- Chat scoring: one forward per conversation, unpacked, identical cost to completion scoring.

## 10. Testing Strategy
- **Real tiny models, never a mocked model** (the house rule in
  `tests/unit/services/test_scoring_completions.py:1-7`): the tiny Llama and WordLevel tokenizer for
  scoring; a tiny Llama with a **character-level** tokenizer covering JSON punctuation for
  structured output — measured to produce valid, schema-conforming JSON under xgrammar from random
  weights.
- **Parity:** chat scoring equals completion scoring of the rendered text with
  `add_special_tokens=False`, bit for bit on CPU, and the call to `_score_prompts` is asserted by
  payload and count.
- **Registry, not lists:** the policy test enumerates `/v1` POST paths from `app.openapi()["paths"]`
  (FastAPI 0.141.1 here; `app.routes` is not a reliable route list on this version) and fails if a
  path has no table column; the error-map test enumerates `MiLLMError.__subclasses__()`
  recursively.
- **Subset honesty:** for each allowlisted keyword, a violating string must be rejected by the
  installed xgrammar; for each refused keyword, the test documents why. An upgrade that stops
  enforcing a keyword goes red.
- **Mutation controls** on the load-bearing lines (FTID §8.3).
- **Hardware acceptance** on mcs-lnxhost02: BRD-04 acceptance 3 (JEV-9B-decision, bfloat16, 200
  prompts), 5 (100 `json_schema` requests on transformers; GGUF refused with no load) and 6 (seed),
  plus the GPU per-token constraint cost and T-61's llama.cpp seed measurement.

## 11. Deployment & DevOps
- `pyproject.toml`: add `xgrammar>=0.2.8,<0.3` and `jsonschema>=4.23,<5`. The image rebuild picks
  them up through CI; no local Docker builds.
- No new environment variable is required. Two optional settings in `millm/core/config.py`:
  `STRUCTURED_OUTPUT_GRAMMAR_CACHE` (default 64) and `IGNORED_FIELDS_HEADER_MAX_BYTES` (default
  1024), mirrored in `.env.example`.
- No Kubernetes manifest change. The CBM stays off in production (BRD-04 §3).
- **Logging:** `request_fields_unused` (warning), `request_field_refused` (info),
  `constrained_generation` (info: format, schema name, tokens, finish reason, mask milliseconds),
  `speculative_disabled_for_constraint` (info).
- **Rollback:** additive. Reverting the image restores `extra="ignore"` behaviour; no data to
  migrate.
- **Docs:** the API reference gains the headers, the outcome table, the subset and the seed scopes;
  `docs/mcp-contract.md` is miStudio's to update for its tools.

## 12. Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| A double BOS in chat scoring shifts every score plausibly | medium | `add_special_tokens=False`; SC-4 asserts token-ID equality |
| xgrammar's kernel or `apache-tvm-ffi` misbehaves on the node's CUDA stack | low–medium | hardware acceptance runs 100 constrained requests on the 3090; refusal path exists |
| gemma-4's tokenizer (`gemma4_unified`) was not measured | medium | a phase-0 spike compiles a schema with the node's gemma-4 tokenizer before phase 5 ships |
| xgrammar silently stops enforcing a keyword after upgrade | low | version cap; subset honesty test; output validation |
| Strict mode breaks a miDataworks request sending a harmless extra | intended | the error names the field |
| `extra="allow"` leaks a field into generation | low | no request dumps today; a test pins message extras never reach the template |
| CBM thread consumes the seeded generator | low (CBM off in production) | scope reported `best-effort` |
| Speculative decoding with a constraint | n/a | draft model dropped for constrained requests, logged |

**Alternatives considered:**
- Raw-body comparison in a middleware for unknown fields — rejected: a second parse that can drift
  from pydantic's aliases and coercions.
- Calling `_score_text_completion` with a synthesised `TextCompletionRequest` — rejected: the
  completion response keys tokens by id or text, never both, so chat's `bytes` would be lost.
- outlines as the constraint layer — rejected (§3.1).

**Complexity:** medium-high, concentrated in `inference_service.py`'s five chat paths.

## 13. Development Phases
1. **Policy and errors** (FR-25.1–FR-25.3, error-map fix). Unblocks every consumer's safety. ~2 days.
2. **`n`, streaming refusals, `max_completion_tokens`, `user`** (FR-25.3.3a/b, FR-25.3.5, FR-25.4).
   ~1 day.
3. **Chat scoring** (FR-25.5–FR-25.9). ~2 days.
4. **Seed and fingerprint** (FR-25.13, FR-25.14 except T-61). ~1.5 days.
5. **Structured output** (FR-25.10–FR-25.12), after the gemma-4 tokenizer spike. ~3 days.
6. **Acceptance:** mutation controls, hardware items, T-61 measurement, docs. ~1.5 days.

Phases 3–5 depend on phase 1 (every new field registers in the table). Milestone: phase 1 deployed
is enough for miDataworks to send strict mode safely.

## 14. Decisions from Clarifying Questions

Clarifying rounds were waived. Each decision cites its source.

| # | Question | Decision | Source |
|---|---|---|---|
| U1 | Constrained-decoding library | xgrammar `>=0.2.8,<0.3` | T-62; §3.1 measurements |
| U2 | Use xgrammar's bundled HF processor? | No; a thin miLLM processor | §3.1 (assert, per-step copy, no final call) |
| U3 | How is the schema subset defined? | miLLM allowlist + output validation | §3.1 probe: three keywords compiled but unenforced |
| U4 | Unknown-field detection mechanism | `extra="allow"` + `model_extra` walk | §12 alternatives |
| U5 | Shared scoring code | Extract `_score_prompts`; both endpoints call it | PADR §10 "One scoring path"; §12 alternatives |
| U6 | Where is the chat template rendered for scoring? | Inside the admission slot | `inference_service.py:3065-3089` (unload deletes the tokenizer) |
| U7 | Seed range | 0 – 2³²−1, refused outside | fits llama.cpp; FR-25.13.1 |
| U8 | Seed scoping | `fork_rng` in the worker thread | FR-25.13.2 |
| U9 | Seed on the CBM | route serial; scope `best-effort` when the CBM runs | FR-25.13.7, FR-25.14.4 |
| U10 | Speculative decoding with a constraint | draft dropped, logged | `inference_service.py:3206-3210` precedent |
| U11 | Where grammars compile | before `_admit()`, off the event loop, cached | §9 |
| U12 | Header encoding | RFC 8941 sf-string list, escaped, 1,024-byte bound | §8 |
| U13 | Strict header name and values | `X-miLLM-Strict`, `true`/`1`/`false`/`0`, else `400` | FPRD FR-25.2.2; miStudio 034 FTDD TD5; T-100 |
| U14 | Missing scoring rows in `ERROR_STATUS_MAP` | Fix here (touched code) | `exception_handlers.py:93-94`; review discipline |
| U15 | Scoring always unsteered? | Yes, on both endpoints | X-09 |
| U16 | Structured output on GGUF; streaming | Refused in v1 | checkpoint default; T-59 |

**Genuinely open:** none that blocks design. Measurements outstanding: T-61 (llama.cpp seed), the
gemma-4 tokenizer spike, GPU per-token cost. **PADR amendments recommended (not made here):**
(a) PADR §2 Technology Stack should list xgrammar and jsonschema; (b) PADR §10 "Constrained decoding"
still says GGUF is refused "unless BRD-04 open question 3 extends support" — the checkpoint closed it
(refused in v1); (c) PADR §10 "One scoring path" names `_score_text_completion` as the shared
function — the design shares its extracted loop, `_score_prompts`; the rationale is unchanged.
