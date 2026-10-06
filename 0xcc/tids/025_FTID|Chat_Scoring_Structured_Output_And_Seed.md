# Technical Implementation Document: Chat Scoring, Structured Output, Seed and Request Validation
## miLLM Feature 25

**Version:** 1.0 · **Created:** 2026-10-06 · **Status:** Planned
**References:** 025_FPRD v1.1, 025_FTDD v1.0 · BRD-04 §5.1–§5.4
**Load-bearing points verified against** miLLM `main` @ `7aa659c` (2026-10-06).
**Re-check every line number before editing.** `inference_service.py` is ~5,700 lines and moves.

---

## 1. Implementation Overview

Six phases, each landing behind tests that fail when its wiring is removed:

1. **Policy** — `request_policy.py`, `extra="allow"`, strict mode, the outcome table, the error-map
   fix.
2. **Small refusals** — `n` on completions, streaming `n`/`extra_messages`, `max_completion_tokens`,
   `user`.
3. **Chat scoring** — `_score_prompts` extracted; chat branch first in `create_chat_completion`.
4. **Seed and fingerprint.**
5. **Structured output** — xgrammar processor, grammar cache, subset, validation.
6. **Acceptance.**

**Principles:**
- **Refuse at the boundary, before auto-load,** whenever the request and the model row decide it.
  The routes already do this for GGUF scoring (`millm/api/routes/openai/completions.py:86-94`).
- **One table, read everywhere.** No endpoint keeps its own copy of the list.
- **Never a mocked model.** Tiny real transformers, as `tests/unit/services/test_scoring_completions.py`
  does.
- **Assert the wiring, not the existence:** payload and call count.

## 2. File Structure and Organization

**New files:**
```
millm/api/request_policy.py          OUTPUT_CHANGING table, NEUTRAL, ENGINE_UNUSED, parse_strict,
                                     evaluate(), encode_field_list()
millm/api/json_schema_subset.py      ALLOWED_KEYWORDS, ANNOTATIONS, check(schema) -> Optional[Refusal]
millm/ml/constrained_decoding.py     GrammarCache, JsonConstraintProcessor, validate_output()
millm/services/system_fingerprint.py build_system_fingerprint(row, loaded) -> str
```
**Tests (new):**
```
tests/unit/api/test_request_policy.py              table, neutral values, strict parsing, header encoding
tests/unit/api/test_request_policy_coverage.py     every /v1 POST path × every list field (OpenAPI-derived)
tests/unit/api/test_unused_fields_http.py          header, strict 400, message-level, log has no values
tests/unit/api/test_error_map_complete.py          every MiLLMError subclass code has a map row
tests/unit/api/test_json_schema_subset.py          allowlist checks + library honesty probe
tests/unit/services/test_chat_scoring.py           parity with completion scoring, routing, unsteered
tests/unit/services/test_seed_and_fingerprint.py   seed repeatability, scope, fork_rng isolation
tests/unit/ml/test_constrained_decoding.py         processor, cache, termination, validation
tests/unit/services/test_structured_output.py      end to end on a tiny model with a char tokenizer
tests/unit/services/test_streaming_refusals.py     stream + n>1 / extra_messages / response_format
```
**Modified files:**
- `millm/api/schemas/openai.py` — fields, `extra="allow"`, validators, response shapes.
- `millm/ml/generation_config.py` — `seed`, `constraint`, `max_completion_tokens`.
- `millm/services/inference_service.py` — `_score_prompts`, `_score_chat_completion`, constraint and
  seed in `_build_generate_kwargs`/`_generate_sync`, CBM gate, streaming refusals, finish reason.
- `millm/api/routes/openai/{chat,completions,embeddings}.py` — policy, headers, fingerprint.
- `millm/core/errors.py`, `millm/api/routes/openai/errors.py` — five errors, seven map rows.
- `millm/core/config.py`, `.env.example` — two optional settings.
- `pyproject.toml` — `xgrammar>=0.2.8,<0.3`, `jsonschema>=4.23,<5`.
- `tests/unit/services/test_scoring_completions.py` — keeps passing unchanged (the extraction must
  not alter completion scoring).
- `manual/` API reference page for OpenAI endpoints — headers, table, subset, seed scopes.

**Import rules:** `request_policy` and `json_schema_subset` import nothing from `services` or `ml`.
`constrained_decoding` imports `xgrammar` lazily inside functions, so the module imports in a
process without a GPU and tests can monkeypatch it.

## 3. Component Implementation Hints

### 3.1 `request_policy.py`
```python
class Endpoint(str, Enum): CHAT = "chat"; COMPLETIONS = "completions"; EMBEDDINGS = "embeddings"
class Engine(str, Enum): TRANSFORMERS = "transformers"; LLAMACPP = "llamacpp"

HONOURED = Honoured()
def refused(reason: str) -> Refused: ...

OUTPUT_CHANGING: dict[str, dict[tuple[Endpoint, Engine], Outcome]] = {
    "logprobs": {(CHAT, TF): HONOURED, (CHAT, LC): refused("llama.cpp exposes no distribution"), ...},
    "seed":     {(CHAT, TF): HONOURED, (CHAT, LC): refused("seed on llama.cpp not yet measured (T-61)"), ...},
    ...  # the FPRD FR-25.3.3 table, every cell filled
}
NEUTRAL = {"n": lambda v: v == 1, "logprobs": lambda v: v is False,
           "response_format": lambda v: _type_of(v) == "text", "tools": lambda v: v == [],
           "logit_bias": lambda v: v == {}}
ENGINE_UNUSED = {(Endpoint.CHAT, Engine.LLAMACPP): frozenset({"chat_template_kwargs"})}
```
- **Every cell is filled.** A test fails if any (field, endpoint, engine) is missing — no default
  outcome exists, so a new endpoint cannot inherit "honoured" by accident.
- `evaluate(request, endpoint, engine, strict) -> PolicyResult(unused: list[str])` or raises.
  Order: list fields first (declared values via `request.model_fields_set`, extras via
  `request.model_extra`), then unused fields.
- **"Present" means sent**, not defaulted: use `model_fields_set` so a default `n=1` is never a
  presence.
- Message walk: `request.messages[i].model_extra` → `messages[i].<key>`;
  `request.extra_messages[j][i].model_extra` → `extra_messages[j][i].<key>`.
- `parse_strict(value: Optional[str]) -> bool`, raising `InvalidParameterError` on an unknown value.
  The error code `INVALID_PARAMETER` already has a map row (`openai/errors.py`, "INVALID_PARAMETER").
- `encode_field_list(locations, max_bytes)` → RFC 8941 list; escape `\` and `"`, percent-encode bytes
  outside 0x20–0x7E; append `"+N more"` when trimmed.

### 3.2 `json_schema_subset.py`
- `ALLOWED_KEYWORDS` per FTDD §4.3; `ANNOTATIONS = {"title","description","default","examples"}`.
- `check(schema)` walks recursively through `properties`, `items`, `additionalProperties`
  (schema form), `anyOf`, `$defs`; `$ref` must start `#/$defs/`. Returns the first offending keyword
  and its JSON pointer; the error lists all of them (cap 20).
- Size guards: serialised ≤ 64 KB, depth ≤ 32, ≤ 256 properties per object.

### 3.3 `constrained_decoding.py`
```python
class GrammarCache:
    def __init__(self, tokenizer, vocab_size: int, stop_token_ids: list[int], capacity: int): ...
    def compile(self, fmt: ResponseFormat) -> "xgr.CompiledGrammar": ...   # LRU by canonical JSON

class JsonConstraintProcessor(LogitsProcessor):
    """One per generate() call. Matchers are stateful."""
    def __init__(self, grammar, batch_size: int, vocab_size: int, device): ...
    def __call__(self, input_ids, scores):
        if self._started:
            for i, m in enumerate(self._matchers):
                if not m.is_terminated() and not m.accept_token(int(input_ids[i, -1])):
                    raise ConstrainedOutputInvalidError(...)   # never `assert`
        self._started = True
        for i, m in enumerate(self._matchers):
            if not m.is_terminated(): m.fill_next_token_bitmask(self._bitmask, i)
        xgr.apply_token_bitmask_inplace(scores, self._bitmask_on_device)
        return scores
```
- **Allocate the bitmask once** on the logits' device; xgrammar fills a CPU bitmask, so copy into a
  preallocated device buffer with `non_blocking=True` rather than `.to()` allocating every step.
- `vocab_size` is the model's logits width (`model.config.vocab_size`), not `len(tokenizer)`; the two
  differ when a head pads its vocabulary. Build `TokenizerInfo.from_huggingface(tokenizer,
  vocab_size=..., stop_token_ids=...)`.
- `stop_token_ids`: the model's `generation_config.eos_token_id` (int or list) plus the tokenizer's
  `eos_token_id`, deduplicated.
- `validate_output(text, fmt)`: `json.loads`; `json_object` requires a dict; `json_schema` runs
  `Draft202012Validator(schema).validate`.

### 3.4 `system_fingerprint.py`
`build_system_fingerprint(row, loaded)` →
`f"millm:{row.name}@{row.revision or 'unrecorded'}:{dtype}/{quant}:{loaded.engine}"`, mapping the
`LoadedModel` sentinel `"unknown"` (`millm/ml/model_loader.py:165-167`) to `unrecorded`. Pure; no I/O.

## 4. Database Implementation Approach

**N/A.** No schema, migration or query change (FTDD §4.1).

## 5. API Implementation Strategy

### 5.1 Schemas (`millm/api/schemas/openai.py`)
- Replace `model_config = {"extra": "ignore"}` at `:39`, `:198`, `:249`, `:296` with
  `{"extra": "allow"}`. Update the module docstring note 5 (`:12`) and the stale comment at `:87-89`
  (an older server silently returning one choice) — after this feature that server reports the field.
- Remove `user` from the three request classes (`:65`, `:232`, `:294`).
- Chat: add the fields from FTDD §4.2; add `wants_scores()` (`logprobs is True or allowed_token_ids
  is not None`) and `validate_scoring_mode` mirroring `:258-282` plus `stream` false; validators:
  `top_logprobs` requires `logprobs is True`; `response_format` (non-`text`) with scoring, `stop` or
  `stream` → error naming `response_format`; `stream` with `n > 1` or `extra_messages` → error.
- Shared validator for both completion requests: `max_completion_tokens` vs `max_tokens` (equal or
  one absent; resolved value written to `max_tokens` so every downstream reader, including the
  scoring check, sees one number); `seed` in `[0, 2**32 - 1]`.
- Text: `n > 1` → error naming `n` (T-56). Existing check at `:269-270` stays for scoring.
- Responses: `ChatLogprobToken`, `ChatLogprobs`, `ChatCompletionChoice.logprobs: Optional[...]`,
  `system_fingerprint: Optional[str]` on both response classes. Serialised with `exclude_none` where
  the existing code does.
- Pydantic `ValueError`s become `400` on `/v1` already (`millm/api/exception_handlers.py:35`), with
  `param` from the error location.

### 5.2 Routes
Shared helper `apply_request_policy(request, endpoint, row, raw_headers) -> list[str]` in
`request_policy.py`, called in each route right after the embedding-only check
(`chat.py:124-125`, `completions.py:71-72`, the equivalent in `embeddings.py`) and **before** the
auto-load block (`chat.py:143`). Then:
- non-streaming: after the service returns, set `X-miLLM-Ignored-Fields` (when non-empty),
  `X-miLLM-Constrained`, `X-miLLM-Seed`, and `result.system_fingerprint`.
- streaming (`chat.py:249-258`): add `X-miLLM-Ignored-Fields` and `X-miLLM-Seed` to
  `stream_headers`. Streaming with `response_format` never reaches here (schema refusal).
- The subset check runs in the chat route between policy and auto-load.
- `completions.py:86-94` GGUF scoring refusal stays; the table now produces the same refusal, so the
  route's own check becomes the table's row. Keep one: delete the route check only after the table
  test proves the same response (status, `param`).

## 6. Frontend Implementation Approach

**N/A.** No Admin UI change (FPRD §4). The headers are an API contract.

## 7. Business Logic Implementation Hints

### 7.1 `_score_prompts` (extracted from `inference_service.py:4964-5022`)
```python
def _score_prompts_sync(self, texts, *, add_special_tokens, allowed, temperature, top_k):
    # body of today's loop: tokenize, empty check, context check,
    # _unsteered_next_token_logits, vocab check, non-finite checks, next_token_scores
    return [(scores, prompt_tokens), ...]
```
Keep the `await asyncio.to_thread(self._unsteered_next_token_logits, inputs)` per prompt exactly as
today; the suppression must be entered in the worker thread (`:5037-5041`). `_score_text_completion`
becomes: build `key()`, call `_score_prompts`, build `TextCompletionChoice`s — output byte-identical to
today (the existing suite is the regression guard).

### 7.2 `_score_chat_completion`
- Branch at the top of `create_chat_completion`, before `:3595`:
  `if request.wants_scores(): return await self._score_chat_completion(request)`.
- Conversations: `[request.messages] + (request.extra_messages or [])`.
- Inside `async with self._admit():` check `self._tokenizer.chat_template` → `NoChatTemplateError`;
  render with `self._format_chat_messages(conv, request.chat_template_kwargs)` (`:5580`); score with
  `add_special_tokens=False`, `allowed=request.allowed_token_ids`, `temperature=request.temperature`,
  `top_k=request.top_logprobs or 0`.
- A failure in conversation k re-raises with `details.index = k` (FR-25.9.4).
- Map: `token = key(id)`; `bytes = list(decoded.encode("utf-8"))` from the decoded text always;
  `top_logprobs` list truncated to `request.top_logprobs or 0`; `logprobs=None` unless
  `request.logprobs is True`; `message.content = decoded chosen token`; `finish_reason="length"`.
- No `_probe_begin`, `_sensing_begin`, `_circuit_sensing_begin`, `_apply_request_steering`.
  Profile and dial fields are already refused by the table for scoring (FR-25.7.2).

### 7.3 Constraint wiring
- `GenerationConfig.from_request` copies `seed`, `response_format` (non-`text`) into `constraint`.
- In the async service method, before `_admit()`:
  `grammar = await asyncio.to_thread(self._grammar_cache().compile, fmt)`.
- `_build_generate_kwargs` (`:2366`): when `gen_config.constraint` is set, append
  `JsonConstraintProcessor(grammar, batch, vocab, device)` to `kwargs["logits_processor"]` (create
  the list if absent) and pop `assistant_model` with a `speculative_disabled_for_constraint` log
  (mirror `:3206-3210`). It is called per `generate()`, so each call gets a fresh processor.
- After decoding each choice: if the last generated id is in `stop_token_ids` → complete → validate;
  else `finish_reason = "length"`. Do not consult `matcher.is_terminated()` (never called after the
  final token).
- `CBM` gate (`_use_cbm_for_request`, `:958-982`): add parameters `constrained: bool`,
  `seeded: bool`; return `False` when `seeded`. For `constrained`, the route refuses before the
  service when the CBM is enabled (FR-25.11.3).
- llama.cpp: the table refuses `response_format` from the row; the service also raises
  `ResponseFormatUnsupportedError` in `_llamacpp_chat_completion` if one arrives (defence in depth).

### 7.4 Seed wiring
```python
def _generate_sync(self, kwargs, seed=None):          # :5492
    if seed is None:
        return self._model.generate(**kwargs)
    with torch.random.fork_rng(devices=self._rng_devices()):
        torch.manual_seed(seed)
        return self._model.generate(**kwargs)
```
- `_rng_devices()` returns the CUDA device indices the model occupies (multi-GPU placement), empty on
  CPU.
- Scope: the serial path reports `request`; `_create_batched_chat_completion` reports `batch-shape`;
  any path while `self._use_cbm()` is true reports `best-effort`.
- Text completions with a prompt list: the seed is applied once per prompt (each prompt is its own
  `generate()`), so prompt i's output does not depend on prompt i−1's length. Document it.
- llama.cpp: `_llamacpp_params` gains `params["seed"] = seed` **only after** T-61 passes; until then
  the table refuses.

### 7.5 Streaming refusals
`stream_chat_completion` (`:4288`) reads neither `n` nor `extra_messages`. The schema validator
refuses them; also add a guard at the top of `stream_chat_completion` raising
`FieldNotHonouredError` so a direct caller (Feature 26's batch runner) cannot bypass the schema.

## 8. Testing Implementation Approach

### 8.1 Fixtures
- Scoring: reuse the tiny Llama and WordLevel tokenizer of `test_scoring_completions.py:26-68`;
  give the tokenizer a minimal `chat_template` (Jinja) for chat-scoring tests, and one without a
  template for T-55.
- Structured output: a character-level `WordLevel` tokenizer over JSON punctuation, digits,
  lowercase letters, space and newline, with `</s>` as EOS; `xgr.TokenizerInfo(encoded_vocab=...,
  vocab_type=RAW, stop_token_ids=[0])`. Measured: random weights under the constraint emit valid,
  schema-conforming JSON (e.g. `{"label": "not_humor", "n": 9785586725}`).
- HTTP: the existing FastAPI test-client pattern (`tests/unit/api/test_gguf_refused_before_load.py`)
  with a spy on `load_model_and_wait` to prove "before auto-load" (call count 0).

### 8.2 Registry-derived tests (no hand-kept lists)
- `/v1` POST paths: `app.openapi()["paths"]`, filtered to `/v1/` with a `post` operation. FastAPI is
  0.141.1 here; do not read `app.routes`.
- Endpoint ↔ table: every derived path maps to an `Endpoint`; an unmapped path fails.
- Errors: walk `MiLLMError.__subclasses__()` recursively; every `code` must be in `ERROR_STATUS_MAP`.
- Subset honesty: iterate `ALLOWED_KEYWORDS`; each has a violating example in a fixture table keyed
  by keyword, and a missing example fails the test.

### 8.3 Mutation controls (each must turn the suite red; record the result)

| # | Mutation | Expected red |
|---|---|---|
| M1 | `evaluate()` skips the strict branch (return unused list instead of raising) | strict 400 tests |
| M2 | A list field outcome check removed (refused fields pass) | coverage test over every field × path |
| M3 | Message-level walk removed | `messages[i]` reporting test |
| M4 | Log event includes field values | privacy test (asserts absence of a sentinel value) |
| M5 | `apply_request_policy` call deleted from one route | that route's HTTP test |
| M6 | `with self._unsteered()` removed from `_unsteered_next_token_logits` (`:5037-5041`) | chat and completion "unsteered" tests |
| M7 | Chat scoring branch moved after the `extra_messages` branch | routing test (`extra_messages` scoring returns scores) |
| M8 | `add_special_tokens=False` → `True` in chat scoring | parity test (token IDs differ) |
| M9 | `_score_chat_completion` reimplements instead of calling `_score_prompts` | call-count/payload test |
| M10 | Processor not appended in `_build_generate_kwargs` | structured-output validation test |
| M11 | Truncation check reads `matcher.is_terminated()` | `finish_reason` test |
| M12 | `fork_rng` removed | "unseeded after seeded is not deterministic" test |
| M13 | `manual_seed` removed | seed repeatability test |
| M14 | A subset keyword added to the allowlist without enforcement (e.g. `uniqueItems`) | honesty test |
| M15 | `ERROR_STATUS_MAP` row for `INVALID_SCORING_REQUEST` removed | error-map test |
| M16 | Streaming `n > 1` refusal removed | streaming refusal test |

Back up the file, edit one line, run the affected tests, restore, and confirm `git diff` is clean
before the next control. Verify each mutation **landed** (re-grep the line) before concluding it
survived.

### 8.4 Hardware tests
Scripts under `tests/hardware/` (not in the unit suite), run on mcs-lnxhost02 against the deployed
pod: chat-scoring parity (200 prompts, JEV-9B-decision bf16), 100 `json_schema` requests, the GPU
per-token constraint cost, seed repeatability, T-61's llama.cpp seed check.

## 9. Configuration and Environment Strategy
- `millm/core/config.py`: `STRUCTURED_OUTPUT_GRAMMAR_CACHE: int = 64`,
  `IGNORED_FIELDS_HEADER_MAX_BYTES: int = 1024`; `.env.example` entries with one-line comments.
- No feature flag. Strict mode is per request.
- `pyproject.toml` dependencies as FTDD §11; CI rebuilds the image.

## 10. Integration Strategy
- **Backwards compatibility:** a client sending only declared fields and no strict header gets the
  same body plus `system_fingerprint`. Open WebUI keeps working (report, not refuse).
- **Feature 26 (batch):** constructs request objects per line; it must call `request_policy.evaluate`
  with `strict=True` (FR-26.1.6) and copy the seed and constrained values into each output line
  (FR-26.10.1). The service-level streaming guard (§7.5) protects its direct calls.
- **Feature 27:** `_score_prompts` is where `return_sae_activations` attaches in scoring mode
  (R-04.26); `return_sae_activations` must be added to the schema so it is never reported
  (027 FPRD).
- **Feature 28:** flips the `steering` outcome cells to honoured for transformers chat and completions
  (FR-28.4.5); scoring stays refused, and the scoring response carries `X-miLLM-Steering: none`
  (028 D13).
- **Feature 30:** flips `dimensions` for declaring models and adds `pooling` as a list field.
- **miStudio 034:** sends `X-miLLM-Strict: true` on every new `millm_*` request (TD5, T-100); header
  name matches.
- **miDataworks `jev_client.py`:** can move to chat scoring; not changed here (other repo).

## 11. Utilities and Helpers Design
- `request_policy.encode_field_list` — the only header encoder for field lists; reuse for Feature
  26's per-line extension values.
- `system_fingerprint.build_system_fingerprint` — reused by Feature 26 output lines.
- `constrained_decoding.validate_output` — reused by Feature 26 for constrained batch lines.

## 12. Error Handling and Logging Strategy
- All refusals are `MiLLMError` subclasses with class-level `code`/`status_code`
  (`millm/core/errors.py:42-57` explains why class attributes matter) and an `ERROR_STATUS_MAP` row.
- `param` names the field; `details.fields` lists all unused locations under strict mode.
- Logs (structlog): `request_fields_unused` (warning; endpoint, request id, locations),
  `request_field_refused` (info; field, endpoint, engine, reason), `constrained_generation` (info;
  type, schema name, tokens, finish reason, total mask ms), `speculative_disabled_for_constraint`
  (info). Never field values, never prompt text.
- `CONSTRAINED_OUTPUT_INVALID` logs the schema name and the first 200 characters' **length** only.

## 13. Performance Implementation Hints
- Policy: O(fields + messages); no regex on hot paths beyond header escaping.
- Grammar compile: cached LRU, compiled outside the slot.
- Bitmask: one device buffer per request; in-place apply.
- Record `mask_ms` per request from a `time.perf_counter()` around fill+apply; the GPU figure is
  read on hardware.

## 14. Code Quality and Standards
- Black (100), Ruff, MyPy strict on new modules.
- Every new module opens with a docstring stating the guarantee it enforces and the defect it
  prevents (house style).
- No `assert` for runtime checks (stripped under `-O`).
- Tests named for the behaviour (`test_a_refused_field_is_400_without_strict`).
- Reachability: each wiring line has a removal test asserting payload and call count (§8.3).

## 15. Decisions from Clarifying Questions

Clarifying rounds were waived. Decisions cite their source.

| # | Question | Decision | Source |
|---|---|---|---|
| I1 | Where does the policy run? | In each route, after the row lookup, before auto-load | FR-25.2.3, FR-25.3.8 |
| I2 | How is "present" judged? | `model_fields_set` for declared fields; `model_extra` for extras | FR-25.3.4 (defaults are not presence) |
| I3 | Missing table cell? | Test failure; no default outcome | FR-25.3.1 |
| I4 | `max_completion_tokens` resolution | Validated and folded into `max_tokens` in the schema | T-58; FR-25.3.3a |
| I5 | `bytes` in chat logprobs | UTF-8 of the decoded text, always | FR-25.8.5 |
| I6 | Truncation detection | Last generated id ∈ stop ids; never matcher state | FTDD §3.1 measurement |
| I7 | Seed per prompt in a completion list | Applied per `generate()` | FR-25.13.3 |
| I8 | llama.cpp seed before T-61 | Refused | T-61 (fail closed) |
| I9 | Route-level GGUF scoring check | Replaced by the table only after a test proves identical output | FR-25.6.3 |
| I10 | Test of the endpoint set | `app.openapi()["paths"]` | FastAPI 0.141.1 |
| I11 | Streaming guard location | Schema and service both | FR-25.3.5; Feature 26 calls the service directly |

**Open:** none. Measurements outstanding: T-61, the gemma-4 tokenizer spike, the GPU per-token cost.
