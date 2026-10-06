# Technical Design: Embedding Options
## miLLM Feature 30

**Version:** 1.0 · **Created:** 2026-10-06 · **Status:** Planned
**References:** 030_FPRD v1.1 · BRD-04 §5.9 and R-04.5 · PPRD v1.5 Feature 30 (with its split note) ·
PADR v1.5 §10 "Dataworks Support" trade-offs ("Packed scoring by default vs one row at a time";
"Degrade optional capabilities rather than refuse to serve") · Feature-PRD decisions 2026-10-06
(`~/app/miDataworks/0xcc/docs/brd-decisions-2026-10-05.md`, P-18) · technical register T-91–T-95
(`~/app/miDataworks/0xcc/docs/fprd-open-questions-2026-10-06.md`, all accepted) · sibling designs
`025_FTDD` (request policy table) and `026_FTDD` (batch runner, packed embedding rows)
**Code verified at** miLLM `7aa659c` (HEAD, 2026-10-06). Re-check line numbers before editing.

---

## 1. Executive Summary

Feature 30 gives `/v1/embeddings` a choice of pooling and normalisation, refuses `dimensions` on every
model, and stops silent truncation. It adds no route, no table and no migration. One setting is new.

| Area | Decision | Rationale |
|---|---|---|
| Pooling | A pure function `pool_hidden(hidden, attention_mask, mode)` in a new `millm/ml/embedding_pooling.py` | Unit-testable with hand-built tensors; Feature 26 passes padded batches to the same function |
| Default output | The unpadded `mean` case keeps today's exact expression, `hidden.mean(dim=1)` | A masked sum-then-divide rounds differently in bfloat16; stored retrieval indexes must keep matching (FR-30.2.2) |
| Normalisation | `finalize_vector(vec, normalize)`: float32 L2 norm; zero or non-finite refused by index | Meets the 1e-5 tolerance in bfloat16 models; never emits `NaN` (FR-30.2.6) |
| `dimensions` | A *refused* row on both engines in Feature 25's `OUTPUT_CHANGING` table (`millm/api/request_policy.py`) | T-91: no model declares support in v1. One table, no second copy (FR-25.3.1) |
| `pooling` on GGUF | *Refused* row for `(embeddings, llamacpp)`, with `mean` as the neutral value | llama.cpp pools at construction (`millm/ml/model_loader.py:2292-2293`); decided from the row before auto-load |
| Input limit | Tokenise every input first, without truncation; refuse all over-limit indices in one error, before any forward | FR-30.3.1–30.3.4; today input 0 runs before input 1 is refused (`inference_service.py:5113`) |
| Error `param` | `MiLLMError` gains an optional `openai_param`; the live handler reads it like `openai_error_type` | The handler sends no `param` today (`millm/api/exception_handlers.py:118-123`) |
| Input count and empty input | A pydantic validator on `EmbeddingRequest` against `EMBEDDINGS_MAX_INPUTS` | Runs before row lookup and auto-load; the validation handler already returns a 400 with `param` (`exception_handlers.py:28-75`) |
| Cap value | `EMBEDDINGS_MAX_INPUTS = 256`, provisional until the §9 measurement | T-93: set from measured latency |
| Unsteered | Every forward stays inside `_unsteered()` in the forward's thread | The 2026-10-04 fix (`inference_service.py:1178-1182`, call site `:5117`) is preserved and re-tested per mode |

**Consumers.** miDataworks 004 uses these options after its milestone M1 (P-18). Nothing here gates M1.

## 2. System Architecture

### 2.1 Request flow after this feature

```
POST /v1/embeddings
  │ 1. pydantic: EmbeddingRequest (pooling, normalize, dimensions, input count, empty input)
  │      → 400 invalid_parameter, param from loc          [exception_handlers.py:28-75]
  │ 2. row lookup                                          [embeddings.py:60-62]
  │ 3. request_policy.evaluate(request, row, "embeddings") → 400 FIELD_NOT_HONOURED
  │      dimensions: refused (both engines)  pooling≠mean on GGUF: refused
  │ 4. auto-load                                           [embeddings.py:80-103]
  │ 5. InferenceService.create_embeddings(request)
  │      async with _admit():
  │        transformers: _embed_inputs(texts, options)
  │          a. tokenise ALL (no truncation) → token counts
  │          b. _check_embedding_lengths(counts, limit) → EmbeddingInputTooLongError (all indices)
  │          c. per input: with no_grad, _unsteered(): forward → hidden_states[-1]
  │                         pool_hidden(hidden, mask, mode) → finalize_vector(vec, normalize)
  │        llama.cpp: _llamacpp_embeddings
  │          a. tokenize ALL with the instance tokenizer → counts; same length check
  │          b. pooling guard (mean only); create_embedding; finalize_vector
  │ 6. EmbeddingResponse (shape unchanged)
```

### 2.2 Components and relationships

- **`millm/ml/embedding_pooling.py` (new).** `PoolingMode`, `pool_hidden`, `finalize_vector`,
  `NonFiniteEmbeddingError`. Pure tensor code; imports only `torch`.
- **`millm/api/schemas/openai.py` (modified).** `EmbeddingRequest` (`:287-296`) gains `pooling`,
  `normalize` and a `model_validator` for count and emptiness.
- **`millm/api/request_policy.py` (Feature 25).** Three rows added: `dimensions`, `pooling`,
  `normalize`.
- **`millm/services/inference_service.py` (modified).** `create_embeddings` (`:5071-5148`) is split:
  a new `_embed_inputs` holds the transformers body. `_llamacpp_embeddings` (`:5150-5224`) gains the
  length check, the pooling guard and `finalize_vector`.
- **`millm/core/errors.py` (modified).** `MiLLMError` gains `openai_param`; new
  `EmbeddingInputTooLongError(ContextLengthExceededError)` and `EmbeddingVectorInvalidError`.
- **`millm/api/exception_handlers.py` (modified).** Pass `param=getattr(exc, "openai_param", None)`
  at `:118-123`.
- **`millm/api/routes/openai/embeddings.py` (modified).** Comments and docstrings corrected
  (FR-30.4); the Feature 25 policy call sits before auto-load.

### 2.3 Integration points

- **Feature 25.** The policy table must exist first. Feature 30 adds rows only.
- **Feature 26.** Its embedding executor calls `_embed_inputs` (026 FTDD §6, "Executors share the
  synchronous code"). Its phase 6 adds `pack_size` and padded batches; `pool_hidden` already takes a
  mask, so packing adds no second pooling path.
- **Feature 23.** GGUF embeddings and `GGUF_ENABLE_EMBEDDINGS` (`config.py:280-290`) are unchanged.
- **Probe runtime.** Embeddings open no probe context, so the hook records nothing
  (`millm/services/probe_runtime.py:839-842`). This feature adds no `_probe_begin`.

## 3. Technical Stack

- Python 3.11, FastAPI, pydantic v2, PyTorch, transformers (`>=5.15.1,<6`, PADR §10), llama-cpp-python
  (`>=0.3.29`, `pyproject.toml:82`).
- **No new dependency.** Pooling and normalisation are a few tensor operations.
- **llama-cpp-python behaviour to verify on the image** (the library is not installed in the
  development venv, checked 2026-10-06). From memory of the 0.3 API, and therefore unverified:
  `Llama.create_embedding` calls `embed(..., truncate=True)`, `Llama.n_ctx()` returns the context
  size, `Llama.n_batch` is an attribute, and `Llama.tokenize(bytes, add_bos=True, special=False)` is
  what `embed` uses. Task 0.1 checks each on the backend image. The design does not depend on the
  truncation default: miLLM refuses an over-limit input before calling the engine.

## 4. Data Design

- **Database:** no change. T-91 removes the truncated-embedding declaration from v1, so no column
  and no migration.
- **Request model** (`EmbeddingRequest`):

  | Field | Type | Default | Validation |
  |---|---|---|---|
  | `pooling` | `Literal["mean", "last", "cls"]` | `"mean"` | enum; other values → 400 `param: pooling` |
  | `normalize` | `bool` | `False` | pydantic bool |
  | `dimensions` | `Optional[int]`, `gt=0` | `None` | unchanged (`openai.py:293`); refused by policy |
  | `input` | `str \| list[str]` | required | non-empty string(s); list length 1..`EMBEDDINGS_MAX_INPUTS` |

- **Validator messages** (T-94, FR-30.3.6, FR-30.3.8):
  - `input` empty string: `"input must not be empty"`.
  - `[]`: `"input must contain at least one string"`.
  - empty element: `"input[3] must not be empty"`; when several, the first index and a count.
  - too many: `"input has 300 items; the limit is 256 (EMBEDDINGS_MAX_INPUTS)"`.
  The validator reads `settings.EMBEDDINGS_MAX_INPUTS` at validation time, not at import, so a test
  can change it.
- **Response:** `EmbeddingResponse` unchanged (`openai.py:449-455`).

## 5. API Design

### 5.1 Policy rows (FR-30.1, FR-30.2.7, FR-30.2.11)

Added to `OUTPUT_CHANGING` in `millm/api/request_policy.py` (025 FTDD §5.1):

| Field | (embeddings, transformers) | (embeddings, llamacpp) | NEUTRAL | Other endpoints |
|---|---|---|---|---|
| `dimensions` | `Refused("no model declares truncated-embedding support (T-91)")` | same | none | refused (025 table) |
| `pooling` | `HONOURED` | `Refused("llama.cpp pools at load time; only 'mean' is available")` | `== "mean"` | not declared there |
| `normalize` | `HONOURED` | `HONOURED` | `is False` | not declared there |

A neutral value is honoured as "no change" (FR-25.3.4), so `pooling: "mean"` passes on GGUF and
`pooling: "last"` is refused. `dimensions` has no neutral value (FR-30.1.6). The refusal message names
the model (FR-30.1.2): the policy module formats `"Model '{name}': {reason}"`.

**If Feature 25 has not landed** when this feature is built, task 1.1 stops. Feature 30 does not
create a second table to work around it.

### 5.2 Errors

| Class | code | status / type | `openai_param` | When |
|---|---|---|---|---|
| `EmbeddingInputTooLongError(ContextLengthExceededError)` | `CONTEXT_LENGTH_EXCEEDED` (inherited; map row exists, `errors.py:91`) | 400 / invalid_request_error | `input[i]` (first index), or `input` for a string | any input over the limit |
| `EmbeddingVectorInvalidError(MiLLMError)` | `EMBEDDING_VECTOR_INVALID` | 500 / server_error | `input[i]` | pooled vector has zero or non-finite norm under `normalize`, or non-finite values at all |

- `EmbeddingInputTooLongError.details`: `{"max_context_tokens", "over_limit": [{"index", "tokens"}],
  "omitted": n}`. At most 16 entries are listed; `omitted` counts the rest.
- Message: `"Input 3 has 9,120 tokens and input 7 has 8,500; this model's limit is 8,192 tokens.
  Inputs are never truncated. Shorten or split them."`
- `EmbeddingVectorInvalidError` is a 500 because the request was valid; the model produced an
  unusable vector. A new `ERROR_STATUS_MAP` row is added; Feature 25's walk over `MiLLMError`
  subclasses (025 FTDD §5.5) then covers it.
- **`openai_param` plumbing.** `MiLLMError.__init__` (`errors.py:17-24`) takes an optional
  `openai_param: Optional[str] = None`, stored on the instance. The handler passes it at
  `exception_handlers.py:118-123`. Existing errors are unaffected (default `None`).

### 5.3 Evaluation order

1. pydantic (count, emptiness, enum) → 400 before row lookup.
2. Row lookup (`embeddings.py:60-62`).
3. Policy evaluation → 400 before auto-load (`dimensions`, GGUF `pooling`).
4. Auto-load.
5. Service: length check inside `_admit()` (needs the tokenizer, which an unload deletes; 025 FTDD
   U6) and before any forward.

### 5.4 Response

Shape unchanged. No provenance header (T-95).

## 6. Component Architecture

### 6.1 `millm/ml/embedding_pooling.py`

```python
PoolingMode = Literal["mean", "last", "cls"]

def pool_hidden(hidden: Tensor, attention_mask: Tensor, mode: PoolingMode) -> Tensor:
    """hidden [B, T, D], attention_mask [B, T] of 0/1 → [B, D] in hidden's dtype."""

def finalize_vector(vec: Tensor, normalize: bool) -> list[float]:
    """[D] → list. normalize: float32 L2 to unit norm. Raises NonFiniteEmbeddingError."""
```

Rules:
- **mean.** If `attention_mask.all()`, return `hidden.mean(dim=1)` exactly (FR-30.2.2). Else
  `(hidden * m).sum(1) / m.sum(1)`, with `m = mask[..., None].to(hidden.dtype)`.
- **last.** Index `T - 1 - mask.flip(1).float().argmax(1)` per row: the last real position under
  left or right padding.
- **cls.** Index `mask.float().argmax(1)`: the first real position.
- A row whose mask is all zero raises `ValueError` (a programming error; inputs are non-empty by
  validation).
- **finalize.** Non-finite values always raise. With `normalize`, cast to float32, compute
  `torch.linalg.vector_norm`, refuse a zero norm, divide. Without it, keep today's conversion
  (`.cpu().tolist()` from the model dtype) so the default path is unchanged.

### 6.2 Service split

- `create_embeddings(request)` keeps its signature and builds `EmbeddingOptions(pooling, normalize)`
  from the request. It dispatches to `_llamacpp_embeddings` on llama.cpp (`:5084-5085` today) or to
  `_embed_inputs` inside `_admit()`.
- `_embed_inputs(texts, options) -> tuple[list[list[float]], list[int]]` holds the transformers body:
  tokenise all, check lengths, then one forward per input under `torch.no_grad()` and `_unsteered()`
  in the same thread, `pool_hidden`, `finalize_vector`. It does not take a slot itself: it is called
  inside one. That is what lets Feature 26's executor call it under its own chunk slot without the
  nested-slot deadlock 026 FTDD §7 describes.
- `_embedding_limit() -> Optional[int]`: `_served_max_context(config)` (`inference_service.py:222`)
  on transformers; on llama.cpp, the smaller of `n_ctx()` and `n_batch` (task 0.1 confirms both, and
  whether `n_batch` really bounds a single embedding input).
- `_check_embedding_lengths(counts, limit)` raises `EmbeddingInputTooLongError` naming every
  over-limit index (bounded). With `limit is None`, nothing is checked and nothing is truncated; the
  FPRD's "limit source" consideration is answered in §12 risk R3.
- base64 encoding stays where it is (`:5131-5133`, `:5211-5213`), applied to the finalised list.

### 6.3 Separation of concerns

Schema validates shape. The policy decides honour or refuse from request and row. The service
measures and runs. The pooling module does arithmetic. Nothing below the route reads HTTP headers.

## 7. State Management

No new state. Options travel with the request object. No caches. The admission slot is held for the
whole request, as today (`inference_service.py:5105`, `:5194`). The cap bounds how long.

## 8. Security Considerations

- No authentication on `/v1` (BRD-04 §7).
- **Privacy.** Error messages and logs name indices and token counts, never input text. The existing
  `embedding_request` log line (`embeddings.py:105-110`) logs only model and count; it stays that way.
- **Resource bounds.** The input-count cap and the per-input length check bound the slot's hold time
  and the activation memory of a single forward. Today a list of any length is accepted.

## 9. Performance & Scalability

- **No extra forward pass.** One forward per input, as today. Tokenising all inputs first costs one
  extra list of tensors in host memory; inputs are capped.
- **Cap measurement (T-93).** On the RTX 3090, for LFM2.5-1.2B-Instruct (bfloat16) and the largest
  transformers model the node serves for embeddings, measure p95 seconds per input at 512 tokens over
  64 inputs. Set `EMBEDDINGS_MAX_INPUTS` to the largest power of two with
  `cap × p95 ≤ 30 s`, clamped to [64, 2048]. 2,048 is OpenAI's per-request ceiling. 30 s is this
  design's target, because a capped request holds the only admission slot (`MAX_CONCURRENT_REQUESTS`
  is 1, `config.py:255`) and an interactive chat waits behind it. The provisional value is 256 until
  the measurement is recorded.
- **Not changed:** the transformers forward runs on the event-loop thread inside `_admit()`
  (`inference_service.py:5117-5120`). Recorded debt; moving it to a worker thread must move
  `_unsteered()` with it, because suppression is per thread (`:1184-1187`).

## 10. Testing Strategy

- **Pure unit tests** for `pool_hidden` and `finalize_vector` with hand-built tensors. Padding
  positions hold large values (1e3) so a mask-blind pool cannot pass by coincidence. Rows differ in
  length by more than one. Both padding sides.
- **Real tokenizer and model.** Reuse the tiny real Llama and WordLevel tokenizer pattern from
  `tests/unit/services/test_scoring_completions.py:30-57`, with `model_max_length` set small. The
  existing mock tokenizer (`test_inference_service.py:150`) cannot truncate, so it agrees with the
  defect by construction.
- **Default regression.** For one unpadded input, the response equals `hidden.mean(dim=1)` computed
  directly from the same forward, element for element, in bfloat16 and float32.
- **Route tests** through the FastAPI client, as `tests/unit/api/test_context_length_refusal.py:204`
  does: refusal before auto-load (`load_model_and_wait` never awaited), and before any forward
  (`model.call_count == 0`).
- **Unsteered.** Extend the two-SAE test (`test_inference_service.py:783-830`) to parametrise over
  the three modes.
- **Wiring.** Spy on `pool_hidden`: called once per input, with the request's mode (payload and
  count). Spy on `finalize_vector`: called with the request's `normalize`.
- **Mutation controls** (FTID §8 lists each with its expected red): truncation re-enabled; length
  check moved into the loop; policy `dimensions` row flipped to honoured; policy call moved after
  auto-load; mask dropped from `mean`; `openai_param` dropped from the handler.
- **Hardware** (task 6.x): unit norms, an over-limit refusal and a GGUF `pooling: last` refusal on
  the node.

## 11. Deployment & DevOps

- One setting, `EMBEDDINGS_MAX_INPUTS` (int, default 256 until §9 sets it), in `millm/core/config.py`
  beside `GGUF_ENABLE_EMBEDDINGS` (`:280-290`), in `.env.example`, and in
  `manual/docs/reference/configuration.md` (beside `:85`). No Kubernetes manifest change: the default
  applies.
- No migration. Ships through the normal pipeline; no feature flag (the default request is
  unchanged).
- **Logging:** one `embedding_refused` warning per refusal with `reason` (`too_long`,
  `vector_invalid`) and indices. No text.
- **Rollback:** reverting the commit restores today's behaviour. No data to migrate back.
- **Client effect:** clients that relied on silent truncation now get 400. miDataworks expects this
  (004 FR-004.14, `over_input_cap`). Open WebUI's retrieval chunks documents before embedding, so it
  is not expected to send over-limit inputs; the hardware check confirms with one real upload.

## 12. Risk Assessment

| ID | Risk | Mitigation |
|---|---|---|
| R1 | The default vectors change by a rounding difference and break stored retrieval indexes | The unpadded `mean` keeps the exact existing expression; a regression test compares element for element |
| R2 | llama-cpp-python truncates or rejects at a bound other than `n_ctx` | miLLM checks before the call; task 0.1 measures the real bound on the image |
| R3 | `_served_max_context` returns `None`, so no limit is known | The input runs at full length, untruncated, as the FPRD requires. A test pins "no truncation" in that case. Every model served on the node today has `max_position_embeddings`; task 0.2 lists them |
| R4 | Feature 25's table is not built yet | Task 1.1 is a precondition check; no second table |
| R5 | A client sends `pooling` to an older server and gets mean silently | That is Feature 25's unknown-field report; strict mode refuses it |
| R6 | Feature 26 packs rows and a pad position enters the pool | `pool_hidden` is mask-based and tested with large pad values on both sides |

**Complexity:** low to moderate. **Alternatives considered:** pooling inside `create_embeddings` with
inline branches (rejected: untestable without a model, and Feature 26 would copy it); building the
`dimensions` honour path now behind a never-true declaration (rejected: code nothing can reach, which
this estate's reachability rule treats as a finding).

## 13. Development Phases

| Phase | Content | Depends on |
|---|---|---|
| 0 | Verify llama-cpp-python behaviour on the image; list served models' limits | — |
| 1 | Policy rows (`dimensions`, `pooling`, `normalize`) and schema fields | Feature 25 policy module |
| 2 | `embedding_pooling.py` with unit tests | — |
| 3 | Service split, length check, errors, `openai_param` plumbing | 1, 2 |
| 4 | llama.cpp path: length check, pooling guard, finalize | 0, 3 |
| 5 | Cap validator, empty input, setting; comments and docs | 1 |
| 6 | Hardware acceptance; cap measurement | 3, 4, 5 |

Estimate: two sessions including the hardware check.

## 14. Decisions from Clarifying Questions

Clarifying rounds were waived. Each decision cites its source.

| # | Question | Decision | Source |
|---|---|---|---|
| TD1 | Where does pooling live? | Pure module `millm/ml/embedding_pooling.py` | FPRD §13 recommended approach; Feature 26 reuse (026 FTDD §6) |
| TD2 | Mask-aware pooling? | Yes; mask passed explicitly | FR-30.2.3; FR-26.5.1 packs embedding rows |
| TD3 | How is the default kept identical? | Unpadded `mean` uses `hidden.mean(dim=1)` | FR-30.2.2; bfloat16 rounding |
| TD4 | Where is `dimensions` refused? | Feature 25's policy table, before auto-load | T-91; FR-25.3.1, FR-25.3.8 |
| TD5 | Build the `dimensions` honour path now? | No | T-91; reachability rule |
| TD6 | GGUF pooling? | `mean` only, via the table's neutral value; service guard as defence | `model_loader.py:2292-2293`; FR-30.2.7 |
| TD7 | `cls` on causal decoders? | Served, documented | T-92 |
| TD8 | Where is the input limit checked? | Service, inside the slot, before any forward, all indices | FR-30.3.3; tokenizer lifetime (025 FTDD U6) |
| TD9 | Error code for over-limit? | Inherited `context_length_exceeded` with `param` | FR-30.3.4; `errors.py:91` |
| TD10 | How does `param` reach the client? | `openai_param` on `MiLLMError`, read by the live handler | `exception_handlers.py:117-123` already reads `openai_error_type` the same way |
| TD11 | Count cap and empty input: where? | pydantic validator | Runs before row lookup and load; handler returns `param` (`exception_handlers.py:28-75`) |
| TD12 | Cap value? | Measured per §9; provisional 256 | T-93 |
| TD13 | Empty input? | 400 | T-94 |
| TD14 | Provenance header? | None | T-95 |
| TD15 | GGUF limit source? | `min(n_ctx(), n_batch)`, verified on the image | FR-30.3.5; task 0.1 |
| TD16 | Normalisation precision? | float32 | FPRD §6 numerics |
| TD17 | Does this block miDataworks M1? | No | P-18 |

**Open items:** none at the design level. Two measurements are tasks, not decisions: the
llama-cpp-python checks (task 0.1) and the cap value (task 6.3).
