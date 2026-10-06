# Technical Implementation: Embedding Options
## miLLM Feature 30

**Version:** 1.0 · **Created:** 2026-10-06 · **Status:** Planned
**References:** 030_FPRD v1.1 · 030_FTDD v1.0 · 025_FTDD §5 (policy table, errors) · 026_FTDD §6–§7
(executors, nested slots) · T-91–T-95, P-18
**Code verified at** miLLM `7aa659c`. Line numbers move; re-check each before editing.

---

## 1. Implementation Overview

Five small changes, in this order:

1. Three rows in Feature 25's policy table and two schema fields.
2. A pure pooling module.
3. The transformers embedding body split into `_embed_inputs`, with a length check before any forward.
4. The llama.cpp path brought to the same rules.
5. The input-count validator, the setting, and corrected comments and docs.

**Principles.**
- *Honoured or refused, never ignored.* Every new field has a policy row.
- *The default does not move.* A request with no new fields returns today's floats.
- *Measure, then refuse, then run.* No forward pass starts until every input has been measured.
- *One pooling path.* Feature 26 calls the same functions; nothing is copied.
- *Assert the call, not the existence.* Each wiring line has a test that turns red when it is removed.

## 2. File Structure and Organization

| File | Status | Change |
|---|---|---|
| `millm/ml/embedding_pooling.py` | new | `PoolingMode`, `pool_hidden`, `finalize_vector`, `NonFiniteEmbeddingError` |
| `millm/api/schemas/openai.py` | modified | `EmbeddingRequest` (`:287-296`): `pooling`, `normalize`, validator |
| `millm/api/request_policy.py` | modified (Feature 25's) | rows `dimensions`, `pooling`, `normalize` |
| `millm/services/inference_service.py` | modified | `create_embeddings` (`:5071-5148`), new `_embed_inputs`, `_embedding_limit`, `_check_embedding_lengths`; `_llamacpp_embeddings` (`:5150-5224`) |
| `millm/core/errors.py` | modified | `openai_param` on `MiLLMError` (`:11-27`); `EmbeddingInputTooLongError`, `EmbeddingVectorInvalidError` |
| `millm/api/routes/openai/errors.py` | modified | `ERROR_STATUS_MAP` row `EMBEDDING_VECTOR_INVALID` |
| `millm/api/exception_handlers.py` | modified | pass `param` at `:118-123` |
| `millm/api/routes/openai/embeddings.py` | modified | comments `:6`, `:56-57`, `:64-74`; policy call before auto-load |
| `millm/core/config.py` | modified | `EMBEDDINGS_MAX_INPUTS` beside `:280-290` |
| `.env.example` | modified | the setting, commented |
| `manual/docs/api/openai-compatible.md` | modified | Embeddings section `:193-199` |
| `manual/docs/reference/configuration.md` | modified | row beside `:85` |
| `tests/unit/ml/test_embedding_pooling.py` | new | pure tests |
| `tests/unit/services/test_embedding_options.py` | new | tiny real Llama; service behaviour |
| `tests/unit/api/test_embedding_options_route.py` | new | route: refusals before load and before forward |
| `tests/unit/services/test_inference_service.py` | modified | two-SAE test (`:783-830`) parametrised by mode |
| `tests/unit/services/test_gguf_refusals.py` | modified | `TestGGUFEmbeddings` (`:763`) gains length, pooling and normalise cases |

Imports: `embedding_pooling` imports only `torch` and `typing`. The service imports it at module
level. No route imports it.

## 3. Component Implementation Hints

### 3.1 `pool_hidden`

```python
def pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, mode: PoolingMode) -> torch.Tensor:
    if hidden.dim() != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError(f"hidden {tuple(hidden.shape)} and mask {tuple(attention_mask.shape)} disagree")
    mask = attention_mask.to(hidden.device).bool()
    if not mask.any(dim=1).all():
        raise ValueError("a row has no real tokens")
    if mode == "mean":
        if mask.all():
            return hidden.mean(dim=1)          # today's exact expression (inference_service.py:5124)
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(dim=1) / m.sum(dim=1)
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    if mode == "cls":
        idx = mask.int().argmax(dim=1)                                     # first real position
    elif mode == "last":
        idx = hidden.shape[1] - 1 - mask.flip(1).int().argmax(dim=1)       # last real position
    else:
        raise ValueError(f"unknown pooling mode {mode!r}")
    return hidden[rows, idx]
```

- `argmax` on an int tensor returns the first maximum. That is what makes `cls` and `last` correct.
  The test with padding on both sides proves it rather than trusting the documentation.
- Do not add `keepdim`, do not `.squeeze()`. Today's `.squeeze()` (`:5124`) turns a width-1 vector
  into a float, which `:5127-5128` patches back. Return `[B, D]` and let the caller index row 0.

### 3.2 `finalize_vector`

```python
def finalize_vector(vec: torch.Tensor, normalize: bool) -> list[float]:
    if not torch.isfinite(vec).all():
        raise NonFiniteEmbeddingError("pooled vector has non-finite values")
    if not normalize:
        return vec.cpu().tolist()               # unchanged conversion for the default path
    v = vec.to(torch.float32)
    norm = torch.linalg.vector_norm(v)
    if not torch.isfinite(norm) or norm == 0:
        raise NonFiniteEmbeddingError("pooled vector has zero or non-finite norm")
    return (v / norm).cpu().tolist()
```

`finalize_vector` takes a `list[float]` too, for the llama.cpp path: accept
`torch.as_tensor(vec, dtype=torch.float32)` at the top when given a list. The non-normalised GGUF
path then returns the engine's floats unchanged.

### 3.3 `EmbeddingOptions`

A frozen dataclass in `embedding_pooling.py`: `pooling: PoolingMode = "mean"`, `normalize: bool = False`.
`create_embeddings` builds it with `EmbeddingOptions(request.pooling, request.normalize)`. Feature 26's
executor builds it from the batch row's body the same way.

## 4. Database Implementation Approach

N/A. No schema change, no migration (T-91 removes the only candidate column). Do not add a
`supports_truncated_embeddings` column "for later": an unread column is a declared mechanism with no
wiring.

## 5. API Implementation Strategy

### 5.1 Schema

```python
class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, list[str]]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = Field(default=None, gt=0)
    pooling: Literal["mean", "last", "cls"] = "mean"
    normalize: bool = False
    user: Optional[str] = None

    @field_validator("input")
    @classmethod
    def _input_shape(cls, value: Union[str, list[str]]) -> Union[str, list[str]]:
        from millm.core.config import settings
        items = [value] if isinstance(value, str) else value
        if not items:
            raise ValueError("input must contain at least one string")
        empty = [i for i, t in enumerate(items) if t == ""]
        if empty:
            where = "input" if isinstance(value, str) else f"input[{empty[0]}]"
            more = f" (and {len(empty) - 1} more)" if len(empty) > 1 else ""
            raise ValueError(f"{where} must not be empty{more}")
        cap = settings.EMBEDDINGS_MAX_INPUTS
        if len(items) > cap:
            raise ValueError(f"input has {len(items)} items; the limit is {cap} (EMBEDDINGS_MAX_INPUTS)")
        return value
```

- Keep `model_config` as Feature 25 sets it (`extra="allow"` after 025). Do not touch it here.
- A default (`mode="after"`) field validator runs after type coercion, so `value` is already a string
  or a list of strings.
- **A field validator, not a model validator.** The validation handler builds `param` from the error's
  `loc` (`exception_handlers.py:54-55`). A field validator's `loc` is `("body", "input")`, so
  `param` is `input`. A model-level validator has no field in its `loc`, so `param` would be `None`.
  The message carries the index.

### 5.2 Policy rows

```python
OUTPUT_CHANGING["dimensions"][("embeddings", "transformers")] = Refused(DIMENSIONS_REASON)
OUTPUT_CHANGING["dimensions"][("embeddings", "llamacpp")]     = Refused(DIMENSIONS_REASON)
OUTPUT_CHANGING["pooling"]   = {("embeddings", "transformers"): HONOURED,
                                ("embeddings", "llamacpp"): Refused(GGUF_POOLING_REASON)}
OUTPUT_CHANGING["normalize"] = {("embeddings", "transformers"): HONOURED,
                                ("embeddings", "llamacpp"): HONOURED}
NEUTRAL["pooling"]   = lambda v: v == "mean"
NEUTRAL["normalize"] = lambda v: v is False
```

Write these in the literal table, not as assignments after it; the form above shows the cells only.
`DIMENSIONS_REASON = "no model declares truncated-embedding support (T-91)"`.
Feature 25's every-entry test then asserts both rows on every endpoint automatically.

### 5.3 Route

`embeddings.py` gains, between the row lookup (`:60-62`) and auto-load (`:80`), the Feature 25 call
`request_policy.evaluate(request, model, endpoint="embeddings")`. If Feature 25 already placed it
there, verify the position and add nothing.

### 5.4 Errors

```python
class MiLLMError(Exception):
    def __init__(self, message, details=None, openai_param: Optional[str] = None):
        ...
        self.openai_param = openai_param

class EmbeddingInputTooLongError(ContextLengthExceededError):
    """Inputs over the model's limit. Never truncated (FR-30.3)."""

class EmbeddingVectorInvalidError(MiLLMError):
    code = "EMBEDDING_VECTOR_INVALID"
    status_code = 500
```

Handler (`exception_handlers.py:118-123`): add `param=getattr(exc, "openai_param", None),`.
Map (`openai/errors.py`): `"EMBEDDING_VECTOR_INVALID": (500, "server_error")`.

## 6. Frontend Implementation Approach

N/A. No Admin UI change (PPRD Feature 30, "UI Tab: none").

## 7. Business Logic Implementation Hints

### 7.1 `_embed_inputs`

```python
def _embed_inputs(self, texts: list[str], options: EmbeddingOptions) -> tuple[list[list[float]], list[int]]:
    device = self._get_input_device()
    encoded = [self._tokenizer(t, return_tensors="pt", truncation=False) for t in texts]
    counts = [int(e.input_ids.shape[1]) for e in encoded]
    self._check_embedding_lengths(counts, self._embedding_limit(), single=len(texts) == 1 and self._input_was_str)
    vectors = []
    for i, enc in enumerate(encoded):
        enc = enc.to(device)
        with torch.no_grad(), self._unsteered():
            out = self._model(**enc, output_hidden_states=True)
        pooled = pool_hidden(out.hidden_states[-1], enc.attention_mask, options.pooling)[0]
        try:
            vectors.append(finalize_vector(pooled, options.normalize))
        except NonFiniteEmbeddingError as exc:
            raise EmbeddingVectorInvalidError(f"Input {i}: {exc}", openai_param=f"input[{i}]") from exc
    return vectors, counts
```

- **`truncation=False` explicitly.** Dropping the argument is not enough if a future tokenizer
  default changes. **No `padding=True`** for single strings: today it adds no pads (one sequence),
  and leaving it out removes a reason to wonder.
- **`self._input_was_str` is illustrative only.** Pass whether the request's `input` was a string as
  an argument (`param_for_string: bool`), so the error says `input` rather than `input[0]`. Do not
  store request state on the singleton service.
- The forward stays on the calling thread with `_unsteered()` entered on that same thread
  (`inference_service.py:1184-1187`). Do not wrap it in `asyncio.to_thread` in this feature.
- `_embed_inputs` is synchronous and takes no slot. `create_embeddings` calls it inside
  `async with self._admit():`.

### 7.2 `_check_embedding_lengths`

```python
MAX_LISTED = 16

def _check_embedding_lengths(self, counts, limit, *, param_for_string):
    if limit is None:
        return
    over = [(i, n) for i, n in enumerate(counts) if n > limit]
    if not over:
        return
    listed, omitted = over[:MAX_LISTED], max(0, len(over) - MAX_LISTED)
    parts = ", ".join(f"input {i} has {n:,} tokens" for i, n in listed)
    more = f" ({omitted} more over the limit not listed)" if omitted else ""
    raise EmbeddingInputTooLongError(
        f"{parts[0].upper()}{parts[1:]}{more}; this model's limit is {limit:,} tokens. "
        "Inputs are never truncated. Shorten or split them.",
        details={"max_context_tokens": limit,
                 "over_limit": [{"index": i, "tokens": n} for i, n in listed],
                 "omitted": omitted},
        openai_param="input" if param_for_string else f"input[{over[0][0]}]",
    )
```

`limit` comes from `_embedding_limit()`:
- transformers: `_served_max_context(getattr(self._model, "config", None))` (`:222-247`);
- llama.cpp: `min(self._model.n_ctx(), self._model.n_batch)` once task 0.1 confirms both names and
  that `n_batch` bounds a single embedding input. If `n_batch` does not bound it, use `n_ctx()` alone.

### 7.3 llama.cpp path

Inside `_llamacpp_embeddings`, before `async with self._admit()` work begins on any input:

1. `if options.pooling != "mean": raise EngineUnsupportedError(...)` — defence in depth; the policy
   row normally refuses this before load.
2. Tokenise each text with `self._model.tokenize(text.encode("utf-8"))` (the call `:4212` already
   uses), collect counts, call `_check_embedding_lengths`.
3. Then the existing loop (`:5194-5214`), with `finalize_vector(vector, options.normalize)` applied
   before base64.

Tokenising runs inside the slot: an unload frees the model the tokenizer belongs to.

### 7.4 `create_embeddings`

Keep the public signature. Build `options` and `texts`, then:
- llama.cpp: `return await self._llamacpp_embeddings(request)` (it reads options itself);
- transformers: `async with self._admit(): vectors, counts = self._embed_inputs(texts, options, ...)`,
  then base64 if asked, then the response with `prompt_tokens = sum(counts)` (FR-30.3.7).

Update the docstring at `:5073-5075`: it says "mean pooling" only (FR-30.4.2).

## 8. Testing Implementation Approach

### 8.1 Pure tests (`tests/unit/ml/test_embedding_pooling.py`)

- Each mode on a `[2, 5, 4]` tensor with known values; expected vectors computed by hand.
- Right padding and left padding: pad positions hold `1e3`; rows of real length 2 and 5. `mean`,
  `last` and `cls` must equal the unpadded per-row answers.
- Unpadded `mean` equals `hidden.mean(dim=1)` with `torch.equal` (not `allclose`), in bfloat16.
- `finalize_vector(normalize=True)`: norm within 1e-5; zero vector and `NaN` raise.
- Order test for the future honour path is **not written** (FR-30.1.3 inactive under T-91).

### 8.2 Service tests (`tests/unit/services/test_embedding_options.py`)

Copy the tiny real Llama and WordLevel tokenizer from `test_scoring_completions.py:30-57`. Set
`max_position_embeddings` to 8 and the tokenizer's `model_max_length` to 8.

- A 12-token input returns `context_length_exceeded`, `param` `input`, and the model's forward is
  never called.
- Inputs `[ok, long, ok, long]`: one error naming indices 1 and 3; zero forwards.
- 20 over-limit inputs: 16 listed, `omitted == 4`.
- Each mode returns the right vector, compared with a manual pool over `output_hidden_states`.
- `pool_hidden` spy: called once per input, with the request's mode. `finalize_vector` spy: called
  with the request's `normalize`.
- `usage.prompt_tokens` equals the sum of full token counts.
- `_served_max_context` returning `None`: a long input runs, untruncated (the token count in
  `usage` equals its full length).

### 8.3 Route tests (`tests/unit/api/test_embedding_options_route.py`)

Pattern: `tests/unit/api/test_context_length_refusal.py:204-210`.
- `dimensions: 64` and `dimensions: <native width>` → 400 `param: dimensions`; `load_model_and_wait`
  not awaited.
- GGUF row with `pooling: "last"` → 400 before load; with `pooling: "mean"` → reaches the engine.
- `input: ""`, `input: []`, `input: ["a", ""]` → 400 `param: input`, message naming index 1 in the
  last case; no load.
- 257 inputs with the cap at 256 → 400 naming 257 and 256; no load. Patch
  `settings.EMBEDDINGS_MAX_INPUTS` to a small value instead of sending 257 strings where convenient.
- `pooling: "max"` → 400 `param: pooling`.

### 8.4 Unsteered

Parametrise `test_embeddings_suppress_attached_sae` (`test_inference_service.py:783-830`) over the
three modes. Its fixture returns a `[1, 5, 64]` hidden state; give the new call path the attention
mask it now needs (`torch.ones(1, 5)`).

### 8.5 Mutation controls

Run each, require red, restore, and verify the restore by `git diff --stat` and re-grepping the line.
Record each in the review notes. Never leave a mutation in the tree.

| # | Mutation | Expected red |
|---|---|---|
| M1 | `truncation=False` → `truncation=True` in `_embed_inputs` | 8.2 12-token refusal (the real tokenizer truncates to 8 and the check passes) |
| M2 | Move `_check_embedding_lengths` inside the per-input loop | 8.2 `[ok, long, ok, long]` (one forward runs before the refusal) |
| M3 | Delete the `_check_embedding_lengths` call in `_embed_inputs` | 8.2 12-token refusal |
| M4 | Delete the same call in `_llamacpp_embeddings` | GGUF over-limit test |
| M5 | `dimensions` rows → `HONOURED` | 8.3 `dimensions` refusal, and Feature 25's every-entry test |
| M6 | Move the policy call after auto-load | 8.3 "`load_model_and_wait` not awaited" |
| M7 | `NEUTRAL["pooling"]` → `lambda v: True` | 8.3 GGUF `pooling: "last"` refusal |
| M8 | `pool_hidden` `mean`: drop the mask branch (always `hidden.mean(1)`) | 8.1 padding tests |
| M9 | Drop `param=` from the handler | 8.2 and 8.3 `param` assertions |
| M10 | Pass a fixed `"mean"` instead of `options.pooling` | 8.2 spy payload |
| M11 | Delete `self._unsteered()` from the forward | 8.4 |
| M12 | Remove the field validator's cap check | 8.3 cap test |

M1, M2, M3 and M4 are the truncation-refusal controls; M5, M6 and M7 are the `dimensions` and policy
controls the FTASKS names.

## 9. Configuration and Environment Strategy

```python
# Maximum strings in one /v1/embeddings request. A capped request holds the only admission slot
# (MAX_CONCURRENT_REQUESTS = 1), so this bounds how long a chat waits behind it. Set from measured
# latency (T-93): largest power of two with cap x p95 seconds-per-input <= 30 s, in [64, 2048].
# 256 is provisional until that measurement is recorded.
EMBEDDINGS_MAX_INPUTS: int = Field(default=256, ge=1, le=2048)
```

Add the commented line to `.env.example` and the row to `manual/docs/reference/configuration.md`.
No Kubernetes change. No feature flag.

## 10. Integration Strategy

- **Feature 25** owns `request_policy.py`. Add rows; never fork the table. If 025's tests enumerate
  fields from the table, the new rows are covered with no new test code.
- **Feature 26** calls `_embed_inputs` from its embedding executor. Leave a `pack_size` parameter
  out; 026 phase 6 adds it with packed, padded batches through the same `pool_hidden`.
- **Existing callers** of `create_embeddings`: the route and the tests. The signature is unchanged.
- **Backward compatibility.** Default requests return identical floats. Behaviour changes only for
  inputs that were silently truncated (now 400), empty input (now 400), `dimensions` (now 400 instead
  of ignored), and lists over the cap.

## 11. Utilities and Helpers Design

`embedding_pooling.py` is the only new helper module. `_embedding_limit` and
`_check_embedding_lengths` are private methods on the service, because they read the loaded model.
Do not add them to `_check_context_length` (`:3029-3063`): that function checks one prompt plus
`max_new_tokens` and names no index.

## 12. Error Handling and Logging Strategy

| Case | Layer | Response |
|---|---|---|
| empty input, too many inputs, bad `pooling` | pydantic | 400 `invalid_parameter`, `param: input` / `pooling` |
| `dimensions` (any model), `pooling` ≠ mean on GGUF | policy | 400 `field_not_honoured`, `param` the field |
| input over the limit | service | 400 `context_length_exceeded`, `param: input[i]` |
| zero or non-finite vector | service | 500 `embedding_vector_invalid`, `param: input[i]` |
| GGUF loaded without embeddings | service | existing `EngineUnsupportedError` (`:5198-5205`) |

Log one `embedding_refused` warning with `reason` and the listed indices. Never log input text.

## 13. Performance Implementation Hints

- Tokenising everything first holds `N` small tensors on the host; with `N ≤ cap` this is negligible.
- Move each encoding to the device just before its forward, not all at once.
- Do not batch inputs in this feature. Batching is packing, and packing is Feature 26's (with its
  measured packed-versus-single difference).
- **Cap measurement** (task 6.3): time 64 inputs of 512 tokens, one request at a time, on the 3090;
  take p95 per input; apply the §9 rule; record the numbers in the review notes and the setting's
  comment.

## 14. Code Quality and Standards

- Black (100), Ruff, MyPy strict (PADR Appendix A). `PoolingMode` is a `Literal`, so MyPy checks the
  dispatch.
- Comments explain why, not what. The corrected route comment (FR-30.4.1) says what the route does
  before auto-load and why; it does not narrate history.
- Remove the trailing-whitespace line at `embeddings.py:74`.
- Re-read the module docstring (`embeddings.py:1-7`) and the route docstring (`:53-58`) after the
  change; both must describe the code as it is.

## 15. Decisions from Clarifying Questions

Clarifying rounds were waived. Each decision cites its source.

| # | Question | Decision | Source |
|---|---|---|---|
| ID1 | Field or model validator for the cap? | Field validator on `input`, so `param` is `input` | `exception_handlers.py:54-55` derives `param` from `loc` |
| ID2 | `truncation` argument? | Explicit `False` | FR-30.3.1; M1 |
| ID3 | Keep `padding=True`? | No; one sequence per call needs none | `inference_service.py:5108-5110` |
| ID4 | Where does the string-vs-list flag live? | An argument, never service state | Service is a process singleton |
| ID5 | How many indices are listed? | 16, plus an omitted count | FR-30.3.4 "bounded" |
| ID6 | Non-finite vector status? | 500 | The request was valid |
| ID7 | GGUF limit? | `min(n_ctx(), n_batch)` after task 0.1 | FTDD TD15 |
| ID8 | Batch inputs in one forward? | No; Feature 26 owns packing | 026 FTDD §6 |
| ID9 | Column for a future declaration? | None | T-91; reachability rule |
| ID10 | Test model? | Tiny real Llama, small `model_max_length` | `test_scoring_completions.py:30-57`; mock tokenizer cannot truncate |
| ID11 | Cap setting bounds? | `ge=1, le=2048` | OpenAI's per-request ceiling; T-93 |

**Open items:** task 0.1 (llama-cpp-python names and truncation bound) and task 6.3 (cap
measurement). Neither changes a requirement.
