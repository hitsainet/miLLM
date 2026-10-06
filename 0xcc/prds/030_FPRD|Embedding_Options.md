# Feature PRD: Embedding Options

**Document ID:** 030_FPRD|Embedding_Options
**Version:** 1.1 (planned)
**Status:** Planned. Feature PRD written 2026-10-06; v1.1 the same day applies the operator's
Feature-PRD decisions (P-18; technical defaults T-91–T-95). FTDD, FTID and FTASKS follow.
**Source:** BRD-04 (miLLM — Dataworks Support) §5.9: R-04.35–R-04.37, plus R-04.5 (`dimensions`)
from §5.1. The *refusal* path for `dimensions` is built in Feature 25 (FR-25.3); this feature owns
the honour-or-refuse decision per model.
**PPRD:** Feature 30 (FR-30.1 – FR-30.4), PPRD v1.5, including its split note · **PADR:** v1.5 §10
"Dataworks Support (Features 25–30)" trade-offs, notably "Packed scoring by default vs one row at a
time" (which covers embedding rows) and "Degrade optional capabilities rather than refuse to serve"
(GGUF embeddings)
**Binding decisions:** checkpoint decisions and technical defaults of 2026-10-06
(`~/app/miDataworks/0xcc/docs/brd-decisions-2026-10-05.md`, "Checkpoint decisions"); Feature-PRD
decisions of 2026-10-06 (same file, "Feature-PRD decisions"), and the technical register
(`~/app/miDataworks/0xcc/docs/fprd-open-questions-2026-10-06.md`, T-91–T-95, all accepted)
**Depends on:** Feature 25 (the output-changing field list and unknown-field reporting); Feature 23
(GGUF embeddings through `_llamacpp_embeddings`). Reuses `_unsteered`.
**Consumers:** miDataworks BRD-03 R-03.18 (embedding near-deduplication, clustering, diversity
figures), through miDataworks feature 004 (FR-004.14 and FR-004.24). Its ADR-027 blocks embedding
operators against miLLM until this feature is served. **P-18: embeddings are not required for
miDataworks M1**; a lexical fallback is allowed and recorded. So nothing here blocks M1.

Code references are to miLLM at `7aa659c` (HEAD, 2026-10-06).

---

## 1. Feature Overview

**Name:** Embedding Options.

**What it is:** four changes to `POST /v1/embeddings`.

1. **Pooling.** The caller chooses how token vectors become one vector: `mean` (the default, as
   today), `last` or `cls`.
2. **Normalisation.** The caller may ask for unit-length (L2-normalised) vectors. The default stays
   unnormalised, as today.
3. **`dimensions`.** A shortened vector is served only for a model that declares support for it.
   Otherwise the request is refused with `400`. When honoured, truncation follows pooling, and the
   result is re-normalised when `normalize` is set.
4. **No silent truncation.** An input longer than the model's limit is refused with `400` naming its
   index. The number of inputs per request is capped, with a stated default.

It also corrects the route's comments, which describe a guard the route no longer contains.

**Problem.** The vector is always a fixed mean over the last hidden layer
(`millm/services/inference_service.py:5123-5124`). A caller cannot ask for last-token pooling,
which suits causal decoders, or for unit vectors, which cosine similarity assumes.

`dimensions` is accepted by the schema (`millm/api/schemas/openai.py:293`) and never read by
`create_embeddings` (`inference_service.py:5071-5148`). A caller asking for 256 dimensions gets the
full width, with a `200`.

Inputs are tokenised with `truncation=True` (`inference_service.py:5108-5110`). Where the tokenizer
has a `model_max_length`, a long input is cut before the context check at `:5113` can see it. The
vector then describes a prefix of the text, and nothing says so. A near-duplicate detector would
call two long documents duplicates because their first pages match.

**Goals:**
- The vector is pooled the way the caller asked, or the request is refused.
- An output-changing field (`dimensions`, `pooling`) is never ignored on any engine.
- No input is truncated silently. The refusal names which input was too long.
- The default request returns exactly the vectors it returns today, so stored retrieval indexes stay
  valid.
- Embeddings stay unsteered and unmonitored.

**Connection to the project.** This is the embeddings part of the Dataworks Support increment
(BRD-04). It obeys the increment's rule from Feature 25: a field is honoured or refused, never
ignored. Feature 26's batch runner executes embedding rows through the same service code
(FR-26.4.5), so every rule here also holds for batch rows.

## 2. User Stories & Scenarios

**US-1. Curation operator embeds rows for near-deduplication.** As miDataworks' embedding
deduplicator (FR-004.14), I send `pooling: "last"` and `normalize: true`, so cosine similarity is a
dot product.
*Acceptance:* every returned vector has L2 norm 1 within 1e-5. The vectors are last-token vectors
(BRD-04 acceptance 13).

**US-2. Curation operator never embeds a prefix.** As the same operator, I send a row longer than
the model's context.
*Acceptance:* the request returns `400 context_length_exceeded`, with `param` naming the input's
index. No vector is returned for any input in that request. No forward pass runs (BRD-04 acceptance
13). miDataworks then skips the row with reason `over_input_cap` (004 FPRD §2 edge cases).

**US-3. Client asks for shortened vectors.** As an OpenAI-SDK client, I send `dimensions: 256`.
*Acceptance:* in v1 no model declares truncated-embedding support (T-91), so on every model the request returns `400` naming `dimensions`, before any model is loaded
(BRD-04 acceptance 2).

**US-4. Existing retrieval client is unaffected.** As Open WebUI's retrieval feature, I send
`input` and `model` only.
*Acceptance:* the vectors equal today's output exactly, element for element.

**US-5. Client asks a GGUF model for last-token pooling.** As a client naming a GGUF model (llama.cpp's
file format), I send `pooling: "last"`.
*Acceptance:* the request returns `400` naming `pooling` and the reason. The resident model is not
evicted to find this out.

**Secondary scenarios.**
- A client sends 10,000 inputs in one request. The request is refused with `400`, naming the cap and
  the count, before any model is loaded.
- A client sends `normalize: true` with `dimensions`. In v1 the request is refused (T-91). Once a
  model can declare support, truncation happens first, then normalisation, so the vector is unit length.
- Feature 26 packs embedding rows of different lengths into one padded forward pass. Each row's
  vector ignores the padding positions.

**Edge cases and error scenarios.**

| Case | Behaviour |
|---|---|
| Two inputs over the limit, at indices 3 and 7 | One `400`. `param` names the first; the message names both (FR-30.3.4). |
| Input 0 fits, input 1 is over the limit | `400`. Input 0 is not embedded first; all inputs are checked before any forward pass (FR-30.3.3). |
| `pooling: "max"` | `400` naming `pooling` (schema validation). |
| `dimensions` larger than the model's width, on a declaring model | `400` naming `dimensions` and the width (FR-30.1.4; inactive in v1, T-91). |
| `dimensions` equal to the width, on a model that declares nothing | `400` (FR-30.1.6). |
| A pooled vector with zero or non-finite norm under `normalize: true` | Error naming the index. Never `NaN` in the output (FR-30.2.6). |
| `pooling: "cls"` on a causal decoder | Served as specified and documented (T-92). |
| `input: ""`, `[]`, or an empty string in a list | `400` naming `input` (T-94). |
| GGUF model loaded with embeddings disabled | Existing refusal naming `GGUF_ENABLE_EMBEDDINGS` (`inference_service.py:5198-5205`), unchanged. |
| A steering profile or circuit is active | Vectors are identical to those with no SAE attached (FR-30.2.8). |

**User journey (miDataworks deduplication).** Resolve the `embeddings` role → hold the model lease
(Feature 29) → send rows with the strict header, `pooling`, `normalize` and the refuse-load policy →
on `400 context_length_exceeded`, skip the named row with `over_input_cap` and resend the rest → on
`503`, honour `Retry-After` → cache vectors keyed by the embedding specification.

## 3. Functional Requirements

Each FR-30.x below is the PPRD v1.5 requirement. The numbered items under it refine it into
testable statements. "Refused" means HTTP `400` with the OpenAI error envelope
(`millm/api/routes/openai/errors.py:26-57`), `param` naming the field.

### 3.1 `dimensions`

**FR-30.1 `dimensions` honoured or refused.** `dimensions` on `/v1/embeddings` SHALL be honoured
only for a model whose metadata declares support for truncated embeddings, and otherwise SHALL
return `400`. (R-04.5)

- **FR-30.1.1** A model *declares* support only through an explicit declaration recorded for it.
  No such declaration exists today: the model row's columns (`millm/db/models/model.py:81-166`)
  carry none. **T-91: there is no declaration in version 1.** No model declares support, so every
  `dimensions` request is refused, on both engines.
- **FR-30.1.2** The refusal SHALL happen before any auto-load, because the decision depends only on
  the request and the model row (FR-25.3.8). The message names the model and says it does not
  declare truncated-embedding support.
- **FR-30.1.3** *(Inactive in v1 under T-91; specified for the increment that adds a declaration.)*
  When honoured, the vector SHALL be truncated *after* pooling to its first
  `dimensions` components. If `normalize` is true, the truncated vector SHALL then be L2-normalised.
  A vector normalised before truncation is not unit length after it, so normalising only before
  truncation is non-conformant.
- **FR-30.1.4** *(Inactive in v1 under T-91.)* A `dimensions` larger than the model's native width SHALL be refused, naming the
  width. If the declaration lists allowed sizes, any other size SHALL be refused, naming them.
- **FR-30.1.5** *(Inactive in v1 under T-91.)* Truncation SHALL apply identically on the transformers and llama.cpp engines. It is
  post-processing in miLLM, after the engine returns the pooled vector.
- **FR-30.1.6** On a model that declares nothing, `dimensions` SHALL be refused whatever its value,
  including the native width. `dimensions` has no neutral value in Feature 25's list (FR-25.3.4).
- **FR-30.1.7** This requirement replaces the *refused until Feature 30* outcome for `dimensions` in
  Feature 25's list (FR-25.3.3, FR-25.3.7). The list module stays the single source. Its entry for
  `dimensions` reads the per-model decision; no endpoint keeps its own copy.
- **FR-30.1.8** *(Inactive in v1 under T-91.)* With `encoding_format: "base64"`, the encoded vector SHALL be the truncated one: its
  decoded length equals `dimensions`.

### 3.2 Pooling and normalisation

**FR-30.2 Pooling and normalisation options.** `/v1/embeddings` SHALL accept `pooling` (`mean`
default, `last`, `cls`) and `normalize` (default false). (R-04.35)

- **FR-30.2.1** `pooling` accepts exactly `mean`, `last` and `cls`. Any other value SHALL be refused
  by schema validation, naming `pooling`.
- **FR-30.2.2** A request with neither field SHALL return exactly today's vectors, element for
  element. Today's computation is the mean over every position of the last hidden layer, for one
  unpadded input at a time (`inference_service.py:5106-5124`). Stored retrieval indexes built from
  today's vectors would otherwise silently stop matching new queries. A regression test pins this.
- **FR-30.2.3** Pooling SHALL be defined over *real* tokens, using the attention mask:
  - `mean`: the average over positions whose mask is 1;
  - `last`: the vector at the last position whose mask is 1;
  - `cls`: the vector at the first position whose mask is 1.

  Defining pooling by mask, not by fixed index, is what lets a padded row give its own vector.
  Feature 26 packs embedding rows by default (FR-26.5.1). Under left padding, index 0 is padding,
  and under right padding the last index is padding.
- **FR-30.2.4** Special tokens are added at tokenisation as today. They count as real tokens. So
  `mean` includes a beginning-of-sequence token where the tokenizer adds one, as today, and `cls` is
  that token's position. The API reference SHALL say this.
- **FR-30.2.5** The API reference SHALL state what `cls` means on a causal decoder. Position 0
  attends only to itself, so its vector depends only on the first token. Where the tokenizer adds a
  fixed beginning-of-sequence token, every input gets the same `cls` vector. **T-92: `cls` is served and
  documented**, not refused.
- **FR-30.2.6** `normalize: true` SHALL L2-normalise the final vector (after any truncation,
  FR-30.1.3). Each returned vector SHALL have norm 1 within 1e-5. A vector whose norm is zero or
  non-finite SHALL produce an error naming its index. The output SHALL never contain `NaN` or
  infinity.
- **FR-30.2.7** On the llama.cpp engine, pooling is fixed when the model is constructed: miLLM loads
  GGUF models with MEAN pooling (`millm/ml/model_loader.py:2292-2293`, constant at `:1904`).
  - `pooling: "mean"` SHALL be honoured.
  - `pooling: "last"` and `"cls"` SHALL be refused, naming `pooling` and the reason. The refusal
    SHALL happen before any auto-load, decided from the row's `gguf_files`
    (`millm/db/models/model.py:126`).
  - `normalize` SHALL be honoured, as post-processing in miLLM.
- **FR-30.2.8** Embeddings SHALL stay unsteered. Every forward pass SHALL run inside `_unsteered()`,
  entered in the thread that runs the forward (`inference_service.py:1176-1199`). No pooling mode
  adds a forward path. The existing two-SAE suppression test (`tests/unit/services/
  test_inference_service.py:783-830`) SHALL cover every pooling mode.
- **FR-30.2.9** Embeddings SHALL record no probe events and no sensing events. Today they open no
  probe context, so the probe hook observes nothing (`millm/services/probe_runtime.py:839-842`), and
  sensing runs inside the suppressed SAE hook. This feature SHALL NOT add a `_probe_begin` call to
  the embedding path.
- **FR-30.2.10** The same options SHALL apply to embedding rows in a batch, through the same service
  code (FR-26.4.5).
- **FR-30.2.11** Under Feature 25, `pooling` and `normalize` become declared, consumed fields on both
  engines (FR-25.1.2). `pooling` is output-changing and SHALL be added to the list (FR-25.3.1); on
  llama.cpp its non-mean values are *refused*, not reported as unused.

### 3.3 Input limits

**FR-30.3 No silent truncation; capped input count.** An input longer than the model's limit SHALL
return `400` naming its index and SHALL never be truncated silently. Inputs per request SHALL be
capped with a stated default. (R-04.36)

- **FR-30.3.1** Inputs SHALL be tokenised without truncation. `truncation=True` at
  `inference_service.py:5108-5110` is removed.
- **FR-30.3.2** The limit is the model's served context: on transformers, `_served_max_context`
  (`inference_service.py:222`, read at `:3042`); on llama.cpp, the loaded instance's context window
  (Feature 23). The token count includes any special tokens tokenisation adds.
- **FR-30.3.3** Every input SHALL be tokenised and checked before any forward pass. Today the check
  runs inside the loop (`inference_service.py:5113`), so input 0 is embedded before input 1 is
  refused.
- **FR-30.3.4** The refusal SHALL use code `context_length_exceeded` and `param` `input[i]` for the
  first over-limit index (`input` when `input` is a string). The message SHALL name every over-limit
  index with its token count, and the limit. The list SHALL be bounded; if indices are left out, the
  message says how many. Today the registered handler sends no `param` for a
  `MiLLMError` (`millm/api/exception_handlers.py:118-123`, registered at `millm/main.py:543`), so the
  design must carry `param` through.
- **FR-30.3.5** On llama.cpp, miLLM SHALL count tokens with the instance's own tokenizer before
  calling the engine, and SHALL ensure the engine does not truncate. Whether llama-cpp-python
  truncates inside `create_embedding` is unverified here (the library is not installed in the
  development environment); the FTDD verifies it.
- **FR-30.3.6** The number of inputs per request SHALL be capped by a setting, with its default
  stated in configuration and in the API reference. **T-93: the FTDD sets the value from measured latency.** A request over
  the cap SHALL be refused naming `input`, the cap and the count, before any auto-load.
- **FR-30.3.7** `usage.prompt_tokens` SHALL remain the total tokens embedded. With truncation gone,
  it equals the sum of each input's full token count.
- **FR-30.3.8** Empty input SHALL be refused with `400` naming `input`, before any auto-load
  (**T-94**). Empty means `""`, `[]`, or an empty string at any index of a list. Today `[]` returns
  `200` with no data after auto-loading the model.

### 3.4 Route comments

**FR-30.4 Comments match code.** The embeddings route's comments SHALL match its code. No GGUF guard
SHALL be described that the route does not contain. (R-04.37)

- **FR-30.4.1** The comment at `millm/api/routes/openai/embeddings.py:64-73` says GGUF embeddings
  are impossible and describes a guard. Neither is true: GGUF embeddings are served
  (`_llamacpp_embeddings`, `inference_service.py:5150`), and no guard follows the comment. It SHALL
  be replaced by an accurate statement of the pre-load refusals this feature adds (FR-30.1.2,
  FR-30.2.7, FR-30.3.6), or removed. The trailing whitespace line at `:74` goes with it.
- **FR-30.4.2** Two more stale statements in the same path were found while writing this PRD:
  - the module docstring says a model must already be loaded (`embeddings.py:6`), while the route
    auto-loads (`:80-83`);
  - the route docstring (`embeddings.py:56-57`) and the service docstring
    (`inference_service.py:5073-5075`) say "mean pooling" only.

  Both SHALL be corrected.
- **FR-30.4.3** The manual's embeddings section (`manual/docs/api/openai-compatible.md:193-199`)
  SHALL describe `pooling`, `normalize`, `dimensions`, the input limit, the cap, and the GGUF
  restrictions.

## 4. User Experience Requirements

No Admin UI change (PPRD Feature 30, "UI Tab: none"). The user experience is the API:

- Error messages name the field, the input index and the limit, so a caller can act without reading
  the code.
- Refusals that the model row can decide happen before any model load. A wrong request never costs
  the caller a model swap.
- The API reference documents every option, default and refusal.

## 5. Data Requirements

- **No new table.**
- **Request fields** on `EmbeddingRequest` (`millm/api/schemas/openai.py:287-296`): `pooling`
  (`"mean" | "last" | "cls"`, default `"mean"`) and `normalize` (boolean, default false).
  `dimensions` already exists (`:293`).
- **Truncated-embedding declaration** per model: none in v1 (T-91), so no migration. A later
  increment that adds one SHALL default it to "not declared", so existing rows are refused rather
  than guessed.
- **Setting:** the input-count cap (FR-30.3.6). Name chosen in the FTDD, beside the existing GGUF
  embedding setting (`millm/core/config.py:280-290`).
- **Response shape:** unchanged (`EmbeddingResponse`, `openai.py:449-455`).

## 6. Technical Constraints

- **PADR v1.5 §10, "Report-then-opt-in refusal of unknown fields":** output-changing fields are
  honoured or refused; the list lives in one module (Feature 25).
- **PADR v1.5 §10, "Packed scoring by default vs one row at a time":** embedding rows may be packed
  in a batch. Pooling must therefore be mask-aware (FR-30.2.3).
- **PADR v1.5 §10, "Degrade optional capabilities rather than refuse to serve":** a GGUF model may be
  loaded without embedding support. That refusal is unchanged.
- **Serial execution:** embedding forwards run inside `_admit()` (`inference_service.py:5105`), as
  today. `MAX_CONCURRENT_REQUESTS` stays 1 (BRD-04 §3).
- **Numerics:** normalisation SHOULD be computed in float32, to meet the 1e-5 norm tolerance in
  bfloat16 models. The default path must not change dtype, because FR-30.2.2 requires identical
  output.
- **Python standards:** Black (line length 100), Ruff, MyPy strict (PADR Appendix A).

## 7. API/Integration Specifications

**`POST /v1/embeddings`** — request:

| Field | Type | Default | New? |
|---|---|---|---|
| `model` | string | required | no |
| `input` | string or list of strings | required | no |
| `encoding_format` | `"float"` or `"base64"` | `"float"` | no |
| `dimensions` | integer > 0 | none | no (now honoured or refused) |
| `pooling` | `"mean"`, `"last"`, `"cls"` | `"mean"` | yes |
| `normalize` | boolean | false | yes |
| `user` | string | none | no |

**Response:** unchanged OpenAI shape.

**Refusals (all `400`, OpenAI envelope):**

| Condition | `param` | Before auto-load? |
|---|---|---|
| `dimensions` on a model that declares nothing | `dimensions` | yes |
| `dimensions` above the width or outside the declared sizes (inactive in v1, T-91) | `dimensions` | yes if the declaration states them |
| `pooling` not in the enum | `pooling` | yes (schema) |
| `pooling` `last` or `cls` on a GGUF model | `pooling` | yes |
| more inputs than the cap | `input` | yes |
| an input over the limit | `input[i]`, code `context_length_exceeded` | no (needs the tokenizer); before any forward pass |
| zero or non-finite norm under `normalize` | `input[i]` | no |

**Integration.** Feature 25's list module gains `pooling` and the per-model `dimensions` outcome.
Feature 26's runner calls the same service function. miStudio's MCP proxy (BRD-MIS-DATAWORKS-001)
owns any MCP tool; this feature adds none. miLLM `docs/mcp-contract.md` is touched only if miStudio's
tools call this route; that decision is miStudio's.

**Authentication:** none on `/v1` (BRD-04 §7, decision 7).

## 8. Non-Functional Requirements

- **Overhead:** pooling, truncation and normalisation are vector operations on one row per input.
  They SHALL add no forward pass.
- **Fail before work:** every refusal decidable from the request or the row happens before
  auto-load; every input-limit refusal happens before any forward pass.
- **Compatibility:** default requests are unchanged, element for element (FR-30.2.2).
- **Privacy:** refusal messages and logs name indices and token counts, never input text.
- **Reliability:** no new background state; nothing to clean up after a failed request.

## 9. Feature Boundaries (Non-Goals)

- Token-array inputs (`input` as a list of integers), which OpenAI accepts. Not requested by BRD-04.
- Choosing a layer other than the last hidden layer.
- Instruction prefixes or prompt templates for embedding models.
- `last` or `cls` pooling on GGUF. That would need a second llama.cpp instance or per-token output;
  it is refused instead.
- Training or converting models for truncated embeddings. This feature only reads a declaration.
- Packing embedding rows. Feature 26 owns packing; this feature only makes pooling correct under it.
- Steered embeddings. Embeddings are always unsteered.
- Provenance headers stating which options were applied (T-95: not added).
- MCP tools (BRD-MIS-DATAWORKS-001).

## 10. Dependencies

- **Feature 25:** the output-changing field list (FR-25.3), unknown-field reporting and strict mode
  (FR-25.1, FR-25.2). FR-30.1 and FR-30.2.11 change entries in that list. Build after, or with, the
  list module.
- **Feature 26:** batch embedding rows reuse this code; packing relies on FR-30.2.3.
- **Feature 29:** the model lease and `Retry-After`, which miDataworks uses around embedding calls.
  No code dependency.
- **Feature 23:** GGUF embeddings (`_llamacpp_embeddings`, `GGUF_ENABLE_EMBEDDINGS`).
- **Libraries:** none new. llama-cpp-python's truncation behaviour must be verified (FR-30.3.5).
- **Downstream:** miDataworks feature 004's embedding deduplicator and clustering wait on this
  feature (its ADR-027).

## 11. Success Criteria

1. **BRD-04 acceptance 13.** An over-limit input returns `400` naming its index. `pooling: last` with
   `normalize: true` returns unit vectors.
2. **BRD-04 acceptance 2 (`dimensions` half).** `dimensions` on a model without declared support
   returns `400`, before any model load.
3. A request with no new fields returns today's vectors exactly.
4. With a two-SAE circuit attached, every pooling mode returns the same vectors as with no SAE.
5. A GGUF model refuses `pooling: last` with no load and no eviction of the resident model.
6. Each wiring item is accepted only by a test that fails when its call or registration line is
   removed, asserting payload and call count (BRD-04 §6 preamble; FR-20.3).
7. Hardware check on a served model: unit norms within 1e-5, and a long input refused, not cut.

## 12. Testing Requirements

**Unit (pooling as a pure function):**
- Each mode on hand-built hidden states with known answers.
- Mixed-length padded rows, left and right padding, where padding positions hold large values. The
  test fails if any padding position enters the pool. A fixture whose padding is zero would agree
  with a mask-blind mean by construction.
- Truncate-then-normalise order, on a vector where normalise-then-truncate gives a different answer.
- Zero and non-finite norms produce the indexed error, never `NaN`.

**Service and route:**
- Default-output regression against the current computation.
- A real tokenizer with a small `model_max_length` proves no truncation. The existing mock tokenizer
  cannot truncate, so it agrees with the defect by construction.
- Two inputs over the limit: both indices named, `param` set, model forward call count 0.
- Input-count cap refused with `load_model_and_wait` never awaited.
- `dimensions` on an undeclared model and `pooling: last` on a GGUF row refused with no load.
- Every pooling mode under the two-SAE suppression test (FR-30.2.8).
- Embeddings with a probe armed write no probe event.
- base64 output decodes to the truncated length.

**Wiring and mutation controls** (per the review discipline in the user's global instructions):
- Remove the `pooling` dispatch: a pooling test goes red.
- Restore `truncation=True`: the no-truncation test goes red.
- Move the limit check back inside the loop: the call-count-0 test goes red.
- Swap truncation and normalisation: the order test goes red.
- Drop the mask from `mean`: the padding test goes red.
- Re-enable `dimensions` for undeclared models in the list module: Feature 25's every-entry test and
  this feature's refusal test go red.

**Integration (hardware):** on a served transformers model and one GGUF model, run the acceptance
items in §11.

**Performance:** none beyond confirming no extra forward pass (call count per input is 1).

## 13. Implementation Considerations

- **Complexity:** low to moderate. The pooling function is small. The risk is in the edges: padding,
  special tokens, `param` plumbing and the llama.cpp truncation question.
- **Recommended approach:** one pure function `pool(hidden, mask, mode)`, unit-tested directly, and
  one post-processing step for truncation and normalisation shared by both engines. Tokenise and
  check every input first, then run the forwards.
- **Error `param`:** the registered handler drops `param` (`exception_handlers.py:118-123`). Either raise a
  structured error the handler can read, or return the envelope from the route.
- **Limit source:** `_served_max_context` can return `None` (`inference_service.py:3043-3044`), and
  the check is then skipped. With truncation removed, such an input runs at full length. The FTDD
  states which served models can reach this and what happens.
- **Observed, out of scope:** the transformers embedding forward runs synchronously on the event-loop
  thread inside `_admit()` (`inference_service.py:5117-5120`), while the llama.cpp path uses
  `asyncio.to_thread` (`:5197`). Recorded here for the FTDD; not changed by this feature.
- **Estimate:** small; one to two sessions including tests and the manual.

## 14. Open Questions

All five questions of v1.0 are resolved. The operator accepted each technical default
(`fprd-open-questions-2026-10-06.md`, T-91–T-95), and P-18 settles the consumer's phase.

| ID | Question (v1.0 number) | Resolution | Effect here |
|---|---|---|---|
| T-91 | Where a model declares truncated-embedding support (OQ 1) | No declaration in v1; all `dimensions` refused | FR-30.1.1; FR-30.1.3–30.1.5 and 30.1.8 inactive |
| T-92 | `cls` on causal decoders (OQ 2) | Serve, documented | FR-30.2.5 |
| T-93 | Input-count cap default (OQ 3) | The FTDD sets it from measured latency | FR-30.3.6 |
| T-94 | Empty input (OQ 4) | Refuse with `400` | FR-30.3.8 |
| T-95 | Provenance header (OQ 5) | Not added | §9 non-goal |
| P-18 | Are embeddings needed for miDataworks M1? | No; a lexical fallback is allowed and recorded | Header "Consumers"; nothing here blocks M1 |

**Still open:** none at the product level. Two technical items move to the FTDD: llama-cpp-python's
truncation behaviour (FR-30.3.5) and the measured cap value (T-93).

## 15. Decisions from Clarifying Questions

Clarifying rounds were waived. Each question is pre-answered from a cited source. Questions with no
source were Open Questions, now resolved in §14.

| # | Question | Answer | Source |
|---|---|---|---|
| D1 | Priority? | Planned with the Dataworks Support increment; blocks miDataworks' embedding operators | PPRD v1.5 Feature 30; miDataworks 004 ADR-027 |
| D2 | Who uses it? | API callers: miDataworks, Open WebUI retrieval, scripts | BRD-04 §3; PPRD Feature 30 |
| D3 | Admin UI? | None | PPRD Feature 30 "UI Tab: none" |
| D4 | Default pooling and normalisation? | `mean`, unnormalised, as today | R-04.35 |
| D5 | Must default output change? | No; identical to today | R-04.35 "as today"; retrieval-index compatibility |
| D6 | `dimensions` on an undeclared model? | Refused, whatever the value | R-04.5; FR-25.3.4 (no neutral value) |
| D7 | Order of truncation and normalisation? | Pool, truncate, then normalise | PPRD v1.5 Feature 30 split note |
| D8 | Long input: truncate or refuse? | Refuse with `400` naming the index | R-04.36 |
| D9 | Check all inputs before running any? | Yes | R-04.36 "never truncated silently"; `inference_service.py:5113` runs per input today |
| D10 | Pooling on GGUF? | `mean` only; `last`/`cls` refused before auto-load | Pooling fixed at construction, `model_loader.py:2292-2293` |
| D11 | Are embeddings steered or monitored? | Never | Existing `_unsteered` guarantee, `inference_service.py:1176-1199`; manual `openai-compatible.md:197-199` |
| D12 | Do options apply to batch rows? | Yes, same service code | FR-26.4.5; PADR §10 "Packed scoring" |
| D13 | Is pooling mask-aware? | Yes | Feature 26 packs embedding rows by default (FR-26.5.1) |
| D14 | Error code for over-limit input? | Existing `context_length_exceeded` | `errors.py:91`; `test_context_length_refusal.py:204-210` |
| D15 | Stale comment: fix or remove? | Replace with an accurate one, or remove; plus two more stale docstrings | R-04.37; `embeddings.py:6, 56-57, 64-73` |
| D16 | Is `pooling` output-changing? | Yes; added to Feature 25's list | R-04.3 rule ("honoured or refused"); FR-25.3.1 |
| D17 | MCP tools? | None here | BRD-04 §4 out of scope; BRD-MIS-DATAWORKS-001 |
| D18 | Truncated-embedding declaration? | None in v1; all `dimensions` refused | T-91 |
| D19 | `cls` on causal decoders? | Served, documented | T-92 |
| D20 | Input-count cap? | Set in the FTDD from measured latency | T-93 |
| D21 | Empty input? | `400` | T-94 |
| D22 | Provenance header? | Not added | T-95 |
| D23 | Is this M1-blocking for miDataworks? | No | P-18 |

### Coverage

Every BRD-04 requirement this feature owns is covered.

| BRD-04 | Topic | PPRD FR | Refined in this PRD | Status |
|---|---|---|---|---|
| R-04.5 | `dimensions` honoured or refused | FR-30.1 | FR-30.1.1 – FR-30.1.8 | covered: refused for every model in v1 (T-91); 30.1.3–30.1.5, 30.1.8 inactive |
| R-04.35 | `pooling`, `normalize` | FR-30.2 | FR-30.2.1 – FR-30.2.11 | covered (`cls` served and documented, T-92) |
| R-04.36 | No silent truncation; input cap | FR-30.3 | FR-30.3.1 – FR-30.3.8 | covered (cap value from FTDD measurement, T-93; empty input refused, T-94) |
| R-04.37 | Comments match code | FR-30.4 | FR-30.4.1 – FR-30.4.3 | covered |

**Acceptance mapping:** BRD-04 §6 item 2 (`dimensions` half) → FR-30.1; item 13 → FR-30.2.6,
FR-30.3.4. Totals: 4 of 4 requirements, 30 refined statements.
