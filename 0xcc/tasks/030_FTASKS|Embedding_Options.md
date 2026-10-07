# Feature 30: Embedding Options — Task List

**Status:** Planned (2026-10-06). Generated in one pass from the FPRD, FTDD and FTID; **the "Go" pause
after the parent tasks was waived** by the coordinator's instruction, as in 024 (D13).
**Inputs:** 030_FPRD v1.1, 030_FTDD v1.0, 030_FTID v1.0 · BRD-04 §5.9 and R-04.5 · PPRD v1.5 Feature 30
**Decisions applied:** T-91 (no truncated-embedding declaration; all `dimensions` refused), T-92 (`cls`
served and documented), T-93 (cap from measured latency), T-94 (empty input refused), T-95 (no
provenance header), P-18 (embeddings not required for miDataworks M1, so nothing here blocks M1).
**Prerequisite:** Feature 25's `millm/api/request_policy.py` (task 0.3 checks it).
**Consumers:** miDataworks 004 FR-004.14 and FR-004.24, after its M1 (P-18). Feature 26's embedding
executor calls `_embed_inputs`.

## Relevant Files
- `millm/ml/embedding_pooling.py`: `PoolingMode`, `EmbeddingOptions`, `pool_hidden`, `finalize_vector` ·
  `tests/unit/ml/test_embedding_pooling.py`
- `millm/api/schemas/openai.py`: `EmbeddingRequest` fields and the `input` validator ·
  `tests/unit/api/test_openai_schemas.py` (`TestEmbeddingRequest`, `:311`)
- `millm/api/request_policy.py` (Feature 25's): rows `dimensions`, `pooling`, `normalize`
- `millm/services/inference_service.py`: `create_embeddings`, `_embed_inputs`, `_embedding_limit`,
  `_check_embedding_lengths`, `_llamacpp_embeddings` · `tests/unit/services/test_embedding_options.py`
- `tests/unit/services/test_inference_service.py`: two-SAE suppression test (`:783-830`) parametrised
- `tests/unit/services/test_gguf_refusals.py`: `TestGGUFEmbeddings` (`:763`)
- `millm/core/errors.py`: `openai_param`; `EmbeddingInputTooLongError`, `EmbeddingVectorInvalidError`
- `millm/api/routes/openai/errors.py`: `ERROR_STATUS_MAP` row
- `millm/api/exception_handlers.py`: `param` passed at `:118-123`
- `millm/api/routes/openai/embeddings.py`: comments `:6`, `:56-57`, `:64-74`; policy call position ·
  `tests/unit/api/test_embedding_options_route.py`
- `millm/core/config.py`, `.env.example`: `EMBEDDINGS_MAX_INPUTS`
- `manual/docs/api/openai-compatible.md` (`:193-199`), `manual/docs/reference/configuration.md` (`:85`)
- `tests/unit/api/test_request_policy.py`, `tests/unit/api/test_request_policy_coverage.py`: the
  `dimensions` reason under T-91; test values for the two new rows
- `tests/unit/api/test_context_length_refusal.py`: the embeddings case asserts the indexed message
- `tests/unit/api/test_openai_schemas.py`: `pooling`, `normalize`, empty input and the cap

### Notes
- Tests: `pytest tests/unit` (backend); `ruff check millm tests`; `mypy millm/`.
- **Reachability is a shipping gate** (global instructions): every wiring line has a test that fails
  when it is removed, asserting the payload and the call count. Task 7.0 records the controls.
- **Verify every restore**: `git diff --stat` clean and the mutated line re-grepped. A control helper
  that `cd`s can silently fail its restore (recorded twice in this estate).
- Re-verify every line number from the FTID before editing; code was read at `7aa659c`.
- **The 2026-10-04 unsteered fix is in the current code**: `_unsteered()` suppresses every attached
  SAE (`millm/services/inference_service.py:1176-1199`; the finding is recorded at `:1178-1182`), and
  the embedding forward enters it at `:5117`; commit `8ea473a`; test at
  `tests/unit/services/test_inference_service.py:783-830` (comment `:793-795`). Task 3.6 keeps it.

### Category Checklist Results
- **Data layer:** N/A — no table, column or migration. T-91 removed the only candidate (a declaration
  column); adding it unread would be a declared mechanism with no wiring.
- **Backend/API:** 1.x (schema, policy), 3.x (service, errors), 4.x (GGUF), 5.x (validator)
- **Frontend/UI:** N/A — PPRD Feature 30 "UI Tab: none"; the surface is the API.
- **Business logic:** 2.x (pooling, normalisation), 3.2–3.3 (length check)
- **Integration wiring:** 0.3 and 1.2 (Feature 25 table), 3.1 (Feature 26 entry point `_embed_inputs`),
  3.5 (exception handler), 1.3 (route position)
- **Error handling and logging:** 3.4 (errors, `param`, map row), 3.7 (`embedding_refused` log), 5.1
- **Testing:** unit throughout; route tests 1.4, 5.3; hardware 8.2–8.4; mutation controls 7.x
- **Performance and security:** 8.3 (cap measurement, T-93); 3.7 (no input text in logs); 5.1 (count
  cap bounds slot hold time). No benchmark beyond the cap: no extra forward pass is added (FTDD §9).
- **Configuration/deployment:** 5.2 (setting, `.env.example`). No Kubernetes change: the default applies.
- **Documentation:** 6.x (route comments, manual API page, configuration reference)

## Tasks

- [ ] 0.0 Preconditions and verification spikes (covers FR-30.3.5, FR-30.3.2; precondition for FR-30.1,
      FR-30.2.11). **No product open questions remain**: FPRD v1.1 §14 resolves all five (T-91–T-95).
  - [?] 0.1 **[needs hardware — operator session. Implemented from the library source as recalled, unverified here (llama-cpp-python is absent from the dev venv): `Llama.n_ctx()` is a method, `n_batch` an attribute, and `embed()` cuts each input to `n_batch` tokens by default (`truncate=True`, not exposed by `create_embedding`). `_embedding_limit` uses `min(n_ctx(), n_batch)`, which is safe either way: if `n_batch` does not bound an input it is merely conservative. Confirm on the backend image]** On the backend image (not the dev venv, where llama-cpp-python is absent), with one GGUF
        model loaded with embeddings: confirm `Llama.n_ctx()`, `Llama.n_batch`, the defaults of
        `Llama.tokenize`, and what `create_embedding` does with an input longer than `n_batch` and
        than `n_ctx` (truncate, raise, or other). Record results; set `_embedding_limit` for GGUF to
        `min(n_ctx(), n_batch)` or `n_ctx()` accordingly (FTDD TD15). (FR-30.3.5)
  - [?] 0.2 **[needs hardware — operator session (lists model rows on the node). Code-side: `_served_max_context(None-config)` → None means served untruncated and unchecked; pinned by `test_with_no_stated_limit_a_long_input_runs_whole`]** List every model row on the node and whether `_served_max_context` returns a value for it
        (FTDD risk R3). Any `None` is recorded with its consequence (served untruncated, unchecked). (FR-30.3.2)
  - [x] 0.3 **Route-existence precondition:** confirm `millm/api/request_policy.py` exists with
        `OUTPUT_CHANGING`, `NEUTRAL` and `evaluate`, and that `evaluate` is called in
        `embeddings.py` before auto-load. If it does not exist, **stop**: do not build a second table
        (FTDD R4). (FR-30.1.7, FR-30.2.11)

- [x] 1.0 Policy rows and schema fields (covers FR-30.1.1, FR-30.1.2, FR-30.1.6, FR-30.1.7,
      FR-30.2.1, FR-30.2.7, FR-30.2.11)
  - [x] 1.1 Add `pooling: Literal["mean","last","cls"] = "mean"` and `normalize: bool = False` to
        `EmbeddingRequest` (`openai.py:287-296`). Schema tests: defaults; `pooling: "max"` → 400
        `param: pooling` through the route. (FR-30.2.1)
  - [x] 1.2 Add the three rows to `OUTPUT_CHANGING` and the two `NEUTRAL` entries (FTID §5.2):
        `dimensions` refused on both engines with reason "no model declares truncated-embedding
        support (T-91)"; `pooling` honoured on transformers, refused on llama.cpp with `mean` neutral;
        `normalize` honoured on both. (FR-30.1.1, FR-30.1.6, FR-30.1.7, FR-30.2.7, FR-30.2.11)
  - [x] 1.3 Confirm or place the policy call between row lookup (`embeddings.py:60-62`) and auto-load
        (`:80`). The refusal message names the model. (FR-30.1.2)
  - [x] 1.4 Route tests (`test_embedding_options_route.py`): `dimensions: 64` and `dimensions` equal to
        the native width → 400 `param: dimensions`, `load_model_and_wait` not awaited (edge case
        "equal to the width, undeclared"; US-3); GGUF row + `pooling: "last"` → 400 before load
        (US-5); GGUF + `pooling: "mean"` reaches the engine. Assert the resident model is not evicted.
        (FR-30.1.2, FR-30.1.6, FR-30.2.7)
  - [x] 1.5 Confirm Feature 25's every-entry test enumerates the new rows (it reads the table). If it
        does not, that is a Feature 25 test finding: record it, do not patch around it here. (FR-30.1.7)

- [x] 2.0 Pooling and normalisation module (covers FR-30.2.2, FR-30.2.3, FR-30.2.4, FR-30.2.6)
  - [x] 2.1 Create `millm/ml/embedding_pooling.py` with `PoolingMode`, `EmbeddingOptions`,
        `pool_hidden`, `finalize_vector`, `NonFiniteEmbeddingError` (FTID §3). (FR-30.2.3, FR-30.2.6)
  - [x] 2.2 Unpadded `mean` returns `hidden.mean(dim=1)`; test with `torch.equal` in bfloat16 and
        float32. (FR-30.2.2; US-4)
  - [x] 2.3 Mask tests: right and left padding, pad positions at `1e3`, rows of real length 2 and 5;
        each mode equals the unpadded per-row answer. (FR-30.2.3)
  - [x] 2.4 `cls` returns the first real position and `last` the last; a hand-built test where a
        BOS-like first token is identical across rows shows `cls` vectors equal (documents T-92's
        behaviour). (FR-30.2.4, FR-30.2.5)
  - [x] 2.5 `finalize_vector`: `normalize=True` norm within 1e-5 (float32); `normalize=False` returns the
        same list today's conversion gives; zero vector and `NaN` raise. (FR-30.2.6; edge case
        "zero or non-finite norm")

- [x] 3.0 Transformers service path, errors and `param` plumbing (covers FR-30.2.2, FR-30.2.8,
      FR-30.2.9, FR-30.2.10, FR-30.3.1, FR-30.3.2, FR-30.3.3, FR-30.3.4, FR-30.3.7)
  - [x] 3.1 Split the body of `create_embeddings` (`inference_service.py:5071-5148`) into
        `_embed_inputs(texts, options, *, param_for_string)` (synchronous, takes no slot), called
        inside `async with self._admit()`. Signature of `create_embeddings` unchanged. This is the
        entry point Feature 26's executor calls. (FR-30.2.10)
  - [x] 3.2 Tokenise every input first with `truncation=False` and no `padding`; collect counts.
        (FR-30.3.1)
  - [x] 3.3 Add `_embedding_limit()` and `_check_embedding_lengths()` (FTID §7.2); call the check
        before any forward. (FR-30.3.2, FR-30.3.3)
  - [x] 3.4 Errors: `openai_param` on `MiLLMError`; `EmbeddingInputTooLongError` (inherits
        `CONTEXT_LENGTH_EXCEEDED`); `EmbeddingVectorInvalidError` with map row
        `EMBEDDING_VECTOR_INVALID` → 500. Message lists up to 16 indices with token counts and the
        limit; `details.omitted` counts the rest. (FR-30.3.4, FR-30.2.6)
  - [x] 3.5 Live handler: pass `param=getattr(exc, "openai_param", None)` at
        `exception_handlers.py:118-123`. Test: an existing error without the attribute still returns
        `param: null`. (FR-30.3.4)
  - [x] 3.6 Forward under `torch.no_grad()` and `self._unsteered()` on the calling thread; then
        `pool_hidden(..., options.pooling)[0]` and `finalize_vector(..., options.normalize)`. Keep
        `usage.prompt_tokens = sum(counts)`. Do not add `_probe_begin`. (FR-30.2.8, FR-30.2.9, FR-30.3.7)
  - [x] 3.7 Log one `embedding_refused` warning per refusal with `reason` and indices, no text.
        Test with `caplog` that no input text appears. (FR-30.3.4)
  - [x] 3.8 Service tests with the tiny real Llama (`test_scoring_completions.py:30-57` pattern,
        `max_position_embeddings` and `model_max_length` = 8):
        - 12-token string → 400 `context_length_exceeded`, `param: input`, forward call count 0 (US-2);
        - `[ok, long, ok, long]` → one error naming 1 and 3, forward call count 0 (edge cases "two
          inputs over" and "input 0 fits, input 1 over");
        - 20 over-limit inputs → 16 listed, `omitted == 4`;
        - each mode matches a manual pool over `output_hidden_states`;
        - `pooling: last, normalize: true` → unit vectors (US-1; BRD-04 acceptance 13);
        - `pool_hidden` spy called once per input with the request's mode; `finalize_vector` spy with
          the request's `normalize` (payload and count);
        - `usage.prompt_tokens` equals the sum of full counts;
        - `_served_max_context` → `None`: a long input runs untruncated. (FR-30.3.1–30.3.4, FR-30.3.7,
          FR-30.2.3)
  - [x] 3.9 Parametrise `test_embeddings_suppress_attached_sae` (`test_inference_service.py:783-830`)
        over the three modes; both SAEs suppressed during each forward (edge case "steering active").
        (FR-30.2.8)
  - [x] 3.10 With a probe armed, an embedding request writes no probe event and opens no probe
        context. (FR-30.2.9)

- [x] 4.0 llama.cpp path (covers FR-30.2.6, FR-30.2.7, FR-30.3.2, FR-30.3.3, FR-30.3.4, FR-30.3.5)
  - [x] 4.1 Service guard in `_llamacpp_embeddings`: `pooling != "mean"` → `EngineUnsupportedError`
        (defence in depth behind 1.2). Test it by calling the service directly. (FR-30.2.7)
  - [x] 4.2 Tokenise all inputs with the instance tokenizer inside the slot, check lengths against the
        limit from 0.1, before any `create_embedding` call. Test with a fake engine: over-limit input
        → 400 naming the index, `create_embedding` never called. (FR-30.3.2, FR-30.3.3, FR-30.3.5)
  - [x] 4.3 Apply `finalize_vector(vector, normalize)` before base64. Tests in `TestGGUFEmbeddings`:
        normalise gives unit norm; default returns the engine's floats unchanged. (FR-30.2.6)
  - [x] 4.4 The existing "loaded without embeddings" refusal (`:5198-5205`) still fires and names
        `GGUF_ENABLE_EMBEDDINGS` (edge case). (FR-30.2.7)

- [x] 5.0 Input count, empty input and configuration (covers FR-30.3.6, FR-30.3.8)
  - [x] 5.1 `@field_validator("input")` on `EmbeddingRequest` (FTID §5.1): empty string, `[]`, empty
        element (index named, plus a count), more items than `settings.EMBEDDINGS_MAX_INPUTS`. (FR-30.3.6,
        FR-30.3.8)
  - [x] 5.2 `EMBEDDINGS_MAX_INPUTS: int = Field(default=256, ge=1, le=2048)` in `config.py` beside
        `:280-290`, with the comment from FTID §9; commented line in `.env.example`. (FR-30.3.6)
  - [x] 5.3 Route tests: `""`, `[]`, `["a", ""]` → 400 `param: input`, the last naming index 1; over-cap
        list → 400 naming count and cap; none loads a model (edge case "empty input"; secondary
        scenario "10,000 inputs"). (FR-30.3.6, FR-30.3.8)

- [x] 6.0 Comments and documentation (covers FR-30.4.1, FR-30.4.2, FR-30.4.3, FR-30.2.4, FR-30.2.5)
  - [x] 6.1 Replace the stale comment at `embeddings.py:64-73` (and the whitespace line `:74`) with an
        accurate one naming the pre-load refusals; correct the module docstring (`:6`, "requires a
        model to already be loaded") and the route docstring (`:56-57`). Reviewer check: every
        statement in the file matches the code. (FR-30.4.1, FR-30.4.2)
  - [x] 6.2 Correct the service docstring (`inference_service.py:5073-5075`). (FR-30.4.2)
  - [x] 6.3 Manual `openai-compatible.md:193-199`: `pooling`, `normalize`, special tokens in pooling,
        what `cls` means on a causal decoder (T-92), `dimensions` refused on every model (T-91), the
        input limit and refusal, the cap, empty input, GGUF restrictions. Keep the "never steered" note.
        (FR-30.4.3, FR-30.2.4, FR-30.2.5)
  - [x] 6.4 **[row added with the provisional 256, marked provisional; the measured value replaces it when 8.3 runs]** `manual/docs/reference/configuration.md`: `EMBEDDINGS_MAX_INPUTS` row beside `:85`, with the
        measured value once 8.3 records it. (FR-30.3.6)

- [x] 7.0 Mutation controls (covers FR-30.1.6, FR-30.1.7, FR-30.2.3, FR-30.2.8, FR-30.3.1, FR-30.3.3,
      FR-30.3.4, FR-30.3.6). Run each, require red, restore, verify the restore, record it.
  - [x] 7.1 **Truncation refusal:** M1 `truncation=True` in `_embed_inputs` → 3.8 12-token test red;
        M3 delete the length-check call → red; M2 move the check into the loop → 3.8 four-input test
        red (a forward runs first); M4 delete the GGUF length check → 4.2 red. (FR-30.3.1, FR-30.3.3)
  - [x] 7.2 **`dimensions` handling:** M5 flip both `dimensions` rows to `HONOURED` → 1.4 red and
        Feature 25's every-entry test red; M6 move the policy call after auto-load → 1.4 "not
        awaited" red; M7 make `NEUTRAL["pooling"]` always true → 1.4 GGUF `last` red. (FR-30.1.6,
        FR-30.1.7, FR-30.2.7)
  - [x] 7.3 M8 drop the mask branch in `mean` → 2.3 red; M10 hardcode `"mean"` at the call → 3.8 spy red;
        M11 delete `_unsteered()` → 3.9 red; M9 drop `param=` in the handler → 3.8 and 5.3 red; M12
        remove the cap check → 5.3 red. (FR-30.2.3, FR-30.2.8, FR-30.3.4, FR-30.3.6)
  - [x] 7.4 **[none survived (28 of 28 red, incl. 16 extra X-controls); M5 did not redden Feature 25's every-entry test, which cannot by construction — finding F1 in the controls record]** Any mutation that survives is a test finding: write the test, re-run the mutation as a
        negative control, and record both.

- [ ] 8.0 Feature Acceptance
  - [x] 8.1 **[unit half done; SC-1/SC-4/SC-7 hardware halves open — table in 0xcc/reviews/030_implementation_controls_2026-10-07.md]** Verify each FPRD success criterion (§11, 1–7) and each user story's acceptance (US-1–US-5),
        one by one, citing the test that proves it.
  - [?] 8.2 **[needs hardware — operator session]** Hardware, on the node: a transformers model returns unit vectors for `pooling: last,
        normalize: true` (BRD-04 acceptance 13); an over-limit input is refused, naming its index;
        `dimensions` returns 400 before load (BRD-04 acceptance 2); a GGUF model refuses `pooling: last`
        without evicting the resident model; one real Open WebUI document upload still embeds.
  - [?] 8.3 **[needs hardware — operator session]** **Cap measurement (T-93):** on the RTX 3090, p95 seconds per input at 512 tokens over 64
        inputs for LFM2.5-1.2B-Instruct (bfloat16) and the largest transformers model served for
        embeddings; set `EMBEDDINGS_MAX_INPUTS` to the largest power of two with cap × p95 ≤ 30 s,
        within [64, 2048]; record the numbers in the review notes, the setting's comment and 6.4.
  - [?] 8.4 **[needs hardware — operator session]** With a profile active and a circuit attached, embeddings equal those with no SAE attached
        (FPRD §11 criterion 4).
  - [x] 8.5 **[4593 passed / 3 skipped / 0 failed; ruff and mypy run from a scratch venv (absent from ~/app/miLLM/venv): no new findings, mypy 616 → 616]** Run the full backend suite, `ruff` and `mypy`.
  - [x] 8.6 Record the mutation-control table (7.x) in the review notes. Report the Document Inventory
        update to the coordinator; this task list does not edit `CLAUDE.md` or the PPRD.

## Implementation Notes (2026-10-07, branch `feat/030-embedding-options`)

Where the code and these documents disagreed, the code won; each is recorded here.

- **3.4 / 3.5 — `param` rides in `details["param"]`, not a new `openai_param` attribute.** The FTID
  (written at `7aa659c`) said the live handler sent no `param` for a `MiLLMError`. Feature 25 has
  since made `millm_error_handler` forward `exc.details["param"]` when it is a string. Adding an
  `openai_param` attribute would have been a second mechanism for the same thing, so the new errors
  set `details["param"]` and the handler is unchanged. M9 therefore mutates that existing line.
- **1.3 — the policy call was already placed by Feature 25**, between the row lookup and the
  auto-load. Nothing moved. FR-30.1.2 also asks that the refusal *name the model*: `evaluate` gained
  an optional `model_name` (passed by `apply_request_policy` from the row), and every refusal
  message now reads "… on /v1/<path> for model '<name>' with the <engine> engine …".
- **1.5 — Feature 25's every-entry test does enumerate the table** (`CASES` over `OUTPUT_CHANGING`),
  and by its own design (`test_every_field_has_a_test_value`) it demands a non-neutral test value for
  each new row. `pooling: "last"` and `normalize: true` were added to `VALUES`; that is the test
  working, not a patch around it.
- **0.1 — the GGUF limit is `min(n_ctx(), n_batch)`**, implemented ahead of the hardware check
  because it is safe under either answer (see 0.1).
- **The existing 2026-09-14 embeddings context test** (`test_context_length_refusal.py`) asserted
  the generation-shaped message ("you requested 70 tokens (70 in the input)"). Feature 30 replaced it
  with the indexed message and a `param`, so that test was updated to the new contract.
- **Usage on GGUF** stays the engine's reported `prompt_tokens` (unchanged); with the length check
  in front, the engine can no longer truncate, so it equals the full count.
- **Tasks 1–5 were committed together.** Committing the policy rows (1.x), which mark `pooling`
  HONOURED, before the service consumed it (3.x) would have produced a commit that accepts `pooling`
  and ignores it — the defect class this feature exists to remove.

## Coverage Audit
- **FRs:**
  30.1.1 → 1.2 · 30.1.2 → 1.3, 1.4 · 30.1.3, 30.1.4, 30.1.5, 30.1.8 → **inactive in v1 (T-91)**;
  no task builds them, and 7.2 proves `dimensions` is refused rather than ignored · 30.1.6 → 1.2, 1.4,
  7.2 · 30.1.7 → 0.3, 1.2, 1.5, 7.2 ·
  30.2.1 → 1.1 · 30.2.2 → 2.2, 3.6 · 30.2.3 → 2.3, 3.8, 7.3 · 30.2.4 → 2.4, 6.3 · 30.2.5 → 2.4, 6.3 ·
  30.2.6 → 2.5, 3.4, 4.3 · 30.2.7 → 1.2, 1.4, 4.1, 4.4, 7.2 · 30.2.8 → 3.6, 3.9, 7.3, 8.4 ·
  30.2.9 → 3.6, 3.10 · 30.2.10 → 3.1 · 30.2.11 → 0.3, 1.2 ·
  30.3.1 → 3.2, 7.1 · 30.3.2 → 0.2, 3.3, 4.2 · 30.3.3 → 3.3, 4.2, 7.1 · 30.3.4 → 3.4, 3.5, 3.7, 3.8,
  7.3 · 30.3.5 → 0.1, 4.2 · 30.3.6 → 5.1, 5.2, 5.3, 6.4, 8.3 · 30.3.7 → 3.6, 3.8 · 30.3.8 → 5.1, 5.3 ·
  30.4.1 → 6.1 · 30.4.2 → 6.1, 6.2 · 30.4.3 → 6.3.
  **30 of 30 refined FRs accounted for: 26 with tasks, 4 inactive by decision T-91.**
- **User stories:** US-1 → 3.8 / 8.2 · US-2 → 3.8 / 8.2 · US-3 → 1.4 / 8.2 · US-4 → 2.2 / 8.2 ·
  US-5 → 1.4 / 8.2. Each has an implementing and a testing task.
- **Edge cases** (implement / test): two over-limit → 3.3 / 3.8 · input 0 fits, 1 over → 3.3 / 3.8 ·
  `pooling: "max"` → 1.1 / 1.1 · `dimensions` above width → inactive (T-91) · `dimensions` equal to
  width, undeclared → 1.2 / 1.4 · zero or non-finite norm → 2.1, 3.4 / 2.5 · `cls` on causal → 2.1 /
  2.4, 6.3 · GGUF without embeddings → existing / 4.4 · steering active → 3.6 / 3.9 · empty input →
  5.1 / 5.3.
- **TDD/TID sections:** Data Design → N/A (checklist) · API Design → 1.x, 3.4, 3.5, 5.x · Component
  Architecture → 2.x, 3.1 · State → N/A (no state; FTDD §7) · Security → 3.7, 5.1 · Performance → 8.3
  · Testing → 2.x, 3.8–3.10, 4.x, 5.3, 7.x · Deployment → 5.2, 6.4 · FTID error table → 3.4, 4.4, 5.1.
- **Open questions:** none remain (FPRD v1.1 §14: T-91–T-95, P-18). The two technical measurements are
  tasks 0.1 and 8.3.
- **Mutation controls on the truncation refusal (7.1) and on `dimensions` handling (7.2):** present.
- **The final parent task is Feature Acceptance.** ✔
