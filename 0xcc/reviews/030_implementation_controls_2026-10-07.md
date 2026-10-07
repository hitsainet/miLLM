# Feature 30 — Implementation Controls Record (2026-10-07)

Branch `feat/030-embedding-options`, cut from `main` at `6cce090`. Worktree `~/app/miLLM-030`.
`millm.__file__` verified as `/home/x-sean/app/miLLM-030/millm/__init__.py` (venv `~/app/miLLM/venv`).

**Baseline** (before any change, `pytest tests/unit`): **4473 passed / 1 failed / 3 skipped**. The one
failure (`test_unload_clears_probe_hooks.py::TestTheHangGuardWritesTheRowsNow::test_it_calls_the_helper_and_not_the_undefined_global`)
passed when re-run alone and in both later full runs; the baseline run overlapped the first edits to
the tree, so it is recorded as contamination or a flake, not a defect of `6cce090`.
**After tasks 1–6:** 4593 passed / 3 skipped / 0 failed.

## Procedure

Runner: `scratchpad/impl-030/ctl/mutate.py` (absolute paths; never `cd`s). For each control: back up
the file; make ONE exact-string replacement that must occur exactly once; confirm it LANDED (file
hash changed and content equals the mutated text); run the control set; require a red; restore from
the backup; verify the restore by sha256 against the pre-mutation hash AND by re-counting the
original string (exactly one). The runner stops on a failed restore. `git status` was clean after
the run.

Control set (11 files, 795 tests green unmutated, ~31 s): `ml/test_embedding_pooling.py`,
`services/test_embedding_options.py`, `services/test_gguf_refusals.py`,
`services/test_inference_service.py`, `api/test_embedding_options_route.py`,
`api/test_request_policy.py`, `api/test_request_policy_coverage.py`,
`api/test_context_length_refusal.py`, `api/test_openai_schemas.py`, `api/test_error_map_complete.py`,
`api/test_exception_handlers.py`.

M1–M12 are the FTID §8.5 controls (M1–M4 truncation refusal, task 7.1; M5–M7 `dimensions` and
policy, task 7.2; M8–M12 task 7.3). X1–X16 are additional wiring and latent-gap controls on lines
the FTID does not name.

## Results — 28 controls, 28 red, 0 survived first time

| # | Mutation | Result | Landed | Restore |
|---|---|---|---|---|
| M1 | `_embed_inputs`: `truncation=False` → `truncation=True` | RED (6 failed: test_embedding_options.py) | ✔ | sha ✔ grep ✔ |
| M2 | `_embed_inputs`: length check moved inside the per-input loop (checks the prefix before each forward) | RED (3 failed: test_embedding_options.py) | ✔ | sha ✔ grep ✔ |
| M3 | `_embed_inputs`: `_check_embedding_lengths(...)` call deleted | RED (6 failed: test_context_length_refusal.py, test_embedding_options.py) | ✔ | sha ✔ grep ✔ |
| M4 | `_llamacpp_embeddings`: `_check_embedding_lengths(...)` call deleted | RED (1 failed: test_gguf_refusals.py) | ✔ | sha ✔ grep ✔ |
| M5 | policy: both `dimensions` embeddings cells → `HONOURED` | RED (10 failed: test_embedding_options_route.py, test_request_policy.py) | ✔ | sha ✔ grep ✔ |
| M6 | route: `apply_request_policy` moved after the auto-load | RED (71 failed: test_embedding_options_route.py, test_request_policy_coverage.py) | ✔ | sha ✔ grep ✔ |
| M7 | policy: `NEUTRAL["pooling"]` → `lambda v: True` | RED (12 failed: test_embedding_options_route.py, test_request_policy_coverage.py) | ✔ | sha ✔ grep ✔ |
| M8 | `pool_hidden` `mean`: mask branch dropped (`if True:` → always `hidden.mean(1)`) | RED (2 failed: test_embedding_pooling.py) | ✔ | sha ✔ grep ✔ |
| M9 | live handler: `param=…` → `param=None` | RED (14 failed: test_context_length_refusal.py, test_embedding_options_route.py, test_error_map_complete.py, test_request_policy_coverage.py) | ✔ | sha ✔ grep ✔ |
| M10 | `_embed_inputs`: `options.pooling` → fixed `"mean"` at the `pool_hidden` call | RED (4 failed: test_embedding_options.py) | ✔ | sha ✔ grep ✔ |
| M11 | `_embed_inputs`: `self._unsteered()` removed from the forward | RED (3 failed: test_inference_service.py) | ✔ | sha ✔ grep ✔ |
| M12 | schema validator: cap check removed (`if False:`) | RED (3 failed: test_embedding_options_route.py, test_openai_schemas.py) | ✔ | sha ✔ grep ✔ |
| X1 | `_llamacpp_embeddings`: service pooling guard disabled | RED (1 failed: test_gguf_refusals.py) | ✔ | sha ✔ grep ✔ |
| X2 | `_llamacpp_embeddings`: `finalize_vector(vector, False)` | RED (1 failed: test_gguf_refusals.py) | ✔ | sha ✔ grep ✔ |
| X3 | `_embed_inputs`: `finalize_vector(pooled, False)` | RED (2 failed: test_embedding_options.py) | ✔ | sha ✔ grep ✔ |
| X4 | schema validator: empty-element check disabled | RED (5 failed: test_embedding_options_route.py, test_openai_schemas.py) | ✔ | sha ✔ grep ✔ |
| X5 | schema validator: empty-list check disabled | RED (2 failed: test_embedding_options_route.py, test_openai_schemas.py) | ✔ | sha ✔ grep ✔ |
| X6 | `apply_request_policy`: `model_name=None` (refusal stops naming the model) | RED (6 failed: test_embedding_options_route.py) | ✔ | sha ✔ grep ✔ |
| X7 | error map: `EMBEDDING_VECTOR_INVALID` row deleted | RED (2 failed: test_error_map_complete.py) | ✔ | sha ✔ grep ✔ |
| X8 | `pool_hidden` `last`: fixed last index (ignores the mask) | RED (1 failed: test_embedding_pooling.py) | ✔ | sha ✔ grep ✔ |
| X9 | `pool_hidden` `cls`: fixed index 0 (ignores the mask) | RED (2 failed: test_embedding_pooling.py) | ✔ | sha ✔ grep ✔ |
| X10 | `_check_embedding_lengths`: unbounded list (`listed = over`) | RED (1 failed: test_embedding_options.py) | ✔ | sha ✔ grep ✔ |
| X11 | `_check_embedding_lengths`: string input named `input[0]` | RED (2 failed: test_context_length_refusal.py, test_embedding_options.py) | ✔ | sha ✔ grep ✔ |
| X12 | `_embedding_limit` (GGUF): `n_ctx()` only, `n_batch` bound dropped | RED (1 failed: test_gguf_refusals.py) | ✔ | sha ✔ grep ✔ |
| X13 | `create_embeddings`: `_embed_inputs` called outside `_admit()` | RED (1 failed: test_embedding_options.py) | ✔ | sha ✔ grep ✔ |
| X14 | `create_embeddings`: `usage = len(counts)` | RED (2 failed: test_embedding_options.py) | ✔ | sha ✔ grep ✔ |
| X15 | policy: `NEUTRAL["normalize"]` → `lambda v: True` | RED (8 failed: test_request_policy_coverage.py) | ✔ | sha ✔ grep ✔ |
| X16 | `_llamacpp_embeddings`: counts not taken from the instance tokenizer (all 0) | RED (1 failed: test_gguf_refusals.py) | ✔ | sha ✔ grep ✔ |

## Findings

- **F1 — Feature 25's every-entry HTTP test cannot catch M5, by construction (a discrepancy with
  030_FTASKS 7.2, not a survivor).** The task list expected M5 to turn
  `test_request_policy_coverage.py::test_the_table_is_enforced_over_http` red. It stays green: that
  test reads each cell's EXPECTED outcome from `OUTPUT_CHANGING` itself, so flipping a cell to
  `HONOURED` changes the expectation along with the behaviour and the request is (correctly, for the
  mutated table) not refused. It enforces that the server obeys the table; it cannot check the
  table's contents. M5 is caught by the two tests that pin the content independently:
  `test_embedding_options_route.py::TestDimensionsIsRefusedBeforeLoad` (6 cases) and the new
  `test_request_policy.py::test_dimensions_is_refused_for_want_of_a_declaration` (2 engines). Any
  future row whose value matters needs a content test of its own; the every-entry test is not one.
- **F2 — the FTID's `param` plumbing (TD10) was already built by Feature 25.** `millm_error_handler`
  forwards `exc.details["param"]`; the new errors use it and no `openai_param` attribute was added.
  M9 therefore mutates Feature 25's line; it turned 14 tests red, 6 of them Feature 30's.
- **F3 — a passing GGUF suite measured nothing about length before this feature.** The existing
  `TestGGUFEmbeddings` fixture is a bare `MagicMock`: `len(handle.tokenize(...))` is 0 and
  `int(handle.n_ctx())` is 1, so any length check passes on it. The fixture now states `n_ctx`,
  `n_batch` and a word tokenizer; X12 and X16 prove the over-limit test reads them.

## Hardware controls not run here

0.1 (llama-cpp-python truncation and names on the backend image), 8.2–8.4. See 030_FTASKS.
