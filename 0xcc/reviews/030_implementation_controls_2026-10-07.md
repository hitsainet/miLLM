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

## Feature Acceptance — non-hardware half (task 8)

### 8.1 Success criteria (FPRD §11) and user stories

| Criterion | Status | Evidence |
|---|---|---|
| SC-1 over-limit → 400 naming its index; `last`+`normalize` → unit vectors | **unit PASS**, hardware open (8.2) | `test_embedding_options.py::TestNoSilentTruncation` (real tokenizer, forward count 0), `test_embedding_options_route.py::TestAnOverLimitInputIsA400NamingItsIndex`, `TestPooling::test_last_with_normalize_returns_unit_vectors`; GGUF `test_an_over_limit_input_is_refused_before_the_engine_runs` |
| SC-2 `dimensions` → 400 before any load | **PASS** | `TestDimensionsIsRefusedBeforeLoad` (3 values × 2 engines, `load_model_and_wait` awaited 0×, resident model not run); M5, M6, X6 |
| SC-3 no new fields → today's vectors exactly | **PASS** | `TestPooling::test_the_default_is_todays_vector_exactly` (tiny real Llama, list equality), `test_embedding_pooling.py::TestTheDefaultDoesNotMove` (`torch.equal`, bf16 and fp32) |
| SC-4 two-SAE circuit: every mode equals no-SAE vectors | **unit PASS (suppression wiring)**, vector equality open (8.4) | `test_inference_service.py::test_embeddings_suppress_attached_sae[mean/last/cls]`; M11 |
| SC-5 GGUF refuses `pooling: last`, no load, no eviction | **PASS** | `TestPoolingOnAGgufRow::test_non_mean_is_refused_before_load_without_eviction[last/cls]`; service guard `test_pooling_other_than_mean_is_refused_by_the_service_too`; M7, X1 |
| SC-6 every wiring line has a test that fails without it | **PASS** | this record: 28 controls, 28 red |
| SC-7 hardware: unit norms, long input refused not cut | **open** | 8.2 |
| US-1 near-dedup embeds with `last`+`normalize` | unit PASS | as SC-1 |
| US-2 never embeds a prefix | unit PASS | as SC-1; M1, M2, M3, M4, X12, X16 |
| US-3 `dimensions` refused | PASS | as SC-2 |
| US-4 existing retrieval client unaffected | PASS | as SC-3 |
| US-5 GGUF `last` refused | PASS | as SC-5 |

### 8.5 Suite, lint, types

- `pytest tests/unit`: **4593 passed / 3 skipped / 0 failed** (baseline 4473 + 1 flake / 3 skipped).
- `ruff` and `mypy` are **not installed in `~/app/miLLM/venv`**; both were run from a scratch venv
  (`ruff`, `mypy --python-executable ~/app/miLLM/venv/bin/python`). New files: ruff clean. Touched
  files: ruff counts identical to `6cce090` (pre-existing debt). `mypy millm/`: **616 errors at
  `6cce090`, 616 after** — none introduced. Repo-wide pre-existing ruff debt: 1,798 findings.

### Hardware session list (operator)

1. **0.1** — on the backend image with a GGUF model loaded with embeddings: `Llama.n_ctx()`,
   `Llama.n_batch`, `tokenize` defaults, and what `create_embedding` does past `n_batch` and
   `n_ctx`. If `n_batch` does not bound one input, `_embedding_limit` may relax to `n_ctx()` alone
   (the current `min` is safe either way).
2. **0.2** — list every model row on the node and whether `_served_max_context` returns a value.
3. **8.2** — transformers model: unit vectors for `pooling: last, normalize: true`; an over-limit
   input refused naming its index; `dimensions` 400 before load; a GGUF model refuses
   `pooling: last` without evicting the resident model; one Open WebUI document upload still embeds.
4. **8.3** — cap measurement on the RTX 3090 (p95 s/input at 512 tokens over 64 inputs, LFM2.5-1.2B
   bf16 and the largest transformers embedding model); set `EMBEDDINGS_MAX_INPUTS` to the largest
   power of two with cap × p95 ≤ 30 s in [64, 2048]; update the setting's comment, `.env.example`
   and `manual/docs/reference/configuration.md`.
5. **8.4** — with a profile active and a circuit attached, embeddings equal those with no SAE.

### Document Inventory update (for the coordinator; this list does not edit CLAUDE.md or the PPRD)

Feature 30: FPRD/FTDD/FTID ✅ (unchanged); FTASKS ⏳ — tasks 1–7 and 0.3 done, 8.1/8.5/8.6 done;
0.1, 0.2, 8.2–8.4 need the GPU node. Not merged; not ✅.
