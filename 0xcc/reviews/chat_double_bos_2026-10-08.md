# Chat double-BOS fix — implementation and controls (2026-10-08)

Branch `fix/chat-double-bos` from `origin/main` `44e4c4a`. Operator-approved 2026-10-08. Not merged,
not deployed; the hardware check (§6) is the operator's.

## 1. The defect

Every chat path rendered the chat template and tokenized the render with the default
`add_special_tokens=True`. Templates that begin with `{{ bos_token }}` (Llama 3, gemma 2/3,
LFM2.5) therefore produced two BOS tokens. Production evidence (2026-10-07): `/api/probes/score` ids
on Llama-3.1-8B began `128000, 128000`; removing the duplicate moved a probe's AUROC on 540 rows
0.9604 → 0.9558. miStudio trains with `add_special_tokens=False`
(`backend/src/services/probe_monitor_render.py:266-270`).

The paths disagreed with each other: chat **scoring** hard-coded `False` (right for Llama 3,
**zero** BOS for a template that writes none, e.g. TinyLlama), `count_prompt_tokens` used `True` for
generation and `False` for scoring, `probe_turns` used `False`, and raw `/v1/completions`
generation ignored the request's `add_special_tokens` while scoring honoured it.

## 2. The rule, and how each real tokenizer behaves under it

`millm/services/prompt_encoding.py`:

* `rendered_chat_adds_special_tokens(tok, text)` → **False** iff `tok.bos_token` is a non-empty
  string and `text.startswith(tok.bos_token)`; otherwise **True** (the tokenizer's own default).
* `encode_rendered_chat` / `rendered_chat_ids` / `encode_rendered_chats` (padded batch; rows that
  disagree are encoded separately and joined with `tokenizer.pad`) apply it and refuse an
  explicit `add_special_tokens`.
* `encode_prompt(rendered_chat=False, add_special_tokens=…)` is raw text: the caller's flag.
* `llamacpp_completion_prompt` strips a leading BOS text from a continuation render only when
  `model.tokenize(b"", add_bos=True, special=True) == [token_bos()]` (llama.cpp would add one).

Measured on the real tokenizers (transformers 5.15.1, offline copies; one user turn, generation
prompt on):

| family | bos_token | render starts with BOS | tokenizer adds BOS to raw text | default `tokenizer(render)` | helper decision | helper ids[:3] | leading BOS |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | '<|begin_of_text|>' (128000) | True | True | [128000, 128000, 128006] (2 BOS) | add_special_tokens=False | [128000, 128006, 9125] | 1 |
| LFM2.5-1.2B-Instruct | '<|startoftext|>' (1) | True | True | [1, 1, 6] (2 BOS) | add_special_tokens=False | [1, 6, 6423] | 1 |
| gemma-3-12b-it | '<bos>' (2) | True | True | [2, 2, 105] (2 BOS) | add_special_tokens=False | [2, 105, 2364] | 1 |
| gemma-4-12b-it | '<bos>' (2) | True | False | [2, 105, 2364] (1 BOS) | add_special_tokens=False | [2, 105, 2364] | 1 |
| granite-4.1-8b | '<|end_of_text|>' (100257) | False | False | [100264, 882, 100265] (0 BOS) | add_special_tokens=True | [100264, 882, 100265] | 0 |
| Qwen2.5 (p2/tok/qwen) | None (None) | False | False | [151644, 8948, 198] (0 BOS) | add_special_tokens=True | [151644, 8948, 198] | 0 |
| TinyLlama-1.1B-Chat | '<s>' (1) | False | True | [1, 529, 29989] (1 BOS) | add_special_tokens=True | [1, 529, 29989] | 1 |
| Phi-4-mini-instruct | '<|endoftext|>' (199999) | False | False | [200021, 13225, 1354] (0 BOS) | add_special_tokens=True | [200021, 13225, 1354] | 0 |

Default `tokenizer(render)` duplicates on Llama 3.1, LFM2.5 and gemma 3; the helper yields exactly
one BOS on every family that uses one and none on those that do not. gemma-4's tokenizer does not
add a BOS to raw text, so it never duplicated. Not available offline: a Qwen3 or gemma-2 tokenizer
(gemma-2 is in the HF cache list but its snapshot was absent).

## 3. Sites changed

| Site | Before | After |
|---|---|---|
| `InferenceService.check_stream_admission` | `self._tokenizer(prompt)` | `encode_rendered_chat` |
| `InferenceService.count_prompt_tokens` (chat) | True for generation / False for scoring | `encode_rendered_chat` |
| `InferenceService.count_prompt_tokens` (text) | True for generation, flag for scoring | request flag always |
| `create_chat_completion` (n ≥ 1) | `self._tokenizer(prompt)` | `encode_rendered_chat` |
| `stream_chat_completion` | `self._tokenizer(prompt)` | `encode_rendered_chat` |
| `_generate_batch_chunk` (batched chat) | `self._tokenizer(prompts, padding…)` | `encode_rendered_chats` |
| `_chunk_batch_for_memory` (length estimate) | `self._tokenizer.encode(p)` | `encode_rendered_chat` |
| `_cbm_chat_completion`, `_cbm_stream_chat_completion` | `self._tokenizer.encode(prompt)` | `encode_rendered_chat` |
| `create_text_completion` (serial generation) | default True | `encode_prompt(rendered_chat=False, add_special_tokens=request.add_special_tokens)` |
| `_cbm_text_completion` | default True | same |
| `_score_prompts` / `_score_specs_packed` | caller's bool | `encode_prompt(rendered_chat=…)`; `ScoreSpec.rendered_chat` added |
| `_score_chat_completion` | `add_special_tokens=False` | `rendered_chat=True` |
| `batch/packing.run_packed_scoring` (chat rows) | `ScoreSpec(text, False, …)` | `rendered_chat=True` |
| `_llamacpp_continuation_prompt` | render + partial | `llamacpp_completion_prompt(...)` |
| `probe_scoring.ProbeInputPreparer._encode` | `tokenizer(text)` | `rendered_chat_ids` (the same function live uses) |

Downstream of these ids, unchanged and now consistent: probe prompt length
(`_probe_note_prompt_length`), the `last_user` span (`probe_turns` renders with
`add_special_tokens=False` and places the span by OFFSET into the served ids, so it resolves for any
BOS count — tested on every family), sensing history (`_sensing_mark_history` reads the same
`input_ids`), request activations (`_activations_begin(request, prompt_tokens)`).

Left as is, with reasons (pinned in the guard's `ALLOWED`): embeddings (`_embed_inputs`, raw text);
`probe_turns` span renders (miStudio's construction, byte-identical cases file); `probe_parity._drift`
uses `apply_chat_template(tokenize=True)`, informational only. `_llamacpp_prompt_tokens` is an
approximate usage count over raw message text (llama.cpp's own tokenizer). `create_chat_completion`
on llama.cpp is unaffected: llama-cpp-python's Jinja2 handler tokenizes its own render with
`add_bos=False` (reasoned from the library; the wheel is not installed here).

## 4. Tests and guards

* `tests/unit/services/test_chat_single_bos.py` — four real `PreTrainedTokenizerFast` families
  (template BOS + tokenizer BOS; tokenizer BOS only; no `bos_token`; unused `bos_token`) over a tiny
  real Llama. Ids are read off the model's first forward (pre-hook) for: non-streaming, streaming,
  n=2, batched (every row, unpadded), admission check + activation cap, CBM chat and stream, CBM text
  and serial text (`add_special_tokens` both ways), chat scoring, packed chat scoring, probe scoring
  == live ids (and assistant-ended inputs), `last_user` resolves on live ids, llama.cpp continuation,
  and real cached tokenizers (`MILLM_REAL_TOKENIZERS` or the HF cache; skips loudly when absent — run
  locally against all eight tokenizers above).
* `tests/unit/services/test_prompt_encoding_guard.py` — AST discovery of every direct tokenizer call
  under `millm/` must EQUAL a justified allowlist; 14 functions must CALL their helper; `probe_turns`
  must keep a literal `add_special_tokens=False`; chat scoring and batch packing must pass a literal
  `rendered_chat=True`.
* **Four existing tests pinned the defect** by computing their expected ids as `tokenizer(render)`:
  `test_probe_input_preparer.encode`, `test_probe_scoring::test_token_ids_echoed_only_when_asked`,
  `test_request_activations::test_two_interleaved_requests_each_get_only_their_own`, and
  `test_chat_scoring` asserted the literal `add_special_tokens=False`. Each now writes out the
  single-BOS expectation.
* Negative control on the whole fix: restoring `inference_service.py`, `probe_scoring.py` and
  `packing.py` to `44e4c4a` with the new tests in place → **15 failed** (sha256 restored and verified).

## 5. Mutation controls

Harness: back up, replace one exact string (asserting it occurs once and the mutation landed), run
the seven affected test files, restore, assert the sha256 equals the pre-mutation hash and the
original text is present. `git status` checked clean after the batch.

| Control | Result | Failing tests | Restored sha256 |
|---|---|---|---|
| M1_helper_always_true | red | 15 | `43a9fb612c31` |
| M2_live_site_bypass | red | 5 | `eea0d7851397` |
| M3_no_bos_branch_broken | red | 14 | `43a9fb612c31` |
| M4_probe_scoring_diverges | red | 7 | `75443d2326c8` |
| M5_batch_decision_true | red | 2 | `43a9fb612c31` |
| M6_chat_scoring_flag | red | 4 | `eea0d7851397` |
| M7_packing_flag | red | 1 | `3826d3082efa` |
| M8_packed_scorer_ignores_flag | red | 1 | `eea0d7851397` |
| M9_text_generation_ignores_flag | red | 1 | `eea0d7851397` |
| M10_llamacpp_no_strip | red | 2 | `43a9fb612c31` |
| M11_count_tokens_bypass | red | 3 | `eea0d7851397` |
| M12_admission_bypass | red | 3 | `eea0d7851397` |
| M13_cbm_text_ignores_flag | **SURVIVED** | 0 | `eea0d7851397` |
| M14_count_text_ignores_flag | red | 1 | `eea0d7851397` |
| M13_rerun_negative_control | red | 1 | `eea0d7851397` |

Required controls: M1 (helper → `add_special_tokens=True`), M2 (`create_chat_completion` routed
around the helper — red in the AST guard twice and in behaviour), M3 (no-BOS-template branch broken),
M4 (probe scoring diverges from live). **One survivor: M13** — `_cbm_text_completion` ignoring
`add_special_tokens` had no test. Closed by `TestContinuousBatchingText`; re-run as a negative control:
red. M7 is caught by the guard only (the packing helper's behaviour is caught by M8's test).

## 6. Hardware check (operator, after deploy)

On the GPU node with Llama-3.1-8B loaded, two probes armed:

1. **Live chat ids.** `POST /v1/chat/completions` with `return_sae_activations` (positions
   `prompt`) for a fixed prompt, e.g. `[{"role":"user","content":"My brother just lost his flat."}]`,
   `max_tokens: 8`, `temperature: 0`. The returned positions' `token_id`s must begin
   `128000, 128006` — exactly one 128000. `usage.prompt_tokens` must be 1 lower than the same
   request before deploy.
2. **Probe scoring equals live.** `POST /api/probes/score` with the same `messages` and
   `return_token_ids: true`: ids begin with one 128000 and EQUAL step 1's prompt ids.
3. **Armed probes' scores before/after.** Record both armed probes' `probe_events` scores (all
   windows) for the fixed prompt before deploy and after; the after scores should equal
   `/api/probes/score` on the single-BOS ids, and the before scores should equal scoring the old
   double-BOS ids sent as `token_ids` (`[128000] + after_ids`). Report the per-window deltas.
4. Optional: gemma or LFM2.5 if loaded — one BOS (`2` / `1`) at position 0.

## 7. Suites

`tests/unit`: **5034 passed / 12 skipped / 0 failed** (the first full run after the fix had 1
failure — the activations test that pinned the duplicate — now updated). Mirror view (`0xcc/` and `docs/` stripped) not
run: no new test reads a document.
