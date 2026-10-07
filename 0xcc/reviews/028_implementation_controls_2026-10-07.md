# Feature 28 — implementation controls and acceptance evidence (2026-10-07)

Branch `feat/028-inline-steering` (cut from `main` at `c3b4cad`). Scope: every parent task before
Feature Acceptance, plus the non-hardware part of 0.0 and of Feature Acceptance (8.x).

## How each control was run

`controls/mutate.py` (scratchpad): back up the file, apply ONE textual edit (the OLD text must occur
exactly once, and the edit must land), run the affected suite, restore the original bytes, then
verify the restore by **sha256** of the whole file and by **re-grepping** that the OLD text is back
exactly once. Any restore failure aborts. `git status --short millm` was confirmed clean after the
run. Affected suite (16 files, `-x`): `test_steering_state`, `test_inline_steering`,
`test_inline_steering_real_model`, `test_steering_report`, `test_steering_report_every_path`,
`test_steering_header_routes`, `batch/test_runner`, `test_request_policy`,
`test_request_policy_coverage`, `test_openai_schemas`, `test_chat_scoring`, `test_steering_epoch`,
`test_steering_dial_serialised`, `test_request_intensity`, `test_gguf_refusals`,
`test_inference_service`. (Deviation from FTASKS 8.2's "full suite" per control: 41 × 4–6 min was
not spent; every control's target is exercised by the files listed, and the full suite was run
green before and after the control session.)

FTASKS 8.2's named controls are (a) `a_clamp`, (b) `b_restore`, (c) `c_epoch_guard`,
(d) `d1/d2/d3`, (e) `e1/e2`, (f) `f_override`, (g) `g1–g3`, (h) `h1/h2`, (i) `i_text_dispatch`,
(j) `j_restore_branch`. The coordinator's extra classes: clamp/restore (`a`, `b`, `c`, `j`, `hh`),
header kind (`e1`, `e2`, `dd`), hash serialisation — float formatting `k`, ordering `l`, dropped
zeros `m`, `m2`, trailing LF `o`, published vector `ff` — "computed from hooks, not echoed"
(`d1`, `d2`, `d3`, `s`), scoring `none` (`n`, `n2`, `gg`).

## Results — 41 controls; 4 survived first time

| ID | File | Mutation | Verdict | First red test | sha after restore |
|---|---|---|---|---|---|
| a_clamp | `millm/services/inference_service.py` | skip clamp_steering on the inline path | **RED** | `tests/unit/services/test_inline_steering.py::TestInlineApply::test_strengths_are_clamped_and_recorded` | 017e97028f71 |
| b_restore | `millm/services/inference_service.py` | skip the restore in _finish_request_steering | **RED** | `tests/unit/services/test_inline_steering.py::TestRestore::test_finish_captures_before_restoring` | 017e97028f71 |
| c_epoch_guard | `millm/services/inference_service.py` | restore without the epoch guard | **RED** | `tests/unit/services/test_inline_steering.py::TestRestore::test_an_epoch_bump_mid_request_skips_the_restore` | 017e97028f71 |
| d1_record_equality | `millm/services/steering_report.py` | label from the record without value equality | **RED** | `tests/unit/services/test_steering_report.py::TestHonesty::test_the_record_lies_and_the_snapshot_wins` | 1b61955dacce |
| d2_hash_from_record | `millm/services/steering_report.py` | hash the request record, not the snapshot | **SURVIVED** | `` → negative control after the new test: **SURVIVED** () | 1b61955dacce |
| d3_echo_request | `millm/services/steering_report.py` | report the request record instead of the snapshot (echo) | **RED** | `tests/unit/services/test_steering_report.py::TestHonesty::test_the_record_lies_and_the_snapshot_wins` | 1b61955dacce |
| e1_kind_serializer | `millm/core/steering_state.py` | hard-code the item kind to inline (serialiser) | **RED** | `tests/unit/core/test_steering_state.py::test_none_and_empty_serialise_as_none` | 698710991aed |
| e2_kind_reader | `millm/services/steering_report.py` | profile record labelled inline | **RED** | `tests/unit/services/test_inline_steering_real_model.py::test_inline_equals_an_equivalent_saved_profile` | 1b61955dacce |
| f_override | `millm/services/inference_service.py` | drop steering from _has_steering_override | **RED** | `tests/unit/services/test_inline_steering.py::TestSerialRouting::test_a_steering_request_never_routes_to_the_batching_manager[steering0-chat]` | 017e97028f71 |
| g1_header | `millm/api/provenance.py` | remove the header line (provenance, both routes + batch) | **RED** | `tests/unit/api/test_steering_header_routes.py::TestHeaderPerKind::test_none[/v1/chat/completions-base0]` | 51b67af9d569 |
| g2_preload_chat | `millm/api/routes/openai/chat.py` | remove the chat pre-load refusal | **RED** | `tests/unit/api/test_steering_header_routes.py::TestRefusedBeforeLoad::test_a_non_empty_set_on_a_non_resident_model[/v1/chat/completions-base0]` | a21c04661a7e |
| g3_preload_comp | `millm/api/routes/openai/completions.py` | remove the completions pre-load refusal | **RED** | `tests/unit/api/test_steering_header_routes.py::TestRefusedBeforeLoad::test_a_non_empty_set_on_a_non_resident_model[/v1/completions-base1]` | a4f7bb527955 |
| h1_stream_chunk | `millm/services/inference_service.py` | remove the serial stream chunk | **RED** | `tests/unit/services/test_steering_report_every_path.py::test_every_generation_site_publishes_a_report[stream_chat_completion-nothing-attached]` | 017e97028f71 |
| h2_cbm_stream_chunk | `millm/services/inference_service.py` | remove the CBM stream chunk | **RED** | `tests/unit/services/test_steering_report_every_path.py::test_every_generation_site_publishes_a_report[_cbm_stream_chat_completion-nothing-attached]` | 017e97028f71 |
| i_text_dispatch | `millm/services/inference_service.py` | drop the dispatcher call in create_text_completion | **RED** | `tests/unit/services/test_inline_steering_real_model.py::test_explicit_unsteered_under_an_active_profile_equals_detached[_completion_text]` | 017e97028f71 |
| j_restore_branch | `millm/services/inference_service.py` | revert the restore branch to circuit | **RED** | `tests/unit/services/test_inline_steering.py::TestInlineApply::test_a_mutation_raising_mid_apply_restores_before_reraising` | 017e97028f71 |
| k_float_format | `millm/core/steering_state.py` | decimal text instead of the bit pattern | **RED** | `tests/unit/core/test_steering_state.py::test_published_vector_canonical_form_byte_for_byte[TV-1]` | 698710991aed |
| l_ordering | `millm/core/steering_state.py` | canonical lines unsorted | **RED** | `tests/unit/core/test_steering_state.py::test_published_vector_canonical_form_byte_for_byte[TV-3]` | 698710991aed |
| m_zeros | `millm/core/steering_state.py` | keep zeros in the applied set | **RED** | `tests/unit/core/test_steering_state.py::test_zeros_are_excluded_from_the_applied_set_and_the_hash` | 698710991aed |
| m2_zeros_canonical | `millm/core/steering_state.py` | keep zeros in the canonical form | **RED** | `tests/unit/core/test_steering_state.py::test_zeros_are_excluded_from_the_applied_set_and_the_hash` | 698710991aed |
| n_scoring_none | `millm/services/inference_service.py` | scoring publishes no report | **RED** | `tests/unit/services/test_steering_report_every_path.py::test_scoring_publishes_none_even_with_live_steering[text]` | 017e97028f71 |
| n2_scoring_provenance | `millm/api/provenance.py` | provenance scoring shortcut removed | **SURVIVED** | `` → negative control after the new test: **RED** (tests/unit/services/batch/test_runner.py::test_a_packed_scoring_line_says_none) | 51b67af9d569 |
| o_lf | `millm/core/steering_state.py` | no LF after the last line | **RED** | `tests/unit/core/test_steering_state.py::test_published_vector_canonical_form_byte_for_byte[TV-1]` | 698710991aed |
| q_others_disabled | `millm/services/inference_service.py` | other entries left steering under inline | **RED** | `tests/unit/services/test_inline_steering.py::TestInlineApply::test_target_runs_exactly_the_set_and_others_are_disabled` | 017e97028f71 |
| r_unsteered | `millm/services/inference_service.py` | explicit unsteered disables nothing | **RED** | `tests/unit/services/test_inline_steering.py::TestExplicitUnsteered::test_every_entry_is_disabled_and_values_kept` | 017e97028f71 |
| s_changed | `millm/services/steering_report.py` | changed never set | **RED** | `tests/unit/services/test_steering_report.py::TestHonesty::test_an_epoch_move_marks_every_member_changed` | 1b61955dacce |
| t_claims_fail_open | `millm/services/steering_report.py` | claims unreadable fails open | **RED** | `tests/unit/services/test_steering_report.py::TestCircuitLabel::test_unreadable_claims_fail_closed` | 1b61955dacce |
| v_stream_dry_run | `millm/api/routes/openai/chat.py` | remove the streaming dry run | **RED** | `tests/unit/api/test_steering_header_routes.py::TestStreaming::test_a_bad_inline_set_is_a_400_before_the_stream_commits[steering0-sae_not_attached]` | a21c04661a7e |
| w_llamacpp_text | `millm/services/inference_service.py` | llama.cpp text completion accepts steering (direct caller) | **SURVIVED** | `` → negative control after the new test: **RED** (tests/unit/services/test_inline_steering_real_model.py::test_llamacpp_text_completion_refuses_steering_for_a_direct_caller) | 017e97028f71 |
| x_clear | `millm/services/inference_service.py` | inline set merged into live values (no clear) | **RED** | `tests/unit/services/test_inline_steering.py::TestInlineApply::test_target_runs_exactly_the_set_and_others_are_disabled` | 017e97028f71 |
| y_admission_epoch | `millm/services/inference_service.py` | no admission epoch recorded | **RED** | `tests/unit/services/test_inline_steering.py::TestDispatcher::test_the_admission_epoch_is_recorded_first` | 017e97028f71 |
| aa_cbm_report | `millm/services/inference_service.py` | CBM chat publishes no report | **RED** | `tests/unit/services/test_steering_report_every_path.py::test_every_generation_site_publishes_a_report[_cbm_chat_completion-nothing-attached]` | 017e97028f71 |
| bb_preload_empty | `millm/api/routes/openai/load_policy.py` | pre-load refusal also refuses features: [] | **RED** | `tests/unit/api/test_steering_header_routes.py::TestRefusedBeforeLoad::test_the_empty_set_on_a_non_resident_model_is_not_refused[/v1/chat/completions-base0]` | 60e36991cadd |
| cc_select_registry | `millm/services/inference_service.py` | unknown sae_id falls back to the first entry | **RED** | `tests/unit/services/test_inline_steering.py::TestInlineApply::test_an_unattached_sae_id_is_refused_naming_it` | 017e97028f71 |
| dd_profile_source | `millm/services/inference_service.py` | dial over the active profile labelled source=request | **SURVIVED** | `` → negative control after the new test: **RED** (tests/unit/services/test_inline_steering_real_model.py::test_a_dial_over_the_active_profile_is_labelled_source_active) | 017e97028f71 |
| ee_mismatch_log | `millm/services/steering_report.py` | profile_sae_mismatch never logged | **RED** | `tests/unit/services/test_steering_report.py::TestLabels::test_active_profile_by_equal_values` | 1b61955dacce |
| ff_doc_vector | `manual/docs/api/openai-compatible.md` | published TV-4 hash drifts | **RED** | `tests/unit/core/test_steering_state.py::test_the_api_reference_publishes_exactly_these_vectors` | 75e02edcaf31 |
| gg_policy_scoring | `millm/api/request_policy.py` | steering honoured on scoring (X-09) | **RED** | `tests/unit/api/test_steering_header_routes.py::TestRefusedBeforeLoad::test_any_steering_field_on_completion_scoring[steering-value0]` | 4bdcc4a20770 |
| hh_rollback | `millm/services/inference_service.py` | no rollback on a partial inline apply | **RED** | `tests/unit/services/test_inline_steering.py::TestInlineApply::test_a_mutation_raising_mid_apply_restores_before_reraising` | 017e97028f71 |
| ii_validate_index | `millm/services/inference_service.py` | index range not validated before mutating | **RED** | `tests/unit/services/test_inline_steering.py::TestInlineApply::test_an_index_past_d_sae_is_refused_and_nothing_changes` | 017e97028f71 |
| jj_unset | `millm/services/steering_report.py` | an unset (NaN) plan intensity reaches the header | **RED** | `tests/unit/services/test_steering_report.py::TestCircuitLabel::test_an_unset_plan_intensity_falls_back_to_the_row` | 1b61955dacce |
### The four survivors

1. **`n2_scoring_provenance` — a real gap.** Removing provenance's X-09 shortcut left the suite
   green, because every scoring test went through the service's scoring entry point, which
   publishes `none` itself. The Batch API's PACKED scorer builds its body without that entry point
   and publishes no report, so packed scoring lines would have said `unknown;reason=read_failed`.
   New test `batch/test_runner.py::test_a_packed_scoring_line_says_none`; negative control re-run:
   **RED**.
2. **`w_llamacpp_text` — a real gap (defence in depth).** `_llamacpp_text_completion` gained a
   steering refusal (text completions only got steering fields in 028); the policy table refuses
   steering on a GGUF row over HTTP, so nothing tested the direct-caller path. New test
   `test_inline_steering_real_model.py::test_llamacpp_text_completion_refuses_steering_for_a_direct_caller`
   (asserts the `param` and that nothing generated); re-run: **RED**.
3. **`dd_profile_source` — a real gap.** A dial with no `profile` scales the ACTIVE profile; the
   record's `source` could be hard-coded to `request` with every test green. New test
   `test_a_dial_over_the_active_profile_is_labelled_source_active` (hash of stored × 0.5 computed
   independently); re-run: **RED**.
4. **`d2_hash_from_record` — an equivalent mutation, measured, not argued away.** The line is
   reachable only after `applied_set(record.applied) == entry.applied`, and `canonical_set_form`
   drops zeros (the only difference `applied_set` can make to already-clamped record values), so
   hashing the record or the snapshot gives the same bytes on every input that reaches it. Re-run
   after the other fixes: still survives. The honesty property it was meant to guard is pinned by
   `d1` and `d3` (both RED): the record never labels an entry whose values differ, and an echo of
   the request is caught.

One further control (`jj_unset`) was added after `tests/integration/test_single_serving_derivation.py`
flagged a fourth `plan_for` consumer: an unset (NaN) plan intensity must not reach the header —
RED against the new test `test_an_unset_plan_intensity_falls_back_to_the_row`.

A planned `r2` (unsteered by suppression instead of disabling) was dropped before running:
suppression is per-thread and the generation runs in a worker thread, so the mutation would not
suppress the forward at all — a control that cannot change the answer proves nothing.

## Reachability (live app, payload and call count)

- Header: `tests/unit/api/test_steering_header_routes.py` — the real `create_app()`, exact header
  VALUES per kind (none, inline+clamp, profile, dial→manual, batched, multi-prompt, scoring `none`,
  read failure → `unknown;reason=read_failed`), `get_list(...) == [one value]` on batched and
  multi-prompt (exactly one report). Removing the header line (`g1`) → RED.
- Pre-load refusal: `load_model_and_wait.call_count == 0` on refusal and `== 1` for `features: []`
  (`g2`, `g3`, `bb` → RED).
- Dispatcher: `TestDispatcher` asserts each branch's call, its arguments and the count; AST guard
  `test_only_the_dispatcher_applies_and_only_the_finisher_restores` pins the exact call counts.
- Every discovered generation site (10, from `tests/support/generation_entry_points.py`) publishes
  the exact expected report, nothing-attached and live-steering, and every stream ends in exactly
  one `millm_steering` chunk last before `[DONE]` (`h1`, `h2`, `aa`, `n` → RED).
- Batch lines (FR-26.10.1): `batch/test_runner.py::test_a_batch_line_carries_the_synchronous_steering_header`
  (inline, profile, none: line header == synchronous header).

## The published vectors (X-07)

Reproduced independently before pinning: stand-alone Python (`struct` + `hashlib`, no miLLM
import) and Node (`DataView.setFloat64` + `crypto`), both byte-identical to FTDD §5.3:

| ID | Hash |
|---|---|
| TV-1 | `sha256:a4eae730e5105f422b93abeece7e03bda3c29b096c07aa9cee6f247e12844105` |
| TV-2 | `sha256:cc6c48faa720096e5c65fc1b2afab1b9b87659fb4465f20db421a2ed7fed78a7` |
| TV-3 | `sha256:b843912201c5c18f872976e35af288e5f484d81e31a03a1dc16c3b9dbd82e710` |
| TV-4 | `sha256:3f51779e6e33ee248acf6f520cb9475b4ef62f68275060bb69d2e228033b78df` |

`test_the_api_reference_publishes_exactly_these_vectors` cross-checks the API reference against the
pinned literals (`ff` → RED); it skips loudly where the manual is absent.

## Acceptance walk (FPRD §11, US-1 – US-7), CPU evidence

| Item | Evidence | Status |
|---|---|---|
| §11.1 inline == profile, greedy; header per kind, streaming and not | `test_inline_equals_an_equivalent_saved_profile` (tiny real Llama + real hooked SAE, separate literals); route + every-path tests for none/profile/inline/manual/circuit | CPU ✅; hardware 8.6 open |
| §11.2 steered pair hashes match recomputation | TV-1/TV-2 pinned; inline header hash == `independent_hash` of the request in route tests | CPU ✅; hardware 8.6(c) open |
| §11.3 `features: []` under active profile == detached | `test_explicit_unsteered_under_an_active_profile_equals_detached[_chat_text]` | CPU ✅; hardware 8.6(d) open |
| §11.4 text completion opt-out, next one steered again | same test, `[_completion_text]` | ✅ |
| §11.5 every wiring line fails when removed, payload + count | controls above | ✅ (one equivalent survivor, d2) |
| US-1 steered pair without profiles | route tests + TV-1/TV-2 | CPU ✅ |
| US-2 inline equals profile | see §11.1 | CPU ✅ |
| US-3 explicitly unsteered | see §11.3 | CPU ✅ |
| US-4 honest under an active profile | `test_a_live_global_profile_is_reported_with_its_intensity` (λ = 0.75) | ✅ |
| US-5 text completion opt-out | see §11.4 | ✅ |
| US-6 streaming chunk | `TestStreaming`, every-path stream sites | ✅ |
| US-7 miStudio `millm_generate` | miStudio 034's code; header is verbatim-forwardable | not in this repo |

## Suites, lint, benchmark

- Backend `tests/unit`: **4747 passed / 5 skipped** at `c3b4cad` (measured before any change) →
  **4966 passed / 5 skipped / 0 failed** after (final, with the control-session tests). `tests/integration` + `tests/schema`: 1 failure found
  (`test_only_the_expected_call_sites_build_plans`, a fourth `plan_for` consumer) — updated with a
  pinned guard (`jj_unset`); 529 passed after.
- admin-ui: 471 passed / 41 files, before and after (no admin-ui change).
- `http-sfv==0.9.9` grammar tests: green with the package on PYTHONPATH (scratch install); the
  shared dev venv lacks it, so they skip loudly there; CI installs the `dev` extra.
- ruff / mypy: neither is installed in the shared venv; run from a scratch install. The repo
  carries ~2,100 pre-existing ruff findings and ~640 mypy errors; **no mypy error on any line 028
  added**, and the new modules' ruff findings are the repo's own `Optional` style and unused
  fixture/lambda arguments in tests.
- Benchmark (8.4, `tests/performance/test_steering_report_overhead.py`, 200 serial chat requests,
  tiny model, CPU): record-labelled inline request p95 delta **0.053 ms**; worst case (nothing
  claims the entry, so circuit + active-profile reads, SQLite) p95 delta **1.589 ms**. Target
  < 5 ms met; the Postgres read on the node is to be measured in the hardware session.

## Discrepancies recorded (code wins)

1. Circuits are read strictly in the reader (every full-serving circuit), not via the memoised
   `_steering_circuit()`, which fails open (a DB blip would relabel circuit steering `manual`).
2. Ambiguous SAE selection raises `InvalidParameterError` (400 `invalid_parameter`), not the FTID's
   `ValidationError`.
3. `reset_steering_memo()` on completions already runs via `provenance.pre_generation`.
4. The discovery helper was extracted from Feature 27's test into `tests/support/`.
5. Route tests are in `tests/unit/api/`, not `tests/integration/api/` (CI runs `tests/unit` only).
6. No OpenAI-API page exists under `manual/docs/features/`; the manual section went into
   `features/feature-steering.md`.
7. The steering chunk is emitted on the CBM and llama.cpp streams too (FR-28.3.1 coverage), after
   the activations chunk where one is present.
8. Tasks 3.0–6.0 are one commit (they share `inference_service.py`); 2.0's commit alone accepts
   the field before the next commit honours it.
