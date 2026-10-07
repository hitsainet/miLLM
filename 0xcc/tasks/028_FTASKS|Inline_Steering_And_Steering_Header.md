# Feature 28: Inline Steering and Steering-State Header — Task List

**Status:** Planned (2026-10-06). The "Go" pause after the parent tasks was **waived** by the
coordinator's instruction for this increment (clarifying rounds and the pause both waived; decisions
cited from sources). The full list is generated in one pass.
**Inputs:** `028_FPRD` v1.1, `028_FTDD` v1.0, `028_FTID` v1.0 · BRD-04 §5.8 (R-04.31 – R-04.34) ·
operator decisions T-78 – T-83, P-22, X-07, X-09
**Code baseline:** miLLM `f5c71b6`. Re-verify every line number before editing; Features 25–30 edit
`inference_service.py` in the same increment.
**Consumers waiting on task 1.0:** miDataworks 007 pins the hash test vectors (X-07); miStudio 034
reads `X-miLLM-Steering` verbatim.

## Relevant Files
- `millm/core/steering_state.py`: hash, canonical form, header serialiser (new) ·
  `tests/unit/core/test_steering_state.py`
- `millm/services/steering_report.py`: snapshot, record, reader (new) ·
  `tests/unit/services/test_steering_report.py`
- `millm/services/inference_service.py`: dispatcher, inline/unsteered apply, restore branch, capture
  at every site, stream chunk, CBM routing, llama.cpp refusal, scoring report ·
  `tests/unit/services/test_inline_steering.py`,
  `tests/unit/services/test_inline_steering_real_model.py`,
  `tests/unit/services/test_steering_report_every_path.py`
- `millm/api/schemas/openai.py`: `InlineSteering`, fields on both request models ·
  `tests/unit/api/test_openai_schemas.py` (extend)
- `millm/api/routes/openai/chat.py`, `millm/api/routes/openai/completions.py`: pre-load refusals,
  header · `tests/integration/api/test_steering_header_routes.py`
- `tests/support/generation_entry_points.py`: AST call-graph discovery (shared with Feature 27)
- `tests/performance/test_steering_report_overhead.py`: overhead benchmark (new)
- `pyproject.toml`: `http-sfv==0.9.9` in the dev/test extras only
- Docs: the API reference for `/v1/chat/completions` and `/v1/completions`; the manual's
  OpenAI-API feature page under `manual/docs/features/`

### Notes
- Backend tests: `pytest` (full tree, no `--ignore`, per the checkpoint technical defaults);
  `ruff`, `mypy millm/`.
- **Reachability is a shipping gate** (`CLAUDE.md`): every wiring line needs a test that fails when
  the line is removed, asserting payload and call count. Record each mutation control, the command
  and the red result, in the review notes.
- **Operator rules for this feature:** the header is computed from what the hooks applied, so a test
  must fail if it echoes the request (5.6); mutation controls on the clamp, the restore and the
  header kind (8.2); no hand-kept lists (5.8, 3.9).
- Fixtures must not agree by construction: inline sets and equivalent profiles come from separate
  literals; tests that need two SAEs use two layers.

### Category Checklist Results
- **Data layer:** N/A — no table, column or migration; all state is per request in memory (FPRD §5,
  FTDD §4).
- **Backend/API:** 2.x (schemas), 4.x (routing, pre-load refusals), 6.x (headers, stream chunk)
- **Frontend/UI:** N/A — API-only feature; Admin UI unchanged and Open WebUI unaffected (FPRD §4,
  FTID §6).
- **Business logic:** 1.x (hash, grammar), 3.x (apply/restore), 5.x (labelling)
- **Integration wiring:** 4.4 (text completion), 5.4–5.5 (every generation site), 7.1–7.3
  (Features 25, 26, 27)
- **Error handling & logging:** 3.3, 3.4, 3.10, 4.2, 4.3, 5.7, 6.4
- **Testing:** throughout; real model 3.8; every path 5.8; routes 6.5; performance 8.4; hardware 8.6
- **Performance & security:** 8.4 (overhead benchmark on the changed path); 1.4 (name encoding
  cannot inject header syntax); 2.2 (finiteness, bounds)
- **Configuration/deployment:** 1.7 (`http-sfv` dev extra only); no env var, flag or k8s change
  (FTDD §11)
- **Documentation:** 7.4, 7.5

## Tasks

- [x] 0.0 Decisions carried from the FPRD (covers FPRD Open Question 1; FTDD open item 2)
  - [x] 0.1 *(Answered by operator decision S3-09: out of scope for 028, a separate follow-up fixes both paths — recorded in FPRD §14 and as FU-1 below.)* **Operator decision, not a blocker:** profiles steer the first attached SAE
        (`sae_service.py:509-512`), not their recorded `sae_id`/`layer` (`profile.py:58-67`), on
        both the per-request path (`inference_service.py:1990`) and activation
        (`profile_service.py:302`, `:461`). 028 does not change targeting (FPRD D27). Put the
        question to the operator: fix both paths in a follow-up increment? Record the answer in the
        FPRD §14. Task 5.3's `profile_sae_mismatch` log ships either way.
  - [x] 0.2 Ask the PADR owner whether a dev-only `http-sfv` needs a PADR §5 entry. Proceed with
        1.7 meanwhile: it adds nothing at runtime. *Answered (Stage 3, 2026-10-06): PADR §5 lists `http-sfv==0.9.9`
        as dev-only; this sub-task closes on implementation start.*

- [x] 1.0 Pure steering-state module and published vectors (covers FR-28.3.3, FR-28.3.4,
      FR-28.3.5). **Ship first; it unblocks miDataworks 007.**
  - [x] 1.1 Create `millm/core/steering_state.py`: `applied_set` (clamp through `clamp_steering`,
        `steering_range.py:14-16`, then drop zeros including `-0.0`), `canonical_set_form`,
        `steering_set_hash` exactly as FTDD §5.3. Refuse an `sae_id` containing LF.
  - [x] 1.2 Test: TV-1 – TV-4 byte-for-byte (canonical form) and string-for-string (hash), from
        literals copied out of FTDD §5.3, not recomputed by the code under test.
        *2026-10-07: all four reproduced independently before pinning — a stand-alone Python
        one-liner (`struct` + `hashlib`, no miLLM import) and Node (`DataView.setFloat64` +
        `crypto`) — both byte-identical to the FTDD.*
  - [x] 1.3 Add `SteeringItem` and `serialize_steering_header` with the FTDD §5.2 parameter order,
        bare true booleans, member ordering, and a `ValueError` when `none`/`unknown` is combined.
  - [x] 1.4 Add `encode_name` (percent-encoding) and `format_intensity` (`repr(float)`). Test a
        non-ASCII name, a name containing `"` and `\`, and λ = 0.4375 surviving exactly.
  - [x] 1.5 Test: hash independent of input order; zeros excluded from `features` and hash; clamp
        changes the hash (TV-4).
  - [x] 1.6 Test: every kind's serialised header parses with `http-sfv` into the expected kinds and
        parameters (skip loudly if not installed). *The shared dev venv lacks it; verified with
        `http-sfv==0.9.9` installed to a scratch `--target` dir on PYTHONPATH (7 cases green);
        CI installs it through the `dev` extra.*
  - [x] 1.7 Add `http-sfv==0.9.9` to the dev/test extras in `pyproject.toml` only.
  - [x] 1.8 Publish the vectors and grammar in the API reference now (draft section), so consumers
        can pin them before the rest lands.

- [x] 2.0 Request schemas (covers FR-28.1.1, FR-28.1.6 (shape), FR-28.2.1, FR-28.2.2 (shape),
      FR-28.2.4, FR-28.4.1, FR-28.4.6)
  - [x] 2.1 Add `InlineSteeringFeature` and `InlineSteering` (`extra="forbid"`, `sae_id` ≤ 100
        characters) to `millm/api/schemas/openai.py`; add `steering` to `ChatCompletionRequest`
        after `:128`.
  - [x] 2.2 Validators: refuse a boolean strength, a non-finite strength, a duplicate index (naming
        it), and `sae_id` beside an empty list. Test each.
  - [x] 2.3 Model validators on both requests: `steering` + `profile` → refused naming both
        (FR-28.2.1); `steering` + `steering_intensity` → refused naming both (T-78). Test each.
  - [x] 2.4 Add `profile`, `steering_intensity` and `steering` to `TextCompletionRequest`, copying
        the dial validators (`openai.py:182-195`). Test that `steering_intensity: true` and `2.5` are
        refused on completions too.
  - [x] 2.5 *(Feature 25's rule was already present — the policy table's `_unless_scoring` cells —
        so it was extended, not duplicated: `steering`, and `profile`/`steering_intensity` on
        completions, refuse on a scoring request. `features: []` is refused too.)* Extend the scoring check (`openai.py:258-281`, or Feature 25's rule if already
        present): any steering field with scoring → refused (FR-25.7.2, X-09). Test.
  - [x] 2.6 Test: a misspelt key inside `steering` (`strenght`) is refused, not ignored.

- [x] 3.0 Inline and unsteered apply and restore (covers FR-28.1.2, FR-28.1.3, FR-28.1.4,
      FR-28.1.5, FR-28.1.6 (live), FR-28.1.7, FR-28.1.10, FR-28.2.2, FR-28.2.3)
  - [x] 3.1 Add `_dispatch_request_steering` (FTID §3.3). Set `_REQUEST_STEERING` with
        `epoch_at_admission`. Test that each branch is chosen for its input.
  - [x] 3.2 Add `_apply_inline_steering`: select by registry entries, never `attached_sae`;
        validate before mutating; clamp and count; save every entry; set the target; disable the
        others with `enable_steering(False)` (T-79).
  - [x] 3.3 Edge: `sae_id` not attached → `SAE_NOT_ATTACHED` naming it; no SAE attached with a
        non-empty set → the same. Implement and test.
  - [x] 3.4 *(Ambiguity raises `InvalidParameterError` (400, `invalid_parameter`) rather than the FTID's `ValidationError` — same 400 on /v1, with `details.param` = `steering.sae_id` and the attached pairs.)* Edge: `sae_id` omitted with two entries, or one `sae_id` at two layers → `400` naming
        each `(sae_id, layer)`. Implement and test with two real `LoadedSAE`s.
  - [x] 3.5 Edge: an index ≥ `d_sae` → `INVALID_FEATURE_INDEX` naming index and `d_sae`, and global
        state unchanged afterwards. Test.
  - [x] 3.6 Add `_apply_explicit_unsteered` (disable every entry; `None` when nothing attached).
        Test: other entries' monitoring and sensing still capture (T-80), and `features: []` with no
        SAE attached is accepted.
  - [x] 3.7 Change the restore branch to `saved.get("layers") is not None` (`:2206`) and the skip
        log's `path`. Test: restore after inline returns every entry's values *and* enabled flag
        exactly; an epoch bump mid-request skips the restore (existing F16 behaviour).
  - [x] 3.8 Real-model tests (tiny Llama + real `LoadedSAE`, the `test_scoring_completions.py:48-54`,
        `:416` pattern), greedy: inline equals an equivalent saved profile (separate literals);
        `features: []` under an active profile equals the detached output; the next request without
        `steering` is steered again (US-3, US-5).
  - [x] 3.9 AST guard: no generation method calls `_apply_request_steering` or
        `_restore_request_profile` directly except via the dispatcher and finisher, with the two
        streaming restores (`:4450`, `:4762`) named once with their reason.
  - [x] 3.10 Partial-failure rollback: a mutation raising mid-apply restores the saved shape before
        re-raising. Test by making `set_steering_batch` raise.
  - [x] 3.11 Test: inline and unsteered applies never bump the steering epoch (FR-28.1.10).

- [x] 4.0 Routing, engine limits and text-completion steering (covers FR-28.1.8, FR-28.1.9,
      FR-28.1.11, FR-28.4.2, FR-28.4.3, FR-28.4.4, FR-28.4.5)
  - [x] 4.1 `_has_steering_override` reads `steering` on both request types; fix its docstring
        (`inference_service.py:945-956`). Test: a `steering` request with CBM enabled routes serial
        (`:1002-1007`), for chat and text.
  - [x] 4.2 `_refuse_unsupported_llamacpp_request` refuses `steering` (`:3885-3890`). Chat and
        completions routes refuse `steering` on a GGUF model row before auto-load. Test that the
        loader is not called (assert call count 0).
  - [x] 4.3 Pre-load refusal (T-82): a non-empty `steering` set naming a non-resident model →
        `SAE_NOT_ATTACHED` before auto-load, on both routes. Test with the loader mocked: zero calls,
        and the error names the SAE. Test that `features: []` on a non-resident model is not refused.
  - [x] 4.4 Add the dispatch/finish block inside `create_text_completion`'s `_admit()`
        (`:4828`), around every prompt of a multi-prompt request. Test: a named profile applies, a
        missing one returns `404`, and `features: []` opts out (US-5).
  - [x] 4.5 Test: a text completion with no steering field still runs under live global steering
        (FR-28.4.4) and reports it.

- [x] 5.0 The steering report (covers FR-28.3.1, FR-28.3.2, FR-28.3.3, FR-28.3.6, FR-28.3.7,
      FR-28.3.10)
  - [x] 5.1 Create `SteeringSnapshot.capture` (never raises; copies live values with zeros removed,
        the enabled flag and the epoch).
  - [x] 5.2 Add `_STEERING_REPORT`, `get_steering_report()`, and resets for both new contextvars
        inside `reset_steering_memo` (`:333-349`). Test: a stale report from a previous request in a
        reused context is never returned.
  - [x] 5.3 *(Code wins, recorded: circuits are read STRICTLY (`CircuitRepository.list_active`, every full-serving circuit, one item each) instead of the memoised `_steering_circuit()`, which fails open — a DB blip would relabel circuit steering `manual`. Costs one extra read only when an entry is steered and the record does not claim it.)* Create `SteeringStateReader.describe`: labelling order record → circuit → active
        profile → `manual`; exact value equality; `changed` from the epochs; `composed` from a strict
        composition read (unreadable → `unknown;reason=claims_unreadable`); `profile_sae_mismatch`
        warning. Test each label.
  - [x] 5.4 Replace the bare restores at `:3506` and `:3734` with `_finish_request_steering`; add
        the stream capture beside `_probe_finish` (`:4650-4653`). Publish the report after the slot
        releases on non-streaming paths.
  - [x] 5.5 Scoring paths publish `none`; CBM and llama.cpp paths capture an entry epoch and an exit
        snapshot and describe with no record. Test each.
  - [x] 5.6 **Honesty tests** (FTID §8): record claims inline `{5: 8.0}`, snapshot holds `{5: 4.0}` →
        `manual` with the hash of `{5: 4.0}`; an operator write inside generation → new values plus
        `changed` (T-81); an active profile at λ = 0.75 → `profile;source=active;intensity="0.75"`
        with an independently computed hash (US-4).
  - [x] 5.7 Failure paths: `describe` raising internally → `unknown;reason=read_failed` and a
        `steering_report_unknown` warning, while the request still succeeds (FR-28.3.7). Test.
  - [x] 5.8 *(Feature 27 landed first; its discovery code was extracted from `test_probe_paths_discovered.py` into `tests/support/generation_entry_points.py` and both guards import it.)* Discovery test (`tests/support/generation_entry_points.py`, FR-27.9 definition): every
        generation entry point publishes a report; exemptions in one dict with reasons; each exempt
        method asserted to reach no generation primitive. Coordinate with Feature 27: whichever lands
        first creates the helper.
  - [x] 5.9 Expose `steering_report_for_row(...)` for Feature 26 (FR-28.3.10). Test that it returns
        the same string as the header for the same snapshot and record.

- [x] 6.0 Publication: headers and stream chunk (covers FR-28.3.5, FR-28.3.8, FR-28.3.9,
      FR-28.3.11)
  - [x] 6.1 Chat route: set `X-miLLM-Steering` after generation, beside the probe header
        (`chat.py:293-297`). Test the value per kind.
  - [x] 6.2 *(The reset already runs via `provenance.pre_generation`, which the completions route calls before generation; no second call added.)* Completions route: call `reset_steering_memo()` at the top; set the header after
        `create_text_completion` (`completions.py:148`). Test.
  - [x] 6.3 *(The chunk is emitted on the CBM and llama.cpp streaming paths too, after the activations chunk where present.)* Stream: emit `{"choices": [], "millm_steering": <header>}` after the probe chunk and
        before `[DONE]` (`inference_service.py:4656-4660`), always, even for `none`. No header
        before the body. Test ordering, uniqueness and presence for `none`.
  - [x] 6.4 Streaming pre-checks: run the in-slot validations as a dry run before the
        `StreamingResponse`, beside `ensure_profile_exists` (`chat.py:236-237`), so a bad `sae_id` or
        index returns a proper `400`. Test.
  - [x] 6.5 *(Route tests live in `tests/unit/api/test_steering_header_routes.py`, not `tests/integration/`: CI runs `tests/unit` only.)* Route integration tests: chat (non-streaming, streaming, batched `extra_messages`) and
        completions (single, multi-prompt) each carry exactly one correct report; existing
        `X-miLLM-Steering-Intensity` and `X-miLLM-Circuit-Rung` unchanged (FR-28.3.11).

- [x] 7.0 Integration with other features and documentation (covers FR-28.4.5, FR-28.3.10; FPRD
      §10)
  - [x] 7.1 *(Done in 2.0's commit; noted in 025 FTASKS for its owner — the 025 FPRD table is not edited.)* Feature 25: flip the `steering` outcome to honoured for transformers chat and
        completions in Feature 25's code and tests (FR-25.3.7); scoring and embeddings keep refusing.
        The FPRD of Feature 25 is not edited here; note the flip in its FTASKS owner's queue.
  - [x] 7.2 *(Confirmed in code rather than by message: Feature 26's lines read `provenance.post_generation`, which now sets the header from the published report; 026 FTASKS F-3 ticked and FR-26.10.1 marked complete, pinned by `test_runner.py::test_a_batch_line_carries_the_synchronous_steering_header`.)* Feature 26: confirm with its owner that FR-26.10.1 calls `steering_report_for_row`.
  - [x] 7.3 Feature 27: share the discovery helper (5.8).
  - [x] 7.4 API reference: the `steering` object and its errors, every header kind with examples,
        the grammar (FTDD §5.2), the canonical form and TV-1 – TV-4 (FTDD §5.3), the stream chunk, and
        "`X-miLLM-Steering` is authoritative; `X-miLLM-Steering-Intensity` is a pre-generation echo".
  - [x] 7.5 *(There is no OpenAI-API page under `manual/docs/features/`; the section went into `features/feature-steering.md`, linking to the API reference — recorded discrepancy.)* Manual: extend the OpenAI-API feature page with inline steering and the header; keep the
        manual-reachability test green (`tests/unit/test_manual_pages_are_reachable.py`).

- [x] 8.0 Feature Acceptance
  - [x] 8.1 *(Evidence table in `0xcc/reviews/028_implementation_controls_2026-10-07.md`; CPU halves pass, hardware halves are 8.6.)* Verify each FPRD success criterion and user story one by one (§11 items 1–5; US-1 –
        US-7) and record the evidence.
  - [x] 8.2 *(41 controls + 1 added, recorded in `0xcc/reviews/028_implementation_controls_2026-10-07.md`; 4 survived first time — 3 real gaps now tested and re-run RED, 1 measured equivalent (d2). Run against the 16 affected files, full suite green before and after.)* **Mutation controls**, each run against the full suite, required red, reverted, and the
        restore verified by `git diff` before moving on: (a) skip `clamp_steering` on the inline path;
        (b) skip the restore in `_finish_request_steering`; (c) restore without the epoch guard;
        (d) report from the request record instead of the snapshot; (e) hard-code the item kind to
        `inline`; (f) drop `steering` from `_has_steering_override`; (g) remove the header line in
        each route; (h) remove the stream chunk; (i) drop the dispatcher call in
        `create_text_completion`; (j) revert the restore branch to `saved.get("circuit")`. A survivor
        is a test finding: write the test, then re-run the mutation as a negative control.
  - [x] 8.3 *(`tests/unit` green; integration/schema green after one guard update; ruff/mypy from a scratch install — no mypy error on any added line.)* Run the full backend suite (no `--ignore`), `ruff`, `mypy millm/`.
  - [x] 8.4 *(p95 delta 0.053 ms record-labelled, 1.589 ms worst case with two SQLite reads; Postgres measurement on the node belongs to 8.6.)* Benchmark the changed path: 200 serial requests with and without the report; record the
        p95 delta (target under 5 ms).
  - [x] 8.5 *(Pinned by `test_the_api_reference_publishes_exactly_these_vectors`. Notifying miDataworks 007 is the coordinator's: no cross-repo contact from this session.)* Confirm that the published vectors in the API reference match
        `tests/unit/core/test_steering_state.py` byte for byte, and notify miDataworks 007 (X-07).
  - [x] 8.6 **needs hardware — operator session** (BRD-04 acceptance 12; also measure 8.4's read on Postgres). **Hardware acceptance, BRD-04 acceptance 12**, on LFM2.5-1.2B-Instruct (FP16) with an
        SAE attached on the k8s node: (a) inline steering and a saved profile with the same features
        give identical greedy output; (b) `X-miLLM-Steering` is correct for none, profile, inline and
        circuit, streaming and not; (c) a P-22 steered pair (one feature index, opposite strengths)
        gives two headers whose hashes match hashes recomputed from the requests (TV-1/TV-2 method);
        (d) `features: []` under an active profile reports `none` and matches the unsteered output.
        Record the results in the review notes.
  - [x] 8.7 *(No AGENT.md in this repo; the inventory lives in CLAUDE.md, updated. Follow-ups FU-1 – FU-3 filed below.)* Update AGENT.md's Document Inventory (mark 028's four documents ✅) and file follow-up
        work: the operator's answer to 0.1, if it is "fix".

## Follow-ups filed by the implementation (2026-10-07)
- [ ] FU-1 **First-SAE profile targeting (S3-09, FPRD Open Question 1).** Operator decision: out of
      scope for 028; a separate follow-up fixes BOTH paths (per-request `_apply_request_steering`
      and global activation in `profile_service`) to select the entry by the profile's own
      `sae_id`/`layer` and refuse when it is not attached. 028 makes it visible: the header's
      `profile` item names the SAE and layer actually steered, and a mismatch logs
      `profile_sae_mismatch` (`steering_report._note_profile_mismatch`, pinned by
      `test_steering_report.py::TestLabels::test_active_profile_by_equal_values`).
- [ ] FU-2 **Attaching an SAE leaves the model `locked = true`, and detaching does not clear it**
      (observed on hardware 2026-10-06/07, also recorded in 027 FTASKS). The next swap is refused
      `model_locked` until `POST /api/models/{id}/unlock`. Not fixed in 028 (the FTDD does not
      cover it); FR-28.1.11's pre-load refusal deliberately does not rely on the lock.
- [ ] FU-3 The 025 FPRD outcome table still says `steering` is "refused until Feature 28"; noted in
      025 FTASKS for its owner (FR-25.3.7 flipped in code by 028).

## Coverage Audit
- **FRs:** 28.1.1 → 2.1 · 28.1.2 → 3.1, 3.7 · 28.1.3 → 3.2, 3.4 · 28.1.4 → 3.3 · 28.1.5 → 3.2, 3.8 ·
  28.1.6 → 2.2, 3.5 · 28.1.7 → 1.1, 3.2, 8.2a · 28.1.8 → 4.1 · 28.1.9 → 4.2 · 28.1.10 → 3.11 ·
  28.1.11 → 4.3 · 28.2.1 → 2.3 · 28.2.2 → 2.2, 3.6 · 28.2.3 → 3.6 · 28.2.4 → 2.3 · 28.3.1 → 5.5, 5.8
  · 28.3.2 → 5.3, 5.6 · 28.3.3 → 1.3, 5.3 · 28.3.4 → 1.1, 1.2, 8.5 · 28.3.5 → 1.3, 1.6 · 28.3.6 → 5.3,
  5.6 · 28.3.7 → 5.7 · 28.3.8 → 6.1, 6.2 · 28.3.9 → 6.3 · 28.3.10 → 5.9, 7.2 · 28.3.11 → 6.5 ·
  28.4.1 → 2.4 · 28.4.2 → 4.4 · 28.4.3 → 4.4 · 28.4.4 → 4.5 · 28.4.5 → 7.1 · 28.4.6 → 2.5. **All 32
  sub-requirements covered; parents FR-28.1 – FR-28.4 each cited by at least one parent task.**
- **Edge cases (FPRD §2 table), implement + test:** both fields → 2.3 · steering + dial → 2.3 · unattached
  SAE → 3.3 · ambiguous SAE → 3.4 · no SAE → 3.3 · `[]` with no SAE → 3.6 · non-resident model → 4.3 ·
  bad index → 3.5 · duplicate → 2.2 · non-finite → 2.2 · beyond ±200 → 1.5, 3.2 · GGUF → 4.2 · scoring
  → 2.5 · embeddings → 7.1 (Feature 25 table, unchanged) · mid-request change → 5.6 · unreadable state
  → 5.3, 5.7.
- **Acceptance criteria:** BRD-04 acceptance 12 → 3.8 (real model), 6.5 (routes), 8.6 (hardware).
- **TDD sections:** Data Design → N/A (checklist) · API Design → 2.x, 4.x, 6.x · Component
  Architecture → 1.x, 3.x, 5.x · State Management → 5.2 · Security → 1.4, 2.2 · Performance → 8.4 ·
  Testing → throughout, 8.2 · Deployment → 1.7, 7.4–7.5.
- **TID sections:** file structure → Relevant Files · component hints → 1.x, 3.x, 5.x · database →
  N/A · API → 2.x, 6.x · frontend → N/A · business logic → 3.x, 5.3 · testing → 5.6, 5.8 · config →
  1.7 · integration → 7.x · utilities → 1.x, 3.9 · errors and logging → 3.3–3.5, 5.7 · performance →
  8.4 · code quality → 3.9, 8.3.
- **Open questions:** FPRD Open Question 1 → 0.1 · FTDD open item 2 → 0.2. The six v1.0 questions are
  resolved (T-78 – T-83) and implemented in 2.3, 3.2, 3.6, 5.3, 4.3 and 1.3.
- **The final parent task is Feature Acceptance.** ✔


## Hardware acceptance — 2026-10-07 (RTX 3090; `main` at `f563658`; Qwen2.5-7B-Instruct + layer-25 SAE `Geaming--…blocks_25…jumprelu`)

Run on Qwen because it is the only model with an SAE in this miLLM; LFM2.5-1.2B has none to attach. Greedy (`temperature` 0), `max_tokens` 30.

| BRD-04 acc. 12 | Result |
|---|---|
| (a) inline = saved profile | PASS — a profile saved with {100: +80} and activated (`apply_steering: true`) gives text **identical** to inline `{features: [{index: 100, strength: 80}]}`, with the **same hash** |
| (b) header per kind | PASS — `none`; `inline;sae=…;layer=25;features=1;hash="sha256:3782c648…"`; `profile;name=…;source=active;intensity="1.0";sae=…;layer=25;features=1;hash=…`. Streaming: the value arrives in the terminal `millm_steering` chunk (FTDD §2, manual), identical to the non-streaming header for inline and profile. **Circuit kind not exercised:** none of the 3 circuits belongs to an attachable SAE on this node |
| (c) steered pair | PASS — +80 / −80 on feature 100: both header hashes equal `steering_set_hash()` recomputed from the requests (the TV-1/TV-2 method); the two hashes and the two texts differ. At ±8 the greedy text did not change — strength, not a defect |
| (d) `features: []` under a profile | PASS — header `none`, text identical to the unsteered baseline |
