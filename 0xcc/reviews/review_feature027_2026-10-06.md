# Feature 27 — Acceptance Review (non-hardware), 2026-10-06

Branch `feat/027-probe-scoring` (cut from `main` @ `08c1c53`). Implementation controls are recorded
task by task in `027_implementation_controls_2026-10-06.md`; this record is the acceptance walk
(FTASKS 8.1), the full re-run of every control (8.2), the suites (8.3) and the integration test
(8.4). Hardware items (0.1, 0.2, 8.5–8.9) are marked `[?]` — needs hardware, operator session.

## 8.1 Success criteria and user stories

| Item | Status | Evidence |
|---|---|---|
| SC-1 (acceptance 10) | ✅ unit · `[?]` hardware 8.6 | `test_request_activations.py::TestIsolation` (two in-flight requests each get their own positions and token ids; a forward without the owner token feeds nothing); over-cap 400 in `test_return_sae_activations.py`. Controls M18, A6, A12 |
| SC-2 (acceptance 11) | `[?]` hardware 8.5 | Plumbing proved on a tiny real model: `tests/integration/test_probe_score_matches_live.py` (unarmed offline == armed live, prompt and all windows, no event written). Control I1 |
| SC-3 (acceptance 17) | ✅ unit · `[?]` hardware 8.7 | `test_probe_path_fixes.py` (event + live route header `not-scored;reason="batched_request"`); discovery guard red on removal (M5) |
| SC-4 | ✅ | CBM chat/text record `continuous_batching` with `PROBE_FORCE_SERIAL=False`; M6, M7 red; concurrent CBM requests both record (L1) |
| SC-5 | ✅ | score: one `_admit` per input (S3); parity: `run_model_work` once (L6); arm: executor is `run_model_work` (M16) and is awaited once (L7); M15 removes `_admit` from the seam → red |
| SC-6 | ✅ | every wiring control asserts payload and count — discovery guard (M1–M11), route reachability (M20), seams A1–A4, executor payload (M16), header (M21) |
| SC-7 (P-03) | ✅ | exactly-on-the-bar fixture fires through the route and live; M12 turns both red; M12b (a second comparison) turns the AST guard red |
| US-1 | ✅ unit | `test_probe_score_writes_nothing.py`: no event, no `begin_request`, armed set unchanged, parity report unchanged; routing unchanged (4.13) |
| US-2 | ✅ unit · `[?]` hardware | integration test above |
| US-3 | ✅ | given mismatched probe refused naming `hf_id` (S12); omitted → `skipped` with `PROBE_MODEL_MISMATCH` (S13) |
| US-4 | ✅ | `rung_language` verbatim in every result (`TestResponse`) |
| US-5 | ✅ | scoring-mode text completion returns activations with `read_point: "unsteered"`, `last` = last prompt position, and `X-miLLM-Steering: none` (A4, A9) |
| US-6 | ✅ unit · `[?]` hardware | as SC-1 |
| US-7 | ✅ unit · `[?]` hardware | as SC-3 |

## 8.2 Every control re-run (60), each verified red, each restore verified by sha256 and re-grep

Runner: `mutate.py` over the union of the task specs plus I1 (the integration test's negative
control: score with `add_special_tokens=False`, i.e. not the live tokenization → red).
**60 / 60 RED, 60 / 60 restores verified.** After the final type-only edits to
`probe_scoring.py`, the 15 controls in that module were re-run: 15 / 15 red.

| Control | Result | Tail |
|---|---|---|
| M1 | RED | 1 failed, 14 passed in 5.49s |
| M2 | RED | 1 failed, 16 passed in 5.47s |
| M3 | RED | 1 failed, 15 passed in 5.38s |
| M4 | RED | 1 failed, 8 passed in 2.73s |
| M5 | RED | 1 failed, 10 passed in 5.38s |
| M6 | RED | 1 failed, 7 passed in 2.78s |
| M7 | RED | 1 failed, 9 passed in 2.85s |
| M8 | RED | 1 failed, 11 passed in 5.48s |
| M9 | RED | 1 failed, 12 passed in 5.28s |
| M10 | RED | 1 failed, 13 passed in 5.34s |
| M11 | RED | 1 failed, 6 passed in 2.44s |
| M21 | RED | 1 failed in 5.35s |
| L1-collision | RED | 1 failed in 2.52s |
| L2-record-closes-other | RED | 1 failed, 2 passed in 2.37s |
| L3-hung-capture | RED | 1 failed in 4.99s |
| L4-batched-finish | RED | 1 failed, 4 passed in 4.97s |
| L5-cbm-record-finally | RED | 1 failed, 5 passed in 4.95s |
| M15 | RED | 1 failed in 2.55s |
| M16 | RED | 1 failed, 8 passed in 5.28s |
| M17 | RED | 1 failed, 1 passed in 2.67s |
| L6-parity-route-seam | RED | 1 failed, 9 passed in 5.51s |
| L7-arm-executor-call | RED | 1 failed, 10 passed in 5.52s |
| L8-report-model | RED | 1 failed, 5 passed in 5.07s |
| M13 | RED | 1 failed, 36 passed in 8.60s |
| M14 | RED | 1 failed, 36 passed in 5.79s |
| M20 | RED | 1 failed, 36 passed in 5.85s |
| S1-text-gate | RED | 1 failed, 6 passed in 5.41s |
| S2-pin | RED | 1 failed, 30 passed in 5.92s |
| S3-slot-per-input | RED | 1 failed, 18 passed in 5.40s |
| S4-null-verdict | RED | 1 failed, 23 passed in 5.46s |
| S5-given-refuses | RED | 1 failed, 14 passed in 2.73s |
| S6-vocab | RED | 1 failed, 11 passed in 2.83s |
| S7-windows | RED | 1 failed, 18 passed in 5.40s |
| S8-echo | RED | 1 failed, 22 passed in 5.46s |
| S9-prompt-boundary | RED | 1 failed, 21 passed in 5.50s |
| S10-prefix-check | RED | 1 failed, 46 passed in 6.67s |
| S11-text-one-user-turn | RED | 1 failed, 47 passed in 6.60s |
| S12-identity-shared | RED | 1 failed, 14 passed in 5.37s |
| S13-skip-records-mismatch | RED | 1 failed, 15 passed in 5.33s |
| M12-live | RED | 1 failed, 10 passed in 2.53s |
| M12-stateless | RED | 1 failed in 5.56s |
| M12b-second-compare | RED | 1 failed in 2.59s |
| P20-event-flag | RED | 1 failed, 2 passed in 2.28s |
| M18 | RED | 1 failed, 10 passed in 5.15s |
| M19 | RED | 1 failed in 2.49s |
| A1-serial-chat-seam | RED | 1 failed, 10 passed in 5.13s |
| A2-stream-seam | RED | 1 failed, 15 passed in 5.26s |
| A3-text-seam | RED | 1 failed, 14 passed in 5.31s |
| A4-scoring-seam | RED | 1 failed, 11 passed in 5.20s |
| A5-stream-owner | RED | 1 failed, 15 passed in 5.33s |
| A6-owner-check | RED | 1 failed, 20 passed in 5.39s |
| A7-serial-routing | RED | 1 failed, 17 passed in 5.30s |
| A8-route-refusal | RED | 1 failed, 22 passed in 6.71s |
| A9-x09-header | RED | 1 failed, 34 passed in 8.98s |
| A10-offset | RED | 1 failed, 2 passed in 2.65s |
| A11-chat-millm-attach | RED | 1 failed, 10 passed in 5.34s |
| A12-cap | RED | 1 failed, 28 passed in 7.38s |
| H1-hook-pre | RED | 1 failed, 10 passed in 5.36s |
| H2-hook-post | RED | 1 failed, 10 passed in 5.37s |
| I1-live-render | RED | 1 failed in 5.14s |

## 8.3 Suites

- `pytest tests/unit` (final, at the acceptance commit): **4474 passed / 3 skipped / 0 failed**; at the task-7
  commit **4474 passed / 3 skipped / 0 failed** (baseline at `08c1c53`: 4313 / 3 / 0).
- Mirror view (`0xcc/` moved out of the tree): **4467 passed / 10 skipped / 0 failed**; `0xcc/`
  restored and `git status` confirmed.
- Cross-repo: `MILLM_REQUIRE_CROSS_REPO_CHECKS=1 pytest tests/unit/test_mcp_contract_consistency.py
  tests/unit/test_mcp_tool_paths_are_real.py` → 35 passed / 1 skipped (the pre-existing
  `MISTUDIO_SETTINGS_AVAILABLE` known limit), miStudio present at `~/app/miStudio`.
- admin-ui: vitest **471 passed / 41 files**; `tsc -b` clean.
- ruff (`--select E,F,B`) clean on every new file; touched files have no new findings versus
  `08c1c53` (the repo is not ruff-clean overall: 617 findings under `tests/unit` pre-existing, and
  CI does not gate on ruff). mypy is not installed in the venv; run via `uvx mypy` on the four new
  modules: clean.
- Integration tier: `tests/integration/test_probe_workflow.py` had a test failing on `main`
  (fixed in task 3); the new `test_probe_score_matches_live.py` passes. ⚠ A run of the WHOLE
  integration tier was killed (exit 137) during task 3 — not re-attempted; only these two files
  are verified.

## Hardware session list (operator)

1. **0.1** — on one miStudio definition, compare each test vector's `token_ids` with live serving's
   render of its `messages` (generation prompt on) and the drift render (off); record which
   reproduces the recorded ids.
2. **0.2 (T-49)** — score one miStudio-evaluated set through `/api/probes/score` as one-user-turn
   input (temporarily via `messages`, since `text` is gated) on the probe's model; AUROC must lie
   inside miStudio's interval. Pass → flip `TEXT_INPUT_VERIFIED`, delete
   `test_text_refused_until_verified_names_the_check`, record the pass in the same commit.
3. **8.5 / BRD-04 acc. 11** — LFM2.5-1.2B: score a definition's test vectors unarmed within the
   parity tolerance; arm and compare live; `probe_events` count unchanged by the stateless call.
4. **8.6 / acc. 10** — two interleaved activation requests; an over-cap request returns 400.
5. **8.7 / acc. 17** — probe armed, batched chat → header and event `batched_request`.
6. **8.8** — scoring throughput and the dynamo-reset share per input (two `ProbeHooker` installs per
   input); hoist the installs out of the loop if it dominates.
7. **8.9** — wait for any rollout to settle first.
