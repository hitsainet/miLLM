# Feature 27 — Implementation Controls Record (2026-10-06)

Branch `feat/027-probe-scoring`, cut from `main` at `08c1c53`. Worktree `~/app/miLLM-027`.
`millm.__file__` verified as `/home/x-sean/app/miLLM-027/millm/__init__.py` (venv `~/app/miLLM/venv`).

**Baseline** (before any change, `pytest tests/unit`): **4313 passed / 3 skipped / 0 failed** (174.6 s).

Procedure for every control: back up the file, make ONE change, confirm it landed (grep), run the
named tests and require a red, restore from the backup, verify the restore by `sha256sum` against
the pre-mutation hash AND by re-grepping the mutated line. A survivor is a test finding: write the
test, re-run the control as a negative control.

## Task 1 — expected red (task 1.6)

The discovery guard (`tests/unit/services/test_probe_paths_discovered.py`) run on the code at
`08c1c53`: discovery finds **10 generation sites** and **3 entry points**; the behavioural scenario
fails on exactly the six sites FR-27.8 names, each with *"generation reached with no probe context
open"*: `_generate_batch_chunk` (batched chat), `_cbm_chat_completion`, `_cbm_text_completion`,
`_llamacpp_chat_completion`, `_llamacpp_stream_chat_completion`, `_llamacpp_text_completion`.
The four already-wired sites pass. The six are committed as `xfail(strict=True)` so the red is
recorded mechanically and each fix must remove its entry.

## Task 2 — probe-path fixes (M1–M11, M21, latent-defect controls L1–L5)

Runner: `scratchpad/impl-027/mutate.py` (absolute paths, one replacement scoped to one method,
landed-check, sha256 + re-grep restore check). Guard = `test_probe_paths_discovered.py`.
Pre-mutation sha of `inference_service.py`: `eccfa5820b1d…`; every restore matched it.

| # | Mutation | Tests | Result | Restore |
|---|---|---|---|---|
| M1 | `create_chat_completion`: `_probe_ctx = self._probe_begin(…)` → `None` | guard | RED (1 failed) | sha ✔ grep ✔ |
| M2 | `stream_chat_completion`: same | guard | RED | ✔ ✔ |
| M3 | `create_text_completion`: same | guard | RED | ✔ ✔ |
| M4 | `_cbm_stream_chat_completion`: detached begin → `None` | guard | RED | ✔ ✔ |
| M5 | `_create_batched_chat_completion`: detached begin → `None` | guard | RED | ✔ ✔ |
| M6 | `_cbm_chat_completion`: detached begin → `None` | guard | RED | ✔ ✔ |
| M7 | `_cbm_text_completion`: detached begin → `None` | guard | RED | ✔ ✔ |
| M8 | `_llamacpp_chat_completion`: detached begin → `None` | guard | RED | ✔ ✔ |
| M9 | `_llamacpp_stream_chat_completion`: detached begin → `None` | guard | RED | ✔ ✔ |
| M10 | `_llamacpp_text_completion`: detached begin → `None` | guard | RED | ✔ ✔ |
| M11 | new method `_m11_new_site` calling `self._generate_sync({})`, no scenario | guard | RED (`test_the_scenario_table_equals_discovery`) | ✔ ✔ |
| M21 | completions route: drop `response.headers["X-miLLM-Probe-Verdicts"] = …` | `TestCompletionsVerdictHeader` | RED | sha ✔ (`d44fab4a8826…`) grep ✔ |
| L1 | CBM stream back to the REGISTERED `_probe_begin` (the latent collision, FR-27.8h) | `TestConcurrentContinuousBatching` | RED | ✔ ✔ |
| L2 | `_probe_record`: `end_request()` unconditional again (closes another request's context) | `test_probe_path_fixes.py` | RED | ✔ ✔ |
| L3 | hung-thread guard: drop `_close_request_captures(…)` | `TestHungThreadClosesCapture` | RED | ✔ ✔ |
| L4 | batched: `_probe_finish` → `None` (header/ContextVar empty) | `test_probe_path_fixes.py` | RED | ✔ ✔ |
| L5 | CBM chat: drop the `finally` record | `test_probe_path_fixes.py` | RED | ✔ ✔ |

**17 controls, 0 survived first time.**

**Code-vs-doc discrepancies found in task 2:**
- FTID §3/§10.1 lists the batched path's generation site as `_create_batched_chat_completion`; the
  primitive is actually called from `_generate_batch_chunk`, which discovery (correctly) reports.
  The scenario table is keyed by the discovered site.
- FTASKS 2.10 names ten "begin calls"; three of them (serial chat, stream, text) are the registered
  `_probe_begin`, seven are the new `_probe_begin_detached`.
- `test_probe_scope_is_honoured.py` asserted ≥4 registered `_probe_begin` sites; the CBM stream path
  moved to the detached seam (as FR-27.8h requires), so it is now ≥3, with the reason recorded.
- 2.9: the Probe Monitors page renders `not_scored_reason` verbatim (`ProbeMonitorsPage.tsx:591`),
  so no code change; an `it.each` over the three new reasons pins it.

## Task 3 — admission and suppression for model work (M15–M17, L6–L8)

| # | Mutation | Tests | Result | Restore |
|---|---|---|---|---|
| M15 | `run_model_work`: `async with self._admit():` → `if True:` | `test_run_model_work.py` | RED | sha ✔ grep ✔ |
| M16 | arm route: delete `executor=inference.run_model_work` | same | RED (payload test) | ✔ ✔ |
| M17 | `run_model_work`: enter `_unsteered()` around the await, run `fn` bare in the worker | same | RED (`test_suppression_is_entered_IN_THE_WORKER_THREAD`) | ✔ ✔ |
| L6 | parity route: `inference.run_model_work(…)` → `asyncio.to_thread(…)` | same | RED (call count) | ✔ ✔ |
| L7 | `ProbeArmingService.arm`: `await executor(…)` → `asyncio.to_thread(…)` | same + `test_probe_arming.py` | RED | ✔ ✔ |
| L8 | `ParityReport.as_details`: `"model": None` | same | RED | ✔ ✔ |

**6 controls, 0 survived first time.** (Runner note: the first M16 attempt aborted before writing —
the scope helper only knew class methods; route handlers are module-level. Fixed the helper,
confirmed the tree untouched, re-ran. Not a survivor.)

**3.5 measured:** on the tiny real Llama with a LoadedSAE steering layer 0 (feature 2 at 40.0), a
layer-1 probe's per-token scores moved by up to **282.32** on the pre-fix path (direct forward, no
seam); through `run_model_work` they equal the unsteered scores exactly.

**Pre-existing defect fixed:** `tests/integration/test_probe_workflow.py::test_an_armed_probe_scores_a_forward_pass`
failed on `main` (asserted one verdict; the windows feature made it three). Now arms with
`windows=[]`. The integration tier is not in `tests/unit`, which is why it went unnoticed.

