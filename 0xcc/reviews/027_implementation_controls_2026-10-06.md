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

## Task 4 — stateless probe scoring (M13, M14, M20, S1–S13)

Tests: `test_probe_scoring.py`, `test_probe_score_writes_nothing.py`, `test_probe_score_route.py`,
`test_probe_input_preparer.py`.

| # | Mutation | Result | Restore |
|---|---|---|---|
| M13 | `_score_one` opens the runtime's `begin_request` | RED (no-event guard: `begins == []`) | sha ✔ grep ✔ |
| M14 | score route writes a `ProbeEvent` row before returning | RED (no-event guard: event count) | ✔ ✔ |
| M20 | delete `@router.post("/score")` | RED (reachability) | ✔ ✔ |
| S1 | `TEXT_INPUT_VERIFIED = True` (lifts the T-49 gate) | RED | ✔ ✔ |
| S2 | pin check reduced to `current is None` | RED (pinning test) | ✔ ✔ |
| S3 | score loop calls `_score_one` directly (no slot per input) | RED (admission count) | ✔ ✔ |
| S4 | `"verdict": bool(v.fires)` (null coerced to false) | RED | ✔ ✔ |
| S5 | given ids skip instead of refusing | RED | ✔ ✔ |
| S6 | vocabulary range check disabled | RED | ✔ ✔ |
| S7 | `windows=None` instead of the request's windows | RED | ✔ ✔ |
| S8 | `token_ids` echo never returned | RED | ✔ ✔ |
| S9 | `set_prompt_length` dropped | RED | ✔ ✔ |
| S10 | prefix check dropped from the assistant-ended boundary | RED | ✔ ✔ |
| S11 | `text` rendered as a system turn | RED (preparer `text == one user turn`) | ✔ ✔ |
| S12 | shared `identity_refusal` never refuses a model mismatch | RED | ✔ ✔ |
| S13 | omitted-id skip not recorded in `skipped` | RED | ✔ ✔ |

**16 controls, 0 survived first time.**

**Discrepancies / deviations (task 4):**
- FTDD §5.1 puts shape refusals (two kinds, `prompt_tokens` past the input, zero inputs) at
  `INVALID_PROBE_SCORE_REQUEST` 400, while FTASKS 4.1 says "schemas … all `extra="forbid"`".
  Both honoured: unknown fields and unknown windows are pydantic 422s; the semantic shape checks
  live in `probe_scoring.check_shape` so they answer 400 as the refusal table says.
- Arming's identity and scope gates were extracted into `probe_arming.identity_refusal` /
  `scope_refusal` (reuse, not copy) — `arm` raises what they return, unchanged in behaviour.
- `scope_refusal` can no longer fire in practice: `RUNTIME_SCORABLE_SCOPES = frozenset(SCOPES)` at
  `probe_scope.py:169` admits every scope, so the arming docstring describing `scope='all'` only is
  stale. Recorded, not changed (out of 027's scope).
- `text` is refused until FTASKS 0.2 passes (T-49): `TEXT_INPUT_VERIFIED = False`.
- With `probe_ids` omitted and more than `PROBE_SCORE_MAX_PROBES` matching probes, the request is
  refused naming the cap (the FTDD does not say what happens; refusing beats silently truncating).

**Test-harness defect found and fixed (task 4):** the first privacy test used
`structlog.testing.capture_logs`; structlog caches each logger on first use, and the full suite went
5 red in unrelated files (`test_model_lease.py`, `test_structured_output.py`). Separately, entering
`TestClient(app)` as a context manager runs the lifespan, which reconfigures logging, and blinded
the same tests. Both removed (recorder on the module logger + caplog; plain `TestClient(app)`);
M13/M14/M20 re-run afterwards, all RED.

## Task 5 — the verdict boundary (M12, M12b, P-20)

| # | Mutation | Tests | Result | Restore |
|---|---|---|---|---|
| M12 (live) | `value >= threshold` → `value > threshold` at `probe_runtime.py:665` | `test_probe_runtime.py` (the exactly-on-the-bar test) | RED | sha ✔ grep ✔ |
| M12 (stateless) | same mutation | `test_probe_score_route.py::test_a_score_EXACTLY_on_the_bar_fires_through_the_route` | RED | ✔ ✔ |
| M12b | a second comparison: `verdict_payload` recomputes `v.score >= v.threshold` | `test_verdict_boundary_is_one_place.py` | RED | ✔ ✔ |
| P20 | event service stores `provisional: False` | `test_probe_events.py` (new provisional test) | RED | ✔ ✔ |

**4 controls, 0 survived first time.** `>=` confirmed at `millm/services/probe_runtime.py:665`,
still the only comparison (the guard asserts the flagged set EQUALS `{_verdict_for}`). 5.4 had no
existing test that an event ROW keeps `provisional`; one was added.

## Task 6 — per-request SAE activations (M18, M19, A1–A12, H1–H2)

Tests: `test_request_activations.py`, `tests/unit/api/test_return_sae_activations.py`.

| # | Mutation | Result | Restore |
|---|---|---|---|
| M18 | `_activations_close` never ends the capture (shared across requests) | RED | sha ✔ grep ✔ |
| M19 | capture phase always `pre` (read point ignored) | RED | ✔ ✔ |
| A1 | serial chat: `_activations_begin` → `None` | RED (seam payload+count) | ✔ ✔ |
| A2 | streaming chat: same | RED | ✔ ✔ |
| A3 | text completion: same | RED | ✔ ✔ |
| A4 | scoring (`_score_prompts`): same | RED | ✔ ✔ |
| A5 | `_generate_in_thread` never sets the owner (streaming records nothing) | RED | ✔ ✔ |
| A6 | `feed_request_capture` drops the owner check (a foreign forward feeds the capture) | RED (isolation) | ✔ ✔ |
| A7 | serial-routing flag for `return_sae_activations` removed | RED | ✔ ✔ |
| A8 | chat route: `refuse_before_generation` removed | RED | ✔ ✔ |
| A9 | completions route: `X-miLLM-Steering: none` dropped | RED | ✔ ✔ |
| A10 | offset advances by kept count instead of pass width | RED | ✔ ✔ |
| A11 | serial chat response loses `millm=_millm` | RED | ✔ ✔ |
| A12 | worst-case entry cap disabled | RED | ✔ ✔ |
| H1 | SAE hook: drop the pre-steering feed | RED | ✔ ✔ |
| H2 | SAE hook: drop the post-steering feed | RED | ✔ ✔ |

**16 controls, 0 survived first time.**

**Design decision beyond the FTDD (recorded):** the FTDD isolates captures by "one open capture per
SAE, opened inside the slot". That does NOT isolate against a continuous-batching generation, which
takes no slot and runs the same SAE hook concurrently — it would have written its positions into the
open capture. Captures therefore carry an owner token held in a ContextVar (`sae_wrapper.CAPTURE_OWNER`);
`asyncio.to_thread` copies it into the worker, the streaming path hands it to its plain `Thread`
explicitly, and the hook feeds only matching forwards. Control A6 proves the check bites.

**Other deviations / discrepancies:**
- FTID §5 asks for a `@model_serializer(mode="wrap")` to omit `millm`; the schemas already use
  pydantic's `exclude_if` for the same purpose (`system_fingerprint`), so `millm` uses it too. Same
  wire result (no key when absent, OpenAI nulls untouched), tested.
- `return_sae_activations` is registered as a row in Feature 25's `OUTPUT_CHANGING` table (honoured
  on transformers chat/completions, refused on llama.cpp and embeddings), which is how FR-27.2g's
  "GGUF refused before any auto-load" is enforced; the HTTP coverage test exercises every cell.
- X-09: `X-miLLM-Steering: none` is set on scoring-mode responses that carry activations only;
  Feature 28 owns the header elsewhere. `/api/probes/score` carries none (tested).
- 2.8's capture slot (`begin_request_capture`/`end_request_capture`) was built in task 2 so the
  hung-thread guard could close it; task 6 added the hook reads.

