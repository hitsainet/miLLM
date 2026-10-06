# Technical Implementation Document: Probe Scoring, Per-Request Activations and Probe-Path Fixes
## miLLM Feature 27

**Version:** 1.0 · **Created:** 2026-10-06 · **Status:** Planned
**References:** 027_FPRD v1.1, 027_FTDD v1.0 · Feature 24 FTID (patterns reused)
**Load-bearing points verified against** miLLM `main` @ `f5c71b6` (2026-10-06).
**Re-check every line number before editing.** `inference_service.py` is 5,678 lines and moves.

---

## 1. Implementation Overview

- **Reuse, never copy.** Probe construction (`armed_probe_from_row`, `millm/services/probe_arming.py:245`),
  the forward (`build_parity_forward`, `millm/services/probe_arm_bridge.py:119`), and the decision
  (`_verdict_for`, `millm/services/probe_runtime.py:537`) are called, not re-implemented. A copy of any
  of them would let offline and live scores drift (BRD-04 RSK-07).
- **One seam for model work outside generation:** `InferenceService.run_model_work(fn)`. It takes the
  slot through `_admit()` (`inference_service.py:649`), runs `fn` in a worker thread, and enters
  `_unsteered()` (`inference_service.py:1175-1199`) **inside that thread**.
- **Observers never raise.** Every probe and activation seam inside a generation path catches and
  logs, as `_probe_begin` does (`inference_service.py:2498-2509`). Validation that should refuse a
  request happens in the route, before the slot.
- **Guards read structure, not text.** The path guard walks the AST for calls; the boundary guard
  walks the AST for comparisons. A substring search matches comments and passes for the wrong reason.

## 2. File Structure and Organization

**New files:**
```
millm/services/probe_scoring.py            ProbeInputPreparer, ProbeScoringService
millm/services/request_activations.py      ActivationSpec, RequestActivationCapture, build_extension
millm/api/schemas/probe_scoring.py         ProbeScoreRequest, ProbeScoreInput, response models
millm/api/schemas/millm_extension.py       MillmExtension, SaeActivationsBlock, ReturnSaeActivations
tests/unit/services/test_probe_scoring.py
tests/unit/services/test_probe_input_preparer.py
tests/unit/services/test_probe_score_writes_nothing.py
tests/unit/services/test_probe_paths_discovered.py
tests/unit/services/test_run_model_work.py
tests/unit/services/test_request_activations.py
tests/unit/services/test_verdict_boundary_is_one_place.py
tests/unit/api/test_probe_score_route.py
tests/unit/api/test_return_sae_activations.py
tests/integration/test_probe_score_matches_live.py
```

**Modified files:**
- `millm/services/inference_service.py`: `run_model_work`, `_unsteered_call`,
  `_probe_begin_detached`, `_probe_record(detached=)`, activation seams, FR-27.8 wiring.
- `millm/services/probe_arm_bridge.py`: `build_probe_forward(model, layers)`; `build_parity_forward`
  delegates.
- `millm/services/probe_arming.py`: `arm(..., executor)` (required keyword).
- `millm/services/probe_parity.py`: `as_details()` gains `model` and `checked_at`.
- `millm/api/routes/management/probes.py`: the score route; parity and arm through `run_model_work`.
- `millm/api/routes/openai/chat.py`, `completions.py`: shape refusals for activations; the completions
  verdict header.
- `millm/api/schemas/openai.py`: request field and response `millm` object.
- `millm/ml/sae_wrapper.py`, `millm/ml/sae_hooker.py`: request capture.
- `millm/core/errors.py`, `millm/api/routes/openai/errors.py`, `millm/core/config.py`, `.env.example`.
- `tests/unit/services/test_probe_wiring.py`: parametrise over discovery.
- `docs/mcp-contract.md`, `manual/docs/features/probe-monitors.md`, the OpenAI API manual page.

**Imports.** Route modules import services lazily inside handlers where the existing file does so
(`probes.py:396`), to keep import cycles out of `inference_service.py`.

## 3. Component Implementation Hints

**`run_model_work`** (place beside `_admit`, `inference_service.py:649`):
```python
async def run_model_work(self, fn: Callable[[], T]) -> T:
    """Model work that is not generation, inside one admission slot, unsteered."""
    async with self._admit():
        return await asyncio.to_thread(self._unsteered_call, fn)

def _unsteered_call(self, fn: Callable[[], T]) -> T:
    with self._unsteered():          # per-thread suppression: enter it HERE, in the worker
        return fn()
```
`test_every_request_queue_slot_is_taken_through_admission`
(`tests/unit/services/test_unload_admission.py:451-481`) stays green because only `_admit` acquires.

**`build_probe_forward`** (`probe_arm_bridge.py:119-140`): loop over `sorted(set(layers))`, install one
`ProbeHooker` hook per layer whose callback is `lambda h, _l=layer: context.observe(_l, h)`, run
`model(input_ids=ids, use_cache=False)` under `torch.inference_mode()`, remove every handle in
`finally`. Keep the device lookup as is.

**`ProbeArmingService.arm`** (`probe_arming.py:285`): add `executor: Callable[[Callable[[], Any]],
Awaitable[Any]]` with no default. At line 400:
```python
parity = await executor(lambda: ProbeParityEngine(forward).run(probe_obj, definition, ...))
```
The arm route passes `executor=inference.run_model_work`; unit tests pass an inline async executor.

**`ProbeInputPreparer`** (pure; takes the tokenizer and the renderer as arguments so tests need no
model):
- `token_ids`: `PreparedInput(ids, prompt_tokens=input.prompt_tokens, last_user=None,
  last_user_reason="token_ids_have_no_turns")`.
- `messages`, last role `assistant`: `full = render(messages, generation_prompt=False)`;
  `head = render(messages[:-1], generation_prompt=True)`; `prompt_tokens = len(head)` only if
  `full[:len(head)] == head`, else `None` (the window then reports `prompt_boundary_unknown`).
- `messages`, other last role: `ids = render(messages, generation_prompt=True)` through
  `_format_chat_messages` (`inference_service.py:5580`) and the tokenizer exactly as live serving does
  (`inference_service.py:3662`); `prompt_tokens = len(ids)`.
- `text`: `messages = [{"role": "user", "content": text}]`, then as above.
- `last_user`: `last_user_token_span(tokenizer, messages, ids, None)` (`probe_turns.py:153`).

**`ProbeScoringService.score`:**
1. `identity, model, tokenizer = await loaded_identity(session)` (`probe_arm_bridge.py:38`).
2. Resolve rows; for each: `check_identity` (`probe_identity.py:154`), `scope_is_runtime_scorable`
   (`probe_scope.py:172`), `build_probe_encoder` (`probe_arm_bridge.py:143`). Given ids → raise the
   matching existing error. Omitted ids → append `{probe_id, code, reason}` to `skipped`.
3. `probes = [armed_probe_from_row(row, encoder=enc, windows=req.windows) for …]`.
4. `pin = (current.model_id, current.loaded_at)`.
5. Per input: prepare (outside the slot; tokenisation needs no slot), then
   `await inference.run_model_work(lambda: self._score_one(i, prepared, probes, model, pin))`.
6. `_score_one` re-checks the pin, builds `ProbeRequestContext(f"score:{i}", probes)`, sets the prompt
   length and last-user span, runs `build_probe_forward(model, layers)`, returns `context.finish()`.

**Activation capture.** `RequestActivationCapture(spec, n_prompt, sae)`:
- `observe(hidden, phase)`: ignore the other phase; compute this pass's absolute positions from a
  running offset; intersect with the requested positions; slice `hidden[0, keep]`; encode in chunks of
  `SAE_ACTIVATIONS_ENCODE_CHUNK` with `sae.encode` (`millm/ml/sae_wrapper.py:364`); select `features`
  columns if given; `torch.topk(k)` on device; one `.cpu()` of `(values, indices)` per pass.
- The offset advances by the pass width **whatever was kept**, or positions drift after the first
  filtered pass.
- `positions="last"` keeps a one-slot buffer and overwrites it each pass.

## 4. Database Implementation Approach

No schema change, no migration. The only stored change is two additive keys in `probes.parity` JSON
(`millm/db/models/probe.py:131`): `model` and `checked_at`, written by `ParityReport.as_details()`
(`probe_parity.py:246`). Readers must treat both as optional; reports written before this feature have
neither, and the score response then says `checked_against: "unknown"`.

## 5. API Implementation Strategy

- **Score route** in `millm/api/routes/management/probes.py` (router prefix `/api/probes`, line 43):
  `@router.post("/score", response_model=ApiResponse)`. Route order does not matter here, because
  `/{probe_id}/…` routes have one more path segment. A test still asserts that `POST /api/probes/score`
  reaches the score handler and no `{probe_id}` handler.
- **Parity route** (`probes.py:385-418`): replace the direct `ProbeParityEngine(...).run(...)` at
  412-416 with `report = await inference.run_model_work(lambda: ProbeParityEngine(fwd).run(...))`.
  Add `inference: InferenceServiceDep` (`millm/api/dependencies.py:178`).
- **Arm route** (`probes.py:340-383`): pass `executor=inference.run_model_work` at 364-373.
- **`/v1` routes:** validate `return_sae_activations` shape in the route (before the slot): refuse
  `n > 1`, `extra_messages`, a prompt list longer than one, llama.cpp, missing or ambiguous SAE, and the
  worst-case cap.
- **Completions header:** after `create_text_completion` (`completions.py:148`), set
  `X-miLLM-Probe-Verdicts` from `build_probe_verdicts_header(get_probe_verdicts())`, as `chat.py:295-297`.
- **Response `millm` object:** on `ChatCompletionResponse` (`openai.py:334`) and
  `TextCompletionResponse` (`openai.py:421`), `millm: Optional[MillmExtension] = None` with a
  `@model_serializer(mode="wrap")` that drops the key when `None`. Do **not** use route-level
  `response_model_exclude_none`: it would also drop fields OpenAI clients expect as `null`.
- **Errors:** `ProbeScoreRequestError` (`INVALID_PROBE_SCORE_REQUEST`, 400) and
  `SaeActivationsRefusedError` (`SAE_ACTIVATIONS_REFUSED`, 400) in `millm/core/errors.py`; an
  `ERROR_STATUS_MAP` row for the latter (`millm/api/routes/openai/errors.py:61`).

## 6. Frontend Implementation Approach

N/A for new UI. The Probe Monitors page already lists `not_scored_reason` values; the new reasons
(`batched_request` on batched chat, `continuous_batching` on CBM chat and text, `engine_unsupported`)
appear without code change. Check the page renders an unknown reason as text, not a blank.

## 7. Business Logic Implementation Hints

- **Window resolution:** `resolve_windows(req.windows, probe_scope=row.scope, calibrated=…)` inside
  `armed_probe_from_row`; do not resolve twice.
- **Verdict field mapping:** `verdict = v.fires` (keep `None`), `rung_language = v.rung_language`
  verbatim.
- **Parity status:** `never_run` when `row.parity is None`; else `passed` / `failed` from
  `row.parity["passed"]`, plus `checked_against = row.parity.get("model") or "unknown"`.
- **Per-input errors:** `MODEL_CHANGED` when the pin differs; `TOKENIZATION_FAILED` when rendering
  raises. Both fill `error` and leave `verdicts` empty.
- **Detached contexts** for paths that never score: `_probe_begin_detached(rid, reason)` returns
  `None` when nothing is armed, else a context over `ProbeRuntimeState().armed()` already marked.
  Call `_probe_finish(ctx)` before the response, and `_probe_record(ctx, verdicts, full_ids,
  detached=True)` in `finally`.

## 8. Testing Implementation Approach

**Discovery guard** (`test_probe_paths_discovered.py`):
```python
PRIMITIVE_METHODS = {"_generate_sync", "_generate_in_thread", "_llamacpp_sync"}
PRIMITIVE_ATTRS = {("_cbm_backend", "generate"), ("_cbm_backend", "generate_stream"),
                   ("_model", "generate"), ("_model", "create_completion"),
                   ("_model", "create_chat_completion")}
```
- Walk each `FunctionDef`/`AsyncFunctionDef` in `InferenceService` **including nested closures**:
  `_llamacpp_text_completion` calls `self._model.create_completion` inside a local `_complete`
  (`inference_service.py:4246`), and the llama stream path opens it inside `_open_stream`.
- A reference counts when it is a `Call.func` **or** an argument of any call (`to_thread(self._x)`,
  `Thread(target=self._x)`).
- The primitive methods themselves are not sites.
- Event log: `threading.Lock` + list. Fakes return minimal shaped outputs (a tensor of ids, a CBM
  `(ids, "stop")` tuple, a llama dict).

**No-event guard:** real SQLite test session and real `ProbeEventRepository`; `begin_request` wrapped
with `wraps=` and asserted `call_count == 0`.

**Boundary guard:** AST over `millm/`; flag any `Compare` with `>`, `>=`, `<`, `<=` whose operands
include a name or attribute containing `threshold` and a name containing `score` or `value`, outside
`ProbeRequestContext._verdict_for`. Assert the flagged set is exactly `{_verdict_for}` — exactly, so a
broken walker that finds nothing fails.

**Mutation controls to record** (back up, mutate, run, restore, verify the restore with `git diff`
and a re-grep of the mutated line):

| # | Mutation | Expected red |
|---|---|---|
| M1–M10 | delete `_probe_begin`/`_probe_begin_detached` in each of the ten generation paths, one at a time | discovery guard |
| M11 | add a method calling `self._generate_sync` with no scenario | discovery guard ("site not in table") |
| M12 | `>=` → `>` at `probe_runtime.py:665` | live and stateless boundary tests |
| M13 | call `state.begin_request` inside `_score_one` | no-event guard |
| M14 | write a `probe_events` row from the score route | no-event guard |
| M15 | drop `async with self._admit()` from `run_model_work` | admission spies |
| M16 | drop `executor=` from the arm route | arm route payload test |
| M17 | enter `_unsteered()` outside the worker thread | suppression-in-thread test |
| M18 | capture shared across requests (never cleared) | activation isolation test |
| M19 | read point ignored (always pre) | read-point test |
| M20 | remove the score router route | reachability test |
| M21 | drop the completions verdict header | completions header test |

## 9. Configuration and Environment Strategy

In the `PROBE_*` block of `millm/core/config.py` (around line 176) and a new `SAE_ACTIVATIONS_*`
block, each with a comment saying what it bounds; commented lines in `.env.example`:
```
PROBE_SCORE_MAX_INPUTS: int = 64        # miStudio's millm_score_probes cap (034 FTDD §9)
PROBE_SCORE_MAX_PROBES: int = 8         # = PROBE_MAX_ARMED
SAE_ACTIVATIONS_MAX_TOP_K: int = 64
SAE_ACTIVATIONS_MAX_ENTRIES: int = 65536
SAE_ACTIVATIONS_ENCODE_CHUNK: int = 512
```
No feature flag. `text` input is enabled only after FTASKS 0.1 passes; until then it is refused with
`INVALID_PROBE_SCORE_REQUEST` naming the pending verification.

## 10. Integration Strategy

- **Feature 24:** parity and arm behaviour change only in admission and suppression. Re-run the whole
  `tests/unit/services/test_probe_*` family.
- **Feature 25:** register `return_sae_activations` in its known-field set. If Feature 25 has not
  landed, add the field to the schema normally; its validation will pick it up.
- **Feature 26:** a batch row for `/api/probes/score` calls `ProbeScoringService.score` with one
  input; never packed.
- **Features 25, 26, 28** also add to the `millm` response object. `MillmExtension` lives in its own
  module so each feature adds a field without touching the others.
- **Contract:** `docs/mcp-contract.md` endpoint inventory (§4 `millm_probes`) and §4d boundary rule;
  `tests/unit/test_mcp_contract_consistency.py` and `test_mcp_tool_paths_are_real.py` must stay green.

## 11. Utilities and Helpers Design

- `probe_scoring.verdict_payload(v: Verdict) -> dict` — the one wire mapping, reused by Feature 26's
  batch output lines.
- `request_activations.resolve_positions(spec, n_prompt, max_new_tokens) -> int` — the worst-case
  count used by the pre-generation cap check; pure.
- `request_activations.select_sae(spec, entries)` — picks the attached entry or raises with the
  candidates.

## 12. Error Handling and Logging Strategy

- Whole-request refusals raise `MiLLMError` subclasses (FTDD §5.1 table).
- Per-input failures are data, not exceptions, so 63 good inputs are not lost to one bad one.
- In generation paths, activation and probe seams log `warning` and continue on any exception.
- Logs: `probe_score requests=… probes=… skipped=… inputs=… elapsed_ms=…` at info; nothing derived from
  content. `sae_activations sae=… positions=… entries=…` at debug.
- The hung-thread guard (`inference_service.py:4750-4755`) also calls `end_request_capture()` on every
  attached SAE.

## 13. Performance Implementation Hints

- Tokenise outside the slot; hold the slot only for the forward.
- Hooks: one per distinct layer per input; measure the dynamo reset cost
  (`millm/ml/probe_hooker.py:100`, `:111`) at acceptance and hoist installs out of the loop if needed.
- Activations: slice before encoding; chunked encode; top-k on device; one host copy per pass.
- `n_tokens` and `token_ids` echo come from the prepared list, not from the device.

## 14. Code Quality and Standards

- Black (100), Ruff, MyPy on new modules; Google docstrings; every guard's docstring says what
  mutation it catches.
- Comments carry the *why* with a citation, as the Feature 24 code does; no restating code.
- No hand-kept list of paths anywhere. The scenario table in the guard is checked for equality against
  discovery.
- Debt recorded, not fixed here: per-row probe scoring inside batched chat (BRD-04 §7).

## 15. Decisions from Clarifying Questions

Rounds waived; all answers come from the FPRD (D1–D22) and FTDD (TD1–TD16). Implementation-level:

| # | Question | Answer | Source |
|---|---|---|---|
| ID1 | Where does admission live for non-generation work? | `InferenceService.run_model_work` | FTDD §6 |
| ID2 | Is `executor` optional on `arm`? | No; required | FTDD §12 |
| ID3 | How is `millm` omitted when absent? | Model-level wrap serializer | §5 (route-level exclude would drop OpenAI nulls) |
| ID4 | How are nested closures handled by discovery? | Walked as part of the enclosing method | §8; `inference_service.py:4246` |
| ID5 | Where do the new defaults come from? | §9 values | FTDD TD14 |
| ID6 | Is `text` enabled before T-49 verification? | No; refused naming the pending check | T-49 |

**Open items:** none beyond FTASKS 0.1.
