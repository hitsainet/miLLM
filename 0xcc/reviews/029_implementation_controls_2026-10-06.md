# Feature 29 — Implementation Controls and Review Record (2026-10-06)

**Feature:** 029 Model Lease, Backpressure and GPU Visibility · **Branch:** `feat/029-model-lease`
(on `feat/025-chat-scoring` @ `39c6f8e`) · **Worktree:** `~/app/miLLM-029`
**Interpreter:** `~/app/miLLM/venv/bin/python`, with `PYTHONPATH` at the worktree;
`millm.__file__` verified as `/home/x-sean/app/miLLM-029/millm/__init__.py` before the first run.
One pytest process at a time throughout.

## 1. Suite counts

| When | `PYTHONPATH=$PWD venv/bin/python -m pytest tests/unit` |
|---|---|
| Baseline (`39c6f8e`, before any edit) | 4052 passed / 3 skipped / 0 failed (155 s) |
| After 1.0 (errors, settings, value policy) | 4103 passed / 3 skipped / 0 failed |
| After 2.0 + 3.0 (registry, service API, enforcement) | 4179 passed / 3 skipped / 0 failed |
| After 4.0 (startup reconciliation, self-heal) | 4184 passed / 3 skipped / 0 failed |
| After 5.0 (lease routes, headers, refuse policy) | 4251 passed / 3 skipped / 0 failed |
| After 6.0 (Retry-After on every 503) | 4276 passed / 3 skipped / 0 failed |
| After 7.0 (queue state, health contract) | 4293 passed / 3 skipped / 0 failed |
| After 8.0 (GPU memory endpoint) | 4311 passed / 3 skipped / 0 failed |
| After 10.0 (manual, error codes, configuration, MCP contract v1.9) | 4311 passed / 3 skipped / 0 failed |
| Final (after controls + M12 follow-up test) | **4312 passed / 3 skipped / 0 failed** |
| Final, mirror view (copy without `0xcc/` and with `docs/` reduced to `docs/schemas/`, `PYTHONPATH` at the copy, `millm.__file__` verified under it) | **4269 passed / 46 skipped / 0 failed** — the 43 extra skips are document-reading tests that skip loudly when the file is absent |

Admin UI (`npx vitest run`): **457 passed / 41 files** before 9.0 (468 − the 11 new), **468 passed / 41 files** after; `tsc -b --noEmit` clean; `eslint` on the touched files reports the same 3 pre-existing errors as the base tree (GGUFQuantPicker ×2, ModelLoadForm), none in new code.

## 2. Task 0.1 — cited lines re-verified at HEAD (`39c6f8e`)

`model_service.py` lines cited by the FTDD/FTID (782, 832–837, 850–853, 884–892, 1194, 1212–1218,
1281–1312, 1442, 1468–1483) all still match. `main.py` (134, 277, 379, 529), `model_loader.py`
(146, 277, 333), `request_queue.py` (37, 147–169), `inference_service.py:222`
(`_stream_error_event`; FTID says 206), `health.py` (127, 261–262, 361–370), `Dockerfile:124`,
`docs/mcp-contract.md:92`, `ModelsPage.tsx:292-296`, `useModels.ts:29-35` match.

Moved (Feature 25 edited the route files):

| Cited | Now |
|---|---|
| `chat.py:101` route function | `chat.py:118` |
| `completions.py:51` | `completions.py:63` |
| `embeddings.py:47` | `embeddings.py:53` |
| auto-load blocks `chat.py:143-146`, `completions.py:112-115`, `embeddings.py:80-83` | `chat.py:181-183`, `completions.py:124-126`, `embeddings.py:87-89` |
| `inference_service.py:206` `_stream_error_event` | `inference_service.py:222` |

## 3. Task 0.3 — Feature 26's names

026 FTDD §7 (lines 184–201, 239, 296) calls `renew_lease(model_id, lease_id, ttl_seconds)`,
`acquire_lease(model_id, holder=…, ttl_seconds=…, reason=…)`, `release_lease(model_id, lease_id)`,
`resolve_lease(lease_id)`, `RequestQueue.holding_count`, `background_holding_count` and
`register_backlog_provider`. All are implemented here with those names and argument orders. No drift.
026 cites "029 FTDD §5.2" for `register_backlog_provider`; the FTDD places it in §7.3 / FTID §7 — a
section reference only, recorded, not a name drift.

## 4. Document ↔ code discrepancies (code wins)

| # | Document says | Code says / what was done |
|---|---|---|
| D-1 | FTID §5 adds only `MODEL_LEASED` and `MODEL_NOT_RESIDENT` to `ERROR_STATUS_MAP` | `tests/unit/api/test_error_map_complete.py` (Feature 25) requires a row for EVERY `MiLLMError` code. `LEASE_NOT_FOUND` (404), `LEASE_EXPIRED` (409) and `INVALID_LEASE_REQUEST` (400) were added too, all `invalid_request_error`. |
| D-2 | FPRD FR-29.4.5 / FTID §3.4: the GGUF refusals sit in the routes (`completions.py:76-94`, `embeddings.py:60-73`) | Feature 25 moved them into the request policy (`apply_request_policy`), which already runs before the auto-load. `apply_load_policy` runs after it, so the GGUF and embedding-only refusals still come first. |
| D-3 | FTDD §5.3: `HUB_UNAVAILABLE` uses "the breaker's remaining recovery time" citing `resilience.py` | The code that raises `HUB_UNAVAILABLE` uses its own breaker, `cluster_hub_service.cluster_hub_circuit`, not `huggingface_circuit`. The remainder is read from that breaker; with it closed (one network failure), the value is 1. |
| D-4 | FTID §3.6 adds `_holding` only; `in_flight = holding_count + background_holding_count` with the second "Feature 26's" | `RequestQueue.background_holding_count` was added here, fixed at 0, so the estimate reads one real attribute instead of a `getattr` default. Feature 26 increments it. |
| D-5 | FTID §3.3: the unload-success `end_for_model` goes "before the auto-unlock update" | Placed immediately after the unload completes, BEFORE the probe-row database write, so a failing write cannot leave a lease on an unloaded model. Same meaning, earlier. |
| D-6 | FTID §5: `LeaseCreateRequest` declares `holder: str`, `reason: str` | Declared `Any` so a missing or non-string holder/reason reaches the service and is `400 INVALID_LEASE_REQUEST` naming the field, like every other bound (a pydantic type error would be a bare 422). |
| D-7 | FTID §11: `lease_summary(record) -> LeaseSummary` used by ModelResponse, health and the GET route; FTID §7.5 types the health field `LeaseStatusResponse` | One class: `LeaseSummary` is an alias of `LeaseStatusResponse`, so the three reads cannot drift. `_with_lease` reads the process registry directly (no service method, no database). |
| D-8 | FTDD §5.1: `DELETE` returns "the ended `LeaseStatusResponse`" | Returns `EndedLeaseResponse` (FTID §2's name): the status fields that still mean something plus `end_reason` and `ended_at`. |
| D-9 | FPRD §2 edge table: "lease requested while a load or unload is running → `503 model_busy` with `Retry-After`"; FTDD §5.1 lists `503 MODEL_BUSY` on the grant | The lease routes are management routes, where `ModelBusyError` is 409 (`errors.py`), and FPRD §9 / D13 keep management `MODEL_BUSY` at 409. The grant answers `409 MODEL_BUSY`, carrying `unloading` when an unload is the cause. A 409 owes no `Retry-After`. Documented as 409 in the manual and contract. |

## 5. Pre-existing defects found and fixed

| # | Defect | Fix |
|---|---|---|
| P-1 | `ModelService.unload_model`'s "already being unloaded" and `load_model_and_wait`'s "still being unloaded" `ModelBusyError`s carried no unload mark, so the new policy would have answered them with the 15 s LOAD value | Both now carry `details["unloading"] = True` → 5 s. |
| P-3 | Five assertions in `test_model_load_gpu_request.py` pinned `load_model` to `(3, gpu=…)` | Now `(3, gpu=…, lease_id=None)`: the route passes the header. |
| P-4 | The manual said a management unload of a locked model returns `409 MODEL_LOCKED` and that locking "prevents unload/delete" (`manual/docs/api/models.md`, `features/model-management.md`, `reference/error-codes.md`). The code has never read `locked` on the management load or unload (FPRD §9 tracked debt), and delete refuses on LOADED, not on `locked` | The three pages now say what the code does, name the gap as tracked debt and point to the lease. |
| P-2 | Five existing assertions in `test_load_refusal_keeps_resident_model.py` pinned `unload_model` to be awaited with exactly `(9)` | Now `(9, lease_id=None)`: the internal unload carries the load's lease ID (FTID §3.3). The assertions' purpose (unloaded once, the right model) is unchanged. |

## 6. Notes

- `_refuse_if_leased` logs `lease_header_unmatched` once per guarded operation, so a swap with a
  stale header logs two (the load and its internal unload), each naming its operation.
- Tasks 2.0 and 3.0 were committed together: both live in `ModelService` and the enforcement tests
  exercise the API the registry tests build on.

## 7. Mutation controls

Runner: `scratchpad/impl-029/mut/run.py` — for each control: copy the file aside, assert the target text occurs exactly once, replace it, run the listed test files (`-x`), restore from the copy, then verify the restore by **sha256 equal to the pre-mutation hash** and by **re-grepping the original line (exactly one occurrence)**. `git status` was checked clean before the batch and after it. One pytest process at a time.

**40 runs over 39 distinct mutations: M1–M18 (FTID §8.4, M13 run on each of the three routes) and E1–E18 (every remaining lease-enforcement path and wiring line). 1 survived first time (M12); its follow-up is below.**

| # | File | Mutation | Landed | Result | First red / summary | Restore sha256 | Re-grep |
|---|---|---|---|---|---|---|---|
| M1 | `millm/services/model_service.py` | delete the _refuse_if_leased call in load_model | yes | **red** | FAILED tests/unit/services/test_lease_enforcement.py::TestForeignLeaseRefusesEveryPath::test_management_load_of_another_model | ok | ok |
| M2 | `millm/services/model_service.py` | delete the _refuse_if_leased call in unload_model | yes | **red** | FAILED tests/unit/services/test_lease_enforcement.py::TestForeignLeaseRefusesEveryPath::test_management_unload | ok | ok |
| M3 | `millm/services/model_service.py` | delete the _refuse_if_leased call in load_model_and_wait | yes | **red** | FAILED tests/unit/services/test_lease_enforcement.py::TestForeignLeaseRefusesEveryPath::test_auto_load | ok | ok |
| M4 | `millm/services/model_service.py` | drop lease_id from the internal unload a load performs | yes | **red** | FAILED tests/unit/services/test_lease_enforcement.py::TestHolderProceeds::test_holder_swap_succeeds_and_ends_the_lease | ok | ok |
| M5 | `millm/services/model_lease.py` | expiry `<=` -> `<` | yes | **red** | FAILED tests/unit/services/test_model_lease.py::TestExpiry::test_honoured_until_the_deadline_and_gone_at_it | ok | ok |
| M6 | `millm/services/model_lease.py` | matches() always True (after the empty-ID guard) | yes | **red** | FAILED tests/unit/services/test_model_lease.py::TestResolveAndMatches::test_matches | ok | ok |
| M7 | `millm/services/model_service.py` | remove end_for_model in unload_model | yes | **red** | FAILED tests/unit/services/test_lease_enforcement.py::TestHolderProceeds::test_unload_ends_the_lease_even_when_the_registry_could_not_see_it | ok | ok |
| M8 | `millm/services/model_lease.py` | remove the residency branch in current() | yes | **red** | FAILED tests/unit/services/test_model_lease.py::TestResidencySelfHeal::test_a_lease_whose_model_left_reads_as_none | ok | ok |
| M9 | `millm/main.py` | delete the clear_leases_on_startup() call in lifespan | yes | **red** | FAILED tests/unit/test_startup_reset.py::TestLeasesEndAtRestart::test_lifespan_calls_it | ok | ok |
| M10 | `millm/services/model_lease.py` | make clear() a no-op (ends nothing) | yes | **red** | FAILED tests/unit/test_startup_reset.py::TestLeasesEndAtRestart::test_clear_leases_on_startup_empties_a_filled_registry | ok | ok |
| M11 | `millm/core/errors.py` | ModelLeasedError subclasses ModelLockedError | yes | **red** | FAILED tests/unit/core/test_backpressure.py::TestLeaseErrors::test_model_leased_is_not_model_locked | ok | ok |
| M12 | `millm/api/schemas/lease.py` | the shared lease serialiser emits a lease_id on GET / list / health | yes | **SURVIVED** | ============================== 45 passed in 6.29s ============================== | ok | ok |
| M13a | `millm/api/routes/openai/chat.py` | remove apply_load_policy from /v1/chat/completions | yes | **red** | ERROR    millm.services.probe_arming:probe_arming.py:549 probes_disarmed_by_unload_failed error=password authentication failed for user "postgres" — a | ok | ok |
| M13b | `millm/api/routes/openai/completions.py` | remove apply_load_policy from /v1/completions | yes | **red** | ERROR    millm.services.probe_arming:probe_arming.py:549 probes_disarmed_by_unload_failed error=password authentication failed for user "postgres" — a | ok | ok |
| M13c | `millm/api/routes/openai/embeddings.py` | remove apply_load_policy from /v1/embeddings | yes | **red** | ERROR    millm.services.probe_arming:probe_arming.py:549 probes_disarmed_by_unload_failed error=password authentication failed for user "postgres" — a | ok | ok |
| M14 | `millm/api/exception_handlers.py` | remove Retry-After from millm_error_handler | yes | **red** | FAILED tests/unit/api/test_retry_after.py::TestEveryProducible503::test_v1_handler_sets_the_codes_own_value[queue_full-5] | ok | ok |
| M15 | `millm/main.py` | remove add_middleware(RetryAfterMiddleware) | yes | **red** | FAILED tests/unit/api/test_retry_after.py::TestEveryProducible503::test_bare_503_gets_the_fallback_and_the_warning | ok | ok |
| M16 | `millm/core/backpressure.py` | backlog_rows() returns 0 when unregistered | yes | **red** | FAILED tests/unit/core/test_backpressure.py::TestBacklogProvider::test_unregistered_is_none_not_zero | ok | ok |
| M17 | `millm/services/gpu_memory.py` | read torch on an untouched card | yes | **red** | FAILED tests/unit/services/test_gpu_memory.py::TestTwoCards::test_only_the_touched_card_is_measured | ok | ok |
| M18 | `millm/services/model_service.py` | grant ignores _loading_model_id | yes | **red** | FAILED tests/unit/services/test_lease_enforcement.py::TestGrant::test_grant_during_a_load_is_refused | ok | ok |
| E1 | `millm/api/routes/management/models.py` | management load route stops passing X-miLLM-Lease | yes | **red** | FAILED tests/unit/api/test_lease_routes.py::TestHeaderPassThrough::test_management_load_refused_without_and_proceeds_with_the_header | ok | ok |
| E2 | `millm/api/routes/management/models.py` | management unload route stops passing X-miLLM-Lease | yes | **red** | FAILED tests/unit/api/test_lease_routes.py::TestHeaderPassThrough::test_management_unload_refused_without_and_proceeds_with_the_header | ok | ok |
| E3 | `millm/api/routes/openai/chat.py` | chat route drops the lease header on the auto-load | yes | **red** | FAILED tests/unit/api/test_load_policy.py::TestRefusePolicy::test_explicit_auto_auto_loads_and_passes_the_lease_header[chat] | ok | ok |
| E4 | `millm/api/routes/openai/completions.py` | completions route drops the lease header on the auto-load | yes | **red** | FAILED tests/unit/api/test_load_policy.py::TestRefusePolicy::test_explicit_auto_auto_loads_and_passes_the_lease_header[completions] | ok | ok |
| E5 | `millm/api/routes/openai/embeddings.py` | embeddings route drops the lease header on the auto-load | yes | **red** | FAILED tests/unit/api/test_load_policy.py::TestRefusePolicy::test_explicit_auto_auto_loads_and_passes_the_lease_header[embeddings] | ok | ok |
| E6 | `millm/services/model_service.py` | load_model_and_wait does not pass lease_id to load_model (holder auto-load refused) | yes | **red** | FAILED tests/unit/services/test_lease_enforcement.py::TestHolderProceeds::test_holder_auto_load_proceeds | ok | ok |
| E7 | `millm/services/model_lease.py` | grant ignores a live lease (X-08) | yes | **red** | FAILED tests/unit/services/test_model_lease.py::TestGrant::test_second_grant_on_a_live_lease_is_refused_naming_the_holder | ok | ok |
| E8 | `millm/services/model_lease.py` | renew extends from the old deadline, not now | yes | **red** | FAILED tests/unit/services/test_model_lease.py::TestRenew::test_new_expiry_is_now_plus_ttl_not_old_plus_ttl | ok | ok |
| E9 | `millm/core/backpressure.py` | the `unloading` mark no longer selects the unload value | yes | **red** | FAILED tests/unit/core/test_backpressure.py::TestValuePerCode::test_the_inference_unloading_mark_is_an_unload | ok | ok |
| E10 | `millm/services/inference_service.py` | in-stream error loses retry_after | yes | **red** | FAILED tests/unit/api/test_retry_after.py::TestInStreamError::test_a_503_code_carries_retry_after_in_the_event | ok | ok |
| E11 | `millm/api/routes/system/health.py` | readiness 503 loses its Retry-After (falls to the middleware) | yes | **red** | FAILED tests/unit/api/test_retry_after.py::TestReadiness::test_unready_probe_has_its_own_value | ok | ok |
| E12 | `millm/ml/model_loader.py` | LoadedModelState.set() stops recording touched cards | yes | **red** | FAILED tests/unit/services/test_gpu_memory.py::TestTouchedSet::test_a_transformers_placement_is_recorded_and_kept_after_unload | ok | ok |
| E13 | `millm/api/routes/openai/errors.py` | model_busy_error builder loses Retry-After | yes | **red** | FAILED tests/unit/api/test_retry_after.py::TestTheRouteBuilders::test_model_busy_from_the_route[exc0-15] | ok | ok |
| E14 | `millm/api/routes/openai/load_policy.py` | refuse policy no longer answers 503 model_loading for a loading model | yes | **red** | FAILED tests/unit/api/test_load_policy.py::TestRefusePolicy::test_refuse_while_the_model_loads_is_503_with_retry_after[chat] | ok | ok |
| E15 | `millm/api/routes/system/health.py` | in_flight reported under CBM (a count that leaves CBM out) | yes | **red** | FAILED tests/unit/api/test_health_queue_state.py::TestCounts::test_cbm_running_nulls_the_unmeasurable_fields | ok | ok |
| E16 | `millm/services/request_queue.py` | the holding counter is never incremented | yes | **red** | FAILED tests/unit/api/test_health_queue_state.py::TestCounts::test_one_holding_two_waiting | ok | ok |
| E17 | `millm/api/routes/management/models.py` | ModelResponse.lease never populated | yes | **red** | FAILED tests/unit/api/test_lease_routes.py::TestReadsNeverCarryTheId::test_model_list_and_single_carry_the_summary_not_the_id | ok | ok |
| E18 | `millm/api/routes/system/health.py` | health lease field never populated | yes | **red** | FAILED tests/unit/api/test_health_queue_state.py::TestDetailedEndpoint::test_the_lease_appears_without_its_id | ok | ok |
| M12b | `millm/api/schemas/lease.py` | the read schema declares a lease_id field (what a real leak through GET needs) | yes | **red** | FAILED tests/unit/api/test_lease_routes.py::TestGrantRoute::test_201_carries_the_lease_id_and_calls_the_service_once | ok | ok |
| M12-rerun | `millm/api/schemas/lease.py` | M12 re-run after the schema test (expected to stay green: a genuine no-op over HTTP) | yes | **SURVIVED** | ============================== 46 passed in 6.71s ============================== | ok | ok |

**M12 survived first time, and the reason is structural, not a gap in a read path.** The mutation made the shared serialiser emit an undeclared `lease_id`; every read route declares a `response_model`, and FastAPI rebuilds the response from the DECLARED schema, dropping the key. The registry also never holds the ID (only its digest), so no read can produce it. That makes the declared schemas the real guard, and nothing pinned them. **Test written:** `test_lease_routes.py::test_only_the_grant_schema_declares_lease_id` reads the live app's OpenAPI and requires `LeaseGrantResponse` to be the only schema with `lease_id`, and the GET route not to reference it. **Negative control M12b** (declare `lease_id` on the read schema — what a real leak needs) → red. **M12 re-run** after the test → still green, recorded as a measured no-op over HTTP (the undeclared key cannot reach a client), not argued away.

**Slow controls:** M13a–c each took ~185 s before going red: with the refuse policy removed, the request reaches the real `load_model_and_wait`, whose stub worker never marks the row LOADED, so it polls to its 180 s timeout. Red, but slow; the policy tests pass in < 6 s on the unmutated tree.

## 8. Feature Acceptance walk (task 11.1)

### FPRD §2 edge cases

| Edge case | Test |
|---|---|
| Lease requested while another holder's lease is live | `test_lease_enforcement.py::TestGrant::test_grant_over_a_live_lease_is_refused`; `test_lease_routes.py::TestGrantRoute::test_409_leased_names_holder_and_expiry` |
| Lease on a model that is not resident | `TestGrant::test_non_resident_model_is_refused_naming_the_resident`, `test_nothing_resident_says_none`; `TestGrantRoute::test_409_not_resident` |
| Lease while a load or unload is running | `TestGrant::test_grant_during_a_load_is_refused` (M18), `test_grant_during_an_unload_is_refused_as_an_unload`; route answers 409 MODEL_BUSY (D-9) |
| `ttl_seconds` above 7200, zero, negative, non-integer | `TestGrant::test_bad_ttl_is_400_naming_field_and_limit`; `TestGrantRoute::test_400_names_the_field_and_the_limit` |
| Renew/release with wrong or expired ID | `test_model_lease.py::TestRenew`, `TestRelease`; `TestRenewAndRelease::test_unknown_id_404_with_the_restart_sentence`, `test_expired_lease_renew_is_409` |
| `/v1` request names the leased model without the ID | `TestHolderProceeds::test_a_request_for_the_leased_resident_model_needs_no_header`; `test_load_policy.py::test_refuse_for_the_resident_model_continues` |
| Refuse policy, model loading now | `TestRefusePolicy::test_refuse_while_the_model_loads_is_503_with_retry_after`, `test_refuse_while_the_slot_holds_this_model_is_503` |
| 503 inside a committed stream | `test_retry_after.py::TestInStreamError` (E10) |
| Restart while a lease is held | `test_startup_reset.py::TestLeasesEndAtRestart` (M9, M10); hardware 11.5 open |
| nvidia-smi absent | `test_gpu_memory.py::TestNoNvidiaSmi`; `test_gpu_memory_route.py::test_no_nvidia_smi_is_200_with_a_reason` |

### FPRD §11 success criteria

| # | Criterion | Unit evidence | Status |
|---|---|---|---|
| 1 | Lease refuses another model's chat and load; TTL lapse lets the load through | `TestForeignLeaseRefusesEveryPath`, `test_an_expired_lease_refuses_nothing`, `test_foreign_lease_refuses_the_auto_load_as_model_leased[*]`, `TestHeaderPassThrough` | unit ✅ · **hardware 11.3 open** |
| 2 | Holder's load proceeds; the lease ends with the unloaded model | `TestHolderProceeds::test_holder_swap_succeeds_and_ends_the_lease` (M4, M7), route equivalent | unit ✅ · hardware 11.3 open |
| 3 | Refuse + non-resident → 409, load count 0 | `TestRefusePolicy::test_refuse_non_resident_is_409_and_nothing_loads[*]` (M13a–c) | unit ✅ · hardware 11.3 open |
| 4 | Eleven concurrent → 503 queue_full + Retry-After ≥ 1 | `test_retry_after.py::TestQueueFullBurst` (real `RequestQueue`, httpx ASGI, 11 requests) | unit ✅ · hardware 11.5 open |
| 5 | Every producible 503 carries its own Retry-After | `TestEveryProducible503`, `TestTheRouteBuilders`, `TestReadiness` (M14, M15, E11, E13) | ✅ |
| 6 | Queue contract; 1 running + 2 waiting → 1 / 2 | `test_health_queue_state.py::TestCounts::test_one_holding_two_waiting`, `TestContract`, `TestBacklog` (M16, E15, E16) | unit ✅ · hardware 11.5 open |
| 7 | GPU memory matches nvidia-smi within 256 MiB; no context on an untouched card | `test_gpu_memory.py` (M17, E12) | unit ✅ · **hardware 11.4 open (needs spike 0.2)** |
| 8 | Models page and health show holder, reason, expiry, never the ID | `TestReadsNeverCarryTheId`, `test_grant_response_json_is_the_only_place_the_id_appears`, `test_only_the_grant_schema_declares_lease_id`, `TestDetailedEndpoint::test_the_lease_appears_without_its_id`, `LeaseBadge.test.tsx` (M12b, E17, E18) | ✅ |

## 9. Tracked debt (task 11.7)

- **The management load and unload ignore `locked`** (`management/models.py` load/unload → `ModelService.load_model`/`unload_model`, neither reads it). A steering model can be evicted from the Admin UI or by `millm_load_model`. Retiring `locked` in favour of the lease (C8) is the intended fix. The manual now says so (P-4).
- **Single-process registry.** Leases live in one uvicorn worker's memory (`Dockerfile:124`, no `--workers`). Several workers would each hold a registry; the registry must move to shared storage before that.
- **Restart ends every lease** (X-01) — by decision, not debt; recorded so a consumer does not read it as one.
- `background_holding_count` is fixed at 0 until Feature 26 counts batch chunks (D-4).
- Lint: `ruff` (miStudio's 0.1.15 binary; miLLM's venv has none) is clean on the new modules except the repo-wide `timezone.utc` style (UP017, kept for consistency with every other module) and ARG002 on pytest fixtures used for their side effects. `mypy` is clean on the six new modules. `eslint` on the touched UI files: the same 3 pre-existing errors as the base tree.

## 10. Hardware — for the operator session

- **0.2 (T-89 spike):** in the pod, `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits` with a model loaded — does miLLM's PID appear? After an unload, `torch.cuda.memory_reserved(i)` vs the node's per-process `used_memory` → `cuda_context_mb`. `/api/health/gpus` already reports `processes` (or `null` with a reason) and `millm_reserved_mb`, so most of this can be read from the endpoint.
- **11.3** lease acceptance (BRD-04 acceptance 14), **11.4** GPU visibility within 256 MiB using 0.2's context size (acceptance 16), **11.5** eleven concurrent requests on the node + restart ends the lease and the badge.
