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
| D-5 | FTID §3.3: the unload-success `end_for_model` goes "before the auto-unlock update" | Placed immediately after the unload completes, BEFORE the probe-row database write, so a failing write cannot leave a lease on an unloaded model. Same meaning, earlier. |
| D-6 | FTID §5: `LeaseCreateRequest` declares `holder: str`, `reason: str` | Declared `Any` so a missing or non-string holder/reason reaches the service and is `400 INVALID_LEASE_REQUEST` naming the field, like every other bound (a pydantic type error would be a bare 422). |
| D-7 | FTID §11: `lease_summary(record) -> LeaseSummary` used by ModelResponse, health and the GET route; FTID §7.5 types the health field `LeaseStatusResponse` | One class: `LeaseSummary` is an alias of `LeaseStatusResponse`, so the three reads cannot drift. `_with_lease` reads the process registry directly (no service method, no database). |
| D-8 | FTDD §5.1: `DELETE` returns "the ended `LeaseStatusResponse`" | Returns `EndedLeaseResponse` (FTID §2's name): the status fields that still mean something plus `end_reason` and `ended_at`. |
| D-4 | FTID §3.6 adds `_holding` only; `in_flight = holding_count + background_holding_count` with the second "Feature 26's" | `RequestQueue.background_holding_count` was added here, fixed at 0, so the estimate reads one real attribute instead of a `getattr` default. Feature 26 increments it. |

## 5. Pre-existing defects found and fixed

| # | Defect | Fix |
|---|---|---|
| P-1 | `ModelService.unload_model`'s "already being unloaded" and `load_model_and_wait`'s "still being unloaded" `ModelBusyError`s carried no unload mark, so the new policy would have answered them with the 15 s LOAD value | Both now carry `details["unloading"] = True` → 5 s. |
| P-3 | Five assertions in `test_model_load_gpu_request.py` pinned `load_model` to `(3, gpu=…)` | Now `(3, gpu=…, lease_id=None)`: the route passes the header. |
| P-2 | Five existing assertions in `test_load_refusal_keeps_resident_model.py` pinned `unload_model` to be awaited with exactly `(9)` | Now `(9, lease_id=None)`: the internal unload carries the load's lease ID (FTID §3.3). The assertions' purpose (unloaded once, the right model) is unchanged. |

## 6. Notes

- `_refuse_if_leased` logs `lease_header_unmatched` once per guarded operation, so a swap with a
  stale header logs two (the load and its internal unload), each naming its operation.
- Tasks 2.0 and 3.0 were committed together: both live in `ModelService` and the enforcement tests
  exercise the API the registry tests build on.
