"""Feature 26 task 5.12: the runner — resume, exactly-once, cancel, expiry, leases, progress.

All through the real app, a real BatchRunner (its `step()` driven by the test, so ordering is
deterministic), a real ModelService and lease registry, and a tiny real Llama.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from millm.core.batch_values import BatchStatus
from millm.services.batch.state import IllegalTransitionError, TRANSITIONS, transition
from millm.services.model_lease import get_lease_registry
from tests.unit.batch_fixtures import (  # noqa: F401
    BatchHarness,
    batch_db,
    batch_dir,
    client_for,
    completion_line,
    harness,
    jsonl,
    upload,
)

N = 5


def _lines(n: int = N):
    return [completion_line(f"c{i}", prompt="w1 " + " ".join(f"w{j}" for j in range(i % 4 + 1)))
            for i in range(n)]


async def _start(harness, lines=None, **kw):
    """Create a batch and let validation finish; returns a FRESH (unopened) client and the id."""
    async with client_for(harness.app()) as client:
        created = await harness.create(client, lines or _lines(), **kw)
    assert created.status_code == 200, created.text
    await harness.settle_validation()
    return client_for(harness.app()), created.json()["id"]


async def _ids(harness, client, batch) -> Counter:
    out = await harness.lines(client, batch.output_file_id)
    err = await harness.lines(client, batch.error_file_id)
    return Counter(line["custom_id"] for line in out + err)


# ------------------------------------------------------------------- the transition table


def test_only_the_fr_26_3_5_transitions_are_allowed():
    allowed = {(a.value, b.value) for a, targets in TRANSITIONS.items() for b in targets}
    assert allowed == {
        ("validating", "in_progress"), ("validating", "failed"), ("validating", "cancelling"),
        ("in_progress", "finalizing"), ("in_progress", "cancelling"), ("in_progress", "expired"),
        ("in_progress", "failed"), ("finalizing", "completed"), ("finalizing", "failed"),
        ("cancelling", "cancelled"),
    }

    class B:
        id, status, waiting_reason = "b", "completed", None

    with pytest.raises(IllegalTransitionError):
        transition(B(), BatchStatus.IN_PROGRESS)


# ------------------------------------------------------------------------- happy path


async def test_a_batch_runs_to_completion_with_openai_shaped_lines(harness):
    client, batch_id = await _start(harness)
    async with client:
        await harness.drain()
        batch = await harness.batch(batch_id)
        out = await harness.lines(client, batch.output_file_id)
        got = (await client.get(f"/v1/batches/{batch_id}")).json()
    assert batch.status == "completed" and batch.error_file_id is None
    assert got["request_counts"] == {"total": N, "completed": N, "failed": 0}
    assert [line["custom_id"] for line in out] == [f"c{i}" for i in range(N)]  # input order
    line = out[0]
    assert set(line) == {"id", "custom_id", "response", "error"} and line["error"] is None
    assert line["response"]["status_code"] == 200
    assert line["response"]["body"]["object"] == "text_completion"
    assert line["response"]["millm"]["packed"] is False
    assert line["response"]["millm"]["headers"]["X-miLLM-Backend"] == harness.inference.backend_name
    for stamp in ("in_progress_at", "finalizing_at", "completed_at"):
        assert got[stamp] is not None


async def test_a_batch_line_equals_the_synchronous_response(harness):
    """FR-26.5.4 / FR-26.10.1: same body (logprobs) and same X-miLLM-* headers."""
    body = completion_line("x", seed=7)["body"]
    async with client_for(harness.app()) as client:
        sync = await client.post("/v1/completions", json=body)
    client, batch_id = await _start(harness, [completion_line("x", seed=7)], pack=False)
    async with client:
        await harness.drain()
        out = await harness.lines(client, (await harness.batch(batch_id)).output_file_id)
    line_body = out[0]["response"]["body"]
    sync_body = sync.json()
    assert line_body["choices"] == sync_body["choices"]
    assert line_body["system_fingerprint"] == sync_body["system_fingerprint"]
    sync_headers = {k: v for k, v in sync.headers.items() if k.lower().startswith("x-millm-")}
    line_headers = {k.lower(): v for k, v in out[0]["response"]["millm"]["headers"].items()}
    assert line_headers == {k.lower(): v for k, v in sync_headers.items()}
    assert "x-millm-seed" in line_headers


async def test_a_row_failing_on_its_own_merits_goes_to_the_error_file_and_the_batch_goes_on(
    harness, monkeypatch
):
    from millm.core.config import settings

    monkeypatch.setattr(settings, "BATCH_PACK_DEFAULT", False)
    too_long = completion_line("long", prompt=" ".join(["w1"] * 400))
    client, batch_id = await _start(harness, [completion_line("a"), too_long, completion_line("b")])
    async with client:
        await harness.drain()
        batch = await harness.batch(batch_id)
        err = await harness.lines(client, batch.error_file_id)
        out = await harness.lines(client, batch.output_file_id)
    assert batch.status == "completed"
    assert [line["custom_id"] for line in out] == ["a", "b"]
    assert err[0]["custom_id"] == "long" and err[0]["response"]["status_code"] == 400
    assert err[0]["error"]["code"] == "context_length_exceeded"


async def test_counters_rise_per_chunk(harness, monkeypatch):
    from millm.core.config import settings

    monkeypatch.setattr(settings, "BATCH_CHUNK_ROWS", 2)
    client, batch_id = await _start(harness, pack=False)
    seen = []
    async with client:
        while await harness.runner.step():
            seen.append((await harness.batch(batch_id)).request_completed)
    assert seen[:3] == [2, 4, 5]


# ------------------------------------------------------------------ crash and resume


class _Crash(BaseException):
    """Stands in for the process dying mid-chunk."""


async def test_exactly_once_across_a_crash_mid_chunk(harness, monkeypatch):
    """FR-26.3.4 (M4/M5): a crash inside a chunk records nothing of it; after a restart every
    custom_id appears exactly once across the output and error files."""
    from millm.core.config import settings
    from millm.services.batch.executors import RowExecutor
    from millm.services.batch.reconcile import reconcile_batches_on_startup
    from millm.services.batch.files import get_file_store

    monkeypatch.setattr(settings, "BATCH_CHUNK_ROWS", 3)
    client, batch_id = await _start(harness, pack=False)
    assert await harness.runner.step()  # chunk 1: rows 1-3 recorded
    real_run_one = RowExecutor.run_one
    calls = {"n": 0}

    async def dying(self, row):
        calls["n"] += 1
        if calls["n"] == 2:
            raise _Crash()
        return await real_run_one(self, row)

    monkeypatch.setattr(RowExecutor, "run_one", dying)
    with pytest.raises(_Crash):
        await harness.runner.step()
    monkeypatch.setattr(RowExecutor, "run_one", real_run_one)
    batch = await harness.batch(batch_id)
    assert batch.request_completed == 3, "the crashed chunk recorded nothing"

    # A restart: new runner, empty memory, leases gone.
    get_lease_registry().clear()
    fresh = BatchHarness(harness.factory)
    await reconcile_batches_on_startup(harness.factory, fresh.runner, get_file_store())
    async with client:
        await fresh.drain()
        batch = await harness.batch(batch_id)
        ids = await _ids(harness, client, batch)
    assert batch.status == "completed"
    assert ids == Counter({f"c{i}": 1 for i in range(N)})
    await fresh.runner.stop()


async def test_reconciliation_branches(harness):
    from millm.db.models.batch import Batch
    from millm.services.batch.files import get_file_store
    from millm.services.batch.reconcile import reconcile_batches_on_startup

    client, b_run = await _start(harness)
    _, b_cancel = await _start(harness)
    _, b_final = await _start(harness)
    _, b_valid = await _start(harness)
    await harness.runner.step()  # one chunk of the oldest
    async with harness.factory() as session:
        (await session.get(Batch, b_cancel)).status = "cancelling"
        final = await session.get(Batch, b_final)
        final.status = "finalizing"
        (await session.get(Batch, b_valid)).status = "validating"
        await session.commit()
    stray = harness_dir = None
    from millm.core.config import settings
    from pathlib import Path

    harness_dir = Path(settings.BATCH_FILES_DIR) / "2026-10"
    harness_dir.mkdir(parents=True, exist_ok=True)
    stray = harness_dir / "file-x.jsonl.partial"
    stray.write_bytes(b"half")
    get_lease_registry().clear()
    fresh = BatchHarness(harness.factory)
    found = await reconcile_batches_on_startup(harness.factory, fresh.runner, get_file_store())
    await fresh.settle_validation()
    assert found["in_progress"] == [b_run] and found["cancelling"] == [b_cancel]
    assert found["finalizing"] == [b_final] and found["validating"] == [b_valid]
    assert (await harness.batch(b_cancel)).status == "cancelled"
    assert (await harness.batch(b_final)).status == "completed"
    assert (await harness.batch(b_valid)).status == "in_progress"
    resumed = await harness.batch(b_run)
    assert resumed.status == "in_progress" and resumed.lease_mode is None
    assert resumed.waiting_reason == "queued"
    assert not stray.exists()
    async with client:
        await fresh.drain()
    assert (await harness.batch(b_run)).status == "completed"
    await fresh.runner.stop()


async def test_a_changed_input_file_fails_the_resumed_batch(harness):
    from pathlib import Path

    from millm.core.config import settings
    from millm.db.models.batch import BatchFile
    from millm.services.batch.files import get_file_store
    from millm.services.batch.reconcile import reconcile_batches_on_startup

    client, batch_id = await _start(harness)
    batch = await harness.batch(batch_id)
    async with harness.factory() as session:
        f = await session.get(BatchFile, batch.input_file_id)
        (Path(settings.BATCH_FILES_DIR) / f.storage_path).write_bytes(b"different\n")
    await reconcile_batches_on_startup(harness.factory, harness.runner, get_file_store())
    batch = await harness.batch(batch_id)
    assert batch.status == "failed"
    assert batch.errors["data"][-1]["code"] == "input_file_changed"
    await client.aclose()


# --------------------------------------------------------------------- cancel / expiry


async def test_cancel_at_a_row_boundary_keeps_completed_rows(harness, monkeypatch):
    """M6's target: cancel lands while row 2 of an unpacked chunk runs; row 3 must never start."""
    from millm.core.config import settings
    from millm.services.batch.executors import RowExecutor

    monkeypatch.setattr(settings, "BATCH_CHUNK_ROWS", 4)
    client, batch_id = await _start(harness, pack=False)
    real = RowExecutor.run_one
    started: list[int] = []

    async def watched(self, row):
        started.append(row.line_no)
        result = await real(self, row)
        if row.line_no == 2:
            async with client_for(harness.app()) as c:
                assert (await c.post(f"/v1/batches/{batch_id}/cancel")).status_code == 200
        return result

    monkeypatch.setattr(RowExecutor, "run_one", watched)
    async with client:
        await harness.drain()
        batch = await harness.batch(batch_id)
        out = await harness.lines(client, batch.output_file_id)
        err = await harness.lines(client, batch.error_file_id)
        again = await client.post(f"/v1/batches/{batch_id}/cancel")
    assert started == [1, 2]
    assert batch.status == "cancelled" and batch.cancelled_at is not None
    assert [line["custom_id"] for line in out] == ["c0", "c1"]
    assert [e["error"]["code"] for e in err] == ["batch_cancelled"] * 3
    assert again.status_code == 409 and again.json()["error"]["code"] == "batch_state_conflict"


async def test_cancel_while_validating_ends_cancelled_without_running(harness):
    async with client_for(harness.app()) as client:
        file_id = (await upload(client, jsonl(_lines()))).json()["id"]
        created = (await client.post("/v1/batches", json={
            "input_file_id": file_id, "endpoint": "/v1/completions",
            "completion_window": "24h"})).json()
        cancelled = await client.post(f"/v1/batches/{created['id']}/cancel")
        await harness.drain()
    assert cancelled.json()["status"] == "cancelling"
    batch = await harness.batch(created["id"])
    assert batch.status == "cancelled" and batch.request_completed == 0


async def test_expiry_writes_unrun_rows_as_batch_expired(harness):
    client, batch_id = await _start(harness, pack=False)
    harness.runner.now = lambda: harness_now() + timedelta(hours=25)
    async with client:
        await harness.drain()
        batch = await harness.batch(batch_id)
        err = await harness.lines(client, batch.error_file_id)
    assert batch.status == "expired" and batch.expired_at is not None
    assert {e["error"]["code"] for e in err} == {"batch_expired"} and len(err) == N


def harness_now():
    from millm.services.batch.state import utcnow

    return utcnow()


# ------------------------------------------------------------------------------ leases


async def _lease(harness):
    return get_lease_registry().current(1)


async def test_the_own_lease_is_held_while_running_and_released_on_completion(harness):
    client, batch_id = await _start(harness)
    async with client:
        assert await harness.runner.step()
        live = await _lease(harness)
        assert live is not None and live.holder == f"millm-batch:{batch_id}"
        assert (await harness.batch(batch_id)).lease_mode == "own"
        await harness.drain()
    assert await _lease(harness) is None
    assert get_lease_registry().last_ended(1).end_reason == "released"


@pytest.mark.parametrize("ending", ["cancel", "expiry", "exception"])
async def test_the_own_lease_is_released_on_every_terminal_path(harness, monkeypatch, ending):
    """M7's target: the release lives in `finally`."""
    from millm.core.config import settings
    from millm.services.batch.executors import RowExecutor

    monkeypatch.setattr(settings, "BATCH_CHUNK_ROWS", 2)

    client, batch_id = await _start(harness, pack=False)
    async with client:
        assert await harness.runner.step()
        assert await _lease(harness) is not None
        if ending == "cancel":
            await client.post(f"/v1/batches/{batch_id}/cancel")
        elif ending == "expiry":
            harness.runner.now = lambda: harness_now() + timedelta(hours=25)
        else:
            async def boom(self, row):
                raise RuntimeError("CUDA error: device-side assert")
            monkeypatch.setattr(RowExecutor, "run_one", boom)
        await harness.drain()
    batch = await harness.batch(batch_id)
    assert batch.status == {"cancel": "cancelled", "expiry": "expired", "exception": "failed"}[ending]
    assert await _lease(harness) is None
    if ending == "exception":
        assert "RuntimeError" in batch.errors["data"][-1]["message"]


async def test_a_caller_lease_is_renewed_and_never_released(harness, monkeypatch):
    """T-64 / M8's target."""
    async with harness.factory() as session:
        grant = await harness.model_service(session).acquire_lease(1, "midataworks", "labels", 300)
    client, batch_id = await _start(harness, headers={"X-miLLM-Lease": grant.lease_id})
    clock = {"t": 0.0}
    harness.runner.monotonic = lambda: clock["t"]
    harness.runner.memory(batch_id).renewed_mono = 0.0
    before = (await _lease(harness)).expires_at
    clock["t"] = 200.0  # past a third of the 300 s TTL
    async with client:
        await harness.drain()
    live = await _lease(harness)
    assert live is not None and live.holder == "midataworks", "the caller's lease was released"
    assert live.expires_at > before, "the caller's lease was never renewed"
    assert (await harness.batch(batch_id)).status == "completed"


async def test_after_a_restart_the_batch_waits_for_the_lease_and_runs_under_a_handed_over_one(
    harness,
):
    """X-01 / T-66 / X-08: the restart ends every lease; another holder takes one; the batch
    waits (`lease_unavailable`) and runs NO row; the holder hands it over; it runs."""
    from millm.services.batch.files import get_file_store
    from millm.services.batch.reconcile import reconcile_batches_on_startup

    client, batch_id = await _start(harness)
    assert await harness.runner.step()
    done_before = (await harness.batch(batch_id)).request_completed
    get_lease_registry().clear()  # the restart
    async with harness.factory() as session:
        grant = await harness.model_service(session).acquire_lease(1, "midataworks", "labels", 600)
    fresh = BatchHarness(harness.factory)
    await reconcile_batches_on_startup(harness.factory, fresh.runner, get_file_store())
    assert await fresh.runner.step() is False
    batch = await harness.batch(batch_id)
    assert batch.waiting_reason == "lease_unavailable"
    assert batch.request_completed == done_before, "a row ran without a lease"
    async with client_for(fresh.app()) as c:
        handed = await c.post(f"/v1/batches/{batch_id}/lease",
                              headers={"X-miLLM-Lease": grant.lease_id})
        bogus = await c.post(f"/v1/batches/{batch_id}/lease", headers={"X-miLLM-Lease": "nope"})
    assert handed.status_code == 200 and handed.json()["millm"]["lease_mode"] == "caller"
    assert bogus.status_code == 404 and bogus.json()["error"]["code"] == "lease_not_found"
    await fresh.drain()
    assert (await harness.batch(batch_id)).status == "completed"
    assert (await _lease(harness)).holder == "midataworks"
    await fresh.runner.stop()
    await client.aclose()


async def test_a_model_not_resident_batch_waits_and_never_loads(harness):
    from millm.ml.model_loader import LoadedModelState

    client, batch_id = await _start(harness)
    saved = LoadedModelState()._loaded
    LoadedModelState()._loaded = None
    load = AsyncMock()
    try:
        from millm.services.model_service import ModelService

        original_load, original_unload = ModelService.load_model, ModelService.unload_model
        ModelService.load_model = load
        ModelService.unload_model = load
        assert await harness.runner.step() is False
        assert (await harness.batch(batch_id)).waiting_reason == "model_not_resident"
    finally:
        ModelService.load_model, ModelService.unload_model = original_load, original_unload
        LoadedModelState()._loaded = saved
    assert load.await_count == 0, "a batch must never load or unload a model"
    async with client:
        await harness.drain()
    assert (await harness.batch(batch_id)).status == "completed"


async def test_a_batch_never_calls_load_or_unload(harness, monkeypatch):
    from millm.services.model_service import ModelService

    spy = AsyncMock()
    monkeypatch.setattr(ModelService, "load_model", spy)
    monkeypatch.setattr(ModelService, "unload_model", spy)
    monkeypatch.setattr(ModelService, "load_model_and_wait", spy)
    client, _ = await _start(harness)
    async with client:
        await harness.drain()
    assert spy.await_count == 0


async def test_an_unloading_model_makes_the_batch_wait_not_fail(harness, monkeypatch):
    """FTASKS 2.6 / 5.6: the unloading refusal at the chunk's admission is a wait."""
    from millm.core.config import settings
    from millm.ml.model_loader import LoadedModelState

    monkeypatch.setattr(settings, "BATCH_CHUNK_ROWS", 2)
    client, batch_id = await _start(harness, pack=False)
    async with client:
        assert await harness.runner.step()  # holds a lease, records a chunk
        LoadedModelState().begin_unload()
        try:
            assert await harness.runner.step() is False
            batch = await harness.batch(batch_id)
            assert batch.status == "in_progress" and batch.waiting_reason == "model_not_resident"
        finally:
            LoadedModelState().cancel_unload()
        await harness.drain()
    assert (await harness.batch(batch_id)).status == "completed"


async def test_interactive_chat_proceeds_while_the_batch_holds_the_lease(harness):
    client, batch_id = await _start(harness)
    async with client:
        assert await harness.runner.step()
        assert await _lease(harness) is not None
        chat = await client.post("/v1/chat/completions", json={
            "model": "tiny", "messages": [{"role": "user", "content": "w1"}], "max_tokens": 2})
        await harness.drain()
    assert chat.status_code == 200, chat.text


# ------------------------------------------------------------------- progress + wiring


async def test_progress_is_emitted_with_its_payload_on_every_transition(harness, monkeypatch):
    from millm.core.config import settings

    monkeypatch.setattr(settings, "BATCH_CHUNK_ROWS", 2)
    monkeypatch.setattr(settings, "BATCH_PROGRESS_MIN_INTERVAL_S", 0.0)
    client, batch_id = await _start(harness, pack=False)
    async with client:
        await harness.drain()
    payloads = [p for p in harness.emitter.payloads if p["id"] == batch_id]
    statuses = [p["status"] for p in payloads]
    assert statuses[0] == "in_progress" and statuses[-1] == "completed"
    assert "finalizing" in statuses
    assert set(payloads[-1]) == {"id", "status", "request_counts"}
    assert payloads[-1]["request_counts"] == {"total": N, "completed": N, "failed": 0}
    # in_progress, three chunks (2+2+1), finalizing, completed — exactly.
    assert statuses == ["in_progress", "in_progress", "in_progress", "in_progress",
                        "finalizing", "completed"]
    assert [p["request_counts"]["completed"] for p in payloads[1:4]] == [2, 4, 5]


async def test_chunk_progress_is_throttled_but_transitions_are_not(harness, monkeypatch):
    from millm.core.config import settings

    monkeypatch.setattr(settings, "BATCH_CHUNK_ROWS", 1)
    monkeypatch.setattr(settings, "BATCH_PROGRESS_MIN_INTERVAL_S", 3600.0)
    client, batch_id = await _start(harness, pack=False)
    async with client:
        await harness.drain()
    statuses = [p["status"] for p in harness.emitter.payloads if p["id"] == batch_id]
    assert statuses == ["in_progress", "finalizing", "completed"]


async def test_health_reports_the_batch_backlog_and_a_chunk_in_flight(harness, monkeypatch):
    """FR-26.4.7 / FTASKS 5.11 / 8.3: `batch_backlog_rows` is a NUMBER once the runner is live
    (null before — 029 FR-29.7.3), counts unrun rows, and `in_flight` sees a chunk's slot."""
    from millm.api.routes.system.health import read_inference_state
    from millm.core.config import settings

    monkeypatch.setattr(settings, "BATCH_CHUNK_ROWS", 2)
    assert read_inference_state(harness.inference).batch_backlog_rows is None
    client, batch_id = await _start(harness, pack=False)
    harness.runner.start()
    try:
        state = read_inference_state(harness.inference)
        assert state.batch_backlog_rows == N
    finally:
        await harness.runner.stop()
    assert read_inference_state(harness.inference).batch_backlog_rows is None
    harness.runner.start = lambda: None  # keep the loop off; drive by hand
    from millm.core.backpressure import register_backlog_provider

    register_backlog_provider(harness.runner.backlog_rows)
    try:
        assert await harness.runner.step()
        assert read_inference_state(harness.inference).batch_backlog_rows == N - 2
        async with harness.inference._admit(background=True):
            assert read_inference_state(harness.inference).in_flight == 1
    finally:
        register_backlog_provider(None)
    await client.aclose()


async def test_a_batch_row_never_runs_on_the_continuous_batching_manager(harness):
    """FTASKS 5.3: with the manager on and the sampling params matching, a request is routed to
    it — unless it is a batch row."""
    from unittest.mock import MagicMock

    from millm.services.batch.state import BATCH_ROW

    inference = harness.inference
    inference._cbm_backend = MagicMock()
    inference._cbm_backend.sampling_params_match.return_value = True
    inference._use_cbm = lambda: True
    assert inference._use_cbm_for_request(temperature=0.0, top_p=1.0) is True
    token = BATCH_ROW.set(("batch_x", 1))
    try:
        assert inference._use_cbm_for_request(temperature=0.0, top_p=1.0) is False
    finally:
        BATCH_ROW.reset(token)
        inference._cbm_backend = None
