"""`ModelService` under attachable weights (031 FTASKS 5.4-5.6, 5.8; addendum §8, D-01, D-03).

Through the REAL `ModelService` (tests/unit/lease_fixtures.py: a fake repository, the background
workers replaced by counting stand-ins) and the REAL weight barrier and weight service, so "no
load started" is a worker call count of 0 and a refusal carries its real payload.

Fixtures disagree with the defect: model 1 is resident, model 2 requested; an attachment is
tried active AND stale; the torn barrier holds a real slot left by a holder that is gone.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import inspect
import json
import textwrap
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import millm.services.weight_barrier as barrier_module
from millm.core.config import settings
from millm.core.errors import (
    AlreadyAttachedError,
    LoadThroughWeightsApiError,
    WeightsTornError,
)
from millm.core.weights_auth import WeightsIdentity
from millm.db.models.model import ModelStatus
from millm.services.model_service import ModelService
from millm.services.weight_attachment import Attachment, WeightService, set_weight_service
from millm.services.weight_barrier import WeightBarrier, set_weight_barrier
from tests.unit.lease_fixtures import (
    build_service,
    clear_resident,
    drain_executor,
    install_registry,
    no_probe_rows,
    set_resident,
)

TRAINER = WeightsIdentity("miforge", "trainer")


class _Inference:
    def __init__(self) -> None:
        self.request_queue = SimpleNamespace(max_concurrent=1)
        self.slot = asyncio.Semaphore(1)

    def cbm_enabled(self) -> bool:
        return False

    def invalidate_request_caches(self) -> int:
        return 0

    @contextlib.asynccontextmanager
    async def hold_admission(self):
        async with self.slot:
            yield


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(barrier_module, "_synchronize_resident_devices", lambda: None)
    clear_resident()
    install_registry()
    service, repo = build_service()
    barrier = WeightBarrier(inference=_Inference)
    set_weight_barrier(barrier)
    clock = Clock()
    weights = WeightService(barrier=lambda: barrier, clock=clock)
    set_weight_service(weights)
    yield SimpleNamespace(service=service, repo=repo, barrier=barrier, weights=weights, clock=clock)
    barrier.reset("test teardown")
    clear_resident()


def _attach(env, *, stale: bool = False) -> None:
    env.weights._attachment = Attachment(
        digest="d" * 64, ref="dddddddd", identity=TRAINER,
        attached_at=datetime(2026, 10, 14, tzinfo=timezone.utc), model_id=1,
        last_heartbeat_mono=env.clock.now, pending=False,
    )
    if stale:
        env.clock.now += float(settings.WEIGHTS_HEARTBEAT_TIMEOUT_S) + 1


async def _torn_with_holder_gone(barrier: WeightBarrier) -> None:
    """The real path: quiesce, commit-begin, watchdog, the holder detaches. The slot stays held."""
    await barrier.quiesce("d" * 64, TRAINER)
    barrier.commit_begin("d" * 64)
    barrier._watchdog_deadline_mono = 0.0
    barrier._on_watchdog(barrier._watchdog_generation)
    barrier.on_holder_detached("d" * 64)
    assert barrier.snapshot().weights_state == "torn" and barrier.holds_slot()


# --- 5.4 / 5.8: the attachment guard -----------------------------------------------------------


class TestRefusedWhileAttached:
    @pytest.mark.parametrize("stale", [False, True], ids=["active", "stale"])
    async def test_load(self, env, stale):
        """M9."""
        _attach(env, stale=stale)
        with pytest.raises(AlreadyAttachedError) as info:
            await env.service.load_model(2)
        assert info.value.details["operation"] == "model_load"
        assert info.value.details["state"] == ("stale" if stale else "active")
        assert info.value.details["label"] == "miforge"
        assert env.service._load_worker.call_count == 0
        assert env.repo.status_writes == [] and ModelService.is_loading is False

    @pytest.mark.parametrize("stale", [False, True], ids=["active", "stale"])
    async def test_unload(self, env, stale):
        """M10."""
        _attach(env, stale=stale)
        with patch.object(env.service, "_unload_worker") as worker, no_probe_rows():
            with pytest.raises(AlreadyAttachedError) as info:
                await env.service.unload_model(1)
        assert info.value.details["operation"] == "model_unload"
        assert worker.call_count == 0 and env.service.loader.is_unloading is False

    @pytest.mark.parametrize("stale", [False, True], ids=["active", "stale"])
    async def test_auto_load(self, env, stale):
        """M11."""
        _attach(env, stale=stale)
        with pytest.raises(AlreadyAttachedError) as info:
            await env.service.load_model_and_wait(2, timeout=0.1)
        assert info.value.details["operation"] == "auto_load"
        assert env.service._load_worker.call_count == 0

    async def test_using_the_resident_model_is_not_a_swap(self, env):
        _attach(env)
        row = await env.service.load_model_and_wait(1, timeout=0.1)
        assert row.id == 1


def _calls_in(method) -> list[ast.Call]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]


def _name(call: ast.Call):
    return getattr(call.func, "attr", None) or getattr(call.func, "id", None)


class TestGuardsAreCalledInEachBody:
    @pytest.mark.parametrize("name,operation", [
        ("load_model", "model_load"), ("unload_model", "model_unload"),
        ("load_model_and_wait", "auto_load"),
        # 032's full reload refuses before its first write through `precheck_load` (R2-A /
        # docs pass: no test pinned this call; removing it left every suite green).
        ("precheck_load", "model_load"),
    ])
    def test_refuse_if_attached(self, name, operation):
        calls = [c for c in _calls_in(getattr(ModelService, name))
                 if _name(c) == "_refuse_if_attached"]
        assert len(calls) == 1 and calls[0].args[0].value == operation

    @pytest.mark.parametrize("name", ["load_model", "load_model_and_wait", "precheck_load"])
    def test_refuse_if_torn(self, name):
        calls = [c for c in _calls_in(getattr(ModelService, name)) if _name(c) == "_refuse_if_torn"]
        assert len(calls) == 1

    def test_the_attachment_guard_sits_before_the_slot_claim_with_no_await(self):
        body = ast.parse(textwrap.dedent(inspect.getsource(ModelService.load_model))).body[0].body
        names = [
            _name(s.value) if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) else None
            for s in body
        ]
        check = names.index("_refuse_if_attached")
        claim = next(i for i, s in enumerate(body) if isinstance(s, ast.Assign) and any(
            getattr(t, "attr", None) == "_loading_model_id" for t in s.targets))
        assert check < claim
        assert not any(isinstance(n, ast.Await) for s in body[check:claim] for n in ast.walk(s))


# --- 5.5 / D-01: torn -------------------------------------------------------------------------


class TestTorn:
    async def test_a_management_load_is_refused_naming_the_recovery(self, env):
        await _torn_with_holder_gone(env.barrier)
        with pytest.raises(WeightsTornError) as info:
            await env.service.load_model(2)
        assert "POST /api/weights/load" in info.value.message
        assert info.value.details["operation"] == "model_load"
        assert env.service._load_worker.call_count == 0

    async def test_the_auto_load_is_refused(self, env):
        await _torn_with_holder_gone(env.barrier)
        with pytest.raises(WeightsTornError) as info:
            await env.service.load_model_and_wait(2, timeout=0.1)
        assert info.value.details["operation"] == "auto_load"
        assert env.service._load_worker.call_count == 0

    async def test_an_unload_is_allowed_skips_the_drain_and_clears_torn(self, env):
        """D-01 + review D10: nothing resident is clean; the torn set's hold is released and
        the drain does not wait on it."""
        await _torn_with_holder_gone(env.barrier)
        queue = env.service._inference_service.request_queue
        queue.occupied_count = 1
        queue.wait_idle = AsyncMock(return_value=True)
        with patch.object(env.service, "_unload_worker"), no_probe_rows():
            await env.service.unload_model(1)
        await asyncio.sleep(0)
        assert queue.wait_idle.await_count == 0
        snap = env.barrier.snapshot()
        assert (snap.weights_state, snap.barrier, snap.hold_owner) == ("clean", "serving", None)
        assert not env.barrier.holds_slot()

    async def test_under_barrier_skips_the_torn_refusal(self, env):
        await _torn_with_holder_gone(env.barrier)
        async with env.barrier.internal_write("weights_load", TRAINER):
            await env.service.load_model(2, under_barrier=True)
        drain_executor(env.service)
        assert env.service._load_worker.call_count == 1


# --- under_barrier (031 G3) --------------------------------------------------------------------


class TestUnderBarrier:
    @pytest.mark.parametrize("call", [
        lambda s: s.load_model(2, under_barrier=True),
        lambda s: s.unload_model(1, under_barrier=True),
        lambda s: s.load_model_and_wait(2, timeout=0.1, under_barrier=True),
    ], ids=["load", "unload", "auto_load"])
    async def test_refused_unless_the_barrier_holds_the_slot(self, env, call):
        with pytest.raises(RuntimeError, match="under_barrier"):
            await call(env.service)
        assert env.service._load_worker.call_count == 0

    async def test_load_refused_with_nothing_resident(self, env):
        """G3 in `load_model` itself: with nothing resident there is no internal unload whose
        own check would refuse, so only `load_model`'s check stands between a caller and a
        load that skips the torn refusal outside the barrier. `precheck_load` the same."""
        from millm.services import model_service as module

        clear_resident()
        with pytest.raises(RuntimeError, match=r"load_model\(under_barrier=True\)"):
            await env.service.load_model(2, under_barrier=True)
        with pytest.raises(RuntimeError, match=r"precheck_load\(under_barrier=True\)"):
            await env.service.precheck_load(2, under_barrier=True)
        assert env.service._load_worker.call_count == 0
        assert module._LOAD_SLOT["model_id"] is None
        assert env.repo.rows[2].status == ModelStatus.READY

    async def test_an_unload_inside_internal_write_skips_wait_idle(self, env):
        """M29: the drain would wait on the barrier's own hold."""
        queue = env.service._inference_service.request_queue
        queue.occupied_count = 1
        queue.wait_idle = AsyncMock(return_value=True)
        with patch.object(env.service, "_unload_worker") as worker, no_probe_rows():
            async with env.barrier.internal_write("weights_load", TRAINER) as handle:
                handle.mark_first_write()
                await env.service.unload_model(1, under_barrier=True)
                assert env.barrier.snapshot().weights_state == "writing", (
                    "on_parameter_set_replaced is a no-op inside an internal write")
        assert worker.call_count == 1 and queue.wait_idle.await_count == 0
        assert env.barrier.snapshot().weights_state == "clean"

    async def test_without_under_barrier_the_drain_runs(self, env):
        queue = env.service._inference_service.request_queue
        queue.occupied_count = 1
        queue.wait_idle = AsyncMock(return_value=True)
        with patch.object(env.service, "_unload_worker"), no_probe_rows():
            await env.service.unload_model(1)
        assert queue.wait_idle.await_count == 1

    async def test_load_passes_under_barrier_to_its_internal_unload(self, env):
        with patch.object(env.service, "unload_model", wraps=env.service.unload_model) as spy:
            with patch.object(env.service, "_unload_worker"), no_probe_rows():
                async with env.barrier.internal_write("weights_load", TRAINER):
                    await env.service.load_model(2, under_barrier=True)
        drain_executor(env.service)
        assert spy.call_count == 1
        assert spy.call_args.kwargs["under_barrier"] is True


# --- 5.6: on_parameter_set_replaced -----------------------------------------------------------


class TestParameterSetReplaced:
    async def test_after_a_load_completes(self, env):
        with patch.object(env.barrier, "on_parameter_set_replaced") as spy:
            await env.service._update_load_complete(model_id=2, memory_used_mb=1)
        assert spy.call_count == 1 and spy.call_args.args == ("model_load",)

    async def test_after_an_unload(self, env):
        with patch.object(env.barrier, "on_parameter_set_replaced") as spy:
            with patch.object(env.service, "_unload_worker"), no_probe_rows():
                await env.service.unload_model(1)
        assert spy.call_count == 1 and spy.call_args.args == ("model_unload",)

    async def test_a_load_resets_the_steps_counter(self, env):
        await env.barrier.quiesce("d" * 64, TRAINER)
        env.barrier.commit_begin("d" * 64)
        env.barrier.commit_end("d" * 64)
        env.barrier.resume("d" * 64)
        assert env.barrier.snapshot().steps_since_load == 1
        await env.service._update_load_complete(model_id=2, memory_used_mb=1)
        assert env.barrier.snapshot().steps_since_load == 0


# --- weights-origin rows (addendum §8) ---------------------------------------------------------


class TestWeightsOriginRows:
    def _carrier(self, env):
        env.repo.rows[2].weights_origin = "checkpoint"

    async def test_a_management_load_is_refused(self, env):
        self._carrier(env)
        with pytest.raises(LoadThroughWeightsApiError) as info:
            await env.service.load_model(2)
        assert info.value.details == {"model_id": 2, "weights_origin": "checkpoint",
                                      "operation": "load"}
        assert "POST /api/weights/load" in info.value.message
        assert env.service._load_worker.call_count == 0

    async def test_the_auto_load_is_refused(self, env):
        self._carrier(env)
        with pytest.raises(LoadThroughWeightsApiError) as info:
            await env.service.load_model_and_wait(2, timeout=0.1)
        assert info.value.details["operation"] == "auto_load"

    async def test_a_reload_of_a_resident_carrier_is_refused_before_it_unloads(self, env):
        """M33: `load_model` refuses a carrier too, but a `reload=True` auto-load unloads the
        resident FIRST; without its own guard the served model was gone before the refusal."""
        from millm.ml.model_loader import LoadedModelState

        env.repo.rows[1].weights_origin = "checkpoint"  # model 1: resident, LOADED
        with patch.object(env.service, "_unload_worker") as worker, no_probe_rows():
            with pytest.raises(LoadThroughWeightsApiError) as info:
                await env.service.load_model_and_wait(1, timeout=0.1, reload=True)
        assert info.value.details["operation"] == "auto_load"
        assert worker.call_count == 0 and env.service._load_worker.call_count == 0
        assert LoadedModelState().loaded_model_id == 1
        assert env.repo.rows[1].status == ModelStatus.LOADED

    async def test_the_weights_service_may_load_it(self, env):
        self._carrier(env)
        with patch.object(env.service, "_unload_worker"), no_probe_rows():
            await env.service.load_model(2, allow_weights_origin=True)
        drain_executor(env.service)
        assert env.service._load_worker.call_count == 1

    async def test_startup_auto_load_refuses_it(self, env, monkeypatch):
        import millm.main as main

        self._carrier(env)
        env.repo.rows[2].status = ModelStatus.READY

        class _Factory:
            async def __aenter__(self):
                return None

            async def __aexit__(self, *exc):
                return False

        monkeypatch.setattr("millm.db.base.async_session_factory", lambda: _Factory())
        monkeypatch.setattr("millm.ml.model_downloader.ModelDownloader", MagicMock)
        monkeypatch.setattr("millm.ml.model_loader.ModelLoader", MagicMock)
        monkeypatch.setattr("millm.services.model_service.ModelService.__init__",
                            lambda self, **kw: None)
        monkeypatch.setattr(ModelService, "get_model",
                            lambda self, model_id: _coro(env.repo.rows[2]))
        with pytest.raises(LoadThroughWeightsApiError):
            await main._auto_load_model("2")



async def _coro(value):
    return value


# --- reload, wait_until_loaded, is_loading -----------------------------------------------------


class TestReloadAndWait:
    async def test_reload_unloads_then_loads_the_resident_row(self, env):
        def finish(model_id, *args, **kwargs):
            set_resident(model_id, f"m{model_id}")
            env.repo.rows[model_id].status = ModelStatus.LOADED

        env.service._load_worker.side_effect = finish
        with patch.object(env.service, "unload_model", wraps=env.service.unload_model) as unload:
            with no_probe_rows():
                row = await env.service.load_model_and_wait(1, timeout=5, reload=True)
        assert row.id == 1 and unload.call_count == 1 and unload.call_args.args == (1,)
        assert env.service._load_worker.call_count == 1

    async def test_without_reload_the_resident_row_returns_at_once(self, env):
        row = await env.service.load_model_and_wait(1, timeout=0.1)
        assert row.id == 1 and env.service._load_worker.call_count == 0

    async def test_wait_until_loaded(self, env):
        env.repo.rows[2].status = ModelStatus.LOADED
        assert (await env.service.wait_until_loaded(2, timeout=5)).id == 2

    def test_is_loading_reads_the_shared_slot_from_class_and_instance(self, env):
        from millm.services import model_service

        assert ModelService.is_loading is False and env.service.is_loading is False
        model_service._LOAD_SLOT["model_id"] = 9
        assert ModelService.is_loading is True and env.service.is_loading is True

    def test_the_attach_transition_probe_reads_it(self, env):
        from millm.services import model_service
        from millm.services.weight_attachment import _default_load_claimed

        model_service._LOAD_SLOT["model_id"] = 9
        assert _default_load_claimed() is True
        model_service._LOAD_SLOT["model_id"] = None
        assert _default_load_claimed() is False


# --- D-03: which loads are attachable ---------------------------------------------------------


def _tokens(*roles: str) -> str:
    return json.dumps([
        {"label": f"l{i}", "role": role, "sha256": f"{i:064x}"} for i, role in enumerate(roles)
    ])


class TestAttachableLoads:
    @pytest.mark.parametrize("quantization,gguf,roles,expected", [
        ("FP16", None, ("trainer",), True),
        ("FP16", None, ("operator", "trainer"), True),
        ("FP16", None, ("operator",), False),
        ("FP16", None, (), False),
        ("Q4", None, ("trainer",), False),
        ("Q8", None, ("trainer",), False),
        ("FP16", "model.gguf", ("trainer",), False),
        # D-37: FP32 can never attach (attach requires fp16/bf16), so it never takes the pool.
        ("FP32", None, ("trainer",), False),
        ("fp32", None, ("operator", "trainer"), False),
        ("fp16", None, ("trainer",), True),
        ("GPTQ", None, ("trainer",), False),
    ])
    def test_the_rule(self, monkeypatch, quantization, gguf, roles, expected):
        monkeypatch.setattr(settings, "WEIGHTS_TOKENS", _tokens(*roles) if roles else "")
        assert ModelService._attachable_load(quantization, gguf) is expected

    @pytest.mark.parametrize("quantization,roles,expected", [
        ("FP16", ("trainer",), True), ("FP16", (), False),
        ("FP32", ("trainer",), False),  # D-37
    ])
    def test_the_worker_passes_it_to_the_loader(self, env, monkeypatch, quantization, roles,
                                                expected):
        monkeypatch.setattr(settings, "WEIGHTS_TOKENS", _tokens(*roles) if roles else "")
        service = ModelService(repository=env.repo, downloader=MagicMock(),
                               loader=MagicMock(), inference_service=MagicMock())
        service._run_async_from_thread = MagicMock()
        service._load_worker(2, "m2", "/abs/path", quantization, 10, False)
        assert service.loader.load.call_count == 1
        assert service.loader.load.call_args.kwargs["attachable"] is expected


# --- R2-B O1: no unload under a barrier operation --------------------------------------------


class TestNoUnloadUnderABarrierOperation:
    """Review R2-B O1: an unload (or a load's internal unload) while a barrier operation holds
    the slot used to wait the drain grace on the barrier's own hold and then free the weights
    under the running copy or reload. Refused `MODEL_BUSY` (reason `weights_operation`) unless
    the caller IS that operation (`under_barrier`). A torn set's leftover holder hold is not an
    operation and stays unloadable (`TestTorn.test_an_unload_is_allowed_*`, review D10).

    MUTATION CONTROL (R2-A): the `_refuse_if_barrier_operation` call removed from
    `unload_model` -> every test here red."""

    @pytest.mark.parametrize("enter", [
        lambda b: b.internal_write("weights_load", TRAINER),
        lambda b: b.serving_pause("weights_save", TRAINER),
    ], ids=["internal_write", "serving_pause"])
    async def test_an_unload_is_refused(self, env, enter):
        from millm.core.errors import ModelBusyError

        with patch.object(env.service, "_unload_worker") as worker, no_probe_rows():
            async with enter(env.barrier):
                assert env.barrier.holds_slot()
                with pytest.raises(ModelBusyError) as info:
                    await env.service.unload_model(1)
        assert info.value.details["reason"] == "weights_operation"
        assert info.value.details["operation"] in {"weights_load", "weights_save"}
        assert worker.call_count == 0 and env.service.loader.is_unloading is False
        assert env.service.loader.loaded_model_id == 1

    async def test_a_swapping_load_is_refused_and_claims_nothing(self, env):
        from millm.core.errors import ModelBusyError

        with patch.object(env.service, "_unload_worker") as worker, no_probe_rows():
            async with env.barrier.internal_write("weights_load", TRAINER):
                with pytest.raises(ModelBusyError):
                    await env.service.load_model(2)
        assert worker.call_count == 0 and env.service._load_worker.call_count == 0
        assert ModelService.is_loading is False
        assert env.service.loader.loaded_model_id == 1

    async def test_the_operations_own_unload_still_runs(self, env):
        with patch.object(env.service, "_unload_worker") as worker, no_probe_rows():
            async with env.barrier.internal_write("weights_load", TRAINER):
                await env.service.unload_model(1, under_barrier=True)
        assert worker.call_count == 1


# --- R2-B O2: a load an internal write started never cleans the barrier ----------------------


class TestUnderBarrierLoadCompletion:
    """Review R2-B O2: a load started under the barrier that outlives its write (032's wait timed
    out; the write ended `torn`) called `on_parameter_set_replaced` when it completed, turning
    `torn` into `clean` over a checkpoint nothing verified. The 032 service owns that
    transition.

    MUTATION CONTROLS (R2-A): `under_barrier=under_barrier` dropped from the `run_in_executor` partial ->
    the wiring test red; the `if not under_barrier` guard in `_update_load_complete` removed ->
    the completion test red."""

    async def test_the_worker_is_told_the_load_is_under_the_barrier(self, env):
        async with env.barrier.internal_write("weights_load", TRAINER):
            await env.service.load_model(2, under_barrier=True)
        drain_executor(env.service)
        assert env.service._load_worker.call_count == 1
        assert env.service._load_worker.call_args.kwargs["under_barrier"] is True

    async def test_an_ordinary_load_is_not(self, env):
        await env.service.load_model(2)
        drain_executor(env.service)
        assert env.service._load_worker.call_args.kwargs["under_barrier"] is False

    async def test_completion_after_the_write_ended_leaves_torn(self, env):
        await _torn_with_holder_gone(env.barrier)
        with patch.object(env.barrier, "on_parameter_set_replaced",
                          wraps=env.barrier.on_parameter_set_replaced) as spy:
            await env.service._update_load_complete(model_id=2, memory_used_mb=1,
                                                    under_barrier=True)
            assert spy.call_count == 0
            assert env.barrier.snapshot().weights_state == "torn"
            await env.service._update_load_complete(model_id=2, memory_used_mb=1)
        assert spy.call_count == 1 and env.barrier.snapshot().weights_state == "clean"
