"""Two loads at once must not both run.

`load_model` checked `_loading_model_id` and only SET it after awaiting the
resident model's unload, which waits up to five seconds for pending inference
to drain. A second load arriving in that window — a double-clicked Switch, or
two OpenAI requests naming different models (`load_model_and_wait`) — passed
the same check, and both loads were submitted to a two-worker executor. They
then ran concurrently, each placed against memory the other was about to take,
and the second `state.set` replaced the first model without unloading it.

The slot is now claimed with no await between the check and the claim, and
released when anything before the background load is submitted fails.

MUTATION CONTROLS (round 2, 2026-09-13), each verified red then restored:
  * claim the slot after the unload again -> "the second load is refused as
    busy" fails with two submitted loads (1 red)
  * drop the release in the except block  -> "a failed unload releases the
    slot" and "a refused placement does not leave the service busy" fail (2 red)
"""

import asyncio
from concurrent.futures import Future
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from millm.core.errors import InsufficientMemoryError, ModelBusyError, ModelNotLoadedError
from millm.db.models.model import ModelStatus, QuantizationType
from millm.ml.model_loader import LoadedModel
from millm.services.model_service import ModelService
from tests.support.factories import make_model
from tests.support.fake_gpus import RTX_3090, TI_3080, fake_gpus

NODE = ((TI_3080, 11_000, 12_288), (RTX_3090, 5_000, 24_576))


class _RecordingExecutor:
    def __init__(self):
        self.calls = []

    def submit(self, fn, *args):
        self.calls.append((fn, args))
        future = Future()
        future.set_result(None)
        return future


def _service(unload_side_effect):
    models = {
        mid: make_model(
            id=mid, name=f"m{mid}", status=ModelStatus.READY,
            quantization=QuantizationType.FP16, estimated_memory_mb=4_000,
        )
        for mid in (3, 4)
    }
    repo = MagicMock()
    repo.get_by_id = AsyncMock(side_effect=lambda mid: models[mid])
    repo.update_status = AsyncMock(side_effect=lambda mid, **_: models[mid])
    loader = MagicMock()
    loader.is_loaded = True
    loader.loaded_model_id = 9
    loader.state.current = LoadedModel(
        9, "resident", MagicMock(), MagicMock(), datetime.utcnow(),
        memory_used_mb=16_000, device="cuda:1", gpu_indices=[1],
        memory_by_device_mb={"cuda:1": 16_000},
    )
    svc = ModelService(repository=repo, downloader=MagicMock(), loader=loader, emitter=None)
    svc.unload_model = AsyncMock(side_effect=unload_side_effect)
    svc._executor = _RecordingExecutor()
    return svc


async def _yield(*_args, **_kwargs):
    # The real unload awaits the executor and a drain loop; one yield is enough
    # for a second request to run up to its own busy check.
    await asyncio.sleep(0)


class TestTheSlotIsClaimedBeforeTheUnload:
    def test_the_second_load_is_refused_as_busy(self):
        svc = _service(_yield)

        async def both():
            return await asyncio.gather(
                svc.load_model(3), svc.load_model(4), return_exceptions=True
            )

        with fake_gpus(*NODE):
            results = asyncio.run(both())

        busy = [r for r in results if isinstance(r, ModelBusyError)]
        assert len(busy) == 1, f"expected one load refused as busy, got {results!r}"
        assert len(svc._executor.calls) == 1, (
            f"{len(svc._executor.calls)} background loads were submitted; two loads "
            "running at once each place against memory the other takes"
        )
        assert svc.unload_model.await_count == 1

    def test_a_failed_unload_releases_the_slot(self):
        svc = _service(ModelNotLoadedError("gone"))
        with fake_gpus(*NODE):
            with pytest.raises(ModelNotLoadedError):
                asyncio.run(svc.load_model(3))
        assert svc._loading_model_id is None, "a failed load left the service permanently busy"

    def test_a_refused_placement_does_not_leave_the_service_busy(self):
        svc = _service(_yield)
        too_big = make_model(
            id=3, status=ModelStatus.READY, quantization=QuantizationType.FP16,
            estimated_memory_mb=18_000,
        )
        svc.repository.get_by_id = AsyncMock(return_value=too_big)
        with fake_gpus(*NODE):
            with pytest.raises(InsufficientMemoryError):
                asyncio.run(svc.load_model(3, gpu=0))
        assert svc._loading_model_id is None
        assert not svc.unload_model.called
        assert svc._executor.calls == []

    def test_a_single_load_still_runs_and_holds_the_slot(self):
        svc = _service(_yield)
        with fake_gpus(*NODE):
            asyncio.run(svc.load_model(3))
        assert len(svc._executor.calls) == 1
        assert svc._loading_model_id == 3, "the background worker's finally owns the release"


class TestTheSlotIsSharedAcrossRequests:
    """`get_model_service` builds a NEW ModelService for every request.

    Round 3, 2026-09-14: the round-2 slot lived on `self`, so the tests above
    (one instance, two coroutines) passed while two HTTP requests — two
    instances — both saw an empty slot and both loads ran. These tests use one
    instance per request, as production does.

    MUTATION CONTROLS (round 3), each verified red then restored:
      * the slot back on the instance (`self.__dict__`)  -> both tests red
      * `__init__` resetting the shared slot again       -> both tests red
    """

    def test_a_second_request_is_refused_while_the_first_loads(self):
        first = _service(_yield)
        second = _service(_yield)
        second.loader = first.loader

        async def two_requests():
            return await asyncio.gather(
                first.load_model(3), second.load_model(4), return_exceptions=True
            )

        with fake_gpus(*NODE):
            results = asyncio.run(two_requests())

        busy = [r for r in results if isinstance(r, ModelBusyError)]
        submitted = len(first._executor.calls) + len(second._executor.calls)
        assert len(busy) == 1 and submitted == 1, (
            f"two requests submitted {submitted} background loads ({results!r}); "
            "the slot must be shared by every ModelService, since each request gets one"
        )

    def test_a_request_constructed_mid_load_does_not_release_it(self):
        loading = _service(_yield)
        with fake_gpus(*NODE):
            asyncio.run(loading.load_model(3))
        # A later request (e.g. a status poll) constructs another service.
        later = _service(_yield)
        with fake_gpus(*NODE):
            with pytest.raises(ModelBusyError):
                asyncio.run(later.load_model(4))
        assert later._executor.calls == []


class TestThePrecheckDoesNotBlockTheEventLoop:
    """The pre-check reads the cards through nvidia-smi (up to a 5 s timeout).

    Review round 3, 2026-09-14: it ran inline in the async request, stalling the
    event loop — and every request with it — on each load.

    MUTATION CONTROL, verified red then restored:
      * `await asyncio.to_thread(_precheck)` -> `_precheck()` -> this test fails
    """

    def test_the_precheck_runs_off_the_event_loop_thread(self):
        import threading

        svc = _service(_yield)
        seen = {}

        def record_thread(_model, _wanted):
            seen["precheck"] = threading.get_ident()

        svc._precheck_placement = record_thread

        async def load():
            seen["loop"] = threading.get_ident()
            await svc.load_model(3)

        with fake_gpus(*NODE):
            asyncio.run(load())

        assert "precheck" in seen, "the pre-check never ran"
        assert seen["precheck"] != seen["loop"], "the pre-check ran on the event loop thread"


class TestAStartThatFailsAfterLoadingPutsTheRowBack:
    """A failure between LOADING and submitting the background load.

    Review round 3, 2026-09-14: the slot was released but the row stayed
    LOADING, and `load_model` refuses a LOADING model as busy — so that model
    could not be loaded again until miLLM restarted.

    MUTATION CONTROL, verified red then restored:
      * drop the restore in the except block -> both tests fail
    """

    def _failing_emitter(self, svc, error):
        svc.emitter = MagicMock()
        svc.emitter.emit_load_progress = AsyncMock(side_effect=error)

    def _statuses(self, svc):
        return [c.kwargs.get("status") for c in svc.repository.update_status.await_args_list]

    def test_an_error_leaves_the_row_in_error_not_loading(self):
        svc = _service(_yield)
        self._failing_emitter(svc, RuntimeError("socket down"))

        with fake_gpus(*NODE):
            with pytest.raises(RuntimeError, match="socket down"):
                asyncio.run(svc.load_model(3))

        statuses = self._statuses(svc)
        assert statuses[-2:] == [ModelStatus.LOADING, ModelStatus.ERROR], statuses
        assert svc._loading_model_id is None
        assert svc._executor.calls == []

    def test_a_cancelled_start_leaves_the_row_ready(self):
        svc = _service(_yield)
        self._failing_emitter(svc, asyncio.CancelledError())

        with fake_gpus(*NODE):
            with pytest.raises(asyncio.CancelledError):
                asyncio.run(svc.load_model(3))

        statuses = self._statuses(svc)
        assert statuses[-2:] == [ModelStatus.LOADING, ModelStatus.READY], statuses
        assert svc._loading_model_id is None
