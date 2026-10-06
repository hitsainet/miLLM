"""Feature 29 task 7.6: the typed queue-state block, the lease field, and their contracts.

The queue under test is a REAL `RequestQueue` with real holders, so `in_flight` and
`queue_waiting` are measured, not set. The health endpoint is read through the real app.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from millm.api.routes.system.health import InferenceState, read_inference_state
from millm.core.backpressure import register_backlog_provider
from millm.services.request_queue import RequestQueue
from tests.unit.lease_fixtures import clear_resident, install_registry, set_resident

INFERENCE_FIELDS = {
    "backend", "cbm_enabled", "cbm_running", "queue_pending", "queue_max_concurrent",
    "queue_max_pending", "in_flight", "queue_waiting", "batch_backlog_rows",
    "estimated_wait_seconds", "error",
}
LEASE_FIELDS = {
    "model_id", "model_name", "holder", "reason", "acquired_at", "renewed_at", "expires_at",
    "ttl_seconds", "seconds_remaining",
}


@pytest.fixture(autouse=True)
def _clean():
    clear_resident()
    register_backlog_provider(None)
    yield
    clear_resident()
    register_backlog_provider(None)


def _inference(queue, cbm_running=False):
    return SimpleNamespace(
        request_queue=queue,
        _cbm_backend=object() if cbm_running else None,
        _use_cbm=lambda: cbm_running,
    )


async def _occupy(queue, n):
    """n requests on the queue: the first holds the slot, the rest wait."""
    release = asyncio.Event()

    async def one():
        async with queue.acquire():
            await release.wait()

    tasks = [asyncio.create_task(one()) for _ in range(n)]
    for _ in range(1000):
        if queue.pending_count == n:
            break
        await asyncio.sleep(0)
    return release, tasks


class TestCounts:
    async def test_one_holding_two_waiting(self):
        queue = RequestQueue(max_concurrent=1, max_pending=10)
        release, tasks = await _occupy(queue, 3)
        state = read_inference_state(_inference(queue))
        assert state.queue_pending == 3, "queue_pending keeps its meaning: waiting + holding"
        assert state.in_flight == 1
        assert state.queue_waiting == 2
        assert state.error is None
        release.set()
        await asyncio.gather(*tasks)
        after = read_inference_state(_inference(queue))
        assert (after.in_flight, after.queue_waiting, after.queue_pending) == (0, 0, 0)

    async def test_durations_are_recorded_and_the_estimate_follows(self):
        queue = RequestQueue(max_concurrent=1, max_pending=10)
        for _ in range(3):
            async with queue.acquire():
                pass
        assert queue.median_hold_seconds() is not None
        release, tasks = await _occupy(queue, 3)
        with patch.object(queue, "median_hold_seconds", return_value=4.0):
            state = read_inference_state(_inference(queue))
        # 4.0 × (2 waiting + 1 in flight) / 1
        assert state.estimated_wait_seconds == 12.0
        release.set()
        await asyncio.gather(*tasks)

    async def test_estimate_null_below_three_samples(self):
        queue = RequestQueue(max_concurrent=1, max_pending=10)
        async with queue.acquire():
            pass
        assert read_inference_state(_inference(queue)).estimated_wait_seconds is None

    async def test_the_idle_cache_release_counts_as_holding(self):
        """T-90: the release takes a slot, so a request arriving then waits — it counts."""
        from millm.services.inference_service import InferenceService

        svc = InferenceService.__new__(InferenceService)
        svc._request_queue = RequestQueue(max_concurrent=1, max_pending=10)
        svc._use_cbm = lambda: False
        svc._loaded_gpu_indices = lambda: [0]
        svc._engine_is_llamacpp = lambda: False
        svc._idle_release_generation = 0
        seen = {}

        def fake_release(indices):
            seen["state"] = read_inference_state(_inference(svc._request_queue))
            return {}

        with patch("millm.services.inference_service._release_cached_gpu_memory", fake_release):
            await svc._release_idle_cache(0)
        assert seen["state"].in_flight == 1 and seen["state"].queue_waiting == 0

    def test_cbm_running_nulls_the_unmeasurable_fields(self):
        queue = RequestQueue(max_concurrent=1, max_pending=10)
        state = read_inference_state(_inference(queue, cbm_running=True))
        assert state.backend == "cbm" and state.cbm_running is True
        assert state.in_flight is None
        assert state.queue_waiting is None
        assert state.estimated_wait_seconds is None
        assert state.queue_pending == 0

    def test_a_read_failure_is_reported_not_omitted(self):
        class Broken:
            @property
            def request_queue(self):
                raise RuntimeError("queue gone")

        state = read_inference_state(Broken())
        assert state.error and "queue gone" in state.error
        assert state.in_flight is None and state.queue_pending is None


class TestBacklog:
    def test_null_when_no_batch_api(self):
        """Control M16."""
        state = read_inference_state(_inference(RequestQueue()))
        assert state.batch_backlog_rows is None

    def test_the_providers_value(self):
        register_backlog_provider(lambda: 25_000)
        assert read_inference_state(_inference(RequestQueue())).batch_backlog_rows == 25_000


class TestContract:
    def test_inference_field_set_is_pinned(self):
        assert set(InferenceState.model_fields) == INFERENCE_FIELDS

    def test_lease_field_set_is_pinned(self):
        from millm.api.schemas.lease import LeaseStatusResponse

        assert set(LeaseStatusResponse.model_fields) == LEASE_FIELDS
        assert "lease_id" not in LeaseStatusResponse.model_fields


class TestDetailedEndpoint:
    @pytest.fixture
    def client(self):
        from millm.api.dependencies import get_inference_service, get_model_loader
        from millm.main import create_app
        from millm.ml.model_loader import ModelLoader

        self.queue = RequestQueue(max_concurrent=1, max_pending=10)
        app = create_app()
        app.dependency_overrides[get_inference_service] = lambda: _inference(self.queue)
        app.dependency_overrides[get_model_loader] = lambda: ModelLoader()
        return TestClient(app)

    def test_the_block_is_always_present_with_every_field(self, client):
        body = client.get("/api/health/detailed").json()
        assert set(body["inference"]) == INFERENCE_FIELDS
        assert body["inference"]["batch_backlog_rows"] is None
        assert body["inference"]["in_flight"] == 0
        assert body["lease"] is None

    def test_the_lease_appears_without_its_id(self, client):
        registry, _ = install_registry()
        set_resident(1, "m1")
        grant = registry.grant(1, "m1", "midataworks", "label run 7", 120)
        response = client.get("/api/health/detailed")
        body = response.json()
        assert set(body["lease"]) == LEASE_FIELDS
        assert body["lease"]["holder"] == "midataworks"
        assert body["lease"]["seconds_remaining"] == 120
        assert grant.lease_id not in response.text

    def test_no_nvidia_smi_on_this_path(self, client):
        with patch("millm.ml.nvidia_smi.subprocess.run") as run:
            client.get("/api/health/detailed")
        assert run.call_count == 0
