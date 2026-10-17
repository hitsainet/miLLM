"""Feature 29 task 8.5: `GET /api/health/gpus` is reachable, runs in a thread, takes no slot."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient

READ = {
    "read_at": datetime(2026, 10, 6, 12, tzinfo=timezone.utc),
    "cards": [{
        "smi_index": 0, "uuid": "GPU-x", "name": "RTX 3090", "total_mb": 24576,
        "used_mb": 100, "free_mb": 24476, "torch_index": 0, "torch_measured": False,
        "millm_allocated_mb": None, "millm_reserved_mb": None, "engine_memory": None,
        "processes": None, "processes_reason": "nvidia-smi --query-compute-apps unavailable",
    }],
    "reason": None,
}


def _app():
    from millm.main import create_app

    return create_app()


def test_the_route_is_in_the_live_app():
    paths = _app().openapi()["paths"]
    assert "get" in paths["/api/health/gpus"]


def test_payload_and_one_threaded_read():
    real_to_thread = asyncio.to_thread
    calls = []

    async def spy(fn, *args, **kwargs):
        calls.append(fn)
        return await real_to_thread(fn, *args, **kwargs)

    with patch("millm.services.gpu_memory.read_gpu_memory", return_value=READ) as read, \
            patch("asyncio.to_thread", spy):
        resp = TestClient(_app()).get("/api/health/gpus")
    assert resp.status_code == 200
    body = resp.json()
    assert body["cards"][0]["millm_reserved_mb"] is None
    assert body["cards"][0]["torch_measured"] is False
    assert body["reason"] is None
    assert read.call_count == 1
    assert calls == [read], "the nvidia-smi read must run in a worker thread, exactly once"


def test_no_nvidia_smi_is_200_with_a_reason():
    with patch("millm.ml.nvidia_smi.query_gpus", return_value=[]):
        resp = TestClient(_app()).get("/api/health/gpus")
    assert resp.status_code == 200
    assert resp.json()["cards"] == [] and resp.json()["reason"] == "nvidia-smi unavailable"


def test_it_takes_no_request_queue_slot():
    from millm.api.dependencies import get_inference_service

    queue = get_inference_service().request_queue
    with patch("millm.services.gpu_memory.read_gpu_memory", return_value=READ), \
            patch.object(type(queue), "acquire") as acquire:
        TestClient(_app()).get("/api/health/gpus")
    assert acquire.call_count == 0
