"""Feature 35: the values that used to be captured once at startup now follow an override.

Each test builds the REAL reader first (the process-wide service, the real retention task), THEN
changes the setting through `runtime_settings` — the order production sees: the singletons exist
before the operator saves anything. Each fails when its fix is reverted:

* MAX_PENDING_REQUESTS — `get_inference_service` passing the number instead of the callable.
* CLUSTER/PROBE_HUB_CACHE_TTL_S — `_ttl` assigned in `__init__` again.
* BATCH_RETENTION_INTERVAL_S — `start_batch_api` passing `float(...)` again.
* CIRCUIT_SENSING_MAX_TOKEN_LAG — an unarmed service reporting its construction-time lag.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from millm.core.config import settings
from millm.services import runtime_settings as rs
from millm.services.request_queue import QueueFullError


@pytest.fixture
def fresh_inference_service():
    from millm.api.dependencies import get_inference_service

    get_inference_service.cache_clear()
    yield get_inference_service
    get_inference_service.cache_clear()


class TestMaxPendingAtAdmission:
    async def test_the_real_queue_admits_by_the_value_in_force(self, fresh_inference_service):
        rs._apply({"MAX_PENDING_REQUESTS": 1})
        queue = fresh_inference_service().request_queue  # built while the limit is 1
        async with queue.acquire():
            with pytest.raises(QueueFullError):
                async with queue.acquire():
                    pass
            rs._apply({"MAX_PENDING_REQUESTS": 3})
            assert queue.max_pending == 3 and queue.is_available
            entered = asyncio.Event()

            async def second() -> None:
                async with queue.acquire():
                    entered.set()

            task = asyncio.create_task(second())
            await asyncio.sleep(0)
            assert queue.pending_count == 2, "the second request was not admitted under the new limit"
        await asyncio.wait_for(task, 1)
        assert entered.is_set()

    async def test_health_reports_the_limit_in_force(self, fresh_inference_service):
        """`/api/health/detailed` reads `queue.max_pending`, which now follows the setting."""
        import httpx

        from millm.main import create_app

        fresh_inference_service()  # the singleton exists before the change
        rs._apply({"MAX_PENDING_REQUESTS": 7})
        app = create_app()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://test") as client:
            body = (await client.get("/api/health/detailed")).json()
        found = _find(body, "queue_max_pending")
        assert found == 7, body


def _find(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            hit = _find(value, key)
            if hit is not None:
                return hit
    if isinstance(obj, list):
        for value in obj:
            hit = _find(value, key)
            if hit is not None:
                return hit
    return None


class TestHubTtlAtTheFreshnessCheck:
    @pytest.mark.parametrize("kind,key", [("cluster", "CLUSTER_HUB_CACHE_TTL_S"),
                                          ("probe", "PROBE_HUB_CACHE_TTL_S")])
    async def test_the_real_singleton_follows_the_ttl(self, kind, key, monkeypatch):
        import millm.api.dependencies as deps

        monkeypatch.setattr(deps, f"_{kind}_hub_service", None)
        service = (deps.get_cluster_hub_service() if kind == "cluster"
                   else await deps.get_probe_hub_service())
        service._cache_put("listing", ["a"])
        assert service._cache_get("listing") == ["a"]  # fresh at 300 s
        rs._apply({key: 0})
        assert service._cache_get("listing") is None, f"{key}=0 must make every entry stale"
        rs._apply({key: 300})
        service._cache_put("listing", ["b"])
        assert service._cache_get("listing") == ["b"]

    def test_an_explicit_ttl_still_wins(self):
        from millm.services.cluster_hub_service import ClusterHubService

        service = ClusterHubService(cache_ttl_s=5)
        rs._apply({"CLUSTER_HUB_CACHE_TTL_S": 0})
        assert service._ttl == 5


class TestRetentionIntervalEachTick:
    async def test_the_running_loop_follows_the_interval(self, monkeypatch):
        """The real `start_batch_api` starts the real loop at 3600 s; lowering the interval makes
        it prune within a tick, with no restart."""
        import millm.services.batch.retention as retention
        from millm.services.batch import reconcile

        monkeypatch.setattr(retention, "RETENTION_TICK_S", 0.01)
        rs._apply({"BATCH_RETENTION_INTERVAL_S": 3600.0})
        loop_prunes: list[int] = []

        async def fake_prune(factory, store, now=None, **kw):
            if "sweep_min_age_s" not in kw:  # the startup pass passes it; the loop does not
                loop_prunes.append(1)
            return {}

        async def fake_reconcile(factory, runner, store):
            return {}

        monkeypatch.setattr(retention, "prune_expired_files", fake_prune)
        monkeypatch.setattr(reconcile, "reconcile_batches_on_startup", fake_reconcile)
        monkeypatch.setattr("millm.services.batch.files.get_file_store", lambda: MagicMock())
        runner = MagicMock()
        await reconcile.start_batch_api("FACTORY", runner)
        try:
            await asyncio.sleep(0.1)
            assert loop_prunes == [], "pruned before the interval"
            rs._apply({"BATCH_RETENTION_INTERVAL_S": 0.02})
            for _ in range(100):
                if loop_prunes:
                    break
                await asyncio.sleep(0.01)
            assert loop_prunes, "the loop kept the interval it was started with"
        finally:
            runner._retention.cancel()
            with pytest.raises(asyncio.CancelledError):
                await runner._retention


class TestCircuitSensingLagWhileUnarmed:
    def test_an_unarmed_service_reports_the_lag_in_force(self):
        from millm.services.circuit_sensing_service import CircuitSensingService

        service = CircuitSensingService()
        assert service.status()["max_token_lag"] == settings.CIRCUIT_SENSING_MAX_TOKEN_LAG
        rs._apply({"CIRCUIT_SENSING_MAX_TOKEN_LAG": 5})
        assert service.status()["max_token_lag"] == 5
