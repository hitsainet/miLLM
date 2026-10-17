"""D-45 (F-A) and D-46 (F-B) on every `/v1` route that auto-loads (review R2-A, operator decisions
2026-10-16).

The real app, the real ModelService over a fake repository (tests/unit/lease_fixtures.py), the
real loader state and a real `WeightBarrier`. Model 1 (`m1`) is resident; requests name model 2
(`m2`) or a bare quantization repo name. "Nothing moved" is: the resident is still model 1, the
background load worker was never called, and no row status was written.

Every production auto-load caller is here — chat, completions and embeddings are the only callers
of `ModelService.load_model_and_wait` (the batch runner never loads, FR-26.7.1). Removing the
guard from `load_model_and_wait`, or a route reaching the load by another path, turns that
route's case red (controls FA-1 / FA-6 in the progress log).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from millm.api.routes.openai.errors import model_not_found_error
from millm.api.routes.openai.model_resolution import resolve_requested_model
from millm.core.errors import ModelBusyError
from millm.db.models.model import ModelStatus
from millm.db.repositories.model_repository import ModelRepository
from millm.ml.model_loader import LoadedModelState
from millm.services.weight_barrier import WeightBarrier, set_weight_barrier
from tests.support.factories import make_model
from tests.unit.lease_fixtures import (
    FakeRepo,
    build_service,
    clear_resident,
    install_registry,
    resident_inference,
)

ROUTES = {
    "chat": ("/v1/chat/completions", lambda m: {"model": m, "messages": [
        {"role": "user", "content": "hi"}]}),
    "completions": ("/v1/completions", lambda m: {"model": m, "prompt": "hi"}),
    "embeddings": ("/v1/embeddings", lambda m: {"model": m, "input": "hi"}),
}

BARE = "org/q-GGUF"
TAGGED = "org/q-GGUF:Q4_K_M"


class _BareFallbackRepo(FakeRepo):
    """`find_by_name` as the repository answers it: exact first, then the one tagged row a bare
    name prefixes (`ModelRepository.find_by_name`; pinned against the real repository in
    `TestRealRepositoryBareName`)."""

    async def find_by_name(self, name):
        exact = await super().find_by_name(name)
        if exact is not None:
            return exact
        tagged = [r for r in self.rows.values()
                  if r.name.startswith(f"{name}:") and getattr(r, "weights_origin", None) is None]
        return tagged[0] if len(tagged) == 1 else None


@pytest.fixture(autouse=True)
def _clean():
    clear_resident()
    yield
    clear_resident()
    set_weight_barrier(None)


@pytest.fixture
def env():
    from millm.api.dependencies import get_inference_service, get_model_service
    from millm.main import create_app

    install_registry()
    service, repo = build_service()
    repo.__class__ = _BareFallbackRepo
    repo.rows[3] = make_model(id=3, name=TAGGED, status=ModelStatus.READY)
    service.load_model_and_wait = AsyncMock(wraps=service.load_model_and_wait)
    barrier = WeightBarrier(inference=lambda: None)
    set_weight_barrier(barrier)
    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: service
    app.dependency_overrides[get_inference_service] = resident_inference
    return TestClient(app), service, repo, barrier


def _post(client, route, model):
    path, body = ROUTES[route]
    return client.post(path, json=body(model))


def _nothing_moved(service, repo) -> None:
    assert LoadedModelState().current.model_id == 1
    assert service._load_worker.call_count == 0
    assert repo.status_writes == []


@pytest.mark.parametrize("route", list(ROUTES))
class TestUnsavedCommitsRefuseTheAutoLoad:
    def test_refused_409_naming_resident_count_and_recovery(self, env, route):
        client, service, repo, barrier = env
        barrier._steps_since_load = 3  # three commit-ends since m1 was loaded, none saved
        resp = _post(client, route, "m2")
        assert resp.status_code == 409, resp.text
        err = resp.json()["error"]
        assert err["code"] == "weights_unsaved"
        assert err["type"] == "invalid_request_error"
        assert err["message"].startswith("Could not load 'm2': ")
        assert "'m1' carries 3 commit(s)" in err["message"]
        assert "POST /api/weights/save" in err["message"]
        assert 'POST /api/weights/load {"model_id": 2}' in err["message"]
        assert "POST /api/models/2/load" in err["message"]
        service.load_model_and_wait.assert_awaited_once_with(2, lease_id=None)
        _nothing_moved(service, repo)

    def test_nothing_unsaved_the_auto_load_proceeds(self, env, route):
        client, service, repo, barrier = env
        service.load_model = AsyncMock(side_effect=ModelBusyError("reached load", details={}))
        resp = _post(client, route, "m2")
        assert resp.status_code == 503, resp.text
        assert resp.json()["error"]["code"] == "model_busy"
        service.load_model.assert_awaited_once()
        assert service.load_model.await_args.args[0] == 2

    def test_naming_the_resident_is_not_refused(self, env, route):
        from millm.api.dependencies import get_inference_service

        client, service, repo, barrier = env
        barrier._steps_since_load = 3
        inference = resident_inference()
        inference.active_circuit_rung = AsyncMock(return_value=None)
        reached = AsyncMock(side_effect=ModelBusyError("reached generation", details={}))
        for name in ("create_chat_completion", "create_text_completion", "create_embeddings"):
            setattr(inference, name, reached)
        client.app.dependency_overrides[get_inference_service] = lambda: inference
        resp = _post(client, route, "m1")
        assert resp.status_code == 503, resp.text
        assert reached.await_count == 1
        assert service.load_model_and_wait.await_count == 0
        _nothing_moved(service, repo)


@pytest.mark.parametrize("route", list(ROUTES))
class TestBareQuantizationNameIsRefusedBeforeAnyLoad:
    def test_not_resident_404_and_the_resident_is_not_evicted(self, env, route):
        """Before D-46 the route auto-loaded the tagged row (evicting m1) and THEN answered this
        404. The answer is unchanged; the eviction is gone."""
        client, service, repo, _ = env
        resp = _post(client, route, BARE)
        assert resp.status_code == 404, resp.text
        err = resp.json()["error"]
        # Byte-for-byte the routes' own post-load answer (`model_not_found_error(name, served)`).
        assert resp.content == model_not_found_error(BARE, TAGGED).body
        assert err == {
            "message": f"The model '{BARE}' does not exist. Available: {TAGGED}",
            "type": "invalid_request_error", "param": "model", "code": "model_not_found",
        }
        assert service.load_model_and_wait.await_count == 0
        _nothing_moved(service, repo)

    def test_resident_tagged_row_named_bare_is_the_same_404(self, env, route):
        """The tagged row resident: the post-load check answered exactly this before D-46."""
        client, service, repo, _ = env
        from tests.unit.lease_fixtures import set_resident

        set_resident(3, TAGGED)
        resp = _post(client, route, BARE)
        assert resp.status_code == 404, resp.text
        assert resp.json()["error"]["message"] == (
            f"The model '{BARE}' does not exist. Available: {TAGGED}")
        assert service.load_model_and_wait.await_count == 0

    def test_the_exact_tagged_name_still_auto_loads(self, env, route):
        client, service, repo, _ = env
        service.load_model = AsyncMock(side_effect=ModelBusyError("reached load", details={}))
        resp = _post(client, route, TAGGED)
        assert resp.status_code == 503, resp.text
        assert service.load_model.await_args.args[0] == 3


class TestRealRepositoryBareName:
    """The resolver over the REAL repository: the bare-name fallback it guards against exists."""

    async def test_bare_name_is_refused_and_the_exact_name_resolves(self, test_session):
        test_session.add(make_model(id=7, name=TAGGED, status=ModelStatus.READY))
        await test_session.commit()
        repo = ModelRepository(test_session)
        assert (await repo.find_by_name(BARE)).name == TAGGED  # the fallback is real

        class _Service:
            find_model_by_name = staticmethod(repo.find_by_name)

        class _NoResident:
            @staticmethod
            def get_loaded_model_info():
                return None

        refused = await resolve_requested_model(BARE, _Service, _NoResident)
        assert refused.refusal_reason == "bare_name" and refused.row is None
        assert refused.refusal.status_code == 404
        assert refused.refusal_message == f"The model '{BARE}' does not exist. Available: {TAGGED}"
        exact = await resolve_requested_model(TAGGED, _Service, _NoResident)
        assert exact.refusal is None and exact.row.name == TAGGED

        # The batch validator's adapter raises it, so every line naming it records the 404.
        from millm.api.routes.openai.model_resolution import batch_model_resolver
        from millm.core.errors import ModelNotFoundError

        with pytest.raises(ModelNotFoundError) as info:
            await batch_model_resolver(_Service, _NoResident)(BARE)
        assert info.value.status_code == 404
        assert info.value.details == {"param": "model", "reason": "bare_name"}
        assert info.value.message == f"The model '{BARE}' does not exist. Available: {TAGGED}"
