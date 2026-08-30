"""Model selection, on-demand loading, and the steering lock.

Three behaviours, all reported broken on 2026-08-30:

  1. miLLM auto-loaded a fixed model (granite-4.1-8b) on every restart.
  2. Selecting a model in a client did NOT load it — the endpoints rejected
     anything not already resident, so the picker was decorative.
  3. The rule that a model must be immovable ONLY while locked for steering,
     and that a lock must hide every other model from /v1/models.

load_model_and_wait() existed for (2) — its docstring says "Used by the
OpenAI-compatible endpoints for auto-load on demand" — and had zero callers.
The lock branch of /v1/models existed for (3) and was never exercised by a
test: every case stubbed get_locked_model to None.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from millm.api.schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatMessage,
    Usage,
)
from millm.core.errors import ModelLockedError
from millm.main import create_app


def _response(model="gemma-4-12B-it"):
    """A REAL response object — FastAPI validates the response model, so a
    MagicMock here fails validation rather than exercising the route."""
    return ChatCompletionResponse(
        id="chatcmpl-test", created=0, model=model,
        choices=[ChatCompletionChoice(
            index=0, message=ChatMessage(role="assistant", content="ok"),
            finish_reason="stop")],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _model(name, mid=1):
    m = MagicMock()
    m.id = mid
    m.name = name
    m.repo_id = f"org/{name}"
    m.loaded_at = datetime(2026, 1, 1)
    m.created_at = datetime(2026, 1, 1)
    return m


def _loaded_info(name):
    info = MagicMock()
    info.name = name
    return info


def _app(*, locked=None, available=(), found=None, loaded=None,
         load_side_effect=None):
    from millm.api.dependencies import get_inference_service, get_model_service

    svc = MagicMock()
    svc.get_locked_model = AsyncMock(return_value=locked)
    svc.get_available_models = AsyncMock(return_value=list(available))
    svc.find_model_by_name = AsyncMock(return_value=found)
    svc.load_model_and_wait = AsyncMock(side_effect=load_side_effect)

    inference = MagicMock()
    inference.backend_name = "serial"
    inference.request_queue = MagicMock(pending_count=0, max_pending=5)
    inference.resolve_request_intensity = AsyncMock(return_value=None)
    state = {"loaded": loaded}
    inference.get_loaded_model_info = lambda: (
        _loaded_info(state["loaded"]) if state["loaded"] else None
    )

    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: svc
    app.dependency_overrides[get_inference_service] = lambda: inference
    return app, svc, inference, state


BODY = {"model": "gemma-4-12B-it", "messages": [{"role": "user", "content": "hi"}]}


class TestSelectingAModelLoadsIt:
    def test_chat_loads_a_model_that_is_not_resident(self):
        target = _model("gemma-4-12B-it", 33)

        async def _load(mid):
            assert mid == 33
            state["loaded"] = "gemma-4-12B-it"

        app, svc, inf, state = _app(found=target, loaded="granite-4.1-8b")
        svc.load_model_and_wait = AsyncMock(side_effect=_load)

        inf.create_chat_completion = AsyncMock(return_value=_response())
        with TestClient(app) as tc:
            tc.post("/v1/chat/completions", json=BODY)

        svc.load_model_and_wait.assert_awaited_once(), (
            "selecting a model did not load it; the picker is decorative"
        )

    def test_an_already_loaded_model_is_not_reloaded(self):
        """The common path must cost nothing."""
        target = _model("gemma-4-12B-it", 33)
        app, svc, inf, state = _app(found=target, loaded="gemma-4-12B-it")
        inf.create_chat_completion = AsyncMock(return_value=_response())
        with TestClient(app) as tc:
            tc.post("/v1/chat/completions", json=BODY)
        svc.load_model_and_wait.assert_not_awaited()

    def test_an_unknown_model_is_still_a_404_not_a_load_attempt(self):
        app, svc, inf, state = _app(found=None, loaded="granite-4.1-8b")
        with TestClient(app) as tc:
            r = tc.post("/v1/chat/completions",
                        json={**BODY, "model": "does-not-exist"})
        assert r.status_code == 404
        svc.load_model_and_wait.assert_not_awaited()


class TestTheSteeringLockIsTheOnlyThingThatPins:
    def test_a_locked_model_refuses_a_switch_rather_than_swapping(self):
        """Swapping weights under an attached SAE would invalidate steering."""
        target = _model("gemma-4-12B-it", 33)
        app, svc, inf, state = _app(
            found=target, loaded="granite-4.1-8b",
            locked=_model("granite-4.1-8b", 3),
            load_side_effect=ModelLockedError(
                "locked", details={"locked_model_name": "granite-4.1-8b"}),
        )
        with TestClient(app) as tc:
            r = tc.post("/v1/chat/completions", json=BODY)
        assert r.status_code in (409, 423, 400), r.status_code
        body = r.text.lower()
        assert "lock" in body or "granite" in body

    def test_only_the_locked_model_is_listed_when_locked(self):
        """A lock must HIDE the others, not merely refuse them.

        This branch existed and no test had ever entered it: every case stubbed
        get_locked_model to None.
        """
        app, svc, inf, state = _app(
            locked=_model("granite-4.1-8b", 3),
            available=[_model("a", 1), _model("b", 2), _model("granite-4.1-8b", 3)],
        )
        with TestClient(app) as tc:
            r = tc.get("/v1/models")
        assert r.status_code == 200
        ids = [m["id"] for m in r.json()["data"]]
        assert ids == ["granite-4.1-8b"], (
            f"a locked model must be the only one advertised; got {ids}"
        )

    def test_every_model_is_listed_when_nothing_is_locked(self):
        app, svc, inf, state = _app(
            locked=None, available=[_model("a", 1), _model("b", 2)])
        with TestClient(app) as tc:
            ids = [m["id"] for m in tc.get("/v1/models").json()["data"]]
        assert sorted(ids) == ["a", "b"]
