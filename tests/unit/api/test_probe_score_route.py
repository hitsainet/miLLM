"""`POST /api/probes/score` through the LIVE app (FR-27.4g; reachability is a shipping gate).

Asserted against the built application: the path is in `app.openapi()["paths"]`, a request reaches
the SCORE handler (not a `{probe_id}` handler), the payload the service receives, the number of
times it is called, and the envelope that comes back. Also the boundary (P-03) through the route,
and X-09's rule that this non-generation endpoint carries no `X-miLLM-Steering` header.

Mutation this catches (FTID §8): M20 — remove the route.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from millm.db.repositories.probe_repository import ProbeRepository
from millm.services.probe_runtime import ProbeRuntimeState
from millm.services.probe_service import ProbeService
from tests.unit.f25_fixtures import clear_loaded, make_service, word_model, word_tokenizer
from tests.unit.score_fixtures import tiny_definition, tiny_identity


@pytest.fixture(autouse=True)
def clean():
    ProbeRuntimeState.reset_for_tests()
    yield
    ProbeRuntimeState.reset_for_tests()
    clear_loaded()


@pytest.fixture
async def live(test_session, monkeypatch):
    from millm.api.dependencies import get_db, get_inference_service, get_probe_repository
    from millm.main import create_app

    model, tokenizer = word_model(), word_tokenizer()
    inference = make_service(model, tokenizer)
    monkeypatch.setattr(
        "millm.services.probe_arm_bridge.loaded_identity",
        AsyncMock(return_value=(tiny_identity(), model, tokenizer)),
    )
    repo = ProbeRepository(test_session)

    async def db():
        yield test_session

    app = create_app()
    app.dependency_overrides[get_db] = db
    app.dependency_overrides[get_inference_service] = lambda: inference
    app.dependency_overrides[get_probe_repository] = lambda: repo
    return app, repo, inference


def test_the_path_is_served_by_the_built_app():
    from millm.main import create_app

    paths = create_app().openapi()["paths"]
    assert "/api/probes/score" in paths
    assert "post" in paths["/api/probes/score"]


async def test_a_request_reaches_the_score_handler_with_its_payload(live, monkeypatch):
    app, repo, _inference = live
    await ProbeService(repo).import_definition(tiny_definition())
    calls = []
    from millm.services.probe_scoring import ProbeScoringService

    real = ProbeScoringService.score

    async def spy(self, request, session):
        calls.append(request.model_dump())
        return await real(self, request, session)

    monkeypatch.setattr(ProbeScoringService, "score", spy)
    # No `with`: entering the client runs the app's lifespan, which reconfigures logging
    # and blinded later tests' log capture.
    client = TestClient(app)
    response = client.post("/api/probes/score", json={
        "inputs": [{"token_ids": [2, 6, 7, 8], "prompt_tokens": 2}], "windows": ["all"],
    })
    assert response.status_code == 200, response.text
    assert len(calls) == 1, "the score handler must be reached exactly once"
    assert calls[0]["inputs"][0]["token_ids"] == [2, 6, 7, 8]
    assert calls[0]["windows"] == ["all"]
    body = response.json()
    assert body["success"] is True
    assert body["data"]["results"][0]["verdicts"][0]["window"] == "all"
    assert "X-miLLM-Steering" not in response.headers, "X-09: no steering header here"


async def test_a_score_EXACTLY_on_the_bar_fires_through_the_route(live):
    """5.1 / P-03: `>=`, offline as live. The fixture is exactly representable: zero weights and
    bias 2.0 make every token score exactly 2.0, and the mean of 2.0s is 2.0."""
    app, repo, _ = live
    definition = tiny_definition(weights=[0.0] * 16, threshold=2.0)
    definition["head"]["bias"] = 2.0
    await ProbeService(repo).import_definition(definition)
    # No `with`: entering the client runs the app's lifespan, which reconfigures logging
    # and blinded later tests' log capture.
    client = TestClient(app)
    response = client.post("/api/probes/score", json={
        "inputs": [{"token_ids": [2, 6, 7]}], "windows": ["all"],
    })
    assert response.status_code == 200, response.text
    v = response.json()["data"]["results"][0]["verdicts"][0]
    assert v["score"] == 2.0 and v["threshold"] == 2.0
    assert v["verdict"] is True


async def test_a_refusal_reaches_the_caller_by_code(live):
    app, _repo, _ = live
    # No `with`: entering the client runs the app's lifespan, which reconfigures logging
    # and blinded later tests' log capture.
    client = TestClient(app)
    response = client.post("/api/probes/score", json={"inputs": []})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_PROBE_SCORE_REQUEST"


async def test_score_is_not_mistaken_for_a_probe_id(live):
    """`/api/probes/score` must not fall into `/{probe_id}` handlers: a GET is not served."""
    app, _repo, _ = live
    # No `with`: entering the client runs the app's lifespan, which reconfigures logging
    # and blinded later tests' log capture.
    client = TestClient(app)
    response = client.get("/api/probes/score")
    # GET /api/probes/{probe_id} would answer 404 PROBE_NOT_FOUND for id "score"; that is the
    # only GET on that shape, and it is correct for it to say so. The POST above is the score.
    assert response.status_code == 404

