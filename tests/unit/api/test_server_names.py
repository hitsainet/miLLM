"""T29: the names the server itself shows are miServer (040 FR-040.34, TPD-15).

The OpenAPI title and contact, `GET /api/health/version`'s `app`, and the `owned_by` a model with
no repository id is listed under on `/v1/models`. The package name stays `millm` until P8 (041).

Control NC-87 (0xcc/reviews/040_P4_controls_2026-11-11.md): revert the OpenAPI title ->
`test_openapi_title` red.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from miserver.api.dependencies import get_inference_service, get_model_service
from miserver.api.routes.openai import models as models_routes
from miserver.api.schemas.openai import ModelObject
from miserver.main import create_app


@pytest.fixture(scope="module")
def app():
    return create_app()


def test_openapi_title(app):
    document = app.openapi()
    assert document["info"]["title"] == "miServer API"
    assert app.title == "miServer API"
    assert "miServer provides a REST API" in document["info"]["description"]
    assert document["info"]["contact"] == {"name": "miServer Support",
                                           "url": "https://github.com/hitsainet/miServer"}
    assert document["info"]["license"]["name"] == "Apache-2.0", "the LICENSE file is Apache-2.0"
    assert "miLLM" not in document["info"]["description"]


def test_version_app_is_miserver(app):
    response = TestClient(app).get("/api/health/version")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["app"] == "miServer", body
    assert body["version"], body


def test_version_app_is_miserver_without_a_version_file(app, monkeypatch):
    """The fallback (no VERSION file: the package's own version) names the app the same way. It was
    a second `return` with its own literal, which no test reached (control NC-87c survived)."""
    import miserver
    from miserver.api.routes.system import health

    monkeypatch.setattr(health.Path if hasattr(health, "Path") else __import__("pathlib").Path,
                        "exists", lambda self: False)
    assert health.get_version() == {"version": miserver.__version__, "app": "miServer"}


def _row(row_id: int, name: str, repo_id: str | None):
    stamp = datetime(2026, 11, 11, tzinfo=UTC)
    return SimpleNamespace(id=row_id, name=name, repo_id=repo_id, loaded_at=None, created_at=stamp,
                           source="local" if repo_id is None else "huggingface", config_json=None)


def _models_client(rows, locked=None) -> TestClient:
    service = SimpleNamespace(get_locked_model=AsyncMock(return_value=locked),
                              get_available_models=AsyncMock(return_value=rows),
                              get_model=AsyncMock(side_effect=lambda model_id: rows[0]))
    inference = SimpleNamespace(get_loaded_model_info=lambda: None)  # nothing resident
    api = FastAPI()
    api.include_router(models_routes.router, prefix="/v1")
    api.dependency_overrides[get_model_service] = lambda: service
    api.dependency_overrides[get_inference_service] = lambda: inference
    return TestClient(api)


def test_models_owned_by_fallback():
    """A model with a repository id is owned by it; one without (a local path) by "miServer": on
    the list, in locked mode, and as the schema's default."""
    assert ModelObject(id="x", created=0).owned_by == "miServer"
    local, hub = _row(1, "local-model", None), _row(2, "hub-model", "org/hub-model")

    listed = _models_client([local, hub]).get("/v1/models")
    assert listed.status_code == 200, listed.text
    assert {m["id"]: m["owned_by"] for m in listed.json()["data"]} == {
        "local-model": "miServer", "hub-model": "org/hub-model"}

    locked = _models_client([local, hub], locked=local).get("/v1/models")
    assert [(m["id"], m["owned_by"]) for m in locked.json()["data"]] == [("local-model", "miServer")]
