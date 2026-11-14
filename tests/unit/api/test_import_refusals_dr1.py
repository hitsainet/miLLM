"""Debt D-R1-a / D-R1-b (038 implementation controls, R1 tracked debt), closed 2026-10-25.

D-R1-a: `POST /api/circuits/import` and `POST /api/clusters/import` answered a document that broke
the v1 contract with HTTP 200 + `success: false`, and each error in `details.errors` carried
pydantic's `input` (for a model-level or container error, the caller's document). Every other
VALIDATION_ERROR is a 422 (`miserver.core.errors.ValidationError.status_code`), and the probe import
(R1-1) already stripped `input`. Both routes now raise `ValidationError` with
`error_reference.contract_errors`, the one helper the probe import uses too.

D-R1-b: `miserver_error_handler`'s `api_error` warning logged every refusal's `details` whole, so any
refusal whose details carried what the caller sent put it in the server log. It now logs
`error_reference.details_shape`: keys and numbers, never strings, lists or mappings.

Every request goes through the LIVE app (`create_app()`), so a test also fails if the routes or the
handler are not wired. Each assertion uses a sentinel the caller sent and checks it is absent.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

SENTINEL = "SENTINEL-DR1-USER-TEXT"


@pytest.fixture
def client():
    from miserver.api.dependencies import get_circuit_service, get_cluster_service
    from miserver.main import create_app
    from miserver.services.circuit_service import CircuitService

    app = create_app()
    # The REAL circuit service: its `import_definition` validates before it touches the
    # repository, so a refused document never reaches the mock.
    circuit_service = CircuitService(repository=MagicMock())
    cluster_service = MagicMock()
    cluster_service.import_definition = AsyncMock()
    cluster_service.import_bundle = AsyncMock()
    app.dependency_overrides[get_circuit_service] = lambda: circuit_service
    app.dependency_overrides[get_cluster_service] = lambda: cluster_service
    # No `with`: the lifespan (model, database, telemetry) is not started.
    yield TestClient(app, raise_server_exceptions=False), cluster_service


def _bad_circuit() -> dict:
    return {
        "kind": "mistudio.circuit-definition",
        "schema_version": "1",
        "name": "dr1",
        "saes": [{"layer": 10, "n_features": 8192}],
        # A whole-member error (the member is not an object) and a field error: the first
        # error's `input` would be the caller's text verbatim.
        "members": [SENTINEL, {"layer": SENTINEL, "feature": {"feature_idx": 1, "strength": 1.0}}],
    }


def _bad_cluster() -> dict:
    return {
        "kind": "mistudio.cluster-definition",
        "schema_version": "1",
        "name": "dr1",
        "members": [SENTINEL, {"feature_idx": SENTINEL, "strength": 1.0}],
    }


def _assert_refused_without_echo(response) -> list[dict]:
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert body["error"]["message"].startswith("Payload does not match the v1 contract")
    errors = body["error"]["details"]["errors"]
    assert errors, body
    assert all("input" not in e and "url" not in e for e in errors), errors
    assert all(e["loc"] and e["msg"] for e in errors), errors  # still names what is wrong
    assert SENTINEL not in response.text
    return errors


@pytest.mark.parametrize(
    "path, payload",
    [("/api/circuits/import", _bad_circuit()), ("/api/clusters/import", _bad_cluster())],
    ids=["circuits", "clusters"],
)
def test_a_contract_error_is_a_422_that_echoes_none_of_the_document(client, path, payload):
    http, _ = client
    _assert_refused_without_echo(http.post(path, json=payload))


def test_a_refused_cluster_is_never_imported(client):
    http, cluster_service = client
    _assert_refused_without_echo(http.post("/api/clusters/import", json=_bad_cluster()))
    assert cluster_service.import_definition.await_count == 0


def test_a_circuit_nesting_bomb_is_a_422(client):
    http, _ = client
    bomb: dict = {}
    node = bomb
    for _ in range(3000):
        node["n"] = {}
        node = node["n"]
    doc = _bad_circuit()
    doc["nested"] = bomb
    response = http.post("/api/circuits/import", json=doc)
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == "VALIDATION_ERROR"
    assert "nests" in error["message"]


@pytest.mark.parametrize(
    "path, payload",
    [("/api/circuits/import", _bad_circuit()), ("/api/clusters/import", _bad_cluster())],
    ids=["circuits", "clusters"],
)
def test_a_refused_import_is_logged_without_the_document(client, path, payload, caplog, capsys):
    http, _ = client
    caplog.set_level(logging.DEBUG)
    with capture_logs() as logs:
        response = http.post(path, json=payload)
    errors = _assert_refused_without_echo(response)
    lines = [e for e in logs if e.get("event") == "api_error"]
    assert len(lines) == 1, logs
    line = lines[0]
    assert line["error_code"] == "VALIDATION_ERROR"
    assert line["status_code"] == 422
    # The shape keeps the key and how many errors there were — enough to debug.
    assert line["details"]["errors"] == f"list[{len(errors)}]"
    out = capsys.readouterr()
    assert SENTINEL not in repr(logs) + out.out + out.err + caplog.text


async def test_the_api_error_line_carries_keys_and_numbers_never_text():
    """D-R1-b at the handler: any refusal, any route family."""
    from miserver.api.exception_handlers import miserver_error_handler
    from miserver.core.errors import ValidationError

    exc = ValidationError(
        "refused",
        details={
            "prompt": SENTINEL,
            "messages": [{"content": SENTINEL}],
            "document": {"text": SENTINEL},
            "layer": 12,
            "score": 0.5,
            "streamable": False,
            "param": None,
        },
    )
    for path in ("/api/probes/import", "/v1/chat/completions"):
        request = SimpleNamespace(url=SimpleNamespace(path=path), method="POST")
        with capture_logs() as logs:
            response = await miserver_error_handler(request, exc)
        (line,) = [e for e in logs if e.get("event") == "api_error"]
        assert line["details"] == {
            "prompt": f"str[{len(SENTINEL)}]",
            "messages": "list[1]",
            "document": "dict[1]",
            "layer": 12,
            "score": 0.5,
            "streamable": False,
            "param": None,
        }
        assert SENTINEL not in repr(logs)
        # The caller still receives the details it caused.
        if path.startswith("/api/"):
            assert SENTINEL in response.body.decode()


# --- The same echo on the two refusals that validate what miLLM holds, not a request body.
# A stored definition or a built slice is still a whole document, and these errors go back in a
# 4xx body too.


def _assert_no_echo(details: dict) -> None:
    errors = details["errors"]
    assert errors, "the refusal must still name what is wrong"
    for error in errors:
        assert "input" not in error and "url" not in error, error
        assert error["loc"] and error["msg"], error
    assert SENTINEL not in str(details)


def test_a_stored_definition_that_no_longer_validates_echoes_none_of_it():
    from miserver.core.errors import ValidationError
    from miserver.services.circuit_service import CircuitService

    stored = SimpleNamespace(id="c1", name="dr1", circuit_meta=_bad_circuit())
    with pytest.raises(ValidationError) as refused:
        CircuitService(repository=MagicMock())._parse_stored(stored)
    assert refused.value.details["circuit_id"] == "c1"
    _assert_no_echo(refused.value.details)


async def test_a_slice_that_cannot_be_projected_echoes_none_of_it(monkeypatch):
    from miserver.core.errors import ValidationError
    from miserver.services.circuit_service import CircuitService

    service = CircuitService(repository=MagicMock())
    monkeypatch.setattr(service, "to_layer_slice", lambda *a, **k: {"name": SENTINEL, "x": SENTINEL})
    circuit = SimpleNamespace(id="c1", name="dr1", rung=0, intensity=1.0)
    with pytest.raises(ValidationError) as refused:
        await service._serve_slices(circuit, MagicMock(), [10], [])
    assert refused.value.details["layer"] == 10
    _assert_no_echo(refused.value.details)
