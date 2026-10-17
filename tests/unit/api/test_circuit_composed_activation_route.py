"""A composed activation SAYS it composed, on the wire (F19 FR-19.3, 019 FTASKS 3.5 / 4.4).

Found on the 3090 on 2026-10-17 (GPU block 2): two circuits sharing L20, the second activated with
``allow_layer_overlap=true``. The service result carried ``composed_layers=[20]`` and
``allowed_layer_overlap=True``; the HTTP body did not, because ``CircuitActivationResponse`` had no
such fields and pydantic dropped them at the response model. The admin UI's composed toast
(``useCircuits.ts``) and any MCP caller read exactly those two fields, so a composition reached the
operator as an ordinary success. Every existing test asserted the SERVICE dict, which is why the
suite stayed green.

The service here is a stand-in; what is under test is the route and its response model, driven
through the live app. The service result is the body the 3090 run produced, minus nothing.
"""

from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from millm.main import create_app

# The service result of the hardware run (circ_f309009bca3e composed onto circ_622dacec50ce's L20).
_COMPOSED = {
    "id": "circ_f309009bca3e",
    "name": "E2E compose B (L20+L25, rung 0)",
    "description": None,
    "rung": 0,
    "rung_language": "associated",
    "rung_next_step": "run the attribution pass (sign agreement + magnitude percentile)",
    "validated": False,
    "edge_count": 0,
    "layers": [20, 25],
    "serveable": True,
    "is_active": True,
    "serving_mode": "full",
    "intensity": 1.0,
    "dial_range": [0.5, 2.0],
    "steering": None,
    "per_sae_warnings": [],
    "created_at": "2026-10-17T11:33:44.294443Z",
    "updated_at": "2026-10-17T11:34:09.299976Z",
    "bound_layers": [20, 25],
    "slice_layer": None,
    "applied_per_layer": {20: {1042: 20.0}, 25: {7604: 60.0}},
    "hazards": [],
    "warnings": ["Layers [20] now carry the SUMMED effect of more than one circuit."],
    "acknowledged_unvalidated": True,
    "composed_layers": [20],
    "allowed_layer_overlap": True,
}


def _client(result):
    from millm.api.dependencies import get_circuit_service

    service = MagicMock()
    service.activate = AsyncMock(return_value=dict(result))
    app = create_app()
    app.dependency_overrides[get_circuit_service] = lambda: service
    return TestClient(app, raise_server_exceptions=False), service


def test_a_composed_activation_reports_its_composed_layers_on_the_wire():
    client, service = _client(_COMPOSED)
    response = client.post(
        "/api/circuits/circ_f309009bca3e/activate",
        params={"acknowledge_unvalidated": "true", "allow_layer_overlap": "true"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["composed_layers"] == [20]
    assert body["data"]["allowed_layer_overlap"] is True
    # The override reached the service exactly once, with the operator's flag.
    service.activate.assert_awaited_once()
    assert service.activate.await_args.kwargs["allow_layer_overlap"] is True
    assert service.activate.await_args.kwargs["acknowledge_unvalidated"] is True


def test_an_ordinary_activation_reports_no_composition():
    plain = {k: v for k, v in _COMPOSED.items() if k not in ("composed_layers", "allowed_layer_overlap")}
    client, service = _client(plain)
    response = client.post(
        "/api/circuits/circ_f309009bca3e/activate", params={"acknowledge_unvalidated": "true"}
    )
    body = response.json()
    assert body["data"]["composed_layers"] == []
    assert body["data"]["allowed_layer_overlap"] is False
    assert service.activate.await_args.kwargs["allow_layer_overlap"] is False
