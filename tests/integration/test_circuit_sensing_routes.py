"""Feature 15 Task 4.2: /api/circuit-sensing route tests.

Envelope shapes, filters, the persistent-intent-vs-armed distinction, and the
toggle's refusal semantics. The unsensable-edge surfacing is the one that
matters most: if the API can return "no events" without also saying which
edges were never watched, the UI cannot avoid presenting absence of
observation as evidence of absence.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from millm.main import create_app


@pytest.fixture
def client():
    import millm.api.dependencies as deps

    deps._circuit_sensing_service = None  # fresh singleton per test
    app = create_app()
    with TestClient(app) as tc:
        yield tc
    deps._circuit_sensing_service = None


def session_ctx(rows=()):
    session = MagicMock()

    async def _execute(*a, **kw):
        result = MagicMock()
        result.__iter__ = lambda self: iter(rows)
        return result

    async def _commit():
        return None

    session.execute = _execute
    session.commit = _commit

    class _Ctx:
        async def __aenter__(self_inner):
            return session

        async def __aexit__(self_inner, *a):
            return False

    return _Ctx()


def make_event(**overrides):
    from millm.db.models.circuit_edge_sensing_event import CircuitEdgeSensingEvent

    base = dict(
        id=1,
        circuit_id="circ_1",
        request_id="req-1",
        phase="decode",
        edge_key="1@10->2@13",
        up_layer=10,
        up_feature_idx=1,
        up_pos=5,
        up_act=1.5,
        down_layer=13,
        down_feature_idx=2,
        down_pos=7,
        down_act=0.9,
        token_lag=2,
        edge_rung=2,
        edge_rung_language="causally validated (edge)",
        edge_type="computed",
        summary="edge fired",
        truncated=False,
    )
    base.update(overrides)
    return CircuitEdgeSensingEvent(**base)


class TestStatusRoute:
    def test_status_envelope_when_unarmed(self, client):
        with patch("millm.db.base.async_session_factory", return_value=session_ctx()):
            response = client.get("/api/circuit-sensing/status")
        body = response.json()
        assert body["success"] is True
        assert body["data"]["armed"] is False
        assert body["data"]["max_token_lag"] > 0
        assert body["data"]["enabled_circuits"] == []

    def test_status_reports_enabled_but_unarmed_circuits(self, client):
        """Persistent operator INTENT must be visible distinctly from runtime
        armed: a circuit can be enabled but unarmed because it is not active."""
        row = MagicMock(id="circ_1", is_active=False)
        row.name = "fear→threat"  # name= is a reserved MagicMock kwarg
        with patch(
            "millm.db.base.async_session_factory", return_value=session_ctx((row,))
        ):
            response = client.get("/api/circuit-sensing/status")
        data = response.json()["data"]
        assert data["armed"] is False
        assert data["enabled_circuits"] == [
            {"id": "circ_1", "name": "fear→threat", "is_active": False}
        ]

    def test_status_surfaces_unsensable_edges(self, client):
        """The critical honesty surface: without this a user reads 'no events'
        as 'the edge never fired'."""
        import millm.api.dependencies as deps
        from millm.services.circuit_sensing_service import (
            CircuitSensingService,
            UnsensableEdge,
        )

        svc = CircuitSensingService()
        svc._unsensable = [
            UnsensableEdge("1@10->2@13", "layer_not_attached", "no SAE on layer 13")
        ]
        deps._circuit_sensing_service = svc

        with patch("millm.db.base.async_session_factory", return_value=session_ctx()):
            response = client.get("/api/circuit-sensing/status")
        rows = response.json()["data"]["unsensable_edges"]
        assert rows == [
            {
                "edge_key": "1@10->2@13",
                "reason": "layer_not_attached",
                "detail": "no SAE on layer 13",
            }
        ]


class TestEventRoutes:
    def test_list_events_envelope_nests_both_endpoints(self, client):
        with patch("millm.db.base.async_session_factory", return_value=session_ctx()):
            with patch(
                "millm.db.repositories.circuit_edge_sensing_repository"
                ".CircuitEdgeSensingRepository"
            ) as Repo:
                repo = Repo.return_value

                async def _list(**kw):
                    return [make_event()]

                async def _count(**kw):
                    return 1

                async def _prune_aged(*a, **kw):
                    return 0

                repo.list_events = _list
                repo.count = _count
                repo.prune_aged = _prune_aged
                response = client.get("/api/circuit-sensing/events")

        data = response.json()["data"]
        assert data["total"] == 1
        event = data["events"][0]
        assert event["up"] == {"layer": 10, "feature_idx": 1, "pos": 5, "act": 1.5}
        assert event["down"] == {"layer": 13, "feature_idx": 2, "pos": 7, "act": 0.9}
        assert event["token_lag"] == 2
        assert event["edge_rung_language"] == "causally validated (edge)"

    def test_list_events_accepts_the_edge_key_filter(self, client):
        captured = {}

        with patch("millm.db.base.async_session_factory", return_value=session_ctx()):
            with patch(
                "millm.db.repositories.circuit_edge_sensing_repository"
                ".CircuitEdgeSensingRepository"
            ) as Repo:
                repo = Repo.return_value

                async def _list(**kw):
                    captured.update(kw)
                    return []

                async def _count(**kw):
                    return 0

                async def _prune_aged(*a, **kw):
                    return 0

                repo.list_events = _list
                repo.count = _count
                repo.prune_aged = _prune_aged
                client.get(
                    "/api/circuit-sensing/events"
                    "?circuit_id=circ_1&edge_key=1@10-%3E2@13&limit=10"
                )

        assert captured["circuit_id"] == "circ_1"
        assert captured["edge_key"] == "1@10->2@13"
        assert captured["limit"] == 10

    def test_a_missing_event_is_a_404_with_the_registered_code(self, client):
        with patch("millm.db.base.async_session_factory", return_value=session_ctx()):
            with patch(
                "millm.db.repositories.circuit_edge_sensing_repository"
                ".CircuitEdgeSensingRepository"
            ) as Repo:

                async def _get(event_id):
                    return None

                Repo.return_value.get = _get
                response = client.get("/api/circuit-sensing/events/999")

        assert response.status_code == 404
        body = response.json()
        assert body["error"]["code"] == "CIRCUIT_SENSING_EVENT_NOT_FOUND"

    def test_event_detail_includes_context(self, client):
        with patch("millm.db.base.async_session_factory", return_value=session_ctx()):
            with patch(
                "millm.db.repositories.circuit_edge_sensing_repository"
                ".CircuitEdgeSensingRepository"
            ) as Repo:

                async def _get(event_id):
                    return make_event(
                        context_text="the cat sat",
                        context_parts={
                            "before": "the ",
                            "span": "cat",
                            "after": " sat",
                        },
                    )

                Repo.return_value.get = _get
                response = client.get("/api/circuit-sensing/events/1")

        data = response.json()["data"]
        assert data["context_text"] == "the cat sat"
        assert data["context_parts"]["span"] == "cat"

    def test_clear_events_returns_the_deleted_count(self, client):
        with patch("millm.db.base.async_session_factory", return_value=session_ctx()):
            with patch(
                "millm.db.repositories.circuit_edge_sensing_repository"
                ".CircuitEdgeSensingRepository"
            ) as Repo:

                async def _clear(circuit_id=None):
                    return 7

                Repo.return_value.clear = _clear
                response = client.delete(
                    "/api/circuit-sensing/events?circuit_id=circ_1"
                )

        assert response.json()["data"] == {"deleted": 7}


class TestToggleRoutes:
    def _circuit_ctx(self, circuit):
        session = MagicMock()

        async def _commit():
            return None

        session.commit = _commit

        class _Ctx:
            async def __aenter__(self_inner):
                return session

            async def __aexit__(self_inner, *a):
                return False

        return _Ctx()

    def test_enabling_an_unknown_circuit_is_a_404(self, client):
        with patch(
            "millm.db.base.async_session_factory", return_value=session_ctx()
        ):
            with patch(
                "millm.db.repositories.circuit_repository.CircuitRepository"
            ) as Repo:

                async def _get(circuit_id):
                    return None

                Repo.return_value.get = _get
                response = client.post("/api/circuit-sensing/nope/enable")

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "CIRCUIT_NOT_FOUND"

    def test_enabling_an_inactive_circuit_persists_intent_without_arming(
        self, client
    ):
        circuit = MagicMock(id="circ_1", is_active=False, circuit_meta={})
        circuit.name = "fear→threat"

        with patch(
            "millm.db.base.async_session_factory",
            return_value=self._circuit_ctx(circuit),
        ):
            with patch(
                "millm.db.repositories.circuit_repository.CircuitRepository"
            ) as Repo:

                async def _get(circuit_id):
                    return circuit

                Repo.return_value.get = _get
                response = client.post("/api/circuit-sensing/circ_1/enable")

        data = response.json()["data"]
        assert data["enabled"] is True
        assert data["armed"] is False
        assert "activated" in data["message"]
        assert circuit.sensing_enabled is True, "intent must persist"

    def test_disabling_reports_disarmed(self, client):
        circuit = MagicMock(id="circ_1", is_active=True, circuit_meta={})
        circuit.name = "fear→threat"

        with patch(
            "millm.db.base.async_session_factory",
            return_value=self._circuit_ctx(circuit),
        ):
            with patch(
                "millm.db.repositories.circuit_repository.CircuitRepository"
            ) as Repo:

                async def _get(circuit_id):
                    return circuit

                Repo.return_value.get = _get
                response = client.post("/api/circuit-sensing/circ_1/disable")

        data = response.json()["data"]
        assert data["enabled"] is False
        assert data["armed"] is False
        assert circuit.sensing_enabled is False

    def test_an_unreadable_definition_refuses_with_a_reason(self, client):
        """The column stays enabled — persistent intent survives — but the
        caller is told exactly why arming refused, not given a silent no-arm."""
        circuit = MagicMock(
            id="circ_1", is_active=True, circuit_meta={"garbage": True}
        )
        circuit.name = "fear→threat"

        with patch(
            "millm.db.base.async_session_factory",
            return_value=self._circuit_ctx(circuit),
        ):
            with patch(
                "millm.db.repositories.circuit_repository.CircuitRepository"
            ) as Repo:

                async def _get(circuit_id):
                    return circuit

                Repo.return_value.get = _get
                response = client.post("/api/circuit-sensing/circ_1/enable")

        body = response.json()
        assert body["success"] is False
        assert "unreadable" in body["error"]["message"].lower()
        assert circuit.sensing_enabled is True, "intent survives a refused arm"


class TestR3TheNewStatusFieldsReachTheHTTPRESPONSE:
    """F17 R3-18. Rounds 1-3 added five status fields — `truncated_layers`,
    `requests_sensed`, `requests_truncated`, `ws_throttled`, and the reasons
    behind `paused_reason` — and NO route test asserts any of them.

    The unit tests prove the service computes them and the schema tests prove
    the model carries them. Neither proves the HTTP response does. That gap is
    the F16 R1 failure mode exactly: a field the service computes, the response
    model does not declare, and Pydantic silently drops on the way out — moved
    up a layer to the route.

    Each of these fields exists so an operator can tell 'quiet' from 'broken'.
    A field that never reaches the wire cannot do that."""

    def _status(self, client):
        with patch("millm.db.base.async_session_factory", return_value=session_ctx()):
            r = client.get("/api/circuit-sensing/status")
        assert r.status_code == 200
        return r.json()["data"]

    def test_the_status_route_carries_every_honesty_field(self, client):
        data = self._status(client)
        for field in (
            "truncated_layers", "requests_sensed", "requests_truncated",
            "ws_throttled", "ws_dropped", "paused_reason", "events_recorded",
        ):
            assert field in data, (
                f"{field!r} never reached the HTTP response — the operator "
                "signal it carries is unreachable"
            )

    def test_the_counters_are_typed_as_the_contract_promises(self, client):
        """`truncated_layers` is a LIST of layers, not a boolean — the whole
        point of BR-006. A client reading it as truthy would report every
        request as truncated."""
        data = self._status(client)
        assert isinstance(data["truncated_layers"], list)
        assert isinstance(data["requests_sensed"], int)
        assert isinstance(data["requests_truncated"], int)
        assert isinstance(data["ws_throttled"], int)

    def test_an_unarmed_circuit_reports_zeroes_not_nulls(self, client):
        """Nulls would force every consumer into defensive checks, and an
        agent reading null as 'unknown' would hedge a claim it can make."""
        data = self._status(client)
        assert data["requests_sensed"] == 0
        assert data["requests_truncated"] == 0
        assert data["truncated_layers"] == []
