"""Route tests for /api/circuits (Feature 13, tasks 5.3/5.4).

Envelope shapes, query params, the rung<2 activation gate surfaced as a
200+envelope refusal, slice-fallback disclosure in the activation response,
raw (non-enveloped) export, and the error paths.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from millm.api.dependencies import get_circuit_service
from millm.api.exception_handlers import millm_error_handler
from millm.api.routes.management.circuits import router
from millm.core.errors import (
    CircuitNotFoundError,
    MiLLMError,
    SAESetIncompleteError,
    UnvalidatedCircuitError,
)


def make_summary(**overrides) -> dict:
    base = dict(
        id="circ_1",
        name="fear→threat",
        description=None,
        rung=2,
        rung_language="causally validated (edge)",
        rung_next_step="run circuit-level faithfulness at promotion",
        validated=True,
        edge_count=1,
        layers=[10, 13],
        serveable=True,
        is_active=False,
        serving_mode=None,
        intensity=1.0,
        per_sae_warnings=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    base.update(overrides)
    return base


def make_doc() -> dict:
    return {
        "kind": "mistudio.circuit-definition",
        "schema_version": "1",
        "name": "fear→threat",
        "saes": [{"layer": 10, "n_features": 8192}],
        "members": [{"layer": 10, "feature": {"feature_idx": 1, "strength": 10.0}}],
    }


@pytest.fixture
def mock_service():
    svc = MagicMock()
    svc.list_circuits = AsyncMock(return_value=[make_summary()])
    svc.repository = MagicMock()
    svc.repository.count = AsyncMock(return_value=1)
    svc.get_active = AsyncMock(return_value=None)
    # F19: the active surface is a LIST.
    svc.list_active = AsyncMock(return_value=[])
    svc.import_definition = AsyncMock(return_value=MagicMock())
    svc.summarize = MagicMock(return_value=make_summary())
    svc.activate = AsyncMock(
        return_value={
            **make_summary(is_active=True, serving_mode="full"),
            "bound_layers": [10, 13],
            "applied_per_layer": {10: {1: 40.0}, 13: {2: 30.0}},
            "hazards": [],
            "warnings": [],
            "acknowledged_unvalidated": False,
        }
    )
    svc.deactivate = AsyncMock(return_value=make_summary())
    svc.set_intensity = AsyncMock(return_value=make_summary(intensity=1.5))
    svc.delete = AsyncMock(return_value={"circuit_id": "circ_1", "deleted": True})
    svc.export_definition = AsyncMock(return_value=make_doc())
    return svc


@pytest.fixture
def client(mock_service):
    app = FastAPI()
    app.include_router(router)
    app.add_exception_handler(MiLLMError, millm_error_handler)
    app.dependency_overrides[get_circuit_service] = lambda: mock_service
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


class TestList:
    async def test_list_envelope(self, client):
        async with client:
            r = await client.get("/api/circuits")
        body = r.json()
        assert body["success"] is True
        assert body["data"]["circuits"][0]["id"] == "circ_1"
        assert body["data"]["total"] == 1
        assert body["data"]["active_circuit_id"] is None

    async def test_rung_language_is_server_rendered(self, client):
        """The client must never have to derive evidence phrasing."""
        async with client:
            r = await client.get("/api/circuits")
        row = r.json()["data"]["circuits"][0]
        assert row["rung_language"] == "causally validated (edge)"
        assert row["validated"] is True
        assert row["rung_next_step"]

    async def test_filters_passed_through(self, client, mock_service):
        async with client:
            await client.get("/api/circuits?min_rung=2&serveable=true&limit=10&offset=5")
        kwargs = mock_service.list_circuits.await_args.kwargs
        assert kwargs["min_rung"] == 2
        assert kwargs["serveable"] is True
        assert kwargs["limit"] == 10 and kwargs["offset"] == 5

    async def test_invalid_min_rung_rejected(self, client):
        async with client:
            r = await client.get("/api/circuits?min_rung=9")
        assert r.status_code == 422

    async def test_active_is_an_EMPTY_LIST_when_none_serving(self, client):
        """F19 R3-07: `/circuits/active` returns a LIST.

        It previously returned the most recently updated row as a single
        object, so with two circuits serving the second was invisible to every
        operator surface that reads this — including the one endpoint whose job
        is to answer "what is steering".
        """
        async with client:
            r = await client.get("/api/circuits/active")
        assert r.json()["data"] == []

    async def test_single_true_keeps_the_pre_F19_shape(self, client):
        """Compatibility for unmigrated callers. It under-reports when several
        circuits serve, which is why it is opt-in rather than the default."""
        async with client:
            r = await client.get("/api/circuits/active?single=true")
        assert r.json()["data"] is None

    async def test_active_lists_EVERY_serving_circuit(self, client, mock_service):
        from unittest.mock import AsyncMock

        mock_service.list_active = AsyncMock(
            return_value=[
                make_summary(id="circ_1", is_active=True, serving_mode="full"),
                make_summary(id="circ_2", is_active=True, serving_mode="full"),
            ]
        )
        async with client:
            r = await client.get("/api/circuits/active")
        ids = [row["id"] for row in r.json()["data"]]
        assert ids == ["circ_1", "circ_2"], (
            "a serving circuit is invisible to the endpoint that answers "
            "'what is steering'"
        )


class TestImport:
    async def test_import_ok(self, client):
        async with client:
            r = await client.post("/api/circuits/import", json=make_doc())
        body = r.json()
        assert body["success"] is True
        assert body["data"]["name"] == "fear→threat"

    async def test_unknown_kind_refused_in_envelope(self, client):
        doc = make_doc()
        doc["kind"] = "mistudio.cluster-definition"
        async with client:
            r = await client.post("/api/circuits/import", json=doc)
        body = r.json()
        assert r.status_code == 200  # house style
        assert body["success"] is False
        assert body["error"]["code"] == "UNKNOWN_KIND"

    async def test_oversize_payload_refused(self, client):
        doc = make_doc()
        doc["narrative"] = "x" * 2_000_000
        async with client:
            r = await client.post("/api/circuits/import", json=doc)
        body = r.json()
        assert body["success"] is False
        assert body["error"]["code"] == "PAYLOAD_TOO_LARGE"

    async def test_on_conflict_param_forwarded(self, client, mock_service):
        async with client:
            await client.post("/api/circuits/import?on_conflict=fail", json=make_doc())
        assert mock_service.import_definition.await_args.kwargs["on_conflict"] == "fail"

    async def test_bad_on_conflict_rejected(self, client):
        async with client:
            r = await client.post("/api/circuits/import?on_conflict=explode", json=make_doc())
        assert r.status_code == 422


class TestActivation:
    async def test_activate_full_serving(self, client):
        async with client:
            r = await client.post("/api/circuits/circ_1/activate")
        data = r.json()["data"]
        assert data["serving_mode"] == "full"
        assert data["bound_layers"] == [10, 13]
        # int keys become strings for JSON
        assert data["applied_per_layer"]["10"]["1"] == 40.0

    async def test_unvalidated_refusal_is_200_envelope_with_rung(
        self, client, mock_service
    ):
        """A rung<2 circuit is refused in the envelope so the client can show
        the rung and re-send with the acknowledgement."""
        mock_service.activate.side_effect = UnvalidatedCircuitError(
            "Circuit 'x' is associated (rung 0), not causally validated.",
            details={"rung": 0, "rung_language": "associated"},
        )
        async with client:
            r = await client.post("/api/circuits/circ_1/activate")
        body = r.json()
        assert r.status_code == 200
        assert body["success"] is False
        assert body["error"]["code"] == "UNVALIDATED_CIRCUIT"
        assert body["error"]["details"]["rung"] == 0
        assert body["error"]["details"]["rung_language"] == "associated"

    async def test_ack_param_forwarded(self, client, mock_service):
        async with client:
            await client.post(
                "/api/circuits/circ_1/activate?acknowledge_unvalidated=true"
            )
        assert mock_service.activate.await_args.kwargs["acknowledge_unvalidated"] is True

    async def test_slice_fallback_disclosed(self, client, mock_service):
        mock_service.activate.return_value = {
            **make_summary(is_active=True, serving_mode="slice_fallback"),
            "bound_layers": [10],
            "slice_layer": 10,
            "hazards": [],
            "warnings": ["Only L[10] of [10, 13] bound — serving the L10 slice"],
            "acknowledged_unvalidated": False,
        }
        async with client:
            r = await client.post("/api/circuits/circ_1/activate")
        data = r.json()["data"]
        assert data["serving_mode"] == "slice_fallback"
        assert data["slice_layer"] == 10
        assert "PARTIAL" in data["warnings"][0] or "slice" in data["warnings"][0]

    async def test_sae_set_incomplete_maps_to_422(self, client, mock_service):
        mock_service.activate.side_effect = SAESetIncompleteError(
            [{"layer": 13, "sae_id": "sae-13", "reason": "unbound"}]
        )
        async with client:
            r = await client.post("/api/circuits/circ_1/activate")
        assert r.status_code == 422
        body = r.json()
        assert body["error"]["code"] == "SAE_SET_INCOMPLETE"
        assert body["error"]["details"]["offenders"][0]["layer"] == 13

    async def test_unknown_circuit_404(self, client, mock_service):
        mock_service.activate.side_effect = CircuitNotFoundError("nope")
        async with client:
            r = await client.post("/api/circuits/ghost/activate")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "CIRCUIT_NOT_FOUND"


class TestIntensityAndLifecycle:
    async def test_set_active_intensity(self, client, mock_service):
        # F20 R2-03: the route reads `list_active()` so it can REFUSE when
        # several circuits serve — a single-element list is the dialable case.
        mock_service.list_active.return_value = [make_summary(is_active=True)]
        async with client:
            r = await client.put(
                "/api/circuits/active/intensity", json={"intensity": 1.5}
            )
        assert r.json()["data"]["intensity"] == 1.5

    async def test_dialling_is_REFUSED_while_several_circuits_serve(
        self, client, mock_service
    ):
        """F20 R2-03. This read `get_active()` — the most recently updated row —
        so with two circuits serving it silently dialled whichever was touched
        LAST and reported a λ change for a circuit the caller never named.

        F19 R3-06 applied this rule to the per-request dial and the rung
        header; the management dial was left behind, and the MCP tool
        description then PROMISED a refusal that did not exist."""
        mock_service.list_active.return_value = [
            make_summary(id="circ_1", name="one", is_active=True),
            make_summary(id="circ_2", name="two", is_active=True),
        ]
        async with client:
            r = await client.put(
                "/api/circuits/active/intensity", json={"intensity": 1.5}
            )
        body = r.json()
        assert body["success"] is False
        assert body["error"]["code"] == "AMBIGUOUS_ACTIVE_CIRCUIT"
        named = {c["id"] for c in body["error"]["details"]["active_circuits"]}
        assert named == {"circ_1", "circ_2"}, (
            "the refusal must name WHICH circuits are serving, or the operator "
            "cannot act on it"
        )

    async def test_no_active_circuit_refused_in_envelope(self, client):
        async with client:
            r = await client.put(
                "/api/circuits/active/intensity", json={"intensity": 1.5}
            )
        body = r.json()
        assert r.status_code == 200
        assert body["success"] is False
        assert body["error"]["code"] == "NO_ACTIVE_CIRCUIT"

    async def test_intensity_out_of_range_rejected(self, client):
        async with client:
            r = await client.put(
                "/api/circuits/active/intensity", json={"intensity": 9.0}
            )
        assert r.status_code == 422

    async def test_deactivate(self, client):
        async with client:
            r = await client.post("/api/circuits/circ_1/deactivate")
        assert r.json()["success"] is True

    async def test_delete(self, client):
        async with client:
            r = await client.delete("/api/circuits/circ_1")
        assert r.json()["data"]["deleted"] is True


class TestExport:
    async def test_export_is_raw_document_no_envelope(self, client):
        """The response IS the portable artifact — no {success,data} wrapper."""
        async with client:
            r = await client.get("/api/circuits/circ_1/export")
        body = r.json()
        assert body["kind"] == "mistudio.circuit-definition"
        assert "success" not in body and "data" not in body

    async def test_export_preserves_unknown_fields(self, client, mock_service):
        doc = make_doc()
        doc["future_field"] = {"kept": True}
        mock_service.export_definition.return_value = doc
        async with client:
            r = await client.get("/api/circuits/circ_1/export")
        assert r.json()["future_field"] == {"kept": True}


class TestR1Fixes:
    """Review round 1 regressions on the route surface."""

    async def test_nesting_bomb_refused(self, client):
        """A deeply-nested payload is cheap in BYTES (3000 levels ≈ 21 KB) so
        the size cap cannot see it — the depth gate must."""
        bomb: dict = {}
        node = bomb
        for _ in range(3000):
            node["n"] = {}
            node = node["n"]
        doc = make_doc()
        doc["nested"] = bomb
        async with client:
            r = await client.post("/api/circuits/import", json=doc)
        body = r.json()
        assert body["success"] is False
        assert body["error"]["code"] == "VALIDATION_ERROR"
        assert "nests" in body["error"]["message"]

    async def test_normal_document_passes_the_depth_gate(self, client):
        async with client:
            r = await client.post("/api/circuits/import", json=make_doc())
        assert r.json()["success"] is True

    async def test_intensity_response_carries_reapplied(self, client, mock_service):
        """`reapplied` was silently dropped by the route filter — a slice
        circuit reported a new intensity the steering never received."""
        mock_service.list_active.return_value = [make_summary(is_active=True)]
        mock_service.set_intensity.return_value = {
            **make_summary(is_active=True, serving_mode="slice_fallback", intensity=0.4),
            "reapplied": False,
            "warnings": ["...recorded but not applied..."],
        }
        async with client:
            r = await client.put("/api/circuits/active/intensity", json={"intensity": 0.4})
        data = r.json()["data"]
        assert data["reapplied"] is False
        assert data["warnings"]

    async def test_deactivate_response_carries_cleared_steering(self, client, mock_service):
        mock_service.deactivate.return_value = {
            **make_summary(), "cleared_steering": True
        }
        async with client:
            r = await client.post("/api/circuits/circ_1/deactivate")
        assert r.json()["data"]["cleared_steering"] is True


class TestF19ContentionRoutes:
    """Feature 19 task 4.2/4.3 — the override parameter, the refusal envelope,
    and the claims view."""

    async def test_allow_layer_overlap_reaches_the_service(
        self, client, mock_service
    ):
        """A parameter the route accepts but never forwards is a silent
        no-op: the operator believes they authorised composition and the
        service refuses anyway (or worse, composes without the record)."""
        async with client:
            await client.post(
                "/api/circuits/circ_1/activate?allow_layer_overlap=true"
            )
        _args, kwargs = mock_service.activate.call_args
        assert kwargs.get("allow_layer_overlap") is True

    async def test_it_defaults_to_FALSE(self, client, mock_service):
        async with client:
            await client.post("/api/circuits/circ_1/activate")
        _args, kwargs = mock_service.activate.call_args
        assert kwargs.get("allow_layer_overlap") is False, (
            "composition must never be the default — it is refused by default "
            "precisely because it is measured to destroy generation"
        )

    async def test_a_contention_refusal_uses_the_ENVELOPE(
        self, client, mock_service
    ):
        """House style: 200 + success:false. Nothing is missing; the operation
        does not apply, and the client needs the details to decide."""
        from millm.core.errors import CircuitLayerContentionError

        mock_service.activate = AsyncMock(
            side_effect=CircuitLayerContentionError(
                contended_layers=[13],
                incumbent_id="circ_abc",
                incumbent_name="fear→threat",
                requested_id="circ_1",
                requested_name="hedging",
            )
        )
        async with client:
            r = await client.post("/api/circuits/circ_1/activate")

        assert r.status_code == 200
        body = r.json()
        assert body["success"] is False
        assert body["error"]["code"] == "CIRCUIT_LAYER_CONTENTION"

        details = body["error"]["details"]
        assert details["incumbent"]["name"] == "fear→threat", (
            "the refusal did not name the incumbent, so the operator cannot "
            "tell what to deactivate"
        )
        assert details["override_param"] == "allow_layer_overlap"
        assert details["rung_header_suppressed_if_overridden"] is True
        # BR-011: the measurement travels WITH the refusal.
        assert "degenerate" in details["measured_hazard"]["two_layers_at_strength_5"]
        assert "indicative, not exhaustive" in details["measured_hazard"]["note"]

    async def test_a_COLLISION_refusal_offers_no_override_route(
        self, client, mock_service
    ):
        from millm.core.errors import CircuitLayerContentionError

        mock_service.activate = AsyncMock(
            side_effect=CircuitLayerContentionError(
                contended_layers=[13],
                colliding_keys=[(13, 42, "circ_abc")],
                incumbent_id="circ_abc",
                incumbent_name="fear→threat",
            )
        )
        async with client:
            r = await client.post("/api/circuits/circ_1/activate")

        details = r.json()["error"]["details"]
        assert details["overridable"] is False
        assert "override_param" not in details, (
            "naming an override parameter on a collision invites trying it, "
            "and if it worked one author's strength would silently win"
        )
        assert details["colliding_keys"][0]["feature_idx"] == 42


class TestF19R2ClaimReleaseEndpoint:
    """F19 R2-10. Every claim-leak path in this feature had exactly ONE remedy:
    a full process restart, which drops every loaded model and every attached
    SAE — a multi-minute GPU outage to clear one stale row.

    The endpoint is scoped to a single circuit ON PURPOSE. A "release
    everything" button is a foot-gun in a feature whose whole point is that
    several circuits serve at once: it would silently strip live circuits of
    the protection they are relying on.
    """

    async def test_it_releases_only_the_named_circuits_claims(
        self, client, mock_service
    ):
        from unittest.mock import AsyncMock, MagicMock

        session = MagicMock()
        session.commit = AsyncMock()
        mock_service.repository.session = session
        mock_service.repository.get = AsyncMock(
            return_value=MagicMock(is_active=False)
        )

        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "millm.services.circuit_claim_registry.CircuitClaimRegistry"
        ) as registry_cls:
            registry_cls.return_value.release = AsyncMock(return_value=[10, 13])
            async with client:
                r = await client.post(
                    "/api/circuits/claims/release?circuit_id=circ_1"
                )

        body = r.json()
        assert body["success"] is True
        assert body["data"]["released_layers"] == [10, 13]
        registry_cls.return_value.release.assert_awaited_once_with("circ_1")

    async def test_releasing_an_ACTIVE_circuits_claims_warns(
        self, client, mock_service
    ):
        """Releasing a claim does not stop steering. If the circuit still reads
        active it is now steering layers it does not hold, and another circuit
        can take them — say so rather than refusing, because an operator doing
        this is usually recovering from exactly that divergence."""
        from unittest.mock import AsyncMock, MagicMock

        session = MagicMock()
        session.commit = AsyncMock()
        mock_service.repository.session = session
        mock_service.repository.get = AsyncMock(
            return_value=MagicMock(is_active=True)
        )

        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "millm.services.circuit_claim_registry.CircuitClaimRegistry"
        ) as registry_cls:
            registry_cls.return_value.release = AsyncMock(return_value=[10])
            async with client:
                r = await client.post(
                    "/api/circuits/claims/release?circuit_id=circ_1"
                )

        warnings = r.json()["data"]["warnings"]
        assert any("still reads ACTIVE" in w for w in warnings)
        assert any("Deactivate it" in w for w in warnings)

    async def test_it_requires_a_circuit_id(self, client):
        """No 'release everything' route: the parameter is required."""
        async with client:
            r = await client.post("/api/circuits/claims/release")
        assert r.status_code == 422

    async def test_a_successful_release_CLEARS_the_degraded_flag(
        self, client, mock_service
    ):
        """F19 R3-01. `note_claims_degraded` had no counterpart, so the flag
        was a LATCH, not a status: once set it reported DEGRADED for the life
        of the process — including after the operator had already fixed the
        problem with this very endpoint, which is the documented remedy.

        The two mechanisms R2 added in the same round contradicted each other:
        the remedy could not clear the signal reporting the condition it
        remedies, so the operator restarts anyway — the multi-minute GPU outage
        R2-10 exists to avoid.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        from millm.api.routes.system import health as health_mod

        health_mod.note_claims_degraded("reconcile failed: boom")
        assert health_mod.CIRCUIT_CLAIMS_DEGRADED["degraded"] is True

        session = MagicMock()
        session.commit = AsyncMock()
        mock_service.repository.session = session
        mock_service.repository.get = AsyncMock(
            return_value=MagicMock(is_active=False)
        )

        try:
            with patch(
                "millm.services.circuit_claim_registry.CircuitClaimRegistry"
            ) as registry_cls:
                registry_cls.return_value.release = AsyncMock(return_value=[10])
                async with client:
                    await client.post(
                        "/api/circuits/claims/release?circuit_id=circ_1"
                    )

            assert health_mod.CIRCUIT_CLAIMS_DEGRADED["degraded"] is False, (
                "the documented remedy ran successfully and /health still "
                "reports DEGRADED — a health signal that cannot recover"
            )
        finally:
            health_mod.CIRCUIT_CLAIMS_DEGRADED["degraded"] = False
            health_mod.CIRCUIT_CLAIMS_DEGRADED["reason"] = None

    async def test_an_UNKNOWN_circuit_id_says_so(self, client, mock_service):
        """F19 R3-02. A typo'd id returned success with `released_layers: []` —
        indistinguishable from "the claim was already gone". An operator
        recovering from a stuck claim, working from a name they may have
        mistyped, could not tell whether they had fixed anything."""
        from unittest.mock import AsyncMock, MagicMock, patch

        session = MagicMock()
        session.commit = AsyncMock()
        mock_service.repository.session = session
        mock_service.repository.get = AsyncMock(return_value=None)

        with patch(
            "millm.services.circuit_claim_registry.CircuitClaimRegistry"
        ) as registry_cls:
            registry_cls.return_value.release = AsyncMock(return_value=[])
            async with client:
                r = await client.post(
                    "/api/circuits/claims/release?circuit_id=typo"
                )

        warnings = r.json()["data"]["warnings"]
        assert any("No circuit 'typo' exists" in w for w in warnings)

    async def test_a_known_circuit_with_no_claims_says_something_DIFFERENT(
        self, client, mock_service
    ):
        """The two 'nothing happened' cases have different remedies, so they
        must not read the same."""
        from unittest.mock import AsyncMock, MagicMock, patch

        session = MagicMock()
        session.commit = AsyncMock()
        mock_service.repository.session = session
        mock_service.repository.get = AsyncMock(
            return_value=MagicMock(is_active=False)
        )

        with patch(
            "millm.services.circuit_claim_registry.CircuitClaimRegistry"
        ) as registry_cls:
            registry_cls.return_value.release = AsyncMock(return_value=[])
            async with client:
                r = await client.post(
                    "/api/circuits/claims/release?circuit_id=circ_1"
                )

        warnings = " ".join(r.json()["data"]["warnings"])
        assert "held no live claims" in warnings
        assert "different circuit" in warnings


class TestF19R3PerRowSteeringVerdict:
    """F19 R3-19. `steering` is a PER-ROW question — "is THIS circuit
    influencing generation?" — and it was answered with the SINGULAR
    `_steering_circuit()` predicate.

    R3-06 made that predicate return None when several circuits serve (correct
    for the dial and the rung header, since no single circuit describes the
    response). Reusing it here made every row report `steering: false` in
    exactly the state the feature exists to support: two circuits both
    genuinely steering, and the endpoint saying neither is.
    """

    async def test_both_serving_circuits_report_steering_true(
        self, client, mock_service
    ):
        from unittest.mock import AsyncMock

        from millm.services.sae_service import AttachedSAEState

        state = AttachedSAEState()
        state.reset_for_tests()
        state.apply_owner("circuit:circ_1", {})
        state.apply_owner("circuit:circ_2", {})
        # `owner_keys` is empty for a no-contribution owner, so give each a
        # layer through the entries map the registry actually reads.
        state._owners["circuit:circ_1"] = {("s10", 10): {1: 40.0}}
        state._owners["circuit:circ_2"] = {("s13", 13): {2: 30.0}}

        mock_service.list_active = AsyncMock(
            return_value=[
                make_summary(id="circ_1", is_active=True, serving_mode="full"),
                make_summary(id="circ_2", is_active=True, serving_mode="full"),
            ]
        )
        try:
            async with client:
                r = await client.get("/api/circuits/active")
            rows = {row["id"]: row["steering"] for row in r.json()["data"]}
            assert rows == {"circ_1": True, "circ_2": True}, (
                "circuits that ARE steering report steering:false — the "
                "endpoint denies the state the feature exists to support"
            )
        finally:
            state.reset_for_tests()

    async def test_a_circuit_that_owns_nothing_reports_false(
        self, client, mock_service
    ):
        """Specificity: an active row is not the same as steering. A
        slice-fallback or unattached circuit owns no keys."""
        from unittest.mock import AsyncMock

        from millm.services.sae_service import AttachedSAEState

        AttachedSAEState().reset_for_tests()
        mock_service.list_active = AsyncMock(
            return_value=[
                make_summary(id="circ_1", is_active=True, serving_mode="full")
            ]
        )
        async with client:
            r = await client.get("/api/circuits/active")
        assert r.json()["data"][0]["steering"] is False
