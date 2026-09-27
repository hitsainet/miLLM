"""The arm / parity / hub routes, exercised through the LIVE app.

⚠ These routes did not exist until 2026-09-27. The arming service, the identity gate, the parity
engine and the k-sparse slice were all built, unit-tested and unreachable — see
`test_probe_route_surface.py` for how a passing subset assertion hid that.

What is asserted here is not "the route returns 200". It is that a REFUSAL reaches the caller
saying which gate refused, because the entire value of the four gates is in what they refuse:

* a probe fitted on another model,
* a probe below rung 2 that nobody has acknowledged,
* a build that does not reproduce miStudio's recorded scores,
* and the case where nothing is loaded at all.

An arm route that swallowed any of these into a 500 would leave an operator unable to tell "wrong
model" from "server broken".
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from millm.core.errors import (
    ProbeLimitError,
    ProbeModelMismatchError,
    ProbeParityFailedError,
    UnvalidatedProbeError,
)
from millm.main import create_app


def _probe_row(**over):
    row = MagicMock()
    row.id = over.get("id", "pr_1")
    row.name = over.get("name", "high-stakes")
    row.hf_id = "LiquidAI/LFM2.5-1.2B-Instruct"
    row.layer = over.get("layer", 11)
    row.rule = "mean"
    row.scope = "all"
    row.basis = over.get("basis", "residual")
    row.streamable = True
    row.threshold = 1.07
    row.target_fpr = 0.01
    row.rung = over.get("rung", 2)
    row.armed = False
    row.paused_reason = None
    row.parity = None
    row.created_at = "2026-09-27T00:00:00Z"
    row.definition = over.get("definition", {"test_vectors": {"tolerance": 1e-3, "vectors": []}})
    return row


def _client(*, probe=None, arm_side_effect=None, identity_side_effect=None):
    from millm.api.dependencies import (
        get_probe_arming_service,
        get_probe_hub_service,
        get_probe_repository,
        get_probe_service,
    )

    repo = MagicMock()
    repo.get = AsyncMock(return_value=probe)
    repo.update = AsyncMock()

    arming = MagicMock()
    armed = MagicMock()
    armed.basis = "residual"
    arming.arm = AsyncMock(side_effect=arm_side_effect, return_value=armed)
    arming.disarm = AsyncMock(return_value=True)

    hub = MagicMock()
    hub.search = AsyncMock(return_value=[{"repo_id": "mistudio/probes"}])
    hub.list_definitions = AsyncMock(return_value=[{"filename": "a.probe.json"}])
    hub.fetch_definition = AsyncMock(return_value=(MagicMock(), {"kind": "x"}, {"repo": "r"}))

    service = MagicMock()
    service.import_definition = AsyncMock(return_value=probe or _probe_row())

    app = create_app()
    app.dependency_overrides[get_probe_repository] = lambda: repo
    app.dependency_overrides[get_probe_arming_service] = lambda: arming
    app.dependency_overrides[get_probe_hub_service] = lambda: hub
    app.dependency_overrides[get_probe_service] = lambda: service
    return TestClient(app, raise_server_exceptions=False), repo, arming, hub, service


@pytest.fixture
def patched_bridge(monkeypatch):
    """A loaded model, without one.

    ⚠ Patched on the ROUTE MODULE's name, not the bridge's. `probes.py` does
    `from ...probe_arm_bridge import loaded_identity`, which binds the function into that module at
    import; patching the bridge's attribute would leave the route calling the original — a
    monkeypatch that silently does nothing, which this estate has shipped twice.
    """
    import millm.api.routes.management.probes as routes
    from millm.services.probe_identity import LoadedIdentity

    identity = LoadedIdentity(
        hf_id="LiquidAI/LFM2.5-1.2B-Instruct", d_model=2048, n_layers=16, chat_template="x"
    )
    model, tokenizer = MagicMock(), MagicMock()
    monkeypatch.setattr(
        routes, "loaded_identity", AsyncMock(return_value=(identity, model, tokenizer))
    )
    monkeypatch.setattr(routes, "build_parity_forward", lambda *_a, **_k: (lambda *a, **k: None))
    monkeypatch.setattr(routes, "build_probe_encoder", AsyncMock(return_value=None))
    return identity


class TestArming:
    def test_a_missing_probe_is_404_not_500(self, patched_bridge):
        client, *_ = _client(probe=None)
        response = client.post("/api/probes/pr_missing/arm", json={})
        assert response.status_code == 404

    def test_arming_passes_the_operator_acknowledgement_through(self, patched_bridge):
        probe = _probe_row(rung=1)
        client, _repo, arming, *_ = _client(probe=probe)
        response = client.post(
            "/api/probes/pr_1/arm",
            json={"acknowledge_below_rung2": True, "reason": "triage only"},
        )
        assert response.status_code == 200
        # ⚠ The PAYLOAD, not just that it was called. A call sending False would satisfy a
        # was-called assertion and leave the acknowledgement impossible to give.
        kwargs = arming.arm.await_args.kwargs
        assert kwargs["acknowledge_below_rung2"] is True
        assert kwargs["reason"] == "triage only"
        assert kwargs["loaded"] is patched_bridge

    def test_the_default_is_NOT_acknowledged(self, patched_bridge):
        """An omitted field must not read as consent."""
        client, _repo, arming, *_ = _client(probe=_probe_row(rung=1))
        client.post("/api/probes/pr_1/arm", json={})
        assert arming.arm.await_args.kwargs["acknowledge_below_rung2"] is False

    @pytest.mark.parametrize(
        ("error", "code"),
        [
            (ProbeModelMismatchError("wrong model", details={"mismatches": ["hf_id"]}),
             "PROBE_MODEL_MISMATCH"),
            (UnvalidatedProbeError("rung 1", details={"rung": 1}), "UNVALIDATED_PROBE"),
            (ProbeParityFailedError("does not reproduce", details={"max_abs_diff": 3.2}),
             "PROBE_PARITY_FAILED"),
            (ProbeLimitError("8 already armed", details={"max_armed": 8}), "PROBE_LIMIT"),
        ],
    )
    def test_every_gate_refuses_by_NAME(self, patched_bridge, error, code):
        """Each refusal names its own gate. A generic 500 would make them indistinguishable.

        ⚠ Asserted on `error.code`, not `code in str(body)`. A substring match over the whole
        response passes when the code happens to appear in a traceback or a message — the
        "satisfied by the wrong occurrence" trap this estate has now hit in six arcs.
        """
        client, *_ = _client(probe=_probe_row(), arm_side_effect=error)
        response = client.post("/api/probes/pr_1/arm", json={})
        body = response.json()
        assert body["success"] is False
        assert body["error"]["code"] == code
        assert response.status_code == error.status_code

    def test_nothing_loaded_is_a_refusal_not_a_crash(self, monkeypatch):
        import millm.api.routes.management.probes as routes
        from millm.core.errors import ProbeNoModelLoadedError

        monkeypatch.setattr(
            routes,
            "loaded_identity",
            AsyncMock(side_effect=ProbeNoModelLoadedError("no model is loaded")),
        )
        client, *_ = _client(probe=_probe_row())
        response = client.post("/api/probes/pr_1/arm", json={})
        assert response.json()["error"]["code"] == "PROBE_NO_MODEL_LOADED"


class TestOnDemandParity:
    def test_parity_can_be_checked_without_arming(self, patched_bridge, monkeypatch):
        import millm.api.routes.management.probes as routes

        report = MagicMock()
        report.as_details.return_value = {"passed": True, "tolerance": 1e-3}
        engine = MagicMock()
        engine.run.return_value = report
        monkeypatch.setattr(routes, "armed_probe_from_row", lambda *_a, **_k: MagicMock())
        monkeypatch.setattr(
            "millm.services.probe_parity.ProbeParityEngine", lambda *_a, **_k: engine
        )
        client, _repo, arming, *_ = _client(probe=_probe_row())
        response = client.post("/api/probes/pr_1/parity", json={})
        assert response.status_code == 200
        assert response.json()["data"]["passed"] is True
        # ⚠ Checking parity must NOT arm. A route that armed as a side effect of a check would put
        # a probe on live traffic that nobody asked to arm.
        arming.arm.assert_not_awaited()
        # And it must not flip the ROW either. `arm.assert_not_awaited()` alone survived a mutation
        # that passed `armed=True` to `repository.update`: the row would then be listed as armed and
        # counted against PROBE_MAX_ARMED while no hook was ever installed — a probe reporting
        # nothing, with no `paused_reason` to say why.
        assert "armed" not in _repo.update.await_args.kwargs

    def test_the_report_is_stored_so_the_operator_can_see_it_later(
        self, patched_bridge, monkeypatch
    ):
        import millm.api.routes.management.probes as routes

        report = MagicMock()
        report.as_details.return_value = {"passed": False, "max_abs_diff": 3.2}
        monkeypatch.setattr(routes, "armed_probe_from_row", lambda *_a, **_k: MagicMock())
        monkeypatch.setattr(
            "millm.services.probe_parity.ProbeParityEngine",
            lambda *_a, **_k: MagicMock(run=MagicMock(return_value=report)),
        )
        probe = _probe_row()
        client, repo, *_ = _client(probe=probe)
        client.post("/api/probes/pr_1/parity", json={})
        assert repo.update.await_args.kwargs["parity"] == {"passed": False, "max_abs_diff": 3.2}


class TestHubRoutes:
    def test_search_reaches_the_hub_service(self):
        client, _repo, _arming, hub, _service = _client()
        response = client.get("/api/probes/hub/search?q=stakes&limit=5")
        assert response.status_code == 200
        assert hub.search.await_args.kwargs == {
            "query": "stakes",
            "base_model": None,
            "limit": 5,
        }

    def test_a_repo_id_with_a_slash_survives_the_path(self):
        """`{repo_id:path}` exists so `org/name` is one parameter, not two segments."""
        client, _repo, _arming, hub, _service = _client()
        response = client.get("/api/probes/hub/mistudio/probes-lfm2/definitions")
        assert response.status_code == 200
        assert hub.list_definitions.await_args.args[0] == "mistudio/probes-lfm2"

    def test_a_hub_import_is_recorded_as_origin_hub(self):
        """Origin is how an operator later tells a file import from a Hub one."""
        client, _repo, _arming, hub, service = _client(probe=_probe_row())
        response = client.post(
            "/api/probes/hub/import",
            json={"repo_id": "mistudio/probes", "filename": "a.probe.json"},
        )
        assert response.status_code == 200
        assert service.import_definition.await_args.kwargs["origin"] == "hub"

    def test_hub_import_defaults_to_rename_never_replace(self):
        client, _repo, _arming, _hub, service = _client(probe=_probe_row())
        client.post(
            "/api/probes/hub/import",
            json={"repo_id": "mistudio/probes", "filename": "a.probe.json"},
        )
        assert service.import_definition.await_args.kwargs["on_conflict"] == "rename"

    def test_replace_is_refused_by_the_schema(self):
        """There is deliberately no `replace`: it would change the detector under a running
        monitor while every event before and after kept the same probe id."""
        client, *_ = _client(probe=_probe_row())
        response = client.post(
            "/api/probes/hub/import",
            json={
                "repo_id": "mistudio/probes",
                "filename": "a.probe.json",
                "on_conflict": "replace",
            },
        )
        assert response.status_code == 422
