"""Lifecycle records (FR-38.20, AT12; FTDD §6.5; FTASKS 8.9), through the real routes and services.

Import, arm (or its refusal), stand-alone parity, recalibrate, disarm, the model-unload disarm
(`code: model_unloaded`) and delete each emit one `mi.probe.lifecycle` record with the probe's
identity. The operator's free text — the arm acknowledgement's `reason`, the recalibration's
history `reason`, a disarm's `paused_reason` — never appears in any exported byte.

Control M20: delete the emit in `probe_recalibration`.
"""

from __future__ import annotations

from collections import Counter

import pytest

from tests.support import capture_pre038_goldens as cap
from tests.unit.telemetry.conftest import flush

REASON = "SENTINEL-OPERATOR-REASON because the customer asked"


def _lifecycle(stub) -> list[dict]:
    return [r["attributes"] for r in stub.logs if r["event_name"] == "mi.probe.lifecycle"]


MISTUDIO_ID = "pm_fixture01"


@pytest.fixture
def recalibratable(monkeypatch):
    """The harness's definitions with the provenance a re-cut is verified against."""
    original = cap._definition

    def with_provenance(spec):
        doc = original(spec)
        doc["provenance"] = {"probe_id": MISTUDIO_ID, "run_id": "pmr_fixture01"}
        return doc

    monkeypatch.setattr(cap, "_definition", with_provenance)


def _recalibration(row) -> dict:
    decision = dict(row.definition["decision"])
    decision["threshold"] = float(decision.get("threshold") or 0.0) + 0.125
    decision.setdefault("target_fpr", 0.01)
    decision.setdefault("realised_fpr", 0.01)
    decision.setdefault("threshold_source", "calibration_set")
    return {"decision": decision,
            "mistudio_probe_id": MISTUDIO_ID,
            "reason": REASON}


async def test_at12_recalibrate_through_the_route_emits_one_record(stub, telemetry_on,
                                                                   recalibratable):
    telemetry_on()
    async with cap.Harness() as h:
        rows = await h.arm([cap.ALPHA])
        before_digest = rows[0].decision_digest
        response = h.client.post(f"/api/probes/{rows[0].id}/recalibrate",
                                 json=_recalibration(rows[0]))
        assert response.status_code == 200, response.text
        flush()
    records = [r for r in _lifecycle(stub) if r["mi.lifecycle.action"] == "recalibrate"]
    assert len(records) == 1, _lifecycle(stub)
    rec = records[0]
    assert rec["mi.lifecycle.outcome"] == "ok"
    assert rec["mi.lifecycle.probe_id"] == rows[0].id
    assert rec["mi.lifecycle.threshold_revision"] == 2
    assert rec["mi.lifecycle.previous_threshold_revision"] == 1
    assert rec["mi.lifecycle.previous_decision_digest"] == before_digest
    assert rec["mi.lifecycle.decision_digest"] != before_digest
    assert rec["mi.lifecycle.behaviour"] == cap.ALPHA_KEY
    assert len(rec["mi.lifecycle.detector_digest"]) == 64
    assert b"SENTINEL-OPERATOR-REASON" not in stub.all_bytes()


async def test_a_refused_recalibration_carries_the_code_only(stub, telemetry_on, recalibratable):
    telemetry_on()
    async with cap.Harness() as h:
        rows = await h.arm([cap.ALPHA])
        body = _recalibration(rows[0])
        body["mistudio_probe_id"] = "pm_000000000000"
        response = h.client.post(f"/api/probes/{rows[0].id}/recalibrate", json=body)
        assert response.status_code == 409, response.text
        flush()
    records = [r for r in _lifecycle(stub) if r["mi.lifecycle.action"] == "recalibrate"]
    assert len(records) == 1, _lifecycle(stub)
    rec = records[0]
    assert rec["mi.lifecycle.outcome"] == "refused"
    assert rec["mi.lifecycle.code"] == response.json()["error"]["code"]
    assert "mi.lifecycle.previous_threshold_revision" not in rec


async def test_every_lifecycle_step_emits_and_no_free_text_leaves(stub, telemetry_on, monkeypatch):
    from millm.services.probe_arming import UNLOAD_REASON, mark_armed_rows_disarmed

    telemetry_on()
    async with cap.Harness() as h:
        rows = await h.arm([cap.ALPHA, cap.BETA])  # two imports
        alpha, beta = rows
        # The routes bind `loaded_identity` at import; the harness patches the bridge's, which
        # reaches the routes only when they are first imported inside it. Patch both.
        from unittest.mock import AsyncMock

        import millm.api.routes.management.probes as probe_routes
        from tests.unit.score_fixtures import tiny_identity

        monkeypatch.setattr(probe_routes, "loaded_identity",
                            AsyncMock(return_value=(tiny_identity(), h.model, h.tokenizer)))
        # A stand-alone parity check, then an arm through the route (the tiny model either
        # reproduces the recorded scores or refuses; both are lifecycle records).
        parity = h.client.post(f"/api/probes/{alpha.id}/parity")
        assert parity.status_code == 200, parity.text
        arm = h.client.post(f"/api/probes/{beta.id}/arm",
                            json={"acknowledge_below_rung2": True, "reason": REASON})
        assert arm.status_code == 409, arm.text
        assert arm.json()["error"]["code"] == "PROBE_PARITY_FAILED", arm.text
        # The operator disarms; the probe is deleted.
        disarm = h.client.post(f"/api/probes/{alpha.id}/disarm", json={"reason": REASON})
        assert disarm.status_code == 200, disarm.text
        deleted = h.client.delete(f"/api/probes/{alpha.id}")
        assert deleted.status_code == 200, deleted.text
        # A model unload disarms whatever rows still claim to be armed.
        from sqlalchemy import update

        from millm.db.models.probe import Probe

        async with h.factory() as session:
            await session.execute(update(Probe).where(Probe.id == beta.id).values(armed=True))
            await session.commit()
        assert await mark_armed_rows_disarmed(h.factory, UNLOAD_REASON,
                                              event="probes_disarmed_by_unload") == [beta.id]
        flush()
    records = _lifecycle(stub)
    actions = Counter(r["mi.lifecycle.action"] for r in records)
    assert actions["import"] == 2
    # The stand-alone check and the arm's own parity gate: each is a record (the tiny model does
    # not reproduce the fixture's recorded scores, so both are refused and the arm with them).
    parity = [r for r in records if r["mi.lifecycle.action"] == "parity"]
    assert [(r["mi.lifecycle.probe_id"], r["mi.lifecycle.outcome"], r["mi.lifecycle.code"])
            for r in parity] == [(alpha.id, "refused", "PROBE_PARITY_FAILED"),
                                 (beta.id, "refused", "PROBE_PARITY_FAILED")]
    assert (actions["arm"], actions["arm_refused"]) == (0, 1)
    assert actions["delete"] == 1
    unload = [r for r in records if r["mi.lifecycle.action"] == "disarm"
              and r.get("mi.lifecycle.code") == "model_unloaded"]
    assert [r["mi.lifecycle.probe_id"] for r in unload] == [beta.id]
    assert any(r["mi.lifecycle.action"] == "disarm" and r["mi.lifecycle.probe_id"] == alpha.id
               for r in records)
    raw = stub.all_bytes()
    assert b"SENTINEL-OPERATOR-REASON" not in raw
    assert UNLOAD_REASON.encode() not in raw
    # ⚠ The free text was never even OFFERED to the allowlist (a `reason` passed as a code would
    # be dropped there, and counted here).
    from millm.telemetry import allowlist

    assert allowlist.attributes_dropped() == 0
    for r in records:
        assert r["mi.lifecycle.subject_kind"] == "probe"
        assert r["mi.lifecycle.outcome"] in ("ok", "refused", "failed")


async def test_an_import_refusal_is_a_record_with_no_probe(stub, telemetry_on):
    telemetry_on()
    async with cap.Harness() as h:
        response = h.client.post("/api/probes/import",
                                 json={"kind": "mistudio.circuit-definition/v1", "x": REASON})
        assert 400 <= response.status_code < 500, response.text
        flush()
    records = [r for r in _lifecycle(stub) if r["mi.lifecycle.action"] == "import"]
    assert len(records) == 1, records
    assert records[0]["mi.lifecycle.outcome"] == "refused"
    assert records[0]["mi.lifecycle.code"] == response.json()["error"]["code"]
    assert "mi.lifecycle.probe_id" not in records[0]
    assert b"SENTINEL-OPERATOR-REASON" not in stub.all_bytes()


async def test_the_arming_service_emits_arm_refusal_and_disarm(stub, telemetry_on, monkeypatch):
    """The service methods themselves (so the route and the arm bridge both emit): `arm` on
    success, `arm_refused` with the CODE, `disarm`. The acknowledgement reason is passed to `arm`
    and `disarm` and must not leave (nor even be OFFERED: `attributes_dropped == 0`)."""
    from unittest.mock import AsyncMock, MagicMock

    from millm.core.errors import ProbeParityFailedError
    from millm.services.probe_arming import ProbeArmingService
    from millm.services.probe_runtime import ProbeRuntimeState

    telemetry_on()
    async with cap.Harness() as h:
        rows = await h.arm([cap.ALPHA, cap.BETA])
        service = ProbeArmingService(MagicMock(update=AsyncMock()), ProbeRuntimeState())
        monkeypatch.setattr(service, "_arm", AsyncMock(return_value="armed"))
        assert await service.arm(rows[0], reason=REASON) == "armed"
        monkeypatch.setattr(service, "_arm", AsyncMock(
            side_effect=ProbeParityFailedError(f"{REASON} does not reproduce")))
        with pytest.raises(ProbeParityFailedError):
            await service.arm(rows[1], reason=REASON)
        assert await service.disarm(rows[0], reason=REASON) is True
        flush()
    records = [r for r in _lifecycle(stub) if r["mi.lifecycle.action"] != "import"]
    got = [(r["mi.lifecycle.action"], r["mi.lifecycle.probe_id"], r["mi.lifecycle.outcome"],
            r.get("mi.lifecycle.code")) for r in records]
    assert got == [("arm", rows[0].id, "ok", None),
                   ("arm_refused", rows[1].id, "refused", "PROBE_PARITY_FAILED"),
                   ("disarm", rows[0].id, "ok", None)], got
    assert records[1]["mi.lifecycle.behaviour"] == cap.BETA_KEY
    assert len(records[1]["mi.lifecycle.detector_digest"]) == 64
    assert b"SENTINEL-OPERATOR-REASON" not in stub.all_bytes()
    from millm.telemetry import allowlist

    assert allowlist.attributes_dropped() == 0, "free text was offered to the emitter"


async def test_a_recalibration_that_fails_after_its_commit_is_one_failed_record(
        stub, telemetry_on, recalibratable, monkeypatch):
    """Review: the record is emitted once the whole operation succeeded — a failure after the
    commit (here the registry refresh) is ONE `failed` record, never an `ok` and a `failed`."""
    from millm.services.probe_runtime import ProbeRuntimeState

    telemetry_on()
    async with cap.Harness() as h:
        rows = await h.arm([cap.ALPHA])

        def boom(*_a, **_k):
            raise RuntimeError("refresh failed")

        monkeypatch.setattr(ProbeRuntimeState, "refresh", boom)
        from fastapi.testclient import TestClient

        client = TestClient(h.client.app, raise_server_exceptions=False)
        response = client.post(f"/api/probes/{rows[0].id}/recalibrate",
                               json=_recalibration(rows[0]))
        assert response.status_code >= 500
        flush()
    records = [r for r in _lifecycle(stub) if r["mi.lifecycle.action"] == "recalibrate"]
    assert [(r["mi.lifecycle.outcome"], r.get("mi.lifecycle.code")) for r in records] == [
        ("failed", "RuntimeError")]


async def test_a_restart_disarm_is_a_record_with_its_cause(stub, telemetry_on):
    """Review R1 of S2: `disarm_probes_on_startup` clears every row a previous process left armed.
    That is a disarm, so it is a `disarm` record (code `process_restarted`) like the unload path;
    before the fix the receiver's last record for the probe was its `arm`. Control: delete the
    emit in `disarm_probes_on_startup`."""
    from sqlalchemy import update

    from millm.db.models.probe import Probe
    from millm.main import disarm_probes_on_startup

    telemetry_on()
    async with cap.Harness() as h:
        alpha, beta = await h.arm([cap.ALPHA, cap.BETA])
        async with h.factory() as session:
            await session.execute(update(Probe).values(armed=False))
            await session.execute(update(Probe).where(Probe.id == beta.id).values(armed=True))
            await session.commit()
        assert await disarm_probes_on_startup(h.factory) == [beta.id]
        flush()
    restart = [r for r in _lifecycle(stub) if r["mi.lifecycle.action"] == "disarm"]
    assert [(r["mi.lifecycle.probe_id"], r["mi.lifecycle.code"], r["mi.lifecycle.outcome"])
            for r in restart] == [(beta.id, "process_restarted", "ok")], restart
    assert restart[0]["mi.lifecycle.detector_digest"] == beta.detector_digest
    assert b"disarmed by a restart" not in stub.all_bytes()


# ── 12.5a F3: a refusal BEFORE `ProbeArmingService.arm` is a record too ─────────────────
#: Found by the 12.5 run (AT12): with a llama.cpp model resident, `loaded_identity` refuses
#: `PROBE_HOOK_UNSUPPORTED` in the route, before the arming service (which emits) is reached, so
#: the refused arm and the refused parity check left no lifecycle record at all. Controls F3a
#: (arm route), F3b (parity route), F3c (behaviour arm route): delete the `lifecycle_refused`.
from millm.services.probe_arm_bridge import loaded_identity as _REAL_LOADED_IDENTITY  # noqa: E402


@pytest.fixture
def llamacpp_resident(monkeypatch):
    """The REAL `loaded_identity` over a resident llama.cpp model (no module tree to hook)."""
    from datetime import datetime

    import millm.api.routes.management.probes as probe_routes
    from millm.ml.model_loader import ENGINE_LLAMACPP, LoadedModel, LoadedModelState
    from tests.support.generation_entry_points import EventLog, FakeLlama

    def install() -> None:
        monkeypatch.setattr(probe_routes, "loaded_identity", _REAL_LOADED_IDENTITY)
        monkeypatch.setattr("millm.services.probe_arm_bridge.loaded_identity",
                            _REAL_LOADED_IDENTITY)
        LoadedModelState().set(LoadedModel(
            model_id=1, model_name="tiny", model=FakeLlama(EventLog()), tokenizer=None,
            loaded_at=datetime(2026, 10, 24), engine=ENGINE_LLAMACPP,
        ))

    return install


async def test_f3_a_refused_arm_and_parity_before_the_service_each_emit_one_record(
        stub, telemetry_on, llamacpp_resident):
    telemetry_on()
    async with cap.Harness() as h:
        rows = await h.arm([cap.ALPHA, cap.BETA])
        alpha, beta = rows
        llamacpp_resident()
        arm = h.client.post(f"/api/probes/{beta.id}/arm",
                            json={"acknowledge_below_rung2": True, "reason": REASON})
        parity = h.client.post(f"/api/probes/{alpha.id}/parity")
        for response in (arm, parity):
            assert response.status_code == 409, response.text
            assert response.json()["error"]["code"] == "PROBE_HOOK_UNSUPPORTED", response.text
        flush()
    records = [r for r in _lifecycle(stub) if r["mi.lifecycle.action"] != "import"]
    assert [(r["mi.lifecycle.action"], r["mi.lifecycle.probe_id"], r["mi.lifecycle.outcome"],
             r["mi.lifecycle.code"]) for r in records] == [
        ("arm_refused", beta.id, "refused", "PROBE_HOOK_UNSUPPORTED"),
        ("parity", alpha.id, "refused", "PROBE_HOOK_UNSUPPORTED"),
    ], records
    for r in records:
        assert r["mi.lifecycle.behaviour"] in (cap.ALPHA_KEY, cap.BETA_KEY)
        assert len(r["mi.lifecycle.detector_digest"]) == 64
    assert b"SENTINEL-OPERATOR-REASON" not in stub.all_bytes()


async def test_f3_a_behaviour_arm_refused_before_selection_emits_one_record(
        stub, telemetry_on, llamacpp_resident):
    telemetry_on()
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])
        llamacpp_resident()
        dry = h.client.post(f"/api/probes/behaviours/{cap.ALPHA_KEY}/arm",
                            json={"dry_run": True, "acknowledge_below_rung2": True})
        real = h.client.post(f"/api/probes/behaviours/{cap.ALPHA_KEY}/arm",
                             json={"acknowledge_below_rung2": True, "reason": REASON})
        for response in (dry, real):
            assert response.status_code == 409, response.text
            assert response.json()["error"]["code"] == "PROBE_HOOK_UNSUPPORTED", response.text
        flush()
    records = [r for r in _lifecycle(stub) if r["mi.lifecycle.action"] != "import"]
    # The dry run records nothing; the real one, one record naming the behaviour and no probe.
    assert [(r["mi.lifecycle.action"], r["mi.lifecycle.behaviour"], r["mi.lifecycle.outcome"],
             r["mi.lifecycle.code"]) for r in records] == [
        ("arm_refused", cap.ALPHA_KEY, "refused", "PROBE_HOOK_UNSUPPORTED")], records
    assert "mi.lifecycle.probe_id" not in records[0]
    assert b"SENTINEL-OPERATOR-REASON" not in stub.all_bytes()


# ── 12.5a F5: one refusal, one code — the record says what the API said ─────────────────
#: A definition with no `provenance.probe_id` was answered `PROBE_RECALIBRATION_MISMATCH` by the
#: route and recorded `PROBE_RECALIBRATION_UNVERIFIABLE`. Control F5: pass the internal refusal
#: (not `as_error()`) to `lifecycle_refused` in `ProbeRecalibrationService.recalibrate`.


def _unverifiable(row) -> dict:
    return _recalibration(row)  # the harness's definitions carry no provenance.probe_id


def _mismatched(row) -> dict:
    return {**_recalibration(row), "mistudio_probe_id": "pm_000000000000"}


def _uncalibrated(row) -> dict:
    body = _recalibration(row)
    body["decision"] = {k: v for k, v in body["decision"].items()
                        if k not in ("target_fpr", "threshold_source")}
    return body


@pytest.mark.parametrize("case, provenance, code", [
    (_unverifiable, False, "PROBE_RECALIBRATION_MISMATCH"),
    (_mismatched, True, "PROBE_RECALIBRATION_MISMATCH"),
    (_uncalibrated, True, "PROBE_THRESHOLD_UNCALIBRATED"),
], ids=["unverifiable", "mismatched", "uncalibrated"])
async def test_f5_a_refused_recalibration_records_the_code_the_api_answers(
        stub, telemetry_on, monkeypatch, case, provenance, code):
    if provenance:
        original = cap._definition
        monkeypatch.setattr(cap, "_definition", lambda spec: {
            **original(spec), "provenance": {"probe_id": MISTUDIO_ID, "run_id": "pmr_fixture01"}})
    telemetry_on()
    async with cap.Harness() as h:
        rows = await h.arm([cap.ALPHA])
        response = h.client.post(f"/api/probes/{rows[0].id}/recalibrate", json=case(rows[0]))
        assert response.status_code == 409, response.text
        flush()
    assert response.json()["error"]["code"] == code, response.text
    records = [r for r in _lifecycle(stub) if r["mi.lifecycle.action"] == "recalibrate"]
    assert [(r["mi.lifecycle.outcome"], r["mi.lifecycle.code"]) for r in records] == [
        ("refused", code)], records


async def test_r1_a_behaviour_arm_refused_by_its_selection_emits_one_record(stub, telemetry_on):
    """12.5a R1: the selection's refusals (here, no imported member for the loaded model) came
    after F3's `try` and before `arm_set`, so they recorded nothing. Control R1-B: move the
    selection back out of the `try`."""
    telemetry_on()
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])
        dry = h.client.post("/api/probes/behaviours/b-nobody/arm",
                            json={"dry_run": True, "acknowledge_below_rung2": True})
        real = h.client.post("/api/probes/behaviours/b-nobody/arm",
                             json={"acknowledge_below_rung2": True, "reason": REASON})
        for response in (dry, real):
            assert response.status_code == 404, response.text
            assert response.json()["error"]["code"] == "PROBE_NOT_FOUND", response.text
        flush()
    records = [r for r in _lifecycle(stub) if r["mi.lifecycle.action"] not in ("import", "arm")]
    assert [(r["mi.lifecycle.action"], r["mi.lifecycle.behaviour"], r["mi.lifecycle.outcome"],
             r["mi.lifecycle.code"]) for r in records] == [
        ("arm_refused", "b-nobody", "refused", "PROBE_NOT_FOUND")], records
    assert "mi.lifecycle.probe_id" not in records[0]
    assert b"SENTINEL-OPERATOR-REASON" not in stub.all_bytes()
