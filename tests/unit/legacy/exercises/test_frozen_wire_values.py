"""Exercises for the frozen wire values (040 FR-040.7): DR-14 (`mistudio_probe_id` / `mistudio_run_id`
on recalibrate), DR-17 (the stored actor id of miStudio's MCP agent; its only plant is `OLD_ACTOR`
below, so the presence check bites on it), DR-32 (probe result
`producer.name = "millm"`), DR-34 (response and Batch key `millm`), DR-35 (stored approval
`executor_key` module paths under `millm.`) and DR-46 (body field `millm_probe_verdicts`).

Each value is written by one side and parsed by another (miStudio, miDataworks, Open WebUI, stored
rows). Every check goes through the live route or the real builder.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
from sqlalchemy import update

from miserver.db.models.approval import ApprovalRequest
from miserver.services import approval_service
from miserver.telemetry import allowlist
from tests.support import capture_pre038_goldens as cap
from tests.unit.api.test_probe_event_filters import client  # noqa: F401 - fixture
from tests.unit.api.test_probe_identity_routes import CARRYING, _doc, _new_decision
from tests.unit.api.test_return_sae_activations import ACT, CHAT, clean, live  # noqa: F401
from tests.unit.approval_fixtures import AGENT, harness, make_model, rows  # noqa: F401
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture

REPO = Path(__file__).resolve().parents[4]
OLD_ACTOR = "agent:mistudio-mcp"
#: A key stored in `approval_requests.executor_key`: `<module>.<qualname>` under the `millm.`
#: package. Rows filed before any rename must still run (permanent, CF-7; 041 adds a resolver).
STORED_KEY = "millm.api.routes.management.models.unload_model"  # p8: keep


def test_recalibrate_accepts_mistudio_probe_fields(client):  # noqa: F811
    probe_id = client.post("/api/probes/import", json=_doc()).json()["data"]["id"]
    producer_id = CARRYING["document"]["provenance"]["probe_id"]
    body = {"decision": _new_decision(), "mistudio_probe_id": producer_id,
            "mistudio_run_id": CARRYING["document"]["provenance"].get("run_id"), "reason": "dr14"}
    ok = client.post(f"/api/probes/{probe_id}/recalibrate", json=body)
    assert ok.status_code == 200, ok.text
    assert client.get(f"/api/probes/{probe_id}").json()["data"]["threshold_revision"] == 2
    # The field is READ, not just accepted: another producer id is a different probe.
    wrong = client.post(f"/api/probes/{probe_id}/recalibrate",
                        json={**body, "mistudio_probe_id": "pm_someone_else"})
    assert wrong.status_code >= 400, wrong.text


async def test_stored_old_actor_id_lists_as_an_agent(harness):  # noqa: F811
    h = harness
    h.holder["row"] = row = await make_model(h.db)
    r = h.client.post(f"/api/models/{row.id}/unload", headers={"X-miLLM-Agent": OLD_ACTOR})
    assert r.status_code == 202, r.text
    (stored,) = await rows(h.db)
    assert stored.requested_by == OLD_ACTOR
    listed = h.client.get("/api/approvals").json()["data"]
    assert [a["requested_by"] for a in listed["approvals"]] == [OLD_ACTOR], listed


async def test_stored_executor_key_runs_when_approved(harness):  # noqa: F811
    h = harness
    h.holder["row"] = row = await make_model(h.db)
    approval_id = h.client.post(f"/api/models/{row.id}/unload", headers=AGENT).json()["data"][
        "approval_id"]
    async with h.db() as session:  # the row as an earlier server stored it
        await session.execute(update(ApprovalRequest).where(ApprovalRequest.id == approval_id)
                              .values(executor_key=STORED_KEY))
        await session.commit()
    r = h.client.post(f"/api/approvals/{approval_id}/approve")
    assert r.status_code == 202, r.text
    await approval_service.wait_all()
    assert r.json()["data"]["status"] == "executed", r.text
    h.service.unload_model.assert_awaited_once_with(row.id, lease_id=None)


async def test_probe_result_producer_name():
    schema = json.loads((REPO / "docs" / "schemas" / "probe-result-v1.json").read_text())
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])
        r = h.post_chat(cap.BODY, {})
    results = r.json()["millm_probe_verdicts"]
    assert results and {res["producer"]["name"] for res in results} == {"millm"}
    for res in results:
        jsonschema.validate(res, schema)
        kept, _dropped = allowlist.filter("log_result", {"mi.probe.producer.name": res["producer"]["name"]})
        assert kept == {"mi.probe.producer.name": "millm"}


async def test_probe_verdicts_body_field():
    async with cap.Harness() as h:
        await h.arm([cap.ALPHA])
        r = h.post_chat(cap.BODY, {})
    assert r.status_code == 200
    assert isinstance(r.json().get("millm_probe_verdicts"), list) and r.json()["millm_probe_verdicts"]


def test_response_and_batch_key_millm(live):  # noqa: F811
    from miserver.services.batch.executors import output_line

    client, *_ = live
    body = client.post("/v1/chat/completions", json={**CHAT, "return_sae_activations": ACT}).json()
    assert "sae_activations" in body["millm"]
    with client.stream("POST", "/v1/chat/completions",
                       json={**CHAT, "stream": True, "return_sae_activations": ACT}) as r:
        chunks = [json.loads(line[6:]) for line in r.iter_lines()
                  if line.startswith("data: {")]
    assert any("millm" in c for c in chunks), "the activations chunk carries key 'millm'"
    line = output_line("c1", 200, {}, packed=False, headers={})
    assert line["response"]["millm"] == {"packed": False, "headers": {}}
