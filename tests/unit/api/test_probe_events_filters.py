"""Find a stored event by result or by trace (FR-38.11, FTASKS 4.10), and the identity names
`GET /api/probes/{id}` reports (FR-38.7, FTASKS 4.9) — both read from the LIVE `app.openapi()`.

Two rows differing ONLY in the filtered column: a filter that is accepted and ignored returns both,
and the test goes red. Control M11: remove either repository clause.
"""

from __future__ import annotations

import pytest

from tests.unit.api.test_probe_event_filters import BASE, _seed, client  # noqa: F401
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture

TRACE_A = "0af7651916cd43dd8448eb211c80319c"
TRACE_B = "4bf92f3577b34da6a3ce929d0e0e4736"


def _params(openapi, path):
    return {p["name"]: p for p in openapi["paths"][path]["get"].get("parameters", [])}


def test_both_filters_are_served_with_their_bounds(client):  # noqa: F811
    params = _params(client.app.openapi(), "/api/probes/events")
    assert params["result_id"]["schema"]["anyOf"][0]["maxLength"] == 36
    assert params["trace_id"]["schema"]["anyOf"][0]["pattern"] == "^[0-9a-f]{32}$"


async def test_result_id_returns_exactly_that_event(client, db):  # noqa: F811
    ids = await _seed(db, {"request_id": "r1"}, {"request_id": "r2"})
    listed = client.get("/api/probes/events").json()["data"]
    assert len(listed) == 2
    target = next(e for e in listed if e["id"] == ids[0])
    got = client.get("/api/probes/events", params={"result_id": target["result_id"]}).json()
    assert [e["id"] for e in got["data"]] == [ids[0]]
    assert got["data"][0]["result_id"] == target["result_id"]


async def test_trace_id_returns_exactly_that_requests_events(client, db):  # noqa: F811
    ids = await _seed(db, {"trace_id": TRACE_A}, {"trace_id": TRACE_B})
    got = client.get("/api/probes/events", params={"trace_id": TRACE_B}).json()["data"]
    assert [e["id"] for e in got] == [ids[1]]
    assert got[0]["trace_id"] == TRACE_B


def test_a_malformed_trace_id_is_refused(client):  # noqa: F811
    assert client.get("/api/probes/events", params={"trace_id": "nope"}).status_code == 422


IDENTITY_NAMES = {"detector_digest", "detector_digest_source", "identity_algorithm",
                  "decision_digest", "decision_revision", "threshold_revision"}


def test_get_probe_documents_the_six_identity_names(client):  # noqa: F811
    """⚠ CROSS-REPO: miStudio 039's `test_millm_probe_identity_names.py` pins these names."""
    openapi = client.app.openapi()
    response = openapi["paths"]["/api/probes/{probe_id}"]["get"]["responses"]["200"]
    ref = response["content"]["application/json"]["schema"]["$ref"].rsplit("/", 1)[-1]
    data = openapi["components"]["schemas"][ref]["properties"]["data"]
    inner = next(s["$ref"] for s in data["anyOf"] if "$ref" in s).rsplit("/", 1)[-1]
    assert IDENTITY_NAMES <= set(openapi["components"]["schemas"][inner]["properties"])


async def test_get_probe_reports_the_stored_identity(client, db):  # noqa: F811
    from millm.services.probe_digest import detector_digest
    from tests.unit.probe_fixtures import probe_definition

    await _seed(db, {}, {})
    probe_id = client.get("/api/probes").json()["data"][0]["id"]
    body = client.get(f"/api/probes/{probe_id}").json()["data"]
    assert IDENTITY_NAMES <= set(body)
    assert body["detector_digest"] == detector_digest(probe_definition(name=body["name"]))
    assert body["detector_digest_source"] == "computed" and body["threshold_revision"] == 1
