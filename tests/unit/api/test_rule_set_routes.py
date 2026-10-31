"""The rule-set routes through the LIVE app (039 FR-39.17, FR-39.19, FR-39.21, FR-39.25; FTASKS
11.5, 12.5, 12.7, 13.1).

Real `create_app()` routes, a real SQLite database. Wiring is asserted on the live `app.openapi()`;
the validator coverage test reads the LIVE registry THROUGH the route.

Controls (039 controls record): unregister import / evaluate / validators / the PUT attachment
route, or drop `include_router(rule_sets_router)` → `test_the_rule_set_routes_are_served` red.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import func, select

from millm.core.config import settings
from millm.db.models.rule_set import BehaviourRuleSignalHistory, RuleSet
from tests.support import behaviour_factory as bf
from tests.support import capture_pre038_goldens as cap
from tests.support import rule_set_factory as rf

pytestmark = pytest.mark.asyncio

ROUTES = {
    "/api/rule-sets/import": "post",
    "/api/rule-sets/validate": "post",
    "/api/rule-sets": "get",
    "/api/rule-sets/{rule_set_id}": "get",
    "/api/rule-sets/{rule_set_id}/evaluate": "post",
    "/api/rule-sets/validators": "get",
    "/api/probes/behaviours/{key}/rule-signals": "put",
}
VECTORS = Path(__file__).resolve().parents[3] / "docs" / "schemas" / "rule-signal-cases-v1.json"


async def _count(h, model=RuleSet) -> int:
    async with h.factory() as session:
        return int((await session.execute(select(func.count()).select_from(model))).scalar_one())


def _post(h, path, doc, **params):
    return h.client.post(path, content=rf.raw(doc), params=params,
                         headers={"content-type": "application/json"})


async def test_the_rule_set_routes_are_served():
    from millm.main import create_app

    paths = create_app().openapi()["paths"]
    for path, method in ROUTES.items():
        assert method in paths.get(path, {}), f"{method.upper()} {path} is not served"
    assert "delete" in paths["/api/rule-sets/{rule_set_id}"]
    assert {"get", "delete"} <= set(paths["/api/probes/behaviours/{key}/rule-signals"])


async def test_validators_are_listed_from_the_live_registry_and_cover_the_vectors():
    async with cap.Harness() as h:
        response = h.client.get("/api/rule-sets/validators")
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        listed = {v["ref"] for v in data["validators"]}
        sections = set(json.loads(VECTORS.read_text())["validators"])
        assert listed == sections and len(listed) >= 13
        assert all(v["params_schema"]["type"] == "object" for v in data["validators"])
        forms = {f["form"]: f["evaluated"] for f in data["forms"]}
        assert forms["code_rule"] == "after_response" and forms["pattern"] == "inline"


async def test_a_set_with_every_declarative_form_imports_and_evaluates():
    async with cap.Harness() as h:
        key = bf.random_key()
        doc = rf.document(key)
        response = _post(h, "/api/rule-sets/import", doc, version=3)
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert data["created"] is True and data["id"].startswith("rs_")
        assert (data["version"], data["behaviour_key"], data["origin"]) == (3, key, "authored")
        rule_set_id = data["id"]
        table = [
            ("marker ZQ-1234 and 79927398713, zorbel quint", True, True, True, True),
            ("plain words only", False, False, False, False),
        ]
        for text, token, cued, number, lexicon in table:
            r = h.client.post(f"/api/rule-sets/{rule_set_id}/evaluate", json={"text": text})
            assert r.status_code == 200, r.text
            got = {x["signal"]: x["fires"] for x in r.json()["data"]["results"]
                   if x["window"] == "prompt"}
            assert (got["token"], got["cued"], got["number"], got["lexicon"]) == (
                token, cued, number, lexicon)
            assert all("spans" not in x for x in r.json()["data"]["results"])
        r = h.client.post(f"/api/rule-sets/{rule_set_id}/evaluate",
                          json={"text": "marker ZQ-1234", "return_rule_positions": True})
        token = next(x for x in r.json()["data"]["results"]
                     if x["signal"] == "token" and x["window"] == "prompt")
        assert token["spans"] == [[7, 14]]
        assert await _count(h) == 1


async def test_validate_and_import_agree_and_validate_writes_nothing():
    async with cap.Harness() as h:
        key = bf.random_key()
        good, bad = rf.document(key), rf.with_wrong_case(key)
        assert _post(h, "/api/rule-sets/validate", good).status_code == 200
        r = _post(h, "/api/rule-sets/validate", bad)
        assert r.status_code == 422 and r.json()["error"]["code"] == "RULE_VECTORS_FAILED"
        assert r.json()["error"]["details"]["case_index"] == 1
        assert r.json()["error"]["details"]["signal"] == "lexicon"
        assert r.json()["error"]["next"] == "millm_validate_rule_set"
        assert await _count(h) == 0
        r2 = _post(h, "/api/rule-sets/import", bad)
        assert (r2.status_code, r2.json()["error"]["code"]) == (422, "RULE_VECTORS_FAILED")
        assert await _count(h) == 0


async def test_an_identical_document_returns_the_existing_id_and_a_wrong_digest_is_refused():
    async with cap.Harness() as h:
        doc = rf.document(bf.random_key())
        first = _post(h, "/api/rule-sets/import", doc).json()["data"]
        again = _post(h, "/api/rule-sets/import", doc).json()["data"]
        assert again["id"] == first["id"] and again["created"] is False
        r = _post(h, "/api/rule-sets/import", rf.document(bf.random_key()), digest="0" * 64)
        assert (r.status_code, r.json()["error"]["code"]) == (422, "RULE_SET_DIGEST_MISMATCH")
        assert r.json()["error"]["details"]["declared"] == "0" * 64
        assert await _count(h) == 1


async def test_the_producers_published_example_set_imports_unchanged():
    """Every rule document the producer publishes in the vectors imports as posted (FR-39.17)."""
    documents = json.loads(VECTORS.read_text())["rule_documents"]
    async with cap.Harness() as h:
        for name, doc in documents.items():
            r = _post(h, "/api/rule-sets/import", doc)
            assert r.status_code == 200, (name, r.text)
        assert await _count(h) == len(documents)


async def test_limits_refuse_at_limit_plus_one(monkeypatch):
    async with cap.Harness() as h:
        monkeypatch.setattr(settings, "RULE_SETS_MAX_IMPORTED", 1)
        assert _post(h, "/api/rule-sets/import", rf.document(bf.random_key())).status_code == 200
        r = _post(h, "/api/rule-sets/import", rf.document(bf.random_key()))
        assert (r.status_code, r.json()["error"]["code"]) == (409, "CATALOGUE_LIMIT")
        assert r.json()["error"]["details"]["limit"] == "RULE_SETS_MAX_IMPORTED"
        monkeypatch.setattr(settings, "RULE_SETS_MAX_IMPORTED", 100)
        monkeypatch.setattr(settings, "RULE_SET_MAX_SIGNALS", 8)  # the factory declares 4 + 5
        r = _post(h, "/api/rule-sets/import", rf.document(bf.random_key()))
        assert r.json()["error"]["details"]["limit"] == "RULE_SET_MAX_SIGNALS"
        monkeypatch.setattr(settings, "RULE_SET_MAX_SIGNALS", 64)
        monkeypatch.setattr(settings, "RULE_SET_MAX_BYTES", 1024)
        r = _post(h, "/api/rule-sets/import", rf.document(bf.random_key()))
        assert (r.status_code, r.json()["error"]["code"]) == (422, "RULE_SET_INVALID")
        assert r.json()["error"]["details"]["rule"] == "rule_set_too_large"


async def test_refusals_name_the_construct_and_never_echo_the_document():
    async with cap.Harness() as h:
        key = bf.random_key()
        doc = rf.document(key)
        doc["signals"][0]["pattern"] = r"(a)\1"
        r = _post(h, "/api/rule-sets/validate", doc)
        assert (r.status_code, r.json()["error"]["code"]) == (422, "RULE_PATTERN_UNSUPPORTED")
        details = r.json()["error"]["details"]
        assert {k: details[k] for k in ("rule", "signal", "position")} == {
            "rule": "backreference", "signal": "tok", "position": 3}
        doc = rf.document(key)
        doc["signals"][2]["validator"]["version"] = 9
        r = _post(h, "/api/rule-sets/validate", doc)
        assert r.json()["error"]["code"] == "RULE_FORM_UNAVAILABLE"
        doc = rf.document(key)
        doc["signals"][0]["kind"] = "telepathy"
        r = _post(h, "/api/rule-sets/validate", doc)
        assert r.json()["error"]["code"] in ("RULE_FORM_UNAVAILABLE", "RULE_SET_INVALID")
        assert "ZQ-" not in r.text


async def test_derived_needs_provenance_and_authored_equals_derived():
    async with cap.Harness() as h:
        key = bf.random_key()
        doc = rf.document(key)
        r = _post(h, "/api/rule-sets/validate", doc, origin="derived")
        assert (r.status_code, r.json()["error"]["code"]) == (422, "RULE_SET_INVALID")
        derived = rf.document(key, name="generated-derived")
        derived["provenance"] = {
            "kind": "derived", "derivation_id": "d1", "method": "test_side", "method_version": 1,
            "source_version_id": "v1", "label_column": "label", "seed": 7,
            "params_sha256": "a" * 64}
        a = _post(h, "/api/rule-sets/import", doc).json()["data"]
        d = _post(h, "/api/rule-sets/import", derived, origin="derived").json()["data"]
        assert (a["origin"], d["origin"]) == ("authored", "derived")
        got = h.client.get(f"/api/rule-sets/{d['id']}").json()["data"]
        assert got["provenance"]["derivation_id"] == "d1"
        text = {"text": "marker ZQ-1234 and 79927398713, zorbel quint"}
        ra = h.client.post(f"/api/rule-sets/{a['id']}/evaluate", json=text).json()["data"]
        rd = h.client.post(f"/api/rule-sets/{d['id']}/evaluate", json=text).json()["data"]
        assert ra["results"] == rd["results"]


async def test_a_code_rule_is_accepted_and_never_run(monkeypatch):
    """D41/D44: a set naming operator code imports; with no code document held (and no runner),
    its expression reports `rule_worker_unavailable` (skipped, unavailable) — never `fires: false`.
    With a document and a runner the route runs it synchronously (D57,
    `test_rule_code_stateless.py`)."""
    async with cap.Harness() as h:
        doc = rf.document(bf.random_key(), code_rule=True)
        r = _post(h, "/api/rule-sets/import", doc)
        assert r.status_code == 200, r.text
        rule_set_id = r.json()["data"]["id"]
        r = h.client.post(f"/api/rule-sets/{rule_set_id}/evaluate", json={"text": "ZQ-1234"})
        operator = [x for x in r.json()["data"]["results"] if x["signal"] == "operator"]
        assert operator and all(
            (x["outcome"], x["reason"], x["status"], x["fires"]) ==
            ("skipped", "rule_worker_unavailable", "unavailable", None) for x in operator)


async def test_get_list_and_delete():
    async with cap.Harness() as h:
        key = bf.random_key()
        a = _post(h, "/api/rule-sets/import", rf.document(key)).json()["data"]
        b = _post(h, "/api/rule-sets/import", rf.document(bf.random_key())).json()["data"]
        page = h.client.get("/api/rule-sets", params={"limit": 1}).json()
        assert len(page["data"]) == 1 and page["meta"]["next"]
        second = h.client.get("/api/rule-sets", params={"limit": 1,
                                                         "cursor": page["meta"]["next"]}).json()
        assert {page["data"][0]["id"], second["data"][0]["id"]} == {a["id"], b["id"]}
        assert h.client.get("/api/rule-sets", params={"behaviour": key}).json()["data"][0][
            "id"] == a["id"]
        detail = h.client.get(f"/api/rule-sets/{a['id']}").json()["data"]
        assert detail["vector_status"] == "passed" and "outputs" in detail["not_used_at_serving"]
        assert h.client.delete(f"/api/rule-sets/{a['id']}").status_code == 200
        r = h.client.get(f"/api/rule-sets/{a['id']}")
        assert (r.status_code, r.json()["error"]["code"]) == (404, "RULE_SET_NOT_FOUND")


async def test_attachments_refuse_another_behaviours_set_and_keep_history():
    async with cap.Harness() as h:
        key, other = bf.random_key(), bf.random_key()
        mine = _post(h, "/api/rule-sets/import", rf.document(key)).json()["data"]
        theirs = _post(h, "/api/rule-sets/import", rf.document(other)).json()["data"]
        r = h.client.put(f"/api/probes/behaviours/{key}/rule-signals",
                         json={"rule_set_ids": [theirs["id"]]})
        assert (r.status_code, r.json()["error"]["code"]) == (409, "RULE_SET_BEHAVIOUR_MISMATCH")
        r = h.client.put(f"/api/probes/behaviours/{key}/rule-signals",
                         json={"rule_set_ids": [mine["id"]], "signals": {mine["id"]: ["token"]}})
        assert r.status_code == 200, r.text
        assert r.json()["data"]["rule_set_ids"] == [mine["id"]]
        r = h.client.put(f"/api/probes/behaviours/{key}/rule-signals",
                         json={"rule_set_ids": [mine["id"]], "signals": {mine["id"]: ["nope"]}})
        assert r.json()["error"]["code"] == "RULE_SET_INVALID"
        # No armed probe: deleting the attached set detaches it, with a history row.
        assert h.client.delete(f"/api/rule-sets/{mine['id']}").status_code == 200
        view = h.client.get(f"/api/probes/behaviours/{key}/rule-signals").json()["data"]
        assert view["rule_set_ids"] == []
        assert [x["action"] for x in view["history"]] == ["rule_set_deleted", "attach"]
        assert await _count(h, BehaviourRuleSignalHistory) == 2
