"""The session read API through the real app (FR-36.11, FR-36.14 timeline, US-36.1).

Wiring controls (036 FTDD §10):
* removing `app.include_router(probe_sessions_router)`               → `test_probe_session_routes_are_served` red
* including it AFTER `probes_router` (`/{probe_id}` swallows `sessions`) → the same test red (404)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from millm.db.models.probe import ProbeEvent
from millm.db.models.probe_session import ProbeSession, ProbeSessionTurn
from millm.db.repositories.probe_repository import ProbeRepository
from millm.services import probe_sessions
from millm.services.probe_service import ProbeService
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture
from tests.unit.probe_fixtures import probe_definition
from tests.support.probe_event_rows import event_row

NOW = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def client(db, monkeypatch):  # noqa: F811
    from fastapi.testclient import TestClient

    from millm.api.dependencies import get_db
    from millm.main import create_app

    monkeypatch.setattr(probe_sessions, "_now", lambda: NOW)
    app = create_app()

    async def _db():
        async with db() as session:
            yield session

    app.dependency_overrides[get_db] = _db
    return TestClient(app)


async def _import(db, name, behaviour=None):  # noqa: F811
    doc = probe_definition(name=name)
    if behaviour is not None:
        doc["behaviour"] = {"key": behaviour, "name": behaviour.upper(), "version": 2}
    async with db() as session:
        return await ProbeService(ProbeRepository(session)).import_definition(
            doc, raw_bytes=1, on_conflict="fail", origin="file"
        )


async def _seed_timeline(db):  # noqa: F811
    """Three turns × two behaviours. Only (turn 2, `pii`) fires; every cell has its own score."""
    pii = await _import(db, "pii-probe", behaviour="pii")
    stakes = await _import(db, "stakes-probe")  # no behaviour: grouped by its own name
    async with db() as session:
        session.add(ProbeSession(id="ps_timeline01", source="header", client_session_id="c",
                                 served_model="tiny", opened_at=NOW, last_seen_at=NOW, n_turns=3))
        await session.commit()
        scores = {(1, pii.id): 0.1, (2, pii.id): 2.5, (3, pii.id): 0.3,
                  (1, stakes.id): 0.4, (2, stakes.id): 0.5, (3, stakes.id): 0.6}
        for index in (1, 2, 3):
            session.add(ProbeSessionTurn(
                session_id="ps_timeline01", request_id=f"req-{index}", turn_index=index,
                prefix_hash="p" * 64, weak_hash="w" * 64, continuation_hash="c" * 64,
                has_transcript=(index == 3), created_at=NOW,
            ))
            for probe in (pii, stakes):
                score = scores[(index, probe.id)]
                session.add(ProbeEvent(**event_row(
                    probe_id=probe.id, request_id=f"req-{index}", scored=True, score=score,
                    threshold=1.0, verdict=score > 1.0, rung=2, window="all",
                    session_id="ps_timeline01", turn_index=index,
                    # 039: the stored result carries the armed behaviour, as `record` writes it.
                    behaviour={"key": "pii", "version": 2} if probe is pii else None,
                )))
        await session.commit()
    return pii, stakes


def test_probe_session_routes_are_served(client):
    paths = client.app.openapi()["paths"]
    assert set(paths["/api/probes/sessions"]) >= {"get"}
    assert set(paths["/api/probes/sessions/{session_id}"]) >= {"get"}
    # ⚠ Served, not merely declared: before the probes router, so `sessions` is not a probe id.
    r = client.get("/api/probes/sessions")
    assert r.status_code == 200, r.text
    assert r.json()["success"] is True and r.json()["data"] == []


async def test_the_timeline_marks_the_fired_cell_at_its_coordinates(client, db):  # noqa: F811
    pii, stakes = await _seed_timeline(db)
    detail = client.get("/api/probes/sessions/ps_timeline01").json()["data"]
    assert [c["column"] for c in detail["columns"]] == ["pii", "stakes-probe"]
    assert detail["columns"][0]["behaviour"] == {"key": "pii", "name": "PII", "version": 2}
    assert detail["columns"][1]["behaviour"] is None
    grid = {
        (t["turn_index"], v["column"]): v for t in detail["turns"] for v in t["verdicts"]
    }
    assert len(grid) == 6
    fired = sorted(k for k, v in grid.items() if v["verdict"] is True)
    assert fired == [(2, "pii")]
    assert grid[(3, "stakes-probe")]["score"] == pytest.approx(0.6)
    assert [t["has_transcript"] for t in detail["turns"]] == [False, False, True]
    # ⚠ The timeline says WHETHER there is text, never what it is.
    flat = repr(detail)
    assert "user_text" not in flat and "reply_text" not in flat


async def test_the_list_summarises_each_behaviour(client, db):  # noqa: F811
    await _seed_timeline(db)
    rows = client.get("/api/probes/sessions").json()["data"]
    assert len(rows) == 1
    row = rows[0]
    assert (row["id"], row["n_turns"], row["n_transcribed"], row["open"]) == (
        "ps_timeline01", 3, 1, True)
    assert row["behaviours"]["pii"] == {"n_fired": 1, "n_scored": 3, "max_score": 2.5}
    assert row["behaviours"]["stakes-probe"]["n_fired"] == 0
    assert "user_text" not in repr(rows) and "reply_text" not in repr(rows)


async def test_session_list_filters(client, db):  # noqa: F811
    await _seed_timeline(db)
    async with db() as session:
        session.add(ProbeSession(id="ps_other00001", source="fingerprint", served_model="big",
                                 opened_at=NOW - timedelta(days=3),
                                 last_seen_at=NOW - timedelta(days=3), n_turns=0))
        await session.commit()

    def ids(**params):
        return [r["id"] for r in client.get("/api/probes/sessions", params=params).json()["data"]]

    assert ids() == ["ps_timeline01", "ps_other00001"]
    assert ids(source="fingerprint") == ["ps_other00001"]
    assert ids(served_model="tiny") == ["ps_timeline01"]
    assert ids(since=(NOW - timedelta(days=1)).isoformat()) == ["ps_timeline01"]
    assert ids(until=(NOW - timedelta(days=2)).isoformat()) == ["ps_other00001"]
    assert ids(behaviour="pii") == ["ps_timeline01"]
    assert ids(fired_only="true") == ["ps_timeline01"]
    assert ids(limit=1, offset=1) == ["ps_other00001"]


def test_an_unknown_session_is_404(client):
    r = client.get("/api/probes/sessions/ps_nope")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "PROBE_SESSION_NOT_FOUND"


async def test_probe_and_event_summaries_carry_the_behaviour(client, db):  # noqa: F811
    pii, stakes = await _seed_timeline(db)
    probes = {p["id"]: p for p in client.get("/api/probes").json()["data"]}
    assert probes[pii.id]["behaviour"] == {"key": "pii", "name": "PII", "version": 2}
    assert probes[stakes.id]["behaviour"] is None
    events = client.get("/api/probes/events", params={"session_id": "ps_timeline01"}).json()
    by_probe = {e["probe_id"]: e for e in events["data"]}
    assert by_probe[pii.id]["behaviour"]["key"] == "pii"
    assert by_probe[pii.id]["session_id"] == "ps_timeline01"
    assert by_probe[stakes.id]["behaviour"] is None


async def test_probe_summaries_carry_the_source_probe_id(client, db):  # noqa: F811
    """036 FR-36.14: `definition.provenance.probe_id` (miStudio's id), deliberately DIFFERENT from
    miLLM's own id in the fixture, so a summary that echoed the local id would fail."""
    doc = probe_definition(name="sourced")
    doc["provenance"]["probe_id"] = "pm_studio_123"
    async with db() as session:
        probe = await ProbeService(ProbeRepository(session)).import_definition(
            doc, raw_bytes=1, on_conflict="fail", origin="file")
    plain = await _import(db, "unsourced")
    assert probe.id != "pm_studio_123"
    rows = {p["id"]: p for p in client.get("/api/probes").json()["data"]}
    assert rows[probe.id]["source_probe_id"] == "pm_studio_123"
    assert rows[plain.id]["source_probe_id"] is None
    assert client.get(f"/api/probes/{probe.id}").json()["data"]["source_probe_id"] == "pm_studio_123"


async def test_the_session_list_is_walked_by_cursor(client, db):  # noqa: F811
    """039 FR-39.15: keyset pages, each session exactly once, newest first."""
    from datetime import timedelta

    async with db() as session:
        for i in range(5):
            session.add(ProbeSession(id=f"ps_cursor{i:05d}", source="header",
                                     client_session_id=f"c{i}", served_model="tiny",
                                     opened_at=NOW, last_seen_at=NOW - timedelta(minutes=i % 3),
                                     n_turns=1))
        await session.commit()
    seen, cursor = [], None
    while True:
        body = client.get("/api/probes/sessions",
                          params={"limit": 2, **({"cursor": cursor} if cursor else {})}).json()
        seen.extend(r["id"] for r in body["data"])
        cursor = body["meta"]["next"]
        if cursor is None:
            break
    assert sorted(seen) == [f"ps_cursor{i:05d}" for i in range(5)] and len(seen) == 5
