"""`GET /api/probes/events` filters (FR-36.6). One test per filter, two rows differing ONLY there.

Each pair is otherwise identical, so a filter that is accepted and ignored returns both rows and
the test goes red — the assertion is that exactly the matching row comes back.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from millm.db.models.probe import ProbeEvent
from millm.db.models.probe_session import ProbeSession
from millm.db.repositories.probe_repository import ProbeRepository
from millm.services.probe_service import ProbeService
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture
from tests.unit.probe_fixtures import probe_definition
from tests.support.probe_event_rows import event_row

T = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def client(db):  # noqa: F811
    from fastapi.testclient import TestClient

    from millm.api.dependencies import get_db
    from millm.main import create_app

    app = create_app()

    async def _db():
        async with db() as session:
            yield session

    app.dependency_overrides[get_db] = _db
    return TestClient(app)


BASE = dict(request_id="r", scored=True, score=0.5, threshold=1.0, verdict=False, rung=2,
            window="all", origin="live", created_at=T)


async def _seed(db, a: dict, b: dict, *, behaviours=(None, None)):  # noqa: F811
    async with db() as session:
        ids = []
        for i, key in enumerate(behaviours):
            doc = probe_definition(name=f"p{i}")
            if key:
                doc["behaviour"] = {"key": key, "name": key, "version": 1}
            probe = await ProbeService(ProbeRepository(session)).import_definition(
                doc, raw_bytes=1, on_conflict="fail", origin="file"
            )
            ids.append(probe.id)
        for sid in ("ps_a", "ps_b"):
            session.add(ProbeSession(id=sid, source="header", opened_at=T, last_seen_at=T))
        await session.commit()
        rows = []
        for i, over in enumerate((a, b)):
            fields = {**BASE, "probe_id": ids[min(i, len(ids) - 1)], **over}
            row = ProbeEvent(**event_row(**fields))
            session.add(row)
            rows.append(row)
        await session.commit()
        return [r.id for r in rows]


def _ids(client, **params):
    r = client.get("/api/probes/events", params=params)
    assert r.status_code == 200, r.text
    return sorted(e["id"] for e in r.json()["data"])


async def test_session_id(client, db):  # noqa: F811
    a, b = await _seed(db, {"session_id": "ps_a"}, {"session_id": "ps_b"})
    assert _ids(client, session_id="ps_a") == [a]
    assert _ids(client) == sorted([a, b])


async def test_since_and_until(client, db):  # noqa: F811
    a, b = await _seed(db, {"created_at": T - timedelta(days=2)}, {"created_at": T})
    assert _ids(client, since=(T - timedelta(days=1)).isoformat()) == [b]
    assert _ids(client, until=(T - timedelta(days=1)).isoformat()) == [a]


@pytest.mark.parametrize("value, which", [("fired", 0), ("not_fired", 1)])
async def test_verdict(client, db, value, which):  # noqa: F811
    ids = await _seed(db, {"verdict": True, "score": 0.5}, {"verdict": False, "score": 0.5})
    assert _ids(client, verdict=value) == [ids[which]]


async def test_verdict_not_scored(client, db):  # noqa: F811
    a, b = await _seed(db, {"scored": False, "score": None, "verdict": None,
                            "not_scored_reason": "x"}, {"scored": True, "verdict": None})
    assert _ids(client, verdict="not_scored") == [a]


async def test_window(client, db):  # noqa: F811
    a, b = await _seed(db, {"window": "prompt"}, {"window": "response"})
    assert _ids(client, window="prompt") == [a]


async def test_origin(client, db):  # noqa: F811
    a, b = await _seed(db, {"origin": "batch", "batch_id": "b1", "batch_line": 1},
                       {"origin": "live"})
    assert _ids(client, origin="live") == [b]


async def test_behaviour(client, db):  # noqa: F811
    a, b = await _seed(db, {}, {}, behaviours=("pii", "stakes"))
    assert _ids(client, behaviour="stakes") == [b]


async def test_score_min_and_max(client, db):  # noqa: F811
    a, b = await _seed(db, {"score": 0.2}, {"score": 0.8})
    assert _ids(client, score_min=0.5) == [b]
    assert _ids(client, score_max=0.5) == [a]


def test_an_unknown_verdict_value_is_refused(client):
    assert client.get("/api/probes/events", params={"verdict": "maybe"}).status_code == 422
