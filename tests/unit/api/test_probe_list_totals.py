"""`GET /api/probes` carries each probe's own result totals and its model's layer count (2026-10-29).

Operator: "The number of results is no longer displayed on the tile until I open it." The count
the tile once showed came from the shared newest-results list and was wrong; the list route now
answers `results_total`, `results_fired` and `last_result_at` per probe, counted for the whole page
in ONE grouped query, plus `n_layers` so a tile can say "layer 4 of 16".

Three probes whose numbers all differ, so totals read from the wrong probe cannot agree by
accident; the fired count is seeded beside a not-fired, a no-threshold (`verdict=None`) and an
unscored row, so counting anything but `verdict is True` goes red.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import event as sa_event

from millm.db.models.probe import ProbeEvent
from millm.db.repositories.probe_repository import ProbeRepository
from millm.services.probe_service import ProbeService
from tests.support.probe_event_rows import event_row
from tests.unit.api.test_probe_event_filters import client  # noqa: F401 - fixture
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture
from tests.unit.probe_fixtures import probe_definition

T = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)
ROW = dict(request_id="r", scored=True, score=0.5, threshold=1.0, verdict=False, rung=2,
           window="all", origin="live")


async def _seed(db):  # noqa: F811
    """Probes a (5 results, 2 fired), b (1 result, none fired), c (none). Returns their ids."""
    async with db() as session:
        ids = {}
        for name, n_layers in (("a", 16), ("b", 24), ("c", 32)):
            doc = probe_definition(name=name)
            doc["model"]["n_layers"] = n_layers
            probe = await ProbeService(ProbeRepository(session)).import_definition(
                doc, raw_bytes=1, on_conflict="fail", origin="file"
            )
            ids[name] = probe.id
        rows = [
            (ids["a"], {"verdict": True}, T),
            (ids["a"], {"verdict": True, "window": "prompt"}, T + timedelta(minutes=5)),
            (ids["a"], {"verdict": False}, T + timedelta(minutes=1)),
            (ids["a"], {"verdict": None, "threshold": None}, T + timedelta(minutes=2)),
            (ids["a"], {"scored": False, "score": None, "verdict": None,
                        "not_scored_reason": "MODEL_NOT_LOADED"}, T + timedelta(minutes=3)),
            (ids["b"], {"verdict": False}, T + timedelta(hours=1)),
        ]
        for probe_id, over, when in rows:
            session.add(ProbeEvent(**event_row(**{**ROW, "probe_id": probe_id, **over,
                                                  "created_at": when})))
        await session.commit()
        return ids


def _when(value):
    """The served time as an aware datetime (SQLite drops the zone; Postgres keeps it)."""
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _by_id(client):  # noqa: F811
    response = client.get("/api/probes")
    assert response.status_code == 200, response.text
    return {p["id"]: p for p in response.json()["data"]}


async def test_each_probe_carries_its_own_totals(client, db):  # noqa: F811
    ids = await _seed(db)
    listed = _by_id(client)

    a, b, c = listed[ids["a"]], listed[ids["b"]], listed[ids["c"]]
    assert (a["results_total"], a["results_fired"]) == (5, 2)
    assert _when(a["last_result_at"]) == T + timedelta(minutes=5)
    assert (b["results_total"], b["results_fired"]) == (1, 0)
    assert _when(b["last_result_at"]) == T + timedelta(hours=1)
    # No results: zero, and no time — never absent, never a guess.
    assert (c["results_total"], c["results_fired"], c["last_result_at"]) == (0, 0, None)


async def test_totals_match_what_the_probes_own_results_list_pages_through(client, db):  # noqa: F811
    """The tile's number and the panel's list must never disagree: same set, same count."""
    ids = await _seed(db)
    listed = _by_id(client)
    for probe_id in ids.values():
        events = client.get("/api/probes/events", params={"probe_id": probe_id, "limit": 500})
        rows = events.json()["data"]
        assert listed[probe_id]["results_total"] == len(rows)
        assert listed[probe_id]["results_fired"] == sum(1 for e in rows if e["verdict"] is True)


async def test_one_grouped_query_for_the_whole_list(client, db):  # noqa: F811
    """⚠ Not one per probe. Counts every statement that reads `probe_events` during the call."""
    await _seed(db)
    engine = db.kw["bind"].sync_engine
    reads: list[str] = []

    def _count(_conn, _cursor, statement, *_args):  # noqa: ANN001
        if "probe_events" in statement and statement.lstrip().upper().startswith("SELECT"):
            reads.append(statement)

    sa_event.listen(engine, "before_cursor_execute", _count)
    try:
        listed = _by_id(client)
    finally:
        sa_event.remove(engine, "before_cursor_execute", _count)
    assert len(listed) == 3
    assert len(reads) == 1, reads
    assert "GROUP BY" in reads[0].upper()


async def test_n_layers_is_served(client, db):  # noqa: F811
    ids = await _seed(db)
    listed = _by_id(client)
    assert {name: listed[pid]["n_layers"] for name, pid in ids.items()} == {
        "a": 16, "b": 24, "c": 32,
    }
    one = client.get(f"/api/probes/{ids['b']}").json()["data"]
    assert one["n_layers"] == 24
    assert (one["results_total"], one["results_fired"]) == (1, 0)


async def test_totals_follow_a_clear(client, db):  # noqa: F811
    ids = await _seed(db)
    assert client.delete("/api/probes/events", params={"probe_id": ids["a"]}).status_code == 200
    listed = _by_id(client)
    assert (listed[ids["a"]]["results_total"], listed[ids["a"]]["last_result_at"]) == (0, None)
    assert listed[ids["b"]]["results_total"] == 1
