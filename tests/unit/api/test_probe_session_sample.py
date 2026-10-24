"""`POST /api/probes/sessions/sample` — seeded, stratified, never padded (036 phase 4, FR-36.12).

⚠ The fixture INTERLEAVES fired and not-fired rows in time, so an implementation that took the
first N by time (or by id) would return a lopsided sample and fail the strata assertions.

Controls (recorded in `0xcc/reviews/036_037_controls_2026-10-21.md`):
* the seeded key → id order                → `test_the_same_seed_returns_the_same_rows_and_another_differs` red
* a short stratum padded from another      → `test_a_short_stratum_is_reported_short_never_padded` red
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from millm.db.models.probe import ProbeEvent
from millm.db.models.probe_session import ProbeSession, ProbeSessionTurn
from millm.db.repositories.probe_repository import ProbeRepository
from millm.services.probe_service import ProbeService
from tests.unit.api.test_probe_session_routes import client  # noqa: F401 - fixture
from tests.unit.f28_fixtures import db  # noqa: F401 - fixture
from tests.unit.probe_fixtures import probe_definition
from tests.support.probe_event_rows import event_row

T = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
#: 40 turns, INTERLEAVED in time: odd = fired (2.0), even = not fired (0.2), every 5th just below
#: the bar (0.95, not fired, inside the 10% band). Available: 16 fired, 8 near, 16 not fired.
N = 40


def _score(i: int) -> tuple[float, bool]:
    if i % 5 == 0:
        return 0.95, False
    return (2.0, True) if i % 2 else (0.2, False)


async def _seed(db, *, transcript_every: int = 1):  # noqa: F811
    async with db() as session:
        doc = probe_definition(name="pii-probe")
        doc["behaviour"] = {"key": "pii", "name": "PII", "version": 1}
        doc["provenance"]["probe_id"] = "pm_studio_77"
        probe = await ProbeService(ProbeRepository(session)).import_definition(
            doc, raw_bytes=1, on_conflict="fail", origin="file")
        other = await ProbeService(ProbeRepository(session)).import_definition(
            probe_definition(name="other"), raw_bytes=1, on_conflict="fail", origin="file")
        session.add(ProbeSession(id="ps_s", source="header", opened_at=T, last_seen_at=T))
        await session.commit()
        for i in range(N):
            score, fired = _score(i)
            session.add(ProbeSessionTurn(
                session_id="ps_s", request_id=f"r{i}", turn_index=i + 1, prefix_hash="p",
                weak_hash="w", has_transcript=(i % transcript_every == 0),
                created_at=T + timedelta(minutes=i)))
            for p in (probe, other):
                session.add(ProbeEvent(**event_row(
                    probe_id=p.id, request_id=f"r{i}", scored=True, score=score, threshold=1.0,
                    verdict=fired, rung=2, window="all", session_id="ps_s", turn_index=i + 1,
                    created_at=T + timedelta(minutes=i))))
        await session.commit()
        return probe.id


def _sample(client, **body):  # noqa: F811
    payload = {"behaviour": "pii", "strata": {"fired": 5, "near_threshold": 3, "not_fired": 5},
               "seed": 7, **body}
    r = client.post("/api/probes/sessions/sample", json=payload)
    assert r.status_code == 200, r.text
    return r.json()["data"]


async def test_the_strata_are_disjoint_and_filled_from_the_right_rows(client, db):  # noqa: F811
    await _seed(db)
    data = _sample(client)
    by = {}
    for row in data["turns"]:
        by.setdefault(row["stratum"], []).append(row)
    assert {k: len(v) for k, v in by.items()} == {"fired": 5, "near_threshold": 3, "not_fired": 5}
    assert all(r["verdict"] is True and r["score"] == 2.0 for r in by["fired"])
    assert all(r["score"] == 0.95 for r in by["near_threshold"])
    assert all(r["score"] == 0.2 and r["verdict"] is False for r in by["not_fired"])
    assert len({r["event_id"] for r in data["turns"]}) == len(data["turns"])
    assert {r["source_probe_id"] for r in data["turns"]} == {"pm_studio_77"}
    assert data["strata"]["fired"] == {"requested": 5, "returned": 5, "short": 0, "available": 16}


async def test_the_same_seed_returns_the_same_rows_and_another_differs(client, db):  # noqa: F811
    await _seed(db)
    first = [r["event_id"] for r in _sample(client)["turns"]]
    again = [r["event_id"] for r in _sample(client)["turns"]]
    other = [r["event_id"] for r in _sample(client, seed=8)["turns"]]
    assert first == again
    assert first != other
    # Not the first N by time/id: the seeded order is not the insertion order.
    fired = [r for r in _sample(client)["turns"] if r["stratum"] == "fired"]
    assert [r["event_id"] for r in fired] != sorted(r["event_id"] for r in fired)


async def test_a_short_stratum_is_reported_short_never_padded(client, db):  # noqa: F811
    await _seed(db)
    data = _sample(client, strata={"fired": 0, "near_threshold": 50, "not_fired": 1})
    assert data["strata"]["near_threshold"] == {
        "requested": 50, "returned": 8, "short": 42, "available": 8}
    assert sum(1 for r in data["turns"] if r["stratum"] == "near_threshold") == 8
    assert sum(1 for r in data["turns"] if r["stratum"] == "not_fired") == 1


async def test_only_turns_with_text_unless_asked_otherwise(client, db):  # noqa: F811
    await _seed(db, transcript_every=4)
    with_text = _sample(client, strata={"fired": 100, "near_threshold": 100, "not_fired": 100})
    assert len(with_text["turns"]) == N // 4
    assert all(r["has_transcript"] for r in with_text["turns"])
    every = _sample(client, require_transcript=False,
                    strata={"fired": 100, "near_threshold": 100, "not_fired": 100})
    assert len(every["turns"]) == N


async def test_probe_id_targets_one_probe(client, db):  # noqa: F811
    pid = await _seed(db)
    data = _sample(client, behaviour=None, probe_id=pid)
    assert {r["probe_id"] for r in data["turns"]} == {pid}


@pytest.mark.parametrize("body", [
    {"strata": {"fired": 1}, "seed": 1},                                         # no target
    {"behaviour": "pii", "probe_id": "x", "strata": {"fired": 1}, "seed": 1},    # both
    {"behaviour": "pii", "strata": {"fired": 1}, "seeed": 1},                    # misspelt seed
    {"behaviour": "pii", "strata": {"fired": 1, "maybe": 2}, "seed": 1},         # unknown stratum
])
def test_malformed_requests_are_refused(client, body):  # noqa: F811
    assert client.post("/api/probes/sessions/sample", json=body).status_code == 422


async def test_a_fired_row_inside_the_band_is_fired_not_near(client, db):  # noqa: F811
    """Priority fired > near_threshold > not_fired: the strata are disjoint."""
    from millm.api.routes.management.probe_sessions import stratum_of

    class E:
        def __init__(self, score, verdict, threshold=1.0):
            self.score, self.verdict, self.threshold = score, verdict, threshold

    assert stratum_of(E(1.05, True), 0.1) == "fired"
    assert stratum_of(E(1.05, False), 0.1) == "near_threshold"
    assert stratum_of(E(0.5, False), 0.1) == "not_fired"
    assert stratum_of(E(0.5, None, threshold=None), 0.1) is None  # ranked, decided nothing


async def test_each_turn_carries_the_fields_miStudio_038_reads(client, db):  # noqa: F811
    """Field by field against miStudio 038's spot-check client (`millm_read.SessionReader.sample` and
    `probe_spot_check.execute`): `data.turns[]` and `data.strata.<name>.{requested,returned,short}`."""
    await _seed(db)
    data = _sample(client, strata={"fired": 1, "near_threshold": 1, "not_fired": 1})
    assert "turns" in data and "rows" not in data
    for turn in data["turns"]:
        assert {"session_id", "request_id", "turn_index", "event_id", "score", "threshold",
                "verdict", "stratum", "window", "threshold_revision"} <= set(turn)
    for name in ("fired", "near_threshold", "not_fired"):
        assert {"requested", "returned", "short"} <= set(data["strata"][name])
