"""Migration 032 (Feature 038): stored events into contract form, once and for good (FR-38.10).

Run on SQLite (in-process, `upgrade()`/`downgrade()` through alembic's `Operations`) and on
PostgreSQL (the real `alembic` chain, `MILLM_REQUIRE_PG=1` in the integration job):

* upgrade, read, downgrade, upgrade again: every `result_id` is byte-stable (OI-1: UUIDv5 over the
  row's own fields, `created_at` normalised to UTC so the two databases agree);
* backfilled outcomes and codes match the table, and NO free text survives;
* `decision_digest` is set only where the event's `threshold_revision` is the probe's current one;
* `migrated = true`, `event_time_source = 'inserted'`, the stored `result` validates strictly;
* PostgreSQL: the import-time digest equals one recomputed from the JSONB row read back (T-S1-14),
  and a 200-character exception no longer loses the request's events (D38.32).

Control M10: mint `result_id` at read time (`uuid4` in the migration) — the stability test red.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import jsonschema
import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from millm.api.schemas.probe_result import strict_schema

REPO = Path(__file__).resolve().parents[2]
MIGRATION = REPO / "millm" / "db" / "migrations" / "versions" / "032_probe_result_contract.py"
STRICT = jsonschema.Draft202012Validator(strict_schema())


def _module():
    spec = importlib.util.spec_from_file_location("m032", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _definition(threshold: float = 1.0) -> dict:
    from tests.unit.score_fixtures import tiny_definition

    doc = tiny_definition("probe-m", threshold=threshold)
    doc["behaviour"] = {"key": "b-mig", "name": "mig", "version": 1}
    return doc


#: (not_scored_reason as stored before 032, scored, score, expected outcome, expected code)
EVENTS = [
    (None, True, 2.5, "completed", None),
    ("judge_traffic", False, None, "skipped", "judge_traffic"),
    ("encoder_failed: RuntimeError: " + "x" * 160, False, None, "failed", "encoder_failed"),
    ("prompt_boundary_invalid: -5", False, None, "failed", "prompt_boundary_invalid"),
    ("prompt_boundary_conflict: 6 then 9", False, None, "failed", "prompt_boundary_conflict"),
    ("continuous_batching", False, None, "skipped", "continuous_batching"),
    ("head_mismatch", False, None, "failed", "head_mismatch"),
    (None, True, float("nan"), "failed", "non_finite_score"),
]


# ── SQLite ───────────────────────────────────────────────────────────────────────


@pytest.fixture
def sqlite_031(tmp_path):
    """A SQLite database in the 031 shape: the ORM schema with 032 taken off by its downgrade."""
    from millm.db.base import Base
    import millm.db.models  # noqa: F401 - every table

    engine = sa.create_engine(f"sqlite:///{tmp_path / 'm.db'}")
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        with Operations.context(MigrationContext.configure(conn)):
            _module().downgrade()
    yield engine
    engine.dispose()


def _run(engine, step: str) -> None:
    with engine.begin() as conn:
        with Operations.context(MigrationContext.configure(conn)):
            getattr(_module(), step)()


def _seed_sqlite(engine) -> None:
    with engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO probes (id, name, definition, hf_id, d_model, n_layers, layer, rule, "
            "scope, rung, threshold, threshold_revision, basis, streamable, armed, revision) "
            "VALUES ('pr_00000000000a', 'probe-m', :d, 'tiny/llama', 16, 2, 1, 'mean', 'all', 2, "
            "1.0, 2, 'residual', 1, 0, :rev)"
        ), {"d": json.dumps(_definition()), "rev": "0" * 40})
        for i, (reason, scored, score, _o, _c) in enumerate(EVENTS):
            conn.execute(sa.text(
                "INSERT INTO probe_events (probe_id, request_id, scored, not_scored_reason, "
                "score, threshold, verdict, rung, window, provisional, threshold_revision, "
                "origin, created_at, summary) VALUES ('pr_00000000000a', :rid, :scored, :reason, "
                ":score, 1.0, :verdict, 2, 'all', 0, :rev, 'live', :created, :summary)"
            ), {"rid": f"chatcmpl-{i}", "scored": scored, "reason": reason, "score": score,
                # the pre-038 `_summary`: the reason verbatim, exception text included
                "summary": f"not scored: {reason}" if not scored else f"score {score:.4f}",
                "verdict": bool(score and score == score and score >= 1.0),
                "rev": 2 if i % 2 == 0 else 1,
                "created": datetime(2026, 10, 1, 12, 0, i, 123456)})
        # An orphaned event (its probe deleted with foreign keys off): digest `unknown`.
        conn.execute(sa.text(
            "INSERT INTO probe_events (probe_id, request_id, scored, score, threshold, rung, "
            "window, provisional, threshold_revision, origin, created_at) VALUES "
            "('pr_0000000000ff', 'chatcmpl-x', 1, 0.5, 1.0, 2, 'all', 0, 1, 'live', :c)"
        ), {"c": datetime(2026, 10, 1, 12, 1)})


def _events(engine) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(sa.text(
            "SELECT id, probe_id, request_id, result_id, outcome, reason, not_scored_reason, "
            "decision_digest, migrated, event_time_source, result, score, threshold_revision, "
            "summary "
            "FROM probe_events ORDER BY id"
        )).mappings().all()
    return [dict(r) for r in rows]


def test_sqlite_upgrade_maps_every_form_and_ids_survive_a_round_trip(sqlite_031):
    engine = sqlite_031
    _seed_sqlite(engine)
    _run(engine, "upgrade")
    first = _events(engine)
    assert first == _events(engine), "ids must be stable across reads"
    for row, (_r, _s, _sc, outcome, code) in zip(first, EVENTS):
        assert row["outcome"] == outcome and row["reason"] == code, row
        assert row["not_scored_reason"] in (None, code), "no free text survives (D38.9)"
        if outcome != "completed":
            # review R1: the summary carried the same free text and was not rewritten
            assert row["summary"] == f"not scored: {code}", row["summary"]
        assert row["migrated"] in (1, True) and row["event_time_source"] == "inserted"
        result = json.loads(row["result"]) if isinstance(row["result"], str) else row["result"]
        assert result["result_id"] == row["result_id"] and result["migrated"] is True
        STRICT.validate({**result, "source_event_id": row["id"]})
        # decision digest only under the bar the event was judged against (revision 2 is current)
        assert (row["decision_digest"] is not None) == (row["threshold_revision"] == 2)
    orphan = first[-1]
    result = json.loads(orphan["result"]) if isinstance(orphan["result"], str) else orphan["result"]
    assert result["detector_digest"] == "unknown" and result["detector_digest_source"] == "unknown"
    with engine.connect() as conn:
        digests = conn.execute(sa.text(
            "SELECT detector_digest, detector_digest_source FROM probes")).all()
    from millm.services.probe_digest import detector_digest

    assert digests == [(detector_digest(_definition()), "computed")]

    _run(engine, "downgrade")
    _run(engine, "upgrade")
    again = _events(engine)
    assert [r["result_id"] for r in again] == [r["result_id"] for r in first], (
        "a downgrade and upgrade must re-mint the same result ids (FR-38.10)"
    )


def test_created_at_is_normalised_before_the_id_is_minted():
    m = _module()
    naive = datetime(2026, 10, 1, 12, 0, 0, 123456)
    aware = naive.replace(tzinfo=timezone.utc)
    assert m.result_id_for(7, "pr_x", "r", naive) == m.result_id_for(7, "pr_x", "r", aware)
    assert uuid.UUID(m.result_id_for(7, "pr_x", None, aware)).version == 5


def test_the_frozen_serializer_matches_the_service():
    """The migration carries a FROZEN copy; at 032 it must equal the live module."""
    from millm.services.probe_digest import digests_of

    assert _module().digests(_definition()) == digests_of(_definition())


# ── PostgreSQL ───────────────────────────────────────────────────────────────────


def _pg():
    from tests.schema.test_weights_origin_migration import _admin, _alembic, _pg_available, _query

    return _admin, _alembic, _pg_available, _query


@pytest.fixture(scope="module")
def pg_db():
    _admin, _alembic, _pg_available, _query = _pg()
    from tests.schema.test_schema_guards import PG_BASE

    if not asyncio.run(_pg_available()):
        message = f"PostgreSQL is not reachable at {PG_BASE}"
        if os.environ.get("MILLM_REQUIRE_PG") == "1":
            pytest.fail(f"{message} and MILLM_REQUIRE_PG=1 — this guard did NOT run")
        pytest.skip(f"{message} — a migration round trip needs a real server")
    name = f"probe_result_{uuid.uuid4().hex[:8]}"
    asyncio.run(_admin(f'CREATE DATABASE "{name}"'))
    try:
        yield name
    finally:
        asyncio.run(_admin(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


async def _exec(db: str, sql: str, params: dict | None = None) -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    from tests.schema.test_schema_guards import PG_BASE

    engine = create_async_engine(f"{PG_BASE}/{db}")
    try:
        async with engine.begin() as conn:
            await conn.execute(sa.text(sql), params or {})
    finally:
        await engine.dispose()


def test_pg_upgrade_round_trip_and_jsonb_digest(pg_db):
    _admin, _alembic, _pg_available, _query = _pg()
    _alembic(pg_db, "upgrade", "031")
    asyncio.run(_exec(pg_db, (
        "INSERT INTO probes (id, name, definition, hf_id, d_model, n_layers, layer, rule, scope, "
        "rung, threshold, threshold_revision, revision) VALUES ('pr_00000000000a', 'probe-m', "
        "CAST(:d AS jsonb), 'tiny/llama', 16, 2, 1, 'mean', 'all', 2, 1.0, 2, :rev)"
    ), {"d": json.dumps(_definition()), "rev": "0" * 40}))
    for i, (reason, scored, score, _o, _c) in enumerate(EVENTS):
        asyncio.run(_exec(pg_db, (
            "INSERT INTO probe_events (probe_id, request_id, scored, not_scored_reason, score, "
            "threshold, rung, \"window\", provisional, threshold_revision, origin, created_at) "
            "VALUES ('pr_00000000000a', :rid, :scored, :reason, :score, 1.0, 2, 'all', false, "
            ":rev, 'live', :created)"
        ), {"rid": f"chatcmpl-{i}", "scored": scored, "reason": reason[:64] if reason else None,
            "score": score, "rev": 2 if i % 2 == 0 else 1,
            "created": datetime(2026, 10, 1, 12, 0, i, 123456, tzinfo=timezone.utc)}))
    _alembic(pg_db, "upgrade", "032")
    read = "SELECT id, result_id, outcome, reason, result FROM probe_events ORDER BY id"
    first = asyncio.run(_query(pg_db, read))
    assert [(r[2], r[3]) for r in first] == [(o, c) for *_x, o, c in EVENTS]
    for row in first:
        STRICT.validate({**row[4], "source_event_id": row[0]})
    # T-S1-14: the digest recomputed from the JSONB row equals the one computed at upgrade.
    from millm.services.probe_digest import digests_of

    stored = asyncio.run(_query(pg_db, "SELECT definition, detector_digest, decision_digest "
                                       "FROM probes"))[0]
    assert digests_of(stored[0]) == (stored[1], stored[2])
    # The SQLite and PostgreSQL ids agree for the same row (UTC normalisation).
    m = _module()
    assert first[0][1] == m.result_id_for(
        first[0][0], "pr_00000000000a", "chatcmpl-0",
        datetime(2026, 10, 1, 12, 0, 0, 123456, tzinfo=timezone.utc))

    _alembic(pg_db, "downgrade", "031")
    _alembic(pg_db, "upgrade", "032")
    again = asyncio.run(_query(pg_db, read))
    assert [r[1] for r in again] == [r[1] for r in first]
    _alembic(pg_db, "upgrade", "head")


def test_pg_a_long_exception_no_longer_loses_the_requests_events(pg_db):
    """D38.32: `encoder_failed: <200 characters>` overflowed `String(64)` on PostgreSQL and the
    whole `create_many` failed. Through the real service, every event of the request is written."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from millm.db.repositories.probe_repository import ProbeEventRepository, ProbeRepository
    from millm.services.probe_event_service import ProbeEventService
    from millm.services.probe_runtime import Verdict
    from tests.schema.test_schema_guards import PG_BASE

    _admin, _alembic, _pg_available, _query = _pg()
    _alembic(pg_db, "upgrade", "head")

    async def run() -> int:
        engine = create_async_engine(f"{PG_BASE}/{pg_db}")
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        try:
            async with factory() as session:
                await session.execute(sa.text(
                    "INSERT INTO probes (id, name, definition, hf_id, d_model, n_layers, layer, "
                    "rule, scope, rung) VALUES ('pr_0000000000bb', 'long', '{}'::jsonb, 'm', 8, 2, "
                    "0, 'mean', 'all', 2) ON CONFLICT DO NOTHING"))
                await session.commit()
                service = ProbeEventService(ProbeRepository(session), ProbeEventRepository(session))
                service._emit_events = lambda payloads: None
                verdicts = [
                    Verdict(probe_id="pr_0000000000bb", name="long", rung=2,
                            rung_language="detects on unseen tasks", scored=False,
                            not_scored_reason="encoder_failed", window=w)
                    for w in ("all", "prompt")
                ]
                return await service.record("chatcmpl-long", verdicts)
        finally:
            await engine.dispose()

    assert asyncio.run(run()) == 2
    rows = asyncio.run(_query(pg_db, "SELECT reason FROM probe_events WHERE request_id = "
                                     "'chatcmpl-long'"))
    assert rows == [("encoder_failed",), ("encoder_failed",)]


def test_an_unlisted_stored_reason_migrates_as_a_failure_never_a_skip():
    """Review R1: the builder reports a code outside the list as `failed`; the migration said
    `skipped` for the same row, so one defect read two ways depending on when it was recorded."""
    m = _module()
    event = dict(id=1, probe_id="pr_x", request_id="r", scored=False,
                 not_scored_reason="some_retired_code: detail", score=None, threshold=1.0,
                 verdict=None, rung=2, n_scored_tokens=0, window="all", provisional=False,
                 threshold_revision=1, served_model=None, origin="live", batch_id=None,
                 batch_line=None, session_id=None, turn_index=None,
                 created_at=datetime(2026, 10, 1, tzinfo=timezone.utc))
    values = m.migrated_row(event, None)
    assert values["outcome"] == "failed" and values["reason"] == "some_retired_code"
    assert values["summary"] == "not scored: some_retired_code"
