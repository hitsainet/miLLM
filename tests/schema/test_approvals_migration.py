"""ENH-008 L9: migration 037 (`approval_requests`) goes up and comes down on PostgreSQL, its CHECK
constraints refuse a status or an action outside the lists, and those lists equal the model's.
ENH-006 D76: migration 038 adds `telemetry_write` to the action list; 037 keeps the six it shipped.
ENH-006 D79: migration 039 adds the nullable `batch_id` ("Approve all settings changes").

The list comparison needs no database and always runs. The round trip skips loudly when no
PostgreSQL is reachable (and fails under `MILLM_REQUIRE_PG=1`).
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

from millm.db.models.approval import APPROVAL_ACTION_NAMES, APPROVAL_STATUSES
from tests.schema.test_schema_guards import PG_BASE
from tests.schema.test_weights_origin_migration import _admin, _alembic, _pg_available, _query

VERSIONS = Path(__file__).resolve().parents[2] / "millm" / "db" / "migrations" / "versions"
MIGRATION = VERSIONS / "037_approval_requests.py"
MIGRATION_038 = VERSIONS / "038_approval_telemetry_write.py"
MIGRATION_039 = VERSIONS / "039_approval_batch_id.py"
#: The six actions 037 shipped with (ENH-008), written out: 037 is a record and never changes.
ACTIONS_037 = ("model_load", "model_unload", "model_delete",
               "probe_limits_write", "capacity_measure", "code_rule_import")


def _migration(path: Path = MIGRATION):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_check_lists_equal_the_model_tuples():
    m = _migration()
    assert (m.revision, m.down_revision) == ("037", "036")
    assert tuple(m.STATUSES) == APPROVAL_STATUSES
    assert tuple(m.ACTIONS) == ACTIONS_037
    m38 = _migration(MIGRATION_038)
    assert (m38.revision, m38.down_revision) == ("038", "037")
    assert tuple(m38.PREVIOUS) == ACTIONS_037
    assert tuple(m38.ACTIONS) == APPROVAL_ACTION_NAMES
    assert set(APPROVAL_ACTION_NAMES) - set(ACTIONS_037) == {"telemetry_write"}
    m39 = _migration(MIGRATION_039)
    assert (m39.revision, m39.down_revision) == ("039", "038")
    assert (m39.TABLE, m39.COLUMN) == ("approval_requests", "batch_id")


def test_the_model_column_matches_039():
    from millm.db.models.approval import ApprovalRequest

    column = ApprovalRequest.__table__.c.batch_id
    assert column.nullable is True and column.type.length == 40


def test_the_action_registry_equals_the_check_list():
    from millm.api.approval_gate import APPROVAL_ACTIONS

    assert tuple(APPROVAL_ACTIONS) == APPROVAL_ACTION_NAMES


async def _exec(db: str, sql: str) -> None:
    engine = create_async_engine(f"{PG_BASE}/{db}")
    try:
        async with engine.begin() as conn:
            await conn.execute(sa.text(sql))
    finally:
        await engine.dispose()


@pytest.fixture(scope="module")
def fresh_db():
    if not asyncio.run(_pg_available()):
        message = f"PostgreSQL is not reachable at {PG_BASE}"
        if os.environ.get("MILLM_REQUIRE_PG") == "1":
            pytest.fail(f"{message} and MILLM_REQUIRE_PG=1 — this guard did NOT run")
        pytest.skip(f"{message} — a migration round trip needs a real server")
    name = f"approvals_{uuid.uuid4().hex[:8]}"
    asyncio.run(_admin(f'CREATE DATABASE "{name}"'))
    try:
        yield name
    finally:
        asyncio.run(_admin(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


_COLUMNS = ("SELECT column_name, data_type, is_nullable, character_maximum_length FROM "
            "information_schema.columns WHERE table_name = 'approval_requests' "
            "ORDER BY column_name")

_ROW = ("INSERT INTO approval_requests (id, action, executor_key, summary, payload, "
        "request_digest, requested_by, status, expires_at) VALUES ('{id}', '{action}', 'k', "
        "'{{}}', '{{}}', '{digest}', 'agent:x', '{status}', NOW())")


def _row(status: str = "pending", action: str = "model_load") -> str:
    return _ROW.format(id="apr_" + uuid.uuid4().hex, action=action, digest="0" * 64,
                       status=status)


def test_037_creates_the_table_refuses_bad_values_and_comes_down(fresh_db):
    _alembic(fresh_db, "upgrade", "036")
    assert asyncio.run(_query(fresh_db, _COLUMNS)) == []

    _alembic(fresh_db, "upgrade", "037")
    columns = {c[0]: c[1:] for c in asyncio.run(_query(fresh_db, _COLUMNS))}
    assert columns["id"] == ("character varying", "NO", 40)
    assert columns["secret_payload"] == ("json", "YES", None)
    assert columns["reason"] == ("character varying", "YES", 1000)
    assert columns["expires_at"] == ("timestamp with time zone", "NO", None)
    assert len(columns) == 16
    indexes = asyncio.run(_query(
        fresh_db, "SELECT indexname FROM pg_indexes WHERE tablename = 'approval_requests' "
                  "ORDER BY indexname"))
    assert indexes == [("approval_requests_pkey",), ("ix_approval_requests_status_created_at",)]
    for status in APPROVAL_STATUSES:
        asyncio.run(_exec(fresh_db, _row(status=status)))
    for action in ACTIONS_037:
        asyncio.run(_exec(fresh_db, _row(action=action)))
    with pytest.raises(Exception, match="ck_approval_requests_action"):
        asyncio.run(_exec(fresh_db, _row(action="telemetry_write")))  # 038's, not yet
    with pytest.raises(Exception, match="ck_approval_requests_status"):
        asyncio.run(_exec(fresh_db, _row(status="approved")))
    with pytest.raises(Exception, match="ck_approval_requests_action"):
        asyncio.run(_exec(fresh_db, _row(action="steer")))

    # The startup reset's statement is valid SQL on PostgreSQL (its JSON literal included).
    from millm.main import STALE_STATE_RESETS

    (reset,) = [s for name, s, _ in STALE_STATE_RESETS if name == "failed_interrupted_approvals"]
    asyncio.run(_exec(fresh_db, reset))
    left = asyncio.run(_query(fresh_db, "SELECT count(*) FROM approval_requests "
                                        "WHERE status = 'executing'"))
    assert left == [(0,)]
    codes = asyncio.run(_query(fresh_db, "SELECT error->>'code' FROM approval_requests "
                                         "WHERE error IS NOT NULL"))
    assert codes == [("interrupted_by_restart",)]

    _alembic(fresh_db, "downgrade", "036")
    assert asyncio.run(_query(fresh_db, _COLUMNS)) == []
    _alembic(fresh_db, "upgrade", "037")


def test_038_accepts_telemetry_write_and_its_downgrade_removes_those_rows(fresh_db):
    """ENH-006 D76: after 038 the seventh action is accepted (and an unknown one still refused);
    the downgrade deletes the `telemetry_write` rows, restores 037's list, and keeps the rest."""
    _alembic(fresh_db, "upgrade", "038")
    for action in APPROVAL_ACTION_NAMES:
        asyncio.run(_exec(fresh_db, _row(action=action)))
    with pytest.raises(Exception, match="ck_approval_requests_action"):
        asyncio.run(_exec(fresh_db, _row(action="steer")))
    _alembic(fresh_db, "downgrade", "037")
    left = asyncio.run(_query(fresh_db, "SELECT action FROM approval_requests ORDER BY action"))
    assert sorted(a for (a,) in left if a != "model_load") == sorted(
        a for a in ACTIONS_037 if a != "model_load")
    assert all(a != "telemetry_write" for (a,) in left)
    with pytest.raises(Exception, match="ck_approval_requests_action"):
        asyncio.run(_exec(fresh_db, _row(action="telemetry_write")))
    _alembic(fresh_db, "upgrade", "038")


def test_039_adds_a_nullable_batch_id_and_its_downgrade_keeps_the_rows(fresh_db):
    """ENH-006 D79: after 039 a row may carry a batch id (and existing rows have none); the
    downgrade drops the column and keeps every request."""
    _alembic(fresh_db, "upgrade", "039")
    columns = {c[0]: c[1:] for c in asyncio.run(_query(fresh_db, _COLUMNS))}
    assert columns["batch_id"] == ("character varying", "YES", 40)
    before = asyncio.run(_query(fresh_db, "SELECT count(*) FROM approval_requests"))
    assert asyncio.run(_query(fresh_db, "SELECT count(*) FROM approval_requests "
                                        "WHERE batch_id IS NOT NULL")) == [(0,)]
    asyncio.run(_exec(fresh_db, _row(action="telemetry_write")))
    asyncio.run(_exec(fresh_db, "UPDATE approval_requests SET batch_id = 'bat_" + "0" * 32
                      + "' WHERE action = 'telemetry_write'"))
    _alembic(fresh_db, "downgrade", "038")
    assert "batch_id" not in {c[0] for c in asyncio.run(_query(fresh_db, _COLUMNS))}
    after = asyncio.run(_query(fresh_db, "SELECT count(*) FROM approval_requests"))
    assert after[0][0] == before[0][0] + 1
    _alembic(fresh_db, "upgrade", "039")
