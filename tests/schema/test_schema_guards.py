"""The migrated schema and the ORM models must describe the same database.

WHY THIS EXISTS

Production is built by the migrations; every test builds its schema from the models
with ``create_all`` (on SQLite). On 2026-09-13 a brand-new production database was
compared with the models and ``alembic check`` reported index names, a unique
constraint and a TIMESTAMP-vs-timestamptz type that disagree — and it could not see
at all that three QuantizationType values the API accepts do not exist in the
Postgres enum. Nothing compared the two, and CI ran no PostgreSQL.

TWO GUARDS, because each is blind where the other sees:

* G1 ``test_alembic_drift_matches_the_ratchet`` — Alembic's own comparison against
  ``alembic_drift_ratchet.json``. The ratchet may only SHRINK: an unlisted difference
  fails, and a listed difference that no longer occurs fails until its line is removed.
* G2 ``test_the_orm_builds_the_same_schema_as_the_migrations`` — a pg_catalog snapshot
  of a ``create_all`` database against the migrated one, which sees partial-index
  predicates, enum labels, CHECK constraints and triggers. Strict xfail until the
  reconciliation lands.

Each run migrates its own scratch database from empty. ``MILLM_REQUIRE_PG=1`` (set in
CI) turns an unreachable server into a FAILURE: a skipped guard reads as a passing one.
"""

import ast
import asyncio
import importlib
import json
import os
import pkgutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy.ext.asyncio import create_async_engine

from millm.db.schema_parity import alembic_drift, create_orm_schema, diff, snapshot

REPO = Path(__file__).resolve().parents[2]
RATCHET = Path(__file__).with_name("alembic_drift_ratchet.json")
PROBE_TABLE = "schema_guard_probe"
PG_BASE = os.environ.get(
    "MILLM_TEST_PG_BASE", "postgresql+asyncpg://postgres:devpassword@localhost:5432"
)


async def _pg_available() -> bool:
    try:
        engine = create_async_engine(f"{PG_BASE}/postgres", isolation_level="AUTOCOMMIT")
        async with engine.begin() as conn:
            await conn.execute(sa.text("SELECT 1"))
        await engine.dispose()
        return True
    except Exception:
        return False


async def _admin(sql: str, params: dict | None = None) -> None:
    engine = create_async_engine(f"{PG_BASE}/postgres", isolation_level="AUTOCOMMIT")
    try:
        async with engine.begin() as conn:
            await conn.execute(sa.text(sql), params or {})
    finally:
        await engine.dispose()


async def _drop(name: str) -> None:
    await _admin(
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
        "WHERE datname = :name AND pid <> pg_backend_pid()",
        {"name": name},
    )
    await _admin(f'DROP DATABASE IF EXISTS "{name}"')


async def _with_conn(db: str, fn, *args, write: bool = False):
    engine = create_async_engine(f"{PG_BASE}/{db}")
    try:
        async with engine.begin() if write else engine.connect() as conn:
            return await conn.run_sync(lambda sync_conn: fn(sync_conn, *args))
    finally:
        await engine.dispose()


def _require_pg() -> None:
    if asyncio.run(_pg_available()):
        return
    message = f"PostgreSQL is not reachable at {PG_BASE}"
    if os.environ.get("MILLM_REQUIRE_PG") == "1":
        pytest.fail(f"{message} and MILLM_REQUIRE_PG=1 — the schema guards did NOT run")
    pytest.skip(f"{message} — the schema guards need a real server")


def _script_head() -> str:
    config = Config(str(REPO / "alembic.ini"))
    config.set_main_option("script_location", str(REPO / "millm" / "db" / "migrations"))
    return ScriptDirectory.from_config(config).get_current_head()


def _current_revision(conn: sa.engine.Connection) -> list[str]:
    return conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalars().all()


@pytest.fixture(scope="module")
def migrated_db():
    _require_pg()
    name = f"schema_guard_{uuid.uuid4().hex[:8]}"
    asyncio.run(_admin(f'CREATE DATABASE "{name}"'))
    try:
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=REPO,
            capture_output=True,
            text=True,
            env={**os.environ, "DATABASE_URL": f"{PG_BASE}/{name}"},
            timeout=600,
        )
        assert result.returncode == 0, result.stderr[-3000:]
        current = asyncio.run(_with_conn(name, _current_revision))
        assert current == [_script_head()], f"migrated to {current}, not head"
        yield name
    finally:
        asyncio.run(_drop(name))


def test_alembic_drift_matches_the_ratchet(migrated_db):
    current = set(asyncio.run(_with_conn(migrated_db, alembic_drift)))
    recorded = set(json.loads(RATCHET.read_text()))

    new = sorted(current - recorded)
    assert not new, (
        f"{len(new)} NEW difference(s) between the migrated schema and the models. Fix the "
        f"model or add the migration in the same commit — do not add them to "
        f"{RATCHET.name}:\n" + "\n".join(new)
    )
    fixed = sorted(recorded - current)
    assert not fixed, (
        f"{len(fixed)} difference(s) in {RATCHET.name} no longer occur. Delete those lines so "
        "the ratchet only shrinks:\n" + "\n".join(fixed)
    )


def test_the_drift_comparison_can_see_a_difference(migrated_db):
    """NEGATIVE CONTROL: a comparison that cannot report a difference passes forever."""
    probe = sa.MetaData()
    sa.Table(PROBE_TABLE, probe, sa.Column("id", sa.Integer, primary_key=True))
    signatures = asyncio.run(_with_conn(migrated_db, alembic_drift, probe))
    assert f"add_table:{PROBE_TABLE}" in signatures
    assert "remove_table:models" in signatures


def test_the_ratchet_is_sorted_and_unique():
    entries = json.loads(RATCHET.read_text())
    assert entries == sorted(set(entries)), f"keep {RATCHET.name} sorted and free of duplicates"


def test_alembic_registers_every_model_module():
    """alembic runs env.py in a fresh interpreter; every model module must register there."""
    import millm.db.models as models_package
    from millm.db.base import Base

    for module in pkgutil.iter_modules(models_package.__path__):
        importlib.import_module(f"millm.db.models.{module.name}")

    code = (
        "import json; from millm.db.alembic_support import target_metadata; "
        "print(json.dumps(sorted(target_metadata().tables)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    alembic_tables = set(json.loads(result.stdout.strip().splitlines()[-1]))
    missing = sorted(set(Base.metadata.tables) - alembic_tables)
    assert not missing, f"alembic's metadata lacks tables defined under millm/db/models: {missing}"


def test_env_py_uses_the_shared_metadata_and_compare_options():
    tree = ast.parse((REPO / "millm" / "db" / "migrations" / "env.py").read_text())

    assigns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "target_metadata" for t in node.targets)
    ]
    assert len(assigns) == 1, "env.py must assign target_metadata exactly once"
    value = assigns[0].value
    assert isinstance(value, ast.Call) and getattr(value.func, "id", None) == (
        "load_target_metadata"
    ), "env.py's target_metadata must come from millm.db.alembic_support.target_metadata()"

    configures = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "configure"
    ]
    assert len(configures) == 2, "expected the offline and online context.configure calls"
    for call in configures:
        assert any(
            kw.arg is None and isinstance(kw.value, ast.Name) and kw.value.id == "COMPARE_OPTIONS"
            for kw in call.keywords
        ), f"context.configure at line {call.lineno} must pass **COMPARE_OPTIONS"
        assert any(kw.arg == "target_metadata" for kw in call.keywords)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "P1a-P1c of the Alembic plan reconcile the models with the migrations; strict so "
        "it cannot quietly start passing"
    ),
)
def test_the_orm_builds_the_same_schema_as_the_migrations(migrated_db):
    scratch = f"orm_parity_{uuid.uuid4().hex[:8]}"
    asyncio.run(_admin(f'CREATE DATABASE "{scratch}"'))
    try:
        asyncio.run(_with_conn(scratch, create_orm_schema, write=True))
        orm = asyncio.run(_with_conn(scratch, snapshot))
    finally:
        asyncio.run(_drop(scratch))
    migrated = asyncio.run(_with_conn(migrated_db, snapshot))

    differences = diff(migrated, orm, "migrated", "orm")
    assert not differences, f"{len(differences)} difference(s):\n" + "\n".join(differences)
