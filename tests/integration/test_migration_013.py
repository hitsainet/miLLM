"""Feature 19 task 6.6 — migration 013, upgrade AND downgrade, RUN not read.

The downgrade's ORDERING is load-bearing and invisible to inspection.
Recreating the partial unique index `uq_circuits_active` while TWO circuits are
active FAILS — the index cannot be built over rows that violate it. So the
downgrade must deactivate all but the most recently activated circuit FIRST,
then drop the table, then recreate the index.

A downgrade that merely reverses the upgrade statements bricks any database
that used the feature it is downgrading away from, and reading the code will
not tell you: the failure only appears against a MULTI-ACTIVE state, which is
exactly the state this feature makes possible and no earlier test could create.

Requires real PostgreSQL: the constraint under test is a partial unique index
over concurrently-active rows, and SQLite's `create_all` path does not exercise
the alembic scripts at all. Skipped rather than faked when unavailable — a
green skip is honest; a SQLite stand-in would report this covered while testing
something else.
"""

import os
import subprocess
import uuid

import pytest
import sqlalchemy as sa

pytestmark = pytest.mark.asyncio

PG_BASE = os.environ.get(
    "MILLM_TEST_PG_BASE", "postgresql+asyncpg://postgres:devpassword@localhost:5432"
)
PG_SYNC_BASE = PG_BASE.replace("+asyncpg", "")


async def _pg_available() -> bool:
    from sqlalchemy.ext.asyncio import create_async_engine

    try:
        engine = create_async_engine(f"{PG_BASE}/postgres", isolation_level="AUTOCOMMIT")
        async with engine.begin() as conn:
            await conn.execute(sa.text("SELECT 1"))
        await engine.dispose()
        return True
    except Exception:
        return False


def _alembic(db: str, *args: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "DATABASE_URL": f"{PG_BASE}/{db}",
        "DATABASE_URL_SYNC": f"{PG_SYNC_BASE}/{db}",
    }
    return subprocess.run(
        ["venv/bin/python", "-m", "alembic", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )


@pytest.fixture
async def scratch_db():
    """A throwaway database, dropped afterwards even on failure."""
    if not await _pg_available():
        pytest.skip("PostgreSQL not reachable — migration 013 needs a real server")

    from sqlalchemy.ext.asyncio import create_async_engine

    name = f"f19mig_{uuid.uuid4().hex[:8]}"
    admin = create_async_engine(f"{PG_BASE}/postgres", isolation_level="AUTOCOMMIT")
    async with admin.begin() as conn:
        await conn.execute(sa.text(f'CREATE DATABASE "{name}"'))
    await admin.dispose()
    try:
        yield name
    finally:
        admin = create_async_engine(f"{PG_BASE}/postgres", isolation_level="AUTOCOMMIT")
        async with admin.begin() as conn:
            await conn.execute(
                sa.text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :n AND pid <> pg_backend_pid()"
                ),
                {"n": name},
            )
            await conn.execute(sa.text(f'DROP DATABASE IF EXISTS "{name}"'))
        await admin.dispose()


async def _seed_two_active(db: str) -> None:
    """The state ONLY Feature 19 can produce: two active circuits."""
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(f"{PG_BASE}/{db}", isolation_level="AUTOCOMMIT")
    async with engine.begin() as conn:
        for cid, layer in (("cA", 10), ("cB", 13)):
            await conn.execute(
                sa.text(
                    "INSERT INTO circuits (id,name,circuit_meta,rung,edge_count,"
                    "layers,per_sae_warnings,serveable,is_active,provenance) VALUES "
                    "(:id, :n, CAST('{}' AS jsonb), 2, 0, CAST(:l AS jsonb), "
                    "CAST('[]' AS jsonb), true, true, CAST('{}' AS jsonb))"
                ),
                {"id": cid, "n": cid, "l": f"[{layer}]"},
            )
        # cB claimed LATER, so it is the one the downgrade must keep.
        await conn.execute(
            sa.text(
                "INSERT INTO circuit_layer_claims (circuit_id,layer,composed,claimed_at)"
                " VALUES ('cA',10,false, now() - interval '1 hour'),"
                " ('cB',13,false, now())"
            )
        )
    await engine.dispose()


async def _query(db: str, sql: str):
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(f"{PG_BASE}/{db}", isolation_level="AUTOCOMMIT")
    async with engine.begin() as conn:
        rows = (await conn.execute(sa.text(sql))).fetchall()
    await engine.dispose()
    return rows


class TestMigration013:
    async def test_upgrade_creates_the_claims_table_and_drops_single_active(
        self, scratch_db
    ):
        result = _alembic(scratch_db, "upgrade", "head")
        assert result.returncode == 0, result.stderr[-2000:]

        table = await _query(scratch_db, "SELECT to_regclass('circuit_layer_claims')")
        assert table[0][0] is not None, "the claims table was not created"

        idx = await _query(
            scratch_db,
            "SELECT indexname FROM pg_indexes WHERE indexname='uq_circuits_active'",
        )
        assert idx == [], (
            "the single-active index survived the upgrade — concurrent serving "
            "is impossible while it stands"
        )

        live = await _query(
            scratch_db,
            "SELECT indexname FROM pg_indexes "
            "WHERE indexname='uq_circuit_layer_claim_live'",
        )
        assert live, "the per-layer exclusive index was not created"

    async def test_downgrade_RUNS_against_a_seeded_two_active_state(
        self, scratch_db
    ):
        """The whole point. Two active circuits is a state that only exists
        BECAUSE of this migration, so the downgrade is the only code that ever
        has to cope with it."""
        assert _alembic(scratch_db, "upgrade", "head").returncode == 0
        await _seed_two_active(scratch_db)

        actives = await _query(
            scratch_db, "SELECT count(*) FROM circuits WHERE is_active"
        )
        assert actives[0][0] == 2, "the fixture did not create the state under test"

        result = _alembic(scratch_db, "downgrade", "012")
        assert result.returncode == 0, (
            "the downgrade FAILED against a multi-active database — it bricks "
            f"any deployment that used this feature:\n{result.stderr[-2000:]}"
        )

        rows = await _query(
            scratch_db, "SELECT id, is_active FROM circuits ORDER BY id"
        )
        assert dict(rows) == {"cA": False, "cB": True}, (
            "the downgrade kept the wrong circuit — it must keep the MOST "
            "RECENTLY ACTIVATED one, which is what the operator last asked for"
        )

        table = await _query(scratch_db, "SELECT to_regclass('circuit_layer_claims')")
        assert table[0][0] is None, "the claims table survived the downgrade"

        idx = await _query(
            scratch_db,
            "SELECT indexname FROM pg_indexes WHERE indexname='uq_circuits_active'",
        )
        assert idx, "the single-active index was not restored"

    async def test_the_deactivation_step_is_LOAD_BEARING(self, scratch_db):
        """BR-005 reachability: the test must FAIL when the step is removed.

        Verified by patching the migration on disk, running the downgrade, and
        restoring — the same negative-control discipline used for code. Without
        this, `test_downgrade_RUNS...` could pass for the wrong reason (e.g. if
        the seed silently failed) and the ordering would be undefended.
        """
        from pathlib import Path

        assert _alembic(scratch_db, "upgrade", "head").returncode == 0
        await _seed_two_active(scratch_db)

        path = Path("millm/db/migrations/versions/013_add_circuit_layer_claims.py")
        original = path.read_text()
        anchor = "    for circuit_id, _latest in active[1:]:"
        assert anchor in original, "the deactivation loop moved — re-aim this control"

        try:
            path.write_text(
                original.replace(anchor, "    for circuit_id, _latest in []:", 1)
            )
            result = _alembic(scratch_db, "downgrade", "012")
        finally:
            path.write_text(original)

        assert result.returncode != 0, (
            "the downgrade SUCCEEDED without deactivating the extra circuits — "
            "the ordering that makes it safe is not actually doing anything"
        )
        assert "uq_circuits_active" in result.stderr, (
            "the downgrade failed for some other reason than the index it is "
            "ordered to protect"
        )

    async def test_upgrade_BACKFILLS_the_active_circuits_claims(self, scratch_db):
        """A circuit already serving holds its layers in fact, so it must hold
        them in the new model too — otherwise the first activation after the
        migration sees an unclaimed layer and the incumbent silently loses the
        protection it had a moment earlier."""
        assert _alembic(scratch_db, "upgrade", "012").returncode == 0

        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(f"{PG_BASE}/{scratch_db}", isolation_level="AUTOCOMMIT")
        async with engine.begin() as conn:
            await conn.execute(
                sa.text(
                    "INSERT INTO circuits (id,name,circuit_meta,rung,edge_count,"
                    "layers,per_sae_warnings,serveable,is_active,provenance) VALUES "
                    "('cLive','live', CAST('{}' AS jsonb), 2, 0, "
                    "CAST('[10, 13]' AS jsonb), CAST('[]' AS jsonb), true, true, "
                    "CAST('{}' AS jsonb))"
                )
            )
        await engine.dispose()

        assert _alembic(scratch_db, "upgrade", "013").returncode == 0

        claims = await _query(
            scratch_db,
            "SELECT circuit_id, layer FROM circuit_layer_claims "
            "WHERE released_at IS NULL ORDER BY layer",
        )
        assert [(r[0], r[1]) for r in claims] == [("cLive", 10), ("cLive", 13)], (
            "the already-active circuit's layers were not backfilled, so the "
            "next activation would see them as free"
        )

    async def test_round_trip(self, scratch_db):
        assert _alembic(scratch_db, "upgrade", "head").returncode == 0
        assert _alembic(scratch_db, "downgrade", "012").returncode == 0
        assert _alembic(scratch_db, "upgrade", "head").returncode == 0

        table = await _query(scratch_db, "SELECT to_regclass('circuit_layer_claims')")
        assert table[0][0] is not None
