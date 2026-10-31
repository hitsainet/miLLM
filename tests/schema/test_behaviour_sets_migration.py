"""Migration 034 (039 S1: behaviour sets, limits in behaviours) up and down, on SQLite and PostgreSQL.

What it must do to existing data (FTASKS 1.3, TD39-10, TD39-14):

* `behaviour_version` is backfilled from each stored definition — `behaviour.version: 3` → 3; no
  behaviour block, or an uncatalogued key (`version: null`) → NULL;
* every non-NULL per-model `max_armed` (a PROBE count) is written to the history, then NULLed —
  a probe count is not a behaviour count — and the downgrade restores it from that history row.

Control (recorded in `0xcc/reviews/039_implementation_controls_2026-10-24.md`): delete the NULL-ing
statement in `upgrade()` → `test_*_nulls_max_armed_*` goes red ("migration keeps `max_armed`").
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.schema.test_probe_sessions_migration import _column, _exec, fresh_db  # noqa: F401
from tests.schema.test_weights_origin_migration import _alembic, _query

MIGRATION = (Path(__file__).resolve().parents[2] / "millm" / "db" / "migrations" / "versions"
             / "034_behaviour_sets.py")


def _module():
    spec = importlib.util.spec_from_file_location("m034", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _definition(behaviour):
    doc = {"name": "x"}
    if behaviour is not None:
        doc["behaviour"] = behaviour
    return doc


SEEDED = {
    "pr_v3": {"key": "b-gen-a", "name": "a", "version": 3},
    "pr_uncat": {"key": "b-gen-a", "name": "a", "version": None},
    "pr_none": None,
}


# ── SQLite ───────────────────────────────────────────────────────────────────────


@pytest.fixture
def sqlite_033(tmp_path):
    """A SQLite database in the 033 shape: the ORM schema with 034 taken off by its downgrade."""
    from millm.db.base import Base
    import millm.db.models  # noqa: F401 - every table

    engine = sa.create_engine(f"sqlite:///{tmp_path / 'm.db'}")
    Base.metadata.create_all(engine)
    _run(engine, "downgrade")
    yield engine
    engine.dispose()


def _run(engine, step: str) -> None:
    with engine.begin() as conn:
        with Operations.context(MigrationContext.configure(conn)):
            getattr(_module(), step)()


def _seed(conn) -> None:
    for probe_id, behaviour in SEEDED.items():
        conn.execute(sa.text(
            "INSERT INTO probes (id, name, definition, hf_id, d_model, n_layers, layer, rule, "
            "scope, rung, basis, streamable, armed, threshold_revision, detector_digest_source) "
            "VALUES (:id, :id, :d, 'tiny/llama', 16, 2, 1, 'mean', 'all', 2, 'residual', true, "
            "false, 1, 'unknown')"
        ), {"id": probe_id, "d": json.dumps(_definition(behaviour))})
    conn.execute(sa.text(
        "INSERT INTO models (id, name, source, repo_id, quantization, cache_path, status, "
        "gguf_label, trust_remote_code, locked, created_at, updated_at) VALUES (7, 'm', "
        "'huggingface', 'tiny/llama', 'FP16', '/x', 'ready', '', false, false, "
        "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
    ))
    conn.execute(sa.text(
        "INSERT INTO model_probe_limits (model_id, max_armed, updated_at) VALUES "
        "(7, 12, CURRENT_TIMESTAMP)"
    ))


def test_sqlite_backfills_the_version_and_nulls_max_armed_into_the_history(sqlite_033):
    engine = sqlite_033
    with engine.begin() as conn:
        _seed(conn)
    _run(engine, "upgrade")
    with engine.connect() as conn:
        versions = dict(conn.execute(sa.text(
            "SELECT id, behaviour_version FROM probes ORDER BY id")).all())
        assert versions == {"pr_none": None, "pr_uncat": None, "pr_v3": 3}
        assert conn.execute(sa.text(
            "SELECT max_armed FROM model_probe_limits WHERE model_id = 7")).scalar_one() is None
        history = conn.execute(sa.text(
            "SELECT model_id, old, new, source FROM model_probe_limits_history")).all()
    assert len(history) == 1
    model_id, old, new, source = history[0]
    as_dict = (lambda v: json.loads(v) if isinstance(v, str) else v)
    assert (model_id, as_dict(old), as_dict(new), source) == (
        7, {"max_armed": 12}, {"max_armed": None}, "migration-034")

    _run(engine, "downgrade")
    with engine.connect() as conn:
        assert conn.execute(sa.text(
            "SELECT max_armed FROM model_probe_limits WHERE model_id = 7")).scalar_one() == 12
        columns = {row[1] for row in conn.execute(sa.text("PRAGMA table_info(probes)")).all()}
    assert "behaviour_version" not in columns and "last_cost" not in columns


# ── PostgreSQL (the production dialect; skipped loudly without a server) ─────────


def _pg_probe(probe_id: str, definition: dict) -> str:
    return (
        "INSERT INTO probes (id, name, definition, hf_id, d_model, n_layers, layer, rule, scope, "
        f"rung) VALUES ('{probe_id}', '{probe_id}', '{json.dumps(definition)}'::jsonb, 'm', 8, 2, "
        "0, 'mean', 'all', 2)"
    )


def test_pg_034_backfills_nulls_max_armed_and_comes_down(fresh_db):  # noqa: F811
    _alembic(fresh_db, "upgrade", "033")
    for probe_id, behaviour in SEEDED.items():
        asyncio.run(_exec(fresh_db, _pg_probe(probe_id, _definition(behaviour))))
    asyncio.run(_exec(fresh_db, (
        "INSERT INTO models (id, name, source, repo_id, quantization, cache_path, status, "
        "gguf_label, locked, trust_remote_code, created_at, updated_at) VALUES (7, 'm', "
        "'huggingface', 'tiny/llama', 'FP16', '/x', 'ready', '', false, false, now(), now())"
    )))
    asyncio.run(_exec(fresh_db, "INSERT INTO model_probe_limits (model_id, max_armed) VALUES (7, 12)"))

    _alembic(fresh_db, "upgrade", "034")
    assert asyncio.run(_query(fresh_db, "SELECT id, behaviour_version FROM probes ORDER BY id")) == [
        ("pr_none", None), ("pr_uncat", None), ("pr_v3", 3),
    ]
    assert asyncio.run(_query(
        fresh_db, "SELECT max_armed FROM model_probe_limits WHERE model_id = 7")) == [(None,)]
    assert asyncio.run(_query(fresh_db, (
        "SELECT model_id, old->>'max_armed', new->>'max_armed', source "
        "FROM model_probe_limits_history"))) == [(7, "12", None, "migration-034")]
    assert asyncio.run(_query(fresh_db, (
        "SELECT indexname FROM pg_indexes WHERE indexname = 'ix_probes_behaviour_key_version'"
    ))) == [("ix_probes_behaviour_key_version",)]

    _alembic(fresh_db, "downgrade", "033")
    assert asyncio.run(_query(
        fresh_db, "SELECT max_armed FROM model_probe_limits WHERE model_id = 7")) == [(12,)]
    assert asyncio.run(_query(fresh_db, _column("probes", "behaviour_version"))) == []
    assert asyncio.run(_query(fresh_db, _column("model_probe_limits_history", "id"))) == []
    _alembic(fresh_db, "upgrade", "head")
