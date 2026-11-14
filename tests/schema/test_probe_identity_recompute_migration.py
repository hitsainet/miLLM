"""Migration 033: stored identity digests equal the case-file algorithm's (038 R-039, FTASKS 11.4).

* the FROZEN copy inside 033 reproduces every case of ITS ERA in miStudio's `probe-identity-cases.json`
  (pinned by name hash; later cases are listed in `POST_033` with a verdict) and
  every miStudio-computed edge in `tests/fixtures/probe_identity_edges.json`, and equals the live
  `probe_digest` on all of them (a migration must not import the service, so the copy is pinned);
* SQLite: rows stored under 032's reading are recomputed, agreeing rows and `producer` rows are
  untouched, an undigestible definition becomes `unknown`, and the moved ids are logged;
* PostgreSQL (`MISERVER_REQUIRE_PG=1`): a probe stored at 031 gets 032's digest, which differs from
  miStudio's for an edge definition; after `upgrade 033` the stored digests equal the service's,
  recomputed from the JSONB row read back.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from miserver.services.probe_digest import digests_of
from tests.support.env_flags import flag
from tests.support.package_root import package_path

REPO = Path(__file__).resolve().parents[2]
VERSIONS = package_path("db", "migrations", "versions")
CASES = json.loads((REPO / "docs" / "schemas" / "probe-identity-cases.json").read_text())["cases"]
EDGES = json.loads((REPO / "tests" / "fixtures" / "probe_identity_edges.json").read_text())["edges"]
EDGE = {e["name"]: e for e in EDGES}


def _module(name: str):
    spec = importlib.util.spec_from_file_location(f"m{name[:3]}", VERSIONS / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def m033():
    return _module("033_probe_identity_recompute.py")


def m032():
    return _module("032_probe_result_contract.py")


def test_it_follows_032():
    assert (m033().revision, m033().down_revision) == ("033", "032")


#: ⚠ 033 IS HISTORY, AND THE CASE FILE IS NOT. The frozen copy in 033 is the projection as it stood on
#: 2026-10-23; the shared case file keeps gaining cases for keys classified later (miStudio G38's
#: `aggregation.skip_template_prefix`, 2026-11-08). Rows stored before a key existed never carry it, so
#: 033 must reproduce the cases of ITS era — not every case written after it. Asking it to failed CI on
#: `9aed77b7` ("Schema guards"), and the fix is never to edit a migration.
#:
#: THE ERA IS PINNED, NOT INFERRED: the sorted names of every case 033 must reproduce, as they stood
#: when 033 was written (`0a7c6413`, 70 cases), hashed below. Every case added since must be named in
#: `POST_033` and given a verdict, so a new case cannot become silently exempt: an unlisted one changes
#: the era's hash and fails `test_the_033_era_is_the_case_set_033_was_written_against`.
ERA_033_SHA256 = "1518cb816b26ecd8212dc1173c7728a1191e3084de06f985c6e5e686aeb3b3c0"
ERA_033_COUNT = 70
#: Cases written after 033, each with what 033 must do with it:
#:   "reproduces" — the new key at its default is not projected, so 033's digest still matches;
#:   "differs"    — the new key is projected by the live digest and unknown to 033, so the detector
#:                  digest MUST differ (the decision digest is untouched).
POST_033 = {
    "added-after-default-aggregation.skip_template_prefix": "reproduces",
    "added-after-aggregation.skip_template_prefix": "differs",
}
ERA_033 = [c for c in CASES if c["name"] not in POST_033]
REPRODUCED = ERA_033 + [c for c in CASES if POST_033.get(c["name"]) == "reproduces"]


def test_the_033_era_is_the_case_set_033_was_written_against():
    import hashlib

    names = sorted(c["name"] for c in ERA_033)
    assert len(names) == ERA_033_COUNT, (
        f"{len(names)} cases are treated as 033's era, which had {ERA_033_COUNT}: a case added to "
        f"probe-identity-cases.json after 033 must be listed in POST_033 with its verdict"
    )
    assert hashlib.sha256("\n".join(names).encode()).hexdigest() == ERA_033_SHA256
    assert set(POST_033) <= {c["name"] for c in CASES}, "a POST_033 entry names no case"
    assert set(POST_033.values()) <= {"reproduces", "differs"}


@pytest.mark.parametrize("case", REPRODUCED, ids=lambda c: c["name"])
def test_the_frozen_copy_reproduces_every_case_of_its_era(case):
    assert m033().digests(case["document"]) == (case["detector_digest"], case["decision_digest"])


@pytest.mark.parametrize(
    "case", [c for c in CASES if POST_033.get(c["name"]) == "differs"], ids=lambda c: c["name"]
)
def test_a_key_added_after_033_is_in_the_live_digest_and_not_the_frozen_one(case):
    """G38: 033 cannot know `aggregation.skip_template_prefix`, and must not — no row stored before G38
    carries it. Its detector digest of a flagged document DIFFERS from the case's; the live digest is
    the case's. Together: the flag is in today's identity and absent from the frozen one."""
    frozen = m033().digests(case["document"])
    assert frozen[0] != case["detector_digest"]
    assert frozen[1] == case["decision_digest"]
    assert digests_of(case["document"]) == (case["detector_digest"], case["decision_digest"])


@pytest.mark.parametrize("edge", EDGES, ids=lambda e: e["name"])
def test_the_frozen_copy_agrees_with_mistudio_and_the_service_on_every_edge(edge):
    got = m033().digests(edge["document"])
    assert got == digests_of(edge["document"])
    if "refused" not in edge:
        assert got == (edge["detector_digest"], edge["decision_digest"]), edge["about"]


def test_the_frozen_copy_refuses_both_halves_where_mistudio_refuses_the_document():
    """Review R1 (6.5), R65-1: the frozen copy is both-or-neither too, document for document."""
    from tests.unit.services.test_probe_identity_cases import _refused_by_mistudio

    for name, document in _refused_by_mistudio():
        assert m033().digests(document) == (None, None), name


def test_032_disagreed_on_the_edges_this_recomputes():
    """The reason 033 exists: S1's frozen reading gives a different digest for each R1 edge."""
    old = m032()
    for name in ("bos-extra-key", "bos-partial", "render-add-special-null", "read-absent"):
        assert old.digests(EDGE[name]["document"])[0] != EDGE[name]["detector_digest"], name


# ── SQLite ───────────────────────────────────────────────────────────────────────


@pytest.fixture
def sqlite_head(tmp_path):
    from miserver.db.base import Base
    import miserver.db.models  # noqa: F401 - every table

    engine = sa.create_engine(f"sqlite:///{tmp_path / 'm.db'}")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def _insert(conn, probe_id: str, definition: dict, det, dec, source: str) -> None:
    conn.execute(sa.text(
        "INSERT INTO probes (id, name, definition, hf_id, d_model, n_layers, layer, rule, "
        "scope, rung, threshold, threshold_revision, basis, streamable, armed, revision, "
        "detector_digest, decision_digest, detector_digest_source, identity_algorithm) "
        "VALUES (:id, :id, :d, 'm', 4, 8, 3, 'mean', 'all', 2, 1.0, 1, 'residual', 1, 0, :rev, "
        ":det, :dec, :src, 'mistudio.probe-identity/1')"
    ), {"id": probe_id, "d": json.dumps(definition), "rev": "0" * 40, "det": det, "dec": dec,
        "src": source})


def test_sqlite_recomputes_what_032_stored_and_nothing_else(sqlite_head, caplog):
    old = m032()
    edge = EDGE["bos-extra-key"]["document"]
    agreeing = CASES[0]["document"]
    nan = json.loads(json.dumps(agreeing))
    nan["aggregation"]["params"] = {"n": 2 ** 53}
    with sqlite_head.begin() as conn:
        _insert(conn, "pr_edge", edge, *old.digests(edge), "computed")
        _insert(conn, "pr_agree", agreeing, *old.digests(agreeing), "computed")
        _insert(conn, "pr_producer", edge, "f" * 64, "e" * 64, "producer")
        _insert(conn, "pr_unsafe", nan, *old.digests(nan), "computed")
    caplog.set_level(logging.INFO, logger="alembic.runtime.migration")
    with sqlite_head.begin() as conn:
        with Operations.context(MigrationContext.configure(conn)):
            m033().upgrade()
    with sqlite_head.connect() as conn:
        rows = {r[0]: r[1:] for r in conn.execute(sa.text(
            "SELECT id, detector_digest, decision_digest, detector_digest_source, "
            "identity_algorithm FROM probes"))}
    assert rows["pr_edge"][:2] == digests_of(edge) == (
        EDGE["bos-extra-key"]["detector_digest"], EDGE["bos-extra-key"]["decision_digest"])
    assert rows["pr_edge"][2] == "computed"
    assert rows["pr_agree"][:2] == digests_of(agreeing)
    assert rows["pr_producer"] == ("f" * 64, "e" * 64, "producer", "mistudio.probe-identity/1"), (
        "a verified producer digest is never recomputed")
    assert rows["pr_unsafe"] == (None, None, "unknown", None)
    line = next(r.getMessage() for r in caplog.records
                if r.getMessage().startswith("probe_identity_recompute"))
    assert "checked=3 moved=2" in line and "pr_edge" in line and "pr_unsafe" in line
    assert "pr_agree" not in line


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
        if flag("REQUIRE_PG") == "1":
            pytest.fail(f"{message} and MISERVER_REQUIRE_PG=1 — this guard did NOT run")
        pytest.skip(f"{message} — a migration round trip needs a real server")
    name = f"probe_identity_{uuid.uuid4().hex[:8]}"
    asyncio.run(_admin(f'CREATE DATABASE "{name}"'))
    try:
        yield name
    finally:
        asyncio.run(_admin(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


def test_pg_stored_digests_equal_the_case_file_algorithm_after_033(pg_db):
    from tests.schema.test_probe_result_contract_migration import _exec

    _admin, _alembic, _pg_available, _query = _pg()
    _alembic(pg_db, "upgrade", "031")
    for probe_id, name in (("pr_00000000000a", "bos-extra-key"),
                           ("pr_00000000000b", "render-add-special-null"),
                           ("pr_00000000000c", "residual")):
        doc = EDGE[name]["document"] if name in EDGE else CASES[0]["document"]
        asyncio.run(_exec(pg_db, (
            "INSERT INTO probes (id, name, definition, hf_id, d_model, n_layers, layer, rule, "
            "scope, rung, threshold, threshold_revision, revision) VALUES (:id, :id, "
            "CAST(:d AS jsonb), 'm', 4, 8, 3, 'mean', 'all', 2, 1.0, 1, :rev)"
        ), {"id": probe_id, "d": json.dumps(doc), "rev": "0" * 40}))
    _alembic(pg_db, "upgrade", "032")
    read = ("SELECT id, definition, detector_digest, decision_digest, detector_digest_source "
            "FROM probes ORDER BY id")
    at_032 = asyncio.run(_query(pg_db, read))
    assert at_032[0][2] != digests_of(at_032[0][1])[0], "032 stored S1's reading of the edge"
    _alembic(pg_db, "upgrade", "033")
    for row in asyncio.run(_query(pg_db, read)):
        assert (row[2], row[3]) == digests_of(row[1]), row[0]
        assert row[4] == "computed"
    assert asyncio.run(_query(pg_db, read))[0][2] == EDGE["bos-extra-key"]["detector_digest"]
    _alembic(pg_db, "downgrade", "032")
    _alembic(pg_db, "upgrade", "head")
