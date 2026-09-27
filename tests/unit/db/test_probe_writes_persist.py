"""Probe writes must survive the request that made them.

⚠ **THIS FILE EXISTS BECAUSE EVERY PROBE WRITE WAS SILENTLY ROLLED BACK IN PRODUCTION.**

`get_db` yields a session and closes it; it does **not** commit. `ProbeRepository` only
called `flush()`, so a write was visible inside its own transaction — which is where every
test looked — and vanished when the request ended.

On the node: `POST /api/probes/import` answered `{"success": true}` with a real probe id, and
`GET /api/probes` returned **zero rows** a second later. Arming then failed
`PROBE_NOT_FOUND` for a probe that had just been created. Imports, the armed flag, parity
reports, acknowledgements and events were all affected, and nothing anywhere reported a
failure.

**No existing test could have caught it.** Unit tests assert inside the transaction that made
the write, where a flush is sufficient and the row is visibly present. The integration tests
use mocked repositories. Catching it needs a SECOND session against the same database, which
is what every test here does — that is the whole design of the file, and the reason it lives
in `tests/unit/db/` rather than beside the service.

`circuit_repository.py` has committed since it was written. This one was the outlier, and the
difference between the two files was invisible to the suite.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from millm.db.repositories.probe_repository import ProbeEventRepository, ProbeRepository


@pytest_asyncio.fixture
async def sessions(test_engine):
    """A factory for INDEPENDENT sessions on one database.

    Each `await sessions()` is a fresh session, as a separate HTTP request would get. A single
    shared session is exactly the blind spot this file exists to cover.
    """
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    opened: list[AsyncSession] = []

    async def make() -> AsyncSession:
        session = factory()
        opened.append(session)
        return session

    yield make
    for session in opened:
        await session.close()


def _fields(**over):
    base = dict(
        id="pr_persist",
        name="persist-me",
        hf_id="LiquidAI/LFM2.5-1.2B-Instruct",
        layer=11,
        rule="mean",
        scope="all",
        basis="residual",
        streamable=True,
        threshold=1.07,
        target_fpr=0.01,
        rung=2,
        armed=False,
        definition={"kind": "mistudio.probe-definition/v1"},
        d_model=2048,
        n_layers=16,
    )
    base.update(over)
    return base


class TestAnImportSurvivesItsRequest:
    @pytest.mark.asyncio
    async def test_a_created_probe_is_visible_to_a_LATER_session(self, sessions):
        """⚠ The exact production failure: import says success, the next request sees nothing."""
        writer = await sessions()
        created = await ProbeRepository(writer).create(**_fields())
        assert created.id == "pr_persist"
        await writer.close()

        reader = await sessions()
        found = await ProbeRepository(reader).get("pr_persist")
        assert found is not None, (
            "the probe was created and returned, and a separate session cannot see it — the "
            "write was rolled back when its request ended, which is what shipped"
        )
        assert found.name == "persist-me"

    @pytest.mark.asyncio
    async def test_it_is_in_the_LIST_a_later_session_reads(self, sessions):
        """`get` by id and `list` are different queries, and the production symptom was an
        empty LIST — `GET /api/probes` returned zero rows."""
        writer = await sessions()
        await ProbeRepository(writer).create(**_fields(id="pr_list"))
        await writer.close()

        reader = await sessions()
        rows = await ProbeRepository(reader).list()
        assert [p.id for p in rows] == ["pr_list"]

    @pytest.mark.asyncio
    async def test_the_armed_flag_survives(self, sessions):
        """Arming that does not persist leaves a probe scoring with no row saying so — and a
        restart then has hooks nobody is tracking, or rows nobody armed."""
        writer = await sessions()
        repo = ProbeRepository(writer)
        probe = await repo.create(**_fields(id="pr_arm"))
        await repo.update(probe, armed=True, paused_reason=None)
        await writer.close()

        reader = await sessions()
        found = await ProbeRepository(reader).get("pr_arm")
        assert found is not None and found.armed is True

    @pytest.mark.asyncio
    async def test_a_parity_report_survives(self, sessions):
        """A refused arm stores its report so an operator can see WHY. A report that does not
        persist makes the refusal unexplainable after the fact."""
        writer = await sessions()
        repo = ProbeRepository(writer)
        probe = await repo.create(**_fields(id="pr_par"))
        await repo.update(probe, parity={"passed": False, "max_abs_diff": 3.2})
        await writer.close()

        reader = await sessions()
        found = await ProbeRepository(reader).get("pr_par")
        assert found is not None
        assert found.parity == {"passed": False, "max_abs_diff": 3.2}

    @pytest.mark.asyncio
    async def test_a_delete_survives(self, sessions):
        """A delete that rolls back is worse than one that fails: the caller is told the probe
        is gone and it is still armed."""
        writer = await sessions()
        repo = ProbeRepository(writer)
        probe = await repo.create(**_fields(id="pr_del"))
        await writer.close()

        deleter = await sessions()
        repo2 = ProbeRepository(deleter)
        await repo2.delete(await repo2.get("pr_del"))
        await deleter.close()

        reader = await sessions()
        assert await ProbeRepository(reader).get("pr_del") is None

    @pytest.mark.asyncio
    async def test_events_survive(self, sessions):
        """An event feed that empties on every request would read as 'nothing detected'."""
        writer = await sessions()
        await ProbeRepository(writer).create(**_fields(id="pr_ev"))
        await ProbeEventRepository(writer).create_many(
            [{"probe_id": "pr_ev", "scored": True, "score": 1.0, "verdict": True, "rung": 2}]
        )
        await writer.close()

        reader = await sessions()
        rows = await ProbeEventRepository(reader).list_events(probe_id="pr_ev", limit=10)
        assert len(rows) == 1
        assert rows[0].probe_id == "pr_ev"


class TestTheRepositoryCommits:
    """The mechanism, asserted directly — so the reason stays legible after a refactor."""

    def test_no_write_path_only_flushes(self):
        """⚠ A `flush()` in a write path is the defect, by construction.

        Asserted on the AST rather than by grep: a comment mentioning `flush` would satisfy a
        text search, and this estate has shipped that mistake in three separate arcs.
        """
        import ast
        from pathlib import Path

        source = Path("millm/db/repositories/probe_repository.py").read_text()
        tree = ast.parse(source)
        flushes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "flush"
        ]
        assert not flushes, (
            f"{len(flushes)} flush() call(s) remain in the probe repository. `get_db` does not "
            "commit, so a flushed write is rolled back when its request ends — silently, with "
            "the route having already returned success."
        )

    def test_every_write_method_commits(self):
        """Presence of `commit`, per method, so adding a write path without one is caught."""
        import ast
        from pathlib import Path

        tree = ast.parse(Path("millm/db/repositories/probe_repository.py").read_text())
        write_methods = {
            "create", "update", "delete", "disarm_all", "create_many", "clear",
            "prune_aged", "prune_to_cap",
        }
        seen: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                continue
            if node.name not in write_methods:
                continue
            commits = any(
                isinstance(c, ast.Call)
                and isinstance(c.func, ast.Attribute)
                and c.func.attr == "commit"
                for c in ast.walk(node)
            )
            if commits:
                seen.add(node.name)
        missing = sorted(write_methods - seen)
        assert not missing, f"write methods with no commit(): {missing}"
