"""The one behaviour selector (039 FR-39.1, FTID §3.1) and the version projection (FTASKS 1.6).

Control: the version check accepts mixed versions → `test_two_versions_are_refused_listing_both`
goes red. The order of refusals is pinned: `INVALID_PROBE_SELECTION` → `CATALOGUE_LIMIT` →
`BEHAVIOUR_VERSION_MIXED`.
"""

from __future__ import annotations

import pytest

from millm.core.config import settings
from millm.core.errors import (
    BehaviourVersionMixedError,
    CatalogueLimitError,
    InvalidProbeSelectionError,
)
from millm.db.repositories.probe_repository import ProbeRepository
from millm.services import probe_selection
from tests.support import behaviour_factory as bf
from tests.unit.score_fixtures import tiny_identity

pytestmark = pytest.mark.asyncio


async def _import(session, key, version, name, **over):
    return await bf.import_probe(session, bf.definition(name, key=key, version=version, **over))


async def _select(session, key, **kw):
    return await probe_selection.select_behaviour(ProbeRepository(session), key,
                                                  loaded=tiny_identity(), **kw)


async def test_import_projects_the_version(test_session):
    key = bf.random_key()
    v3 = await _import(test_session, key, 3, "a")
    uncatalogued = await _import(test_session, key, None, "b")
    assert (v3.behaviour_key, v3.behaviour_version) == (key, 3)
    assert (uncatalogued.behaviour_key, uncatalogued.behaviour_version) == (key, None)


async def test_hub_import_projects_the_version_too(test_session):
    """The Hub route imports through the same `ProbeService.import_definition` (one path)."""
    from unittest.mock import AsyncMock, MagicMock

    from fastapi.testclient import TestClient

    from millm.api.dependencies import get_db, get_probe_hub_service
    from millm.main import create_app

    key = bf.random_key()
    doc = bf.definition("hub-probe", key=key, version=4)
    hub = MagicMock()
    hub.fetch_definition = AsyncMock(return_value=(MagicMock(), doc, {"repo": "r"}))

    async def db():
        yield test_session

    app = create_app()
    app.dependency_overrides[get_db] = db
    app.dependency_overrides[get_probe_hub_service] = lambda: hub
    response = TestClient(app).post("/api/probes/hub/import",
                                    json={"repo_id": "org/probes", "filename": "a.probe.json"})
    assert response.status_code == 200, response.text
    row = await ProbeRepository(test_session).get(response.json()["data"]["id"])
    assert row.behaviour_version == 4


async def test_one_version_selects_without_naming_it(test_session):
    key = bf.random_key()
    rows = [await _import(test_session, key, 2, f"p{i}") for i in range(3)]
    selection = await _select(test_session, key)
    assert selection.version == 2 and selection.probe_ids == sorted(r.id for r in rows)


async def test_two_versions_are_refused_listing_both(test_session):
    key = bf.random_key()
    one = await _import(test_session, key, 1, "v1")
    two = [await _import(test_session, key, 2, f"v2-{i}") for i in range(2)]
    with pytest.raises(BehaviourVersionMixedError) as refused:
        await _select(test_session, key)
    versions = {v["version"]: v["probe_ids"] for v in refused.value.details["versions"]}
    assert versions == {1: [one.id], 2: sorted(r.id for r in two)}
    assert refused.value.next_tool == "millm_arm_behaviour"


async def test_version_narrows_the_set(test_session):
    key = bf.random_key()
    await _import(test_session, key, 1, "v1")
    two = await _import(test_session, key, 2, "v2")
    assert (await _select(test_session, key, version=2)).probe_ids == [two.id]


async def test_version_none_selects_only_the_uncatalogued_version(test_session):
    """OQ-39.3 (a): `null` is a version of its own — and absent is not `null` (the sentinel)."""
    key = bf.random_key()
    await _import(test_session, key, 1, "v1")
    uncatalogued = await _import(test_session, key, None, "null")
    assert (await _select(test_session, key, version=None)).probe_ids == [uncatalogued.id]
    with pytest.raises(BehaviourVersionMixedError):
        await _select(test_session, key)


async def test_probe_ids_narrow_and_a_foreign_id_is_refused(test_session):
    key, other = bf.random_key(), bf.random_key()
    mine = [await _import(test_session, key, 1, f"m{i}") for i in range(2)]
    foreign = await _import(test_session, other, 1, "f")
    assert (await _select(test_session, key, probe_ids=[mine[1].id])).probe_ids == [mine[1].id]
    with pytest.raises(InvalidProbeSelectionError) as refused:
        await _select(test_session, key, probe_ids=[mine[0].id, foreign.id])
    assert refused.value.details["probe_ids"] == [foreign.id]


async def test_a_probe_for_another_model_is_skipped_not_selected(test_session):
    key = bf.random_key()
    mine = await _import(test_session, key, 1, "mine")
    other = await _import(test_session, key, 1, "other", hf_id="other/model")
    selection = await _select(test_session, key)
    assert selection.probe_ids == [mine.id] and selection.skipped_identity == (other.id,)


async def test_the_per_behaviour_limit_refuses_at_limit_plus_one(test_session, monkeypatch):
    monkeypatch.setattr(settings, "PROBE_MAX_PER_BEHAVIOUR", 2)
    key = bf.random_key()
    rows = [await _import(test_session, key, 1, f"p{i}") for i in range(3)]
    with pytest.raises(CatalogueLimitError) as refused:
        await _select(test_session, key)
    assert refused.value.details["limit"] == "PROBE_MAX_PER_BEHAVIOUR"
    # At the limit (not over it) the selection stands.
    narrowed = await _select(test_session, key, probe_ids=[r.id for r in rows[:2]])
    assert len(narrowed.probes) == 2


async def test_the_refusals_come_in_their_order(test_session, monkeypatch):
    """A foreign id outranks the limit, and the limit outranks mixed versions."""
    monkeypatch.setattr(settings, "PROBE_MAX_PER_BEHAVIOUR", 1)
    key, other = bf.random_key(), bf.random_key()
    await _import(test_session, key, 1, "a")
    await _import(test_session, key, 2, "b")
    foreign = await _import(test_session, other, 1, "c")
    with pytest.raises(InvalidProbeSelectionError):
        await _select(test_session, key, probe_ids=[foreign.id])
    with pytest.raises(CatalogueLimitError):
        await _select(test_session, key)


async def test_the_member_map_follows_import_and_delete(test_session):
    from millm.services.probe_service import ProbeService

    key = bf.random_key()
    rows = [await _import(test_session, key, 1, f"p{i}") for i in range(3)]
    assert probe_selection.members_of_group(rows[0].id) == tuple(sorted(r.id for r in rows))
    await ProbeService(ProbeRepository(test_session)).delete(rows[2].id)
    assert probe_selection.members_of_group(rows[0].id) == tuple(sorted(r.id for r in rows[:2]))
