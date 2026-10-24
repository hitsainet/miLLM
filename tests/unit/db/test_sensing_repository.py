"""
Feature 11 Task 1.3: SensingRepository unit tests — CRUD, retention
(cap + age prune), and CASCADE with profile deletion.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from millm.db.models.profile import Profile
from millm.db.models.sensing_event import SensingEvent
from millm.db.repositories.sensing_repository import SensingRepository


@pytest.fixture
async def profile(test_session):
    row = Profile(
        id="prof_sense01",
        name="sensed cluster",
        steering={"7": 50.0},
        source_kind="cluster",
        sensing_enabled=True,
    )
    test_session.add(row)
    await test_session.flush()
    return row


@pytest.fixture
def repo(test_session):
    return SensingRepository(test_session)


def make_event(profile_id="prof_sense01", **overrides):
    base = dict(
        profile_id=profile_id,
        request_id="req-1",
        phase="decode",
        pos_start=10,
        pos_end=12,
        fired_members=[[7, 4.2], [9, 3.1]],
        fired_count=2,
        score=2.5,
        summary="fear: 2/3 members fired",
    )
    base.update(overrides)
    return base


class TestCrud:
    async def test_create_many_and_list_newest_first(self, repo, profile):
        await repo.create_many([
            make_event(request_id="req-1"),
            make_event(request_id="req-2"),
        ])
        events = await repo.list_events(profile_id=profile.id)
        assert len(events) == 2
        # newest first (same timestamp resolution -> id desc breaks the tie)
        assert events[0].id > events[1].id

    async def test_list_filters_by_profile(self, repo, profile, test_session):
        other = Profile(id="prof_other01", name="other", steering={})
        test_session.add(other)
        await test_session.flush()
        await repo.create_many([
            make_event(),
            make_event(profile_id="prof_other01"),
        ])
        assert len(await repo.list_events(profile_id=profile.id)) == 1
        assert len(await repo.list_events()) == 2

    async def test_context_round_trips(self, repo, profile):
        rows = await repo.create_many([make_event(
            context_text="the deep ocean current",
            context_token_ids=[101, 202, 303],
        )])
        row = await repo.get(rows[0].id)
        assert row.context_text == "the deep ocean current"
        assert row.context_token_ids == [101, 202, 303]

    async def test_clear_scoped_and_global(self, repo, profile, test_session):
        other = Profile(id="prof_other02", name="other2", steering={})
        test_session.add(other)
        await test_session.flush()
        await repo.create_many([make_event(),
                                make_event(profile_id="prof_other02")])
        assert await repo.clear(profile_id=profile.id) == 1
        assert await repo.count() == 1
        assert await repo.clear() == 1
        assert await repo.count() == 0


class TestRetention:
    async def test_prune_enforces_cap_keeping_newest(self, repo, profile,
                                                     test_session):
        await repo.create_many([make_event(request_id=f"req-{i}")
                                for i in range(10)])
        deleted = await repo.prune(profile.id, cap=4, max_age_days=7)
        assert deleted == 6
        remaining = await repo.list_events(profile_id=profile.id, limit=100)
        assert len(remaining) == 4
        # the survivors are the newest ids
        ids = sorted(e.id for e in remaining)
        assert ids == sorted(ids, reverse=False)
        assert min(ids) > 6 - 1  # oldest 6 gone

    async def test_prune_drops_aged_rows(self, repo, profile, test_session):
        rows = await repo.create_many([make_event(), make_event()])
        old = datetime.now(timezone.utc) - timedelta(days=30)
        rows[0].created_at = old
        await test_session.flush()
        deleted = await repo.prune(profile.id, cap=100, max_age_days=7)
        assert deleted == 1
        assert await repo.count(profile_id=profile.id) == 1

    async def test_prune_scoped_to_profile(self, repo, profile, test_session):
        other = Profile(id="prof_other03", name="other3", steering={})
        test_session.add(other)
        await test_session.flush()
        await repo.create_many([make_event(request_id=f"req-{i}")
                                for i in range(5)])
        await repo.create_many([make_event(profile_id="prof_other03")])
        await repo.prune(profile.id, cap=2, max_age_days=7)
        assert await repo.count(profile_id=profile.id) == 2
        assert await repo.count(profile_id="prof_other03") == 1


class TestCascade:
    async def test_profile_delete_cascades_events(self, repo, profile,
                                                  test_session):
        """FTID pitfall 7: deleting a cluster profile removes its events."""
        await repo.create_many([make_event(), make_event()])
        await test_session.flush()
        # Delete through the ORM — the path ProfileRepository.delete uses.
        # The ORM cascade covers SQLite (FK pragma off); postgres also has
        # the migration's FK ondelete=CASCADE for any non-ORM delete.
        await test_session.delete(profile)
        await test_session.flush()
        result = await test_session.execute(select(SensingEvent))
        assert result.scalars().all() == []


async def test_context_parts_round_trips(repo, profile):
    """Enh R1: the highlight segments must survive the JSON column."""
    rows = await repo.create_many([make_event(
        context_parts={"before": "the deep ", "span": "ocean",
                       "after": " current"},
    )])
    row = await repo.get(rows[0].id)
    assert row.context_parts == {"before": "the deep ", "span": "ocean",
                                 "after": " current"}
    payload = row.to_dict(include_context=True)
    assert payload["context_parts"]["span"] == "ocean"
    # old rows (pre-migration-010) carry None and must not break to_dict
    old_rows = await repo.create_many([make_event()])
    assert old_rows[0].to_dict()["context_parts"] is None
